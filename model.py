import os
import json
import logging
import torch
import torch.nn as nn
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)
from peft import LoraConfig, get_peft_model, TaskType
from pathlib import Path

class ScoreHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def initialize_model(config):
    processor = Qwen2VLProcessor.from_pretrained(
        config.model_name, max_pixels=512 * 512 * 3, trust_remote_code=True
    )
    num_added = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [config.level_token]}
    )
    level_id = processor.tokenizer.convert_tokens_to_ids(config.level_token)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if num_added > 0:
        model.resize_token_embeddings(len(processor.tokenizer))

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_cfg)

    score_head = ScoreHead(model.config.hidden_size).to(device)
    score_head.to(torch.bfloat16)

    return model, processor, score_head

def load_model(save_dir: str = "./weights/final"):
    save_path = Path(save_dir)

    processor = Qwen2VLProcessor.from_pretrained(save_path, trust_remote_code=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        save_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    score_head = ScoreHead(model.config.hidden_size).to(model.device).to(torch.bfloat16)
    score_head.load_state_dict(torch.load(Path(save_dir) / "score_head.pt", map_location=model.device))

    print(f"Model loaded from {save_path}")

    return model, processor, score_head

def save_model(model, processor, score_head, save_dir: str = "./weights/final"):
    if hasattr(model, 'merge_and_unload'):
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(save_dir)
    else:
        model.save_pretrained(save_dir)
        
    processor.save_pretrained(save_dir)
    torch.save(score_head.state_dict(), Path(save_dir) / "score_head.pt")