import os
import json
import logging
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import wandb
from scipy.stats import pearsonr, spearmanr
from PIL import ImageFile
import math
ImageFile.LOAD_TRUNCATED_IMAGES = True

mp.set_start_method("spawn", force=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class Config:
    model_name = "./weights/Qwen2.5-VL-7B-Instruct"
    data_dir   = "./data"
    img_root   = "./data/image_prompts_1st"
    output_dir = "./weights/LongT2IBench-checkpoints"
    train_file = "train.json"
    num_epochs = 1
    batch_size = 2
    grad_accum = 8

    lora_lr = 2e-4
    score_head_lr = 1e-4
    
   
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.1
    
 
    warmup_ratio = 0.1
    max_len = 4096
    eval_steps = 100
    level_token = "<level>"
    
 
    weight_decay = 0.01
    max_grad_norm = 1.0
    use_fp16 = False
    

    resume_from_checkpoint = None  
    ignore_mismatched_sizes = True  

config = Config()
Path(config.output_dir).mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SYSTEM_PROMPT = (
    "You are an expert evaluator specializing in assessing the alignment between a generated image and its corresponding text prompt. "
    "The ideal scenario of alignment occurs when every element present in the text is accurately represented within the image."
    "Please output a single token <level> followed by a JSON explanation."
)

SCORE_QUESTION_PROMPT = (
    """\nAnalyze the given image and text, and evaluate their alignment using a hierarchical approach:

    1. Entity Alignment Analysis: First, identify all entities mentioned in the text. 

    2. For each entity, check if they appear in the image. Determine if it is present (align) or absent (misalign).
    - An entity is considered "align" ONLY if it is visibly present in the image
    - An entity is considered "misalign" if it is mentioned in the text but cannot be seen in the image

    3. Attribute Alignment Analysis: For each entity that IS present in the image, evaluate whether its attributes match the text description.
    - IMPORTANT: If an entity is classified as "misalign" in the entity_score, ALL its attributes must automatically be classified as "misalign" in the attribute_score
    - Example: If "a red cat" is described but no cat appears in the image, the attribute "red" must be listed in misalign section

    4. Relation Alignment Analysis: For each relation between entities mentioned in the text, evaluate whether the relation is correctly depicted in the image.
    - IMPORTANT: If ANY entity in a relation is classified as "misalign" in the entity_score, the ENTIRE relation must automatically be classified as "misalign" in the relation_score
    - Example: If "a cat sitting on a chair" is described but there's no chair in the image (chair is "misalign"), the relation "sitting on" must be listed in misalign section

    Format your answer exactly as follows:
    '<level> all entity:
    [comma-separated list of all entities]

    entity_score:
      align: [list of aligned entities]
      misalign: [list of misaligned entities]

    attribute_score:
      align: [[attributes]entity, [attributes]entity, ...]
      misalign: [[attributes]entity, [attributes]entity, ...]

    relation_score:
      align: [entity1[relation]entity2, entity1[relation]entity2, ...]
      misalign: [entity1[relation]entity2, entity1[relation]entity2, ...]'
    """
)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_model_safely():
   
    processor = Qwen2VLProcessor.from_pretrained(
        config.model_name, 
        max_pixels=1024 * 1024 * 3,
        trust_remote_code=True
    )
    
   
    num_added = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": [config.level_token]}
    )
    level_id = processor.tokenizer.convert_tokens_to_ids(config.level_token)
    
    print(f"Added {num_added} special tokens. Level token ID: {level_id}")
    

    torch_dtype = torch.float16 if config.use_fp16 else torch.bfloat16
    
    try:
       
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            ignore_mismatched_sizes=config.ignore_mismatched_sizes
        )
        print("⚠️ Model loaded with ignored mismatched sizes")
    
  
    if num_added > 0:
        old_vocab_size = model.get_input_embeddings().weight.shape[0]
        print(f"Original vocab size: {old_vocab_size}")
        
    
        try:
            model.resize_token_embeddings(len(processor.tokenizer))
            new_vocab_size = model.get_input_embeddings().weight.shape[0]
            print(f"Resized embeddings to: {new_vocab_size}")
        except Exception as e:
            print(f"Error resizing embeddings: {e}")
            model = resize_embeddings_manually(model, len(processor.tokenizer))
    
    return model, processor, level_id

def resize_embeddings_manually(model, new_vocab_size):

    try:
      
        old_embeddings = model.get_input_embeddings()
        old_vocab_size, embedding_dim = old_embeddings.weight.shape
        
   
        new_embeddings = nn.Embedding(new_vocab_size, embedding_dim, dtype=old_embeddings.weight.dtype)
        
    
        with torch.no_grad():
            new_embeddings.weight[:old_vocab_size] = old_embeddings.weight
            if new_vocab_size > old_vocab_size:
                new_embeddings.weight[old_vocab_size:].normal_(mean=0.0, std=0.02)
        

        model.set_input_embeddings(new_embeddings)
        

        if hasattr(model, 'lm_head') and model.lm_head is not None:
            old_lm_head = model.lm_head
            new_lm_head = nn.Linear(embedding_dim, new_vocab_size, bias=False, dtype=old_lm_head.weight.dtype)
            
            with torch.no_grad():
                new_lm_head.weight[:old_vocab_size] = old_lm_head.weight
                if new_vocab_size > old_vocab_size:
                    new_lm_head.weight[old_vocab_size:].normal_(mean=0.0, std=0.02)
            
            model.lm_head = new_lm_head
        
        print(f"Manually resized embeddings: {old_vocab_size} -> {new_vocab_size}")
        return model
        
    except Exception as e:
        print(f"Failed to manually resize embeddings: {e}")
        return model


model, processor, level_id = load_model_safely()


peft_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model.gradient_checkpointing_enable()
model = get_peft_model(model, peft_cfg)


model.print_trainable_parameters()

# ---------------- Score Head ----------------
class ScoreHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

score_head = ScoreHead(model.config.hidden_size).to(device)
score_head.to(torch.float16 if config.use_fp16 else torch.bfloat16)

# ---------------- Dataset ----------------
class AlignDataset(Dataset):
    def __init__(self, split="cleaned_merged_data_wjj3_train.json"):
        with open(Path(config.data_dir) / split, encoding="utf-8") as f:
            raw_data = json.load(f)
        
        self.data = []
       
        for item in raw_data:
            struct = item.get("en_structured_data", {})
            
        
            if "entity_scores" in struct and "entity_alignment" not in struct:
            
                e_scores = struct.get("entity_scores", {})
                e_align = [k for k, v in e_scores.items() if v == 1]
                e_misalign = [k for k, v in e_scores.items() if v == 0]
                struct["entity_alignment"] = {"align": e_align, "misalign": e_misalign}
                
             
                a_scores = struct.get("attribute_scores", {})
                a_align_list = []
                a_misalign_list = []
                for entity, attrs in a_scores.items():
                  
                    valid_attrs = [k for k, v in attrs.items() if v == 1]
                    if valid_attrs:
                        a_align_list.append(f"[{', '.join(valid_attrs)}]{entity}")
                    
            
                    invalid_attrs = [k for k, v in attrs.items() if v == 0]
                    if invalid_attrs:
                        a_misalign_list.append(f"[{', '.join(invalid_attrs)}]{entity}")
                struct["attribute_alignment"] = {"align": a_align_list, "misalign": a_misalign_list}

              
                r_scores = struct.get("relation_scores", [])
                r_align_list = []
                r_misalign_list = []
                for r in r_scores:
                    rel_str = f"{r['entity1']}[{r['relation']}]{r['entity2']}"
                    if r.get("score") == 1:
                        r_align_list.append(rel_str)
                    else:
                        r_misalign_list.append(rel_str)
                struct["relation_alignment"] = {"align": r_align_list, "misalign": r_misalign_list}
            
            self.data.append(item)
        # -------------------------------------------------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["en_prompt_text"]
        img_path = Path(config.img_root) / item["img_path"]
        image = Image.open(img_path).convert("RGB")
        score = float(item["Align ratio"])
        
        entity_all = "all entity:\n "+ ",".join(item['en_structured_data']["entities"])
        entity_text = "entity_score:\n  align: [" + ", ".join(item['en_structured_data']["entity_alignment"]["align"]) + "]\n"
        entity_text += "  misalign: [" + ", ".join(item['en_structured_data']["entity_alignment"]["misalign"]) + "]"
        
        attribute_text = "attribute_score:\n  align: [" + ", ".join(item['en_structured_data']["attribute_alignment"]["align"]) + "]\n"
        attribute_text += "  misalign: [" + ", ".join(item['en_structured_data']["attribute_alignment"]["misalign"]) + "]"
        
        relation_text = "relation_score:\n  align: [" + ", ".join(item['en_structured_data']["relation_alignment"]["align"]) + "]\n"
        relation_text += "  misalign: [" + ", ".join(item['en_structured_data']["relation_alignment"]["misalign"]) + "]"
        
        full_text = entity_all +"\n\n" + entity_text + "\n\n" + attribute_text + "\n\n" + relation_text
        answer = full_text
        
        return {"image": image, "text": text, "answer": answer, "score": score}

def collate_fn(batch):
    images = [b["image"] for b in batch]
    texts = [b["text"] for b in batch]
    answers = [b["answer"] for b in batch]
    scores = torch.tensor([b["score"] for b in batch], dtype=torch.float32)

    prompt_msgs = []
    for txt, img in zip(texts, images):
        msg = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Text:{txt}{SCORE_QUESTION_PROMPT}"},
                    {"type": "image", "image": img},
                ],
            },
        ]
        prompt_msgs.append(msg)

    full_msgs = []
    for m, ans in zip(prompt_msgs, answers):
        if not ans.startswith(config.level_token):
            ans = f"{config.level_token} {ans.replace(config.level_token, '').strip()}"
        
        full = m + [
            {"role": "assistant", "content": [{"type": "text", "text": ans}]}
        ]
        full_msgs.append(full)

    full_texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in full_msgs
    ]
    
    full_inputs = processor(
        text=full_texts,
        images=[m[1]["content"][1]["image"] for m in full_msgs],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.max_len,
    ).to(device)

    labels = full_inputs.input_ids.clone()
    
    assistant_start_positions = []
    for i, full_text in enumerate(full_texts):
        assistant_marker = "<|im_start|>assistant"
        assistant_pos = full_text.find(assistant_marker)
        if assistant_pos != -1:
            content_start = assistant_pos + len(assistant_marker)
            tokens_before = processor.tokenizer(full_text[:content_start], add_special_tokens=False).input_ids
            assistant_start_positions.append(len(tokens_before))
        else:
            assistant_start_positions.append(full_inputs.input_ids.size(1) // 2)
    
    for i, start_pos in enumerate(assistant_start_positions):
        labels[i, :start_pos] = -100
    
    return {
        "input_ids": full_inputs.input_ids,
        "attention_mask": full_inputs.attention_mask,
        "pixel_values": full_inputs.pixel_values,
        "image_grid_thw": full_inputs.image_grid_thw,
        "labels": labels,
        "scores": scores,
    }


train_ds = AlignDataset(config.train_file)
train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)


lora_optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.lora_lr,
    weight_decay=config.weight_decay,
    betas=(0.9, 0.95)
)

score_optimizer = torch.optim.AdamW(
    score_head.parameters(),
    lr=config.score_head_lr,
    weight_decay=config.weight_decay,
    betas=(0.9, 0.95)
)

total_steps = len(train_dl) * config.num_epochs // config.grad_accum
warmup_steps = int(total_steps * config.warmup_ratio)

lora_scheduler = get_cosine_schedule_with_warmup(lora_optimizer, warmup_steps, total_steps)
score_scheduler = get_cosine_schedule_with_warmup(score_optimizer, warmup_steps, total_steps)

print(f"Training Setup:")
print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")
print(f"LoRA LR: {config.lora_lr:.2e}, Score LR: {config.score_head_lr:.2e}")


def save_model_safely(model, processor, score_head, save_dir: str):
   
    os.makedirs(save_dir, exist_ok=True)
    
    try:
    
        if hasattr(model, 'merge_and_unload'):
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(save_dir)
        else:
            model.save_pretrained(save_dir)
            
        processor.save_pretrained(save_dir)
        torch.save(score_head.state_dict(), Path(save_dir) / "score_head.pt")
        
 
        config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
        with open(Path(save_dir) / "training_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        print(f"Model saved successfully to {save_dir}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        backup_dir = save_dir + "_backup"
        try:
            os.makedirs(backup_dir, exist_ok=True)
            torch.save(model.state_dict(), Path(backup_dir) / "model_state.pt")
            torch.save(score_head.state_dict(), Path(backup_dir) / "score_head.pt")
            print(f"Saved backup to {backup_dir}")
        except Exception as backup_e:
            print(f"Backup save also failed: {backup_e}")

def load_checkpoint_safely(checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("No checkpoint to resume from")
        return None
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None


def find_level_position(seq):
    idx = (seq == level_id).nonzero(as_tuple=True)[0]
    return idx[-1].item() if idx.numel() else -1


def train_epoch(epoch):
    model.train()
    score_head.train()
    
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch+1}/{config.num_epochs}")):
        
        try:
    
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
                labels=batch["labels"],
                output_hidden_states=True,
            )

            lm_loss = outputs.loss
            last_hidden = outputs.hidden_states[-1]

     
            pos_list, h_list = [], []
            for i, lbl in enumerate(batch["labels"]):
                pos = find_level_position(lbl)
                if pos != -1:
                    pos_list.append(pos)
                    h_list.append(last_hidden[i, pos])

            if h_list:
                h = torch.stack(h_list)
                pred_scores = score_head(h)
                tgt_scores = batch["scores"][: len(h)].to(device).to(torch.float16 if config.use_fp16 else torch.bfloat16)
                score_loss = nn.functional.mse_loss(pred_scores, tgt_scores)
            else:
                score_loss = torch.tensor(0.0, device=device)

    
            alpha = 10.0
            loss = lm_loss + alpha * score_loss
            
        
            if torch.isnan(loss):
                print(f"⚠️ NaN loss detected at step {step}, skipping...")
                continue
            
     
            loss.backward()
            
            if (step + 1) % config.grad_accum == 0:
          
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(score_head.parameters(), max_norm=config.max_grad_norm)
                
       
                lora_optimizer.step()
                score_optimizer.step()
                lora_scheduler.step()
                score_scheduler.step()
                
                lora_optimizer.zero_grad()
                score_optimizer.zero_grad()

            total_loss += loss.item()
            
        except Exception as e:
            print(f"⚠️ Error at step {step}: {e}")
    
            lora_optimizer.zero_grad()
            score_optimizer.zero_grad()
            continue
        
  
        if (step + 1) % config.eval_steps == 0:
            current_lora_lr = lora_scheduler.get_last_lr()[0]
            current_score_lr = score_scheduler.get_last_lr()[0]
            
            logger.info(
                f"Step {step+1}: loss={total_loss/(step+1):.4f}, "
                f"lm_loss={lm_loss.item():.4f}, score_loss={score_loss.item():.4f}, "
                f"lora_lr={current_lora_lr:.2e}, score_lr={current_score_lr:.2e}"
            )


        if step > 0 and step % 7000 == 0:
            try:
              
                save_model_safely(model, processor, score_head, f"{config.output_dir}/checkpoint_step_{step}")
            except Exception as e:
                print(f"⚠️ Error during checkpoint save: {e}")
            
    return total_loss / len(train_dl) if len(train_dl) > 0 else 0.0

def evaluate(epoch):
    pass

# ---------------- Main ----------------
if __name__ == "__main__":
    try:
        wandb.init(project="qwen-vl-align", name="qwen-vl-lora-safe")
        

        if config.resume_from_checkpoint:
            checkpoint = load_checkpoint_safely(config.resume_from_checkpoint)
     

        for epoch in range(config.num_epochs):
            try:
                train_loss = train_epoch(epoch)
           
            
                logger.info(f"Epoch {epoch+1} completed. Saving model...")
                save_model_safely(model, processor, score_head, f"{config.output_dir}/epoch_{epoch+1}")
                
                logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.4f}")
                
            except Exception as e:
                logger.error(f"Error in epoch {epoch+1}: {e}")
                continue
                
        logger.info(f"Training completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        try:
            save_model_safely(model, processor, score_head, f"{config.output_dir}/emergency_save")
        except:
            pass