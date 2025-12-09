import os
import json
import logging
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Setup ---
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    data_dir   = "./data/split"
    img_root   = "./data"
    trained_model_path = "./weights/LongT2IBench-checkpoints"
    output_dir = "./score_results"
    batch_size = 1
    max_len = 4096
    level_token = "<level>" 
    use_fp16 = False 
    test_file = "test.json" 

config = Config()
Path(config.output_dir).mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Prompts ---
SYSTEM_PROMPT = (
    "You are an expert evaluator specializing in assessing the alignment between a generated image and its corresponding text prompt. "
    "The ideal scenario of alignment occurs when every element present in the text is accurately represented within the image."
)

SCORE_QUESTION_PROMPT = (
"""I will give you a text description and an image. You need to analyze their alignment relationship by breaking down the text into fine-grained meaningful components.

    Text: "{text}"

    CRITICAL REQUIREMENTS:
    1. Break down the text into individual meaningful words and short phrases
    2. EXCLUDE: articles (the), prepositions (in, on, at, with, of, etc.), conjunctions (and, or, but), and auxiliary verbs (is, are, was, were)
    3. INCLUDE: nouns, adjectives, verbs, adverbs, and meaningful compound phrases
    4. For compound concepts, analyze BOTH the whole phrase AND individual meaningful parts:
    - "water's edge" → [water's edge], [water], [edge] 
    - "golden light" → [golden light], [golden], [light]
    - "gently crashing" → [gently crashing], [gently], [crashing]

    5. Be VERY granular - split into the finest meaningful components possible
    6. Each component should be analyzed on a separate line
    7. Format: **[component] is align** or **[component] is not align**

    EXAMPLE of required granularity:
    Text: "A serene beach at sunset with waves gently crashing"
    Expected output:
    **[serene] is align**
    **[beach] is align**
    **[sunset] is align**
    **[waves] is align**
    **[gently] is not align**
    **[crashing] is not align**
    **[gently crashing] is not align**

    RESPOND WITH ONLY THE ALIGNMENT RESULTS - NO EXPLANATIONS, NO OTHER TEXT."""
)

# --- Score Head ---
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

# --- Model Loading ---
def load_model_for_inference():
    save_path = Path(config.trained_model_path)

    processor = Qwen2VLProcessor.from_pretrained(
        save_path, 
        max_pixels=1024 * 1024 * 3,
        trust_remote_code=True
    )
    
    level_id = processor.tokenizer.convert_tokens_to_ids(config.level_token)
    if level_id is None:
        raise ValueError(f"Special token '{config.level_token}' not found in tokenizer.")
    print(f"Loaded token ID for '{config.level_token}': {level_id}")

    torch_dtype = torch.float16 if config.use_fp16 else torch.bfloat16
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            save_path, 
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    model.eval()

    score_head = ScoreHead(model.config.hidden_size).to(device)
    score_head.to(torch_dtype) 
    score_head_path = Path(config.trained_model_path) / "score_head.pt"
    
    if score_head_path.exists():
        score_head.load_state_dict(torch.load(score_head_path, map_location=device))
        print(f"Score Head loaded from {score_head_path}")
    else:
        raise FileNotFoundError(f"Score Head not found at {score_head_path}")
    
    score_head.eval()

    return model, processor, level_id, score_head

# --- Dataset (Modified as requested) ---
class AlignDataset(Dataset):
    def __init__(self, split_file): 
        file_path = Path(config.data_dir) / split_file
        with open(file_path, encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        text = item["en_prompt_text"]
        
        img_path = Path(config.img_root) / item["img_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}. Creating blank image.")
            image = Image.new('RGB', (224, 224), color='black')

        score = float(item.get("Align ratio", 0.0))
        
        answer = config.level_token
        
        return {"image": image, "text": text, "answer": answer, "score": score}

def collate_fn(batch, processor_arg): 
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
                    {"type": "text", "text": SCORE_QUESTION_PROMPT.format(text=txt)},
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
        processor_arg.apply_chat_template(m, tokenize=False, add_generation_prompt=False) 
        for m in full_msgs
    ]
    
    full_inputs = processor_arg( 
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
            tokens_before = processor_arg.tokenizer(full_text[:content_start], add_special_tokens=False).input_ids
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
        "original_texts": texts 
    }

def find_level_position(seq, level_id_val):
    idx = (seq == level_id_val).nonzero(as_tuple=True)[0]
    return idx[-1].item() if idx.numel() else -1

# --- Inference Loop ---
def test_model():
    model, processor, level_id, score_head = load_model_for_inference()
    
    val_ds = AlignDataset(config.test_file) 
    
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, 
                        collate_fn=lambda b: collate_fn(b, processor), 
                        num_workers=0)

    preds, refs = [], []
    all_results = [] 

    logger.info(f"Starting evaluation on {config.test_file}...")

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dl, desc="Evaluating")):
            try:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    image_grid_thw=batch["image_grid_thw"],
                    output_hidden_states=True, 
                )
                
                last_hidden = outputs.hidden_states[-1]

                h_list = []
                valid_indices = []

                for j, lbl in enumerate(batch["labels"]):
                    pos = find_level_position(lbl, level_id) 
                    if pos != -1:
                        h_list.append(last_hidden[j, pos])
                        valid_indices.append(j)
                    else:
                        logger.warning(f"Batch {i}: <level> token not found.")
                
                if h_list:
                    h = torch.stack(h_list)
                    pred_scores = score_head(h)
                    
                    current_batch_refs = batch["scores"][valid_indices]
                    
                    batch_preds = pred_scores.cpu().tolist()
                    batch_refs = current_batch_refs.tolist()
                    
                    preds.extend(batch_preds)
                    refs.extend(batch_refs)

                    for k_idx, batch_idx in enumerate(valid_indices):
                        all_results.append({
                            "original_text": batch["original_texts"][batch_idx],
                            "true_score": batch_refs[k_idx],
                            "predicted_score": batch_preds[k_idx]
                        })
                    
            except Exception as e:
                logger.error(f"Error in batch {i}: {e}")
                continue
    
    if len(preds) > 1 and len(refs) > 1:
        srcc, _ = spearmanr(preds, refs)
        plcc, _ = pearsonr(preds, refs)
        logger.info(f"\n--- Evaluation Results ---")
        logger.info(f"Samples: {len(preds)}")
        logger.info(f"SRCC: {srcc:.4f}")
        logger.info(f"PLCC: {plcc:.4f}")
    else:
        srcc, plcc = 0.0, 0.0
        logger.warning("Not enough samples.")

    output_filepath = Path(config.output_dir) / "prediction_results.json"
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    return srcc, plcc

if __name__ == "__main__":
    test_model()