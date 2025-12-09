import os
import json
import torch
import time
from pathlib import Path
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)
from tqdm import tqdm
from datetime import datetime

class Config:
    model_path = "./weights/LongT2IBench-checkpoints"
    data_dir = "./data/split"
    img_root = "./data" 
    test_file = "test.json"
    output_dir = "./generation_results"
    
    max_new_tokens = 1024
    temperature = 0.6
    top_p = 0.9
    do_sample = True
    
    max_samples = None
    batch_size = 1
    level_token = "<level>"
    
    max_retries = 3
    retry_delay = 2

config = Config()
 
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

def load_model():
    print(f"Loading model from {config.model_path}...")
    
    processor = Qwen2VLProcessor.from_pretrained(config.model_path, trust_remote_code=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("Model loaded")
    return model, processor

def generate_batch_text(model, processor, images, texts, retry=0):
    try:
        batched_messages = []
        for text, image in zip(texts, images):
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Text:{text}{SCORE_QUESTION_PROMPT}"},
                        {"type": "image", "image": image},
                    ],
                },
            ]
            batched_messages.append(messages)
        
        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]
        
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        
        generated_texts = []
        for i in range(len(generated_ids)):
            single_generated_ids = generated_ids[i, inputs.input_ids[i].shape[0]:]
            raw_text = processor.tokenizer.decode(single_generated_ids, skip_special_tokens=True)
            
            start_keyword = "all entity:"
            start_index = raw_text.find(start_keyword)
            
            if start_index != -1:
                generated_text = raw_text[start_index:]
            else:
                generated_text = raw_text
                
            generated_text = generated_text.strip()
            
            generated_texts.append(generated_text)
            
            print(f"Generated text for batch {i}: {generated_text[:300]}")
        
        return generated_texts
    
    except Exception as e:
        if retry < config.max_retries:
            print(f"{retry+1}/{config.max_retries} retry: {e}")
            time.sleep(config.retry_delay)
            return generate_batch_text(model, processor, images, texts, retry+1)
        else:
            print(f"Retry {config.max_retries} times: {e}")
            raise

def main():
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    model, processor = load_model()
    model.eval()
    
    test_file = Path(config.data_dir) / config.test_file
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if config.max_samples:
        test_data = test_data[:config.max_samples]
    
    print(f"Testing on {len(test_data)} samples with batch size {config.batch_size}...")
    
    results = []
    
    for i in tqdm(range(0, len(test_data), config.batch_size), desc="Processing batches"):
        batch_items = test_data[i : i + config.batch_size]
        
        batch_texts = []
        batch_images = []
        batch_gt_scores = []
        batch_img_paths = []
        batch_indices = []

        for item in batch_items:
            try:
                text = item["en_prompt_text"]
                img_path = Path(config.img_root) / item["img_path"]
                image = Image.open(img_path).convert("RGB")
                gt_score = float(item["Align ratio"])
                original_index = test_data.index(item)
                
                batch_texts.append(text)
                batch_images.append(image)
                batch_gt_scores.append(gt_score)
                batch_img_paths.append(str(img_path))
                batch_indices.append(original_index)
            except Exception as e:
                print(f"Error loading sample at index {test_data.index(item)}: {e}")
                results.append({
                    "id": test_data.index(item),
                    "image_path": str(img_path) if 'img_path' in locals() else "unknown",
                    "input_text": text if 'text' in locals() else "unknown",
                    "generated_text": None,
                    "ground_truth_score": None,
                    "success": False,
                    "error": str(e),
                    "retries": 0
                })
                continue

        if not batch_texts:
            continue
        
        retry_count = 0
        while retry_count <= config.max_retries:
            try:
                generated_texts = generate_batch_text(model, processor, batch_images, batch_texts)
                for j, (text, img_path, gt_score, generated_text, original_index) in enumerate(
                    zip(batch_texts, batch_img_paths, batch_gt_scores, generated_texts, batch_indices)
                ):
                    result = {
                        "id": original_index,
                        "image_path": img_path,
                        "input_text": text,
                        "generated_text": generated_text,
                        "ground_truth_score": gt_score,
                        "success": True,
                        "retries": retry_count
                    }
                    results.append(result)
                
                break
                
            except Exception as e:
                retry_count += 1
                if retry_count <= config.max_retries:
                    print(f"{retry_count}/{config.max_retries} retry: {e}")
                    time.sleep(config.retry_delay)
                else:
                    print(f"Retry {config.max_retries} times: {e}")
                    for j, (text, img_path, gt_score, original_index) in enumerate(
                        zip(batch_texts, batch_img_paths, batch_gt_scores, batch_indices)
                    ):
                        results.append({
                            "id": original_index,
                            "image_path": img_path,
                            "input_text": text,
                            "generated_text": None,
                            "ground_truth_score": gt_score,
                            "success": False,
                            "error": str(e),
                            "retries": config.max_retries
                        })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(config.output_dir) / f"generation_results_{timestamp}.json"
    
    final_results = {
        "config": {
            "model_path": config.model_path,
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "batch_size": config.batch_size,
            "max_retries": config.max_retries,
            "retry_delay": config.retry_delay,
            "timestamp": timestamp
        },
        "summary": {
            "total_samples": len(test_data),
            "successful": len([r for r in results if r["success"]]),
            "failed": len([r for r in results if not r["success"]]),
            "retried_samples": len([r for r in results if r.get("retries", 0) > 0 and r["success"]]),
        },
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_file}")
    print(f"Summary: {final_results['summary']['successful']}/{final_results['summary']['total_samples']} successful")
    print(f"Retried samples that succeeded: {final_results['summary']['retried_samples']}")

if __name__ == "__main__":
    main()