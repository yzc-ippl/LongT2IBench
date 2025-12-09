import os
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class AlignDataset(Dataset):
    def __init__(self, split="cleaned_merged_data_wjj3_train.json", data_dir="./LongT2IBench", img_root="./LongT2IBench/image_prompts_1st"):
        with open(Path(data_dir) / split, encoding="utf-8") as f:
            self.data = json.load(f)
        self.img_root = img_root

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["en_prompt_text"]
        img_path = Path(self.img_root) / item["img_path"]
        image = Image.open(img_path).convert("RGB")
        score = float(item["Align ratio"])
        entity_all = "all entity:\n " + ",".join(item['en_structured_data']["entities"])
        
        entity_text = "entity_score:\n  align: [" + ", ".join(item['en_structured_data']["entity_alignment"]["align"]) + "]\n"
        entity_text += "  misalign: [" + ", ".join(item['en_structured_data']["entity_alignment"]["misalign"]) + "]"
        
        attribute_text = "attribute_score:\n  align: [" + ", ".join(item['en_structured_data']["attribute_alignment"]["align"]) + "]\n"
        attribute_text += "  misalign: [" + ", ".join(item['en_structured_data']["attribute_alignment"]["misalign"]) + "]"
        
        relation_text = "relation_score:\n  align: [" + ", ".join(item['en_structured_data']["relation_alignment"]["align"]) + "]\n"
        relation_text += "  misalign: [" + ", ".join(item['en_structured_data']["relation_alignment"]["misalign"]) + "]"
        
        full_text = entity_all + "\n\n" + entity_text + "\n\n" + attribute_text + "\n\n" + relation_text
        
        return {"image": image, "text": text, "answer": full_text, "score": score}