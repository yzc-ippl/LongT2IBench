class Config:
    model_name = "./Qwen2.5-VL-7B-Instruct"
    data_dir = "./LongT2IBench"
    img_root = "./LongT2IBench/image_prompts_1st"
    output_dir = "./qwen_align_out"
    num_epochs = 3
    batch_size = 2
    grad_accum = 8
    lr = 2e-4
    score_head_lr = 2e-4
    max_len = 2048
    lora_r = 32
    lora_alpha = 64
    lora_dropout = 0.05
    eval_steps = 100
    save_steps = 500
    level_token = "<level>"