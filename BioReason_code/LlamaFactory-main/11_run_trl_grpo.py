import os
import json
import re
import torch
import transformers
from PIL import Image  # 新增：用于加载真实图像
from datasets import Dataset
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# 核心修复 1：强行注入缺失的字典，解决 PEFT 兼容报错
transformers.PreTrainedModel.warnings_issued = {}

# ==========================================
# 1. 配置路径 (请根据你的实际情况修改)
# ==========================================
MODEL_PATH = "models/Qwen2.5-VL-7B-BioReason-SFT" 
TRAIN_DATA_PATH = "data/B_bioreason_reasoning.json"
OUTPUT_DIR = "saves/bioreason/grpo_trl"

# ==========================================
# 2. 准备数据集 (注入真实 PIL 图像)
# ==========================================
print("正在处理数据集并加载图片...")
with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

processed_data = []
for item in raw_data:
    match = re.search(r'<answer>\s*([A-E])\s*</answer>', item['output'], re.IGNORECASE)
    if not match:
        continue
    correct_letter = match.group(1).upper()
    
    image_path = item["images"][0] if item.get("images") else None
    content = []
    images_list = []
    
    if image_path:
        try:
            # 核心修复 2：加载真实的 PIL 图像
            img = Image.open(image_path).convert("RGB")
            # 强行缩小图像分辨率，保证单卡绝对不会爆显存，极大加速 GRPO 采样！
            img.thumbnail((512, 512)) 
            images_list.append(img)
            # 告诉 prompt 这里有一张图即可，不需要路径了
            content.append({"type": "image"})
        except Exception as e:
            print(f"图片加载失败，跳过: {image_path}")
            continue
            
    clean_instruction = item["instruction"].replace("<image>\n", "").replace("<image>", "").strip()
    content.append({"type": "text", "text": clean_instruction})
    
    processed_data.append({
        "prompt": [{"role": "user", "content": content}],
        "images": images_list, # 核心修复 3：专设 images 字段让 TRL 捕获
        "answer": correct_letter 
    })

train_dataset = Dataset.from_list(processed_data)
print(f"有效 GRPO 训练数据: {len(train_dataset)} 条")

# ==========================================
# 3. 定义“裁判” (Reward Functions)
# ==========================================
def format_reward_func(completions, **kwargs):
    rewards = []
    for completion in completions:
        # 剥离出纯文本内容
        if isinstance(completion, list) and len(completion) > 0:
            content = completion[0].get("content", "")
        else:
            content = str(completion)
            
        if re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", content, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def accuracy_reward_func(completions, answer, **kwargs):
    rewards = []
    for comp, ans in zip(completions, answer):
        if isinstance(comp, list) and len(comp) > 0:
            content = comp[0].get("content", "")
        else:
            content = str(comp)
            
        match = re.search(r"<answer>\s*([A-E])\s*</answer>", content)
        if match and match.group(1).upper() == ans.upper():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

# ==========================================
# 4. 初始化模型、处理器和 LoRA
# ==========================================
print("正在加载 Processor 和 Model...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)

# ==========================================
# 5. 配置 GRPO 训练超参数
# ==========================================
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_prompt_length=1024,       
    max_completion_length=512,    
    num_generations=2,            # 保持为 2，最适合单卡生存
    fp16=True,
    logging_steps=1,
    save_steps=100,
    max_steps=500,
    report_to="none",
    remove_unused_columns=False   # 核心修复 4：绝对禁止 TRL 丢弃我们的 images 字段！
)

# ==========================================
# 6. 启动训练！
# ==========================================
trainer = GRPOTrainer(
    model=MODEL_PATH,
    reward_funcs=[format_reward_func, accuracy_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    processing_class=processor,
    peft_config=peft_config
)

print("🚀 开始 GRPO 训练！见证模型自我进化的时刻...")
trainer.train()

trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "checkpoint-final"))
processor.save_pretrained(os.path.join(OUTPUT_DIR, "checkpoint-final"))
print("🎉 训练完成，权重已保存！")