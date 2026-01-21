# 一个qlora的样例代码，还没测试

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset  # 假设使用 HF datasets

# 1. 加载 tokenizer 和量化配置
model_id = "meta-llama/Llama-3-8B"  # 或其他基础模型
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 2. 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quant_config, device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# 3. LoRA 配置（针对牙科聊天：q_proj/v_proj 等关键层）
lora_config = LoraConfig(
    r=16,  # 秩
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Llama 注意力层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. 加载牙科数据集（黑盒蒸馏生成的对，如 prompt-response）
dataset = load_dataset("your-dental-dataset")  # 格式：{"text": "prompt + response"}

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# 5. 训练
args = TrainingArguments(
    output_dir="./dental-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    save_steps=500
)
trainer = Trainer(model=model, args=args, train_dataset=tokenized_dataset["train"])
trainer.train()

# 保存 LoRA 适配器
model.save_pretrained("dental-lora-adapter")
