import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import json

print("开始训练...")

# 1. 模型
model_name = "../DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. 数据
with open('train_data.json', 'r') as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

def preprocess(examples):
    texts = []
    for i in range(len(examples['instruction'])):
        inst = examples['instruction'][i]
        inp = examples['input'][i]
        out = examples['output'][i]
        if inp:
            text = f"指令：{inst}\n输入：{inp}\n回答：{out}"
        else:
            text = f"指令：{inst}\n回答：{out}"
        texts.append(text)
    
    tokens = tokenizer(texts, truncation=True, padding="max_length", max_length=128)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(preprocess, batched=True)

# 3. LoRA
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. 训练
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    logging_steps=1,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

print("开始训练...")
trainer.train()

# 5. 保存
model.save_pretrained("my_lora_model")
tokenizer.save_pretrained("my_lora_model")
print("训练完成！模型已保存。")
