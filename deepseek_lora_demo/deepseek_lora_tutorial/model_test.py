# test_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

print("加载训练好的模型...")

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "../DeepSeek-R1-Distill-Qwen-1.5B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载LoRA权重
model = PeftModel.from_pretrained(base_model, "./my_lora_model")

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("./my_lora_model")

print("模型加载完成！")

# 测试函数
def ask_model(question, max_length=100):
    """向模型提问"""
    # 构建输入
    prompt = f"指令：{question}\n回答："
    
    # 编码
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 解码
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取回答部分
    if "回答：" in response:
        answer = response.split("回答：")[-1].strip()
    else:
        answer = response
    
    return answer

# 开始测试
print("\n 开始测试训练好的模型...")
print("=" * 50)

# 测试训练过的问题
test_questions = [
    "写诗",
    "翻译",
    "解释",
    "写代码",
    "问候"
]

for i, question in enumerate(test_questions):
    print(f"\n测试 {i+1}: {question}")
    answer = ask_model(question)
    print(f"模型回答: {answer}")

# 测试新问题
print("\n 测试新问题（模型没学过的）...")
new_questions = [
    "介绍一下你自己",
    "今天的天气怎么样",
    "Python是什么",
    "写一个计算圆面积的函数"
]

for i, question in enumerate(new_questions):
    print(f"\n新问题 {i+1}: {question}")
    answer = ask_model(question)
    print(f"模型回答: {answer}")

print("\n测试完成！")
