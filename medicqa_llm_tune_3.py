from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys


torch.cuda.set_per_process_memory_fraction(0.99)   # 限制最多使用95%内存
torch.backends.cuda.matmul.allow_tf32 = True       # 开启 TF32 矩阵乘法加速
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # 启用cudnn自动优化
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {device}")


model_path = './medicqa_llm_tuned_model'
if not os.path.exists(model_path):
    print(f"模型文件 {model_path} 不存在， 请确认先训练完成")
    sys.exit(0)

max_field_length = 250     # # 训练数据中最长token是238个词,因此设置最长处理长度是250个，节省空间

# 推理函数
def generate_response(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        padding='longest',  # 按批次内最长样本padding
        truncation=True,
        max_length=max_field_length  # 与生成的max_length一致
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids = inputs['input_ids'],
            attention_mask = inputs['attention_mask'],
            max_length=max_field_length,
            num_beams=3,
            temperature=0.7,
            repetition_penalty=1.2,
            top_k=50,
            top_p=0.95,
            do_sample=False,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(' ', '')

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 半精度量化
    device_map='auto'  # 自动分配到GPU
).to(device)
model = torch.compile(model, mode='max-autotune')  # 编译模型（推理优化模式,首次运行耗时10-20秒，后续推理加速20-30%）


# 示例使用
while True:
    user_input = input("请输入您的问题(输入q退出): ")
    if user_input.lower() == "q":
        break
    response = generate_response(user_input)
    print("AI回答:", response)