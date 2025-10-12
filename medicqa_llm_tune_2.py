from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import pandas as pd
import torch
import os
import sys
from datetime import datetime
import json

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
test_data_path = 'medicqa_test_dataset_1000.csv'
reference_result_path = 'medicqa_reference_result.json'

max_field_length = 250     # # 训练数据中最长token是238个词,因此设置最长处理长度是250个，节省空间
BATCH_SIZE = 64


# 批量推理函数（一次处理batch_size个样本）
def batch_generate(prompts, batch_size=BATCH_SIZE):  # 可根据GPU显存调整batch_size
    results = []
    with tqdm(total=len(prompts), desc="推理中") as pbar:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            # 批量tokenize（统一长度，加速处理）
            inputs = tokenizer(
                batch,
                return_tensors='pt',
                padding='longest',  # 按批次内最长样本padding
                truncation=True,
                max_length=max_field_length  # 与生成的max_length一致
            ).to(device)  # 移到GPU

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=max_field_length,
                    num_beams=3,
                    temperature=0.7,
                    do_sample=False,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # 解码批量结果
            batch_responses = [
                tokenizer.decode(output, skip_special_tokens=True).replace(' ', '')
                for output in outputs
            ]
            results.extend(batch_responses)
            pbar.update(len(batch))  # 每批处理完更新进度
    return results

# 评估函数
def evaluate_model(results):
    correct = 0
    all_predictions = []
    all_references = []
    for item in results:
        pred = item['模型答案']
        ref = item['真实答案']
        if ref.lower() in pred.lower() or pred.lower() in ref.lower():
            correct += 1
        all_predictions.append(pred)
        all_references.append(ref)
    accuracy = correct / len(results)
    print(f'评估结果 - 准确率: {accuracy: .2%}')
    return accuracy

#1. 加载模型和测试集
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # 半精度量化
    device_map='auto'  # 自动分配到GPU
).to(device)
model = torch.compile(model, mode='max-autotune')  # 编译模型（推理优化模式,首次运行耗时10-20秒，后续推理加速20-30%）
df = pd.read_csv(test_data_path)

#2. 批量推理并保存结果
startTime = datetime.now()
print('推理开始时间：', startTime.strftime("%Y-%m-%d %H:%M:%S"))
results = []
prompts = [f"{title}:{ask}" for title, ask in zip(df['title'], df['ask'])]
answers = batch_generate(prompts, batch_size=24)
results = [
    {'问题': q, '真实答案': df['answer'].iloc[i], '模型答案': a}
    for i, (q, a) in enumerate(zip(prompts, answers))
]
endTime = datetime.now()
print('推理结束时间：', endTime.strftime("%Y-%m-%d %H:%M:%S"))
print('推理耗时：', endTime - startTime)

with open(reference_result_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

#3. 执行评估
startTime = datetime.now()
print('评估开始时间：', startTime.strftime("%Y-%m-%d %H:%M:%S"))
evaluate_model(results)
endTime = datetime.now()
print('评估结束时间：', endTime.strftime("%Y-%m-%d %H:%M:%S"))
print('评估耗时：', endTime - startTime)