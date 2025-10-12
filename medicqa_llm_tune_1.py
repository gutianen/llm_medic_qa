import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, TrainerCallback
from datasets import Dataset
from tqdm.auto import tqdm
from datetime import datetime
import os
import torch
import sys
torch.cuda.set_per_process_memory_fraction(0.99)   # 限制最多使用95%内存
torch.backends.cuda.matmul.allow_tf32 = True       # 开启 TF32 矩阵乘法加速
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # 启用cudnn自动优化
device = 'cuda' if torch.cuda.is_available() else 'cpu'    # 设备检查
print(f"device: {device}")

model_path = '../../../models/Qwen2.5-0.5B'
data_path = 'medicqa_dataset_1000.csv'
out_dir = './medicqa_llm_tuned'
output_model_path = './medicqa_llm_tuned_model'

os.environ['TOKENIZERS_PARALLELISM'] = 'false'    # 防止tokenizer并行问题
max_field_length = 250     # # 训练数据中最长token是238个词,因此设置最长处理长度是250个，节省空间



# 数据预处理方法
def preprocess_function(examples):
    texts = [f'问题：{title}:{ask}\n答案：{answer}' for title, ask, answer in zip(examples['title'], examples['ask'], examples['answer'])]
    # print(texts)
    return tokenizer(texts, truncation=True, max_length=max_field_length, padding='max_length')

# 进度条回调类
class ProgressCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.epoch_bar = tqdm(total=state.num_train_epochs, desc='训练轮次')
        self.step_bar = tqdm(total=state.max_steps, desc='训练步数')
    def on_step_end(self, args, state, control, **kwargs):
        self.step_bar.update(1)
    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_bar.update(1)

# 自定义GPU缓存清理回调（解决自动清理问题）
class GpuCacheCleanCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # 每N步清理一次（如每10步，避免频繁清理影响速度）
        if state.global_step % 10 == 0:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    def on_epoch_end(self, args, state, control,** kwargs):
        # 每个epoch结束后清理一次
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# 训练配置：
training_args = TrainingArguments(
    output_dir = out_dir,
    torch_compile=True,  # 编译模型为优化后的CUDA代码（首次运行有编译耗时，后续加速）
    per_device_train_batch_size = 2,        #
    gradient_accumulation_steps = 8,        # 增加梯度累计，
    gradient_checkpointing=True,            # 启用梯度检查点, 必须开， 会牺牲20%计算时间，但能显著降低内存消耗30-40%
    fp16 = True,                           # 启用混合精度， 提高运算速度
    num_train_epochs = 8,
    logging_steps = 20,
    save_steps = 200,
    learning_rate = 3e-5,                   # 降低学习率
    optim = 'adamw_torch_fused',            # 融合版AdamW，比普通版快10-20%（需PyTorch≥2.0）
    warmup_ratio = 0.05,
    report_to = 'none',
    dataloader_num_workers = 8,             # 增加工作进程，增大数据传输给GPU
    dataloader_pin_memory=True              # 锁定内存，加速数据从CPU到GPU的传输
)

# 1.数据加载（3阶段，带进度条）
print('1.加载数据集...')
with tqdm(total=3, desc='数据准备') as pbar:
    df = pd.read_csv(data_path)   # 阶段1：读取
    pbar.update(1)
    dataset = Dataset.from_pandas(df)        # 阶段2： 转换
    pbar.update(1)
    dataset = dataset.train_test_split(test_size=0.1)    # 阶段3：拆分
    pbar.update(1)

# 2.加载模型和分词器
print('2.加载模型...')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# 未对任何层进行冻结操作（没有设置 param.requires_grad=true）,全参数微调
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)

# 3.数据预处理
print('3.预处理数据...')
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['title', 'ask', 'answer'], desc='数据处理')

####  token词元长度测试代码 - start
# import numpy as np
# lengths = []
# print("数据集所有字段名：", dataset.column_names)  # 会输出包含 "问题" 和 "答案" 的列表
# for example in dataset['train']:  # 遍历每条样本
#     title = example['title']
#     ask = example['ask']
#     answer = example['answer']
#     full_text = f'问题：{title}:{ask}\n答案：{answer}'
#
#     # 编码文本（不截断，不padding）
#     tokens = tokenizer(full_text, truncation=False, padding=False, return_tensors=None)
#     # input_ids的长度就是token数量
#     token_length = len(tokens['input_ids'])
#     lengths.append(token_length)
# print(f"平均长度: {np.mean(lengths)}")
# print(f"90%样本长度≤: {np.percentile(lengths, 90)}")
# print(f"最长长度: {np.max(lengths)}")
# sys.exit(0)
####  token词元长度测试代码 - end

# 4.训练微调模型
print('4.开始微调训练...')
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset['train'], data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False), callbacks=[ProgressCallback, GpuCacheCleanCallback])
startTime = datetime.now()
print('训练开始时间：', startTime.strftime("%Y-%m-%d %H:%M:%S"))
trainer.train()
endTime = datetime.now()
print('训练结束时间：', endTime.strftime("%Y-%m-%d %H:%M:%S"))
print('训练耗时：', endTime - startTime)

model.save_pretrained('./medicqa_llm_tuned_model')
tokenizer.save_pretrained('./medicqa_llm_tuned_model')
print("训练完成！模型已保存")