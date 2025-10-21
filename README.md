# llm_medic_qa
医疗科普知识问答大模型

## 训练代码：
* medicqa_llm_tune_1.py
* medicqa_llm_tune_2.py
* medicqa_llm_tune_3.py

## CI脚本
ci/  
├─ Jenkinsfile-CI-Docker      # 流水线脚本（核心）
├─ dify/                      # Dify 自定义配置（覆盖模型对接 VLLM）
│  └─ docker-compose.override.yml  # 让 Dify 指向 VLLM 服务
└─ vllm/                      # VLLM 镜像配置
   └─ Dockerfile.vllm         # 简化版 VLLM Dockerfile

## 知识库
knowledge_base/
├─ medic_qa.csv
├─ medic_qa_1.csv
└─ medic_qa_test.csv