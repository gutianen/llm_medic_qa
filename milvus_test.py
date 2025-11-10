from pymilvus import MilvusClient
import time
import numpy as np

# -------------------------- 参数化配置（可按需修改） --------------------------
# Milvus连接信息
MILVUS_URI = "http://172.17.0.1:19530"
USER = "root"
PASSWORD = "Milvus"
DB_NAME = "default"
COLLECTION_NAME = "Vector_index_63d7adb9_e67b_45d2_853b_d5e358354b7a_Node"
VECTOR_FIELD = "vector"  # 向量字段名

# HNSW索引参数（待测试的参数）
TEST_M = 8
TEST_EF_CONSTRUCTION = 64

# 检索参数
QUERY_TEXT = "小儿肥胖懒怎样治疗"  # 搜索条件（参数化）
TOP_K = 10  # 召回数量
EF_SEARCH = 128  # 检索阶段探索深度（固定，保证对比公平性）

# -------------------------- 嵌入模型调用（需替换为你的Embedding-3调用逻辑） --------------------------
def get_embedding(text: str) -> list:
    """将文本转换为向量（请替换为你的Embedding-3模型调用代码）"""
    # 示例：假设使用OpenAI的Embedding-3模型（需安装openai库并配置密钥）
    import openai
    openai.api_key = "你的OpenAI密钥"  # 替换为实际密钥
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small"  # 或"text-embedding-3-large"
    )
    return response.data[0].embedding

# -------------------------- 核心测试逻辑 --------------------------
if __name__ == "__main__":
    # 1. 连接Milvus
    client = MilvusClient(
        uri=MILVUS_URI,
        user=USER,
        password=PASSWORD,
        db_name=DB_NAME
    )
    print(f"成功连接Milvus，集合：{COLLECTION_NAME}")

    # 2. 生成查询向量
    query_vector = get_embedding(QUERY_TEXT)
    print(f"生成查询向量（长度：{len(query_vector)}维），文本：{QUERY_TEXT}")

    # 3. 暴力搜索获取基准结果（真实相似向量Top-K）
    print("\n===== 开始暴力搜索（获取基准结果） =====")
    # 3.1 确保向量字段无索引（强制全量扫描）
    existing_indexes = client.list_indexes(collection_name=COLLECTION_NAME)
    if any(idx["field_name"] == VECTOR_FIELD for idx in existing_indexes):
        client.drop_index(
            collection_name=COLLECTION_NAME,
            field_name=VECTOR_FIELD
        )
        print("已删除现有索引，确保暴力搜索生效")

    # 3.2 执行暴力搜索（全量计算相似度）
    brute_force_start = time.time()
    brute_results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field=VECTOR_FIELD,
        param={"metric_type": "IP"},  # 与索引度量方式一致
        limit=TOP_K  # 取Top-K作为基准
    )
    brute_force_time = (time.time() - brute_force_start) * 1000  # 转换为毫秒
    # 提取基准结果的向量ID（用于后续对比）
    ground_truth_ids = {res["id"] for res in brute_results[0]}
    print(f"暴力搜索完成，耗时：{brute_force_time:.2f}ms，基准Top-{TOP_K}向量ID：{ground_truth_ids}")

    # 4. 重建HNSW索引（使用测试参数M和efConstruction）
    print(f"\n===== 重建HNSW索引（M={TEST_M}, efConstruction={TEST_EF_CONSTRUCTION}） =====")
    client.create_index(
        collection_name=COLLECTION_NAME,
        field_name=VECTOR_FIELD,
        index_type="HNSW",
        metric_type="IP",
        params={
            "M": TEST_M,
            "efConstruction": TEST_EF_CONSTRUCTION
        }
    )
    # 验证索引参数
    index_details = client.describe_index(
        collection_name=COLLECTION_NAME,
        index_name=VECTOR_FIELD
    )
    print(f"索引重建完成，实际参数：M={index_details['M']}, efConstruction={index_details['efConstruction']}")

    # 5. 执行HNSW检索并计算召回率和时间
    print(f"\n===== 执行HNSW检索（topK={TOP_K}） =====")
    # 5.1 多次检索取平均时间（减少偶然误差）
    total_time = 0
    hnsw_results_list = []
    for i in range(5):  # 重复5次取平均
        start = time.time()
        hnsw_results = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            anns_field=VECTOR_FIELD,
            param={
                "metric_type": "IP",
                "ef": EF_SEARCH  # 检索阶段探索深度
            },
            limit=TOP_K
        )
        end = time.time()
        total_time += (end - start) * 1000  # 累计毫秒数
        hnsw_results_list.append(hnsw_results[0])
    avg_time = total_time / 5  # 平均时间

    # 5.2 计算召回率（取最后一次检索结果计算，因多次结果应一致）
    hnsw_ids = {res["id"] for res in hnsw_results_list[-1]}
    hit_count = len(ground_truth_ids & hnsw_ids)  # 交集数量
    recall_rate = hit_count / TOP_K * 100  # 召回率（百分比）

    # 6. 输出最终结果
    print("\n===== 测试结果 =====")
    print(f"测试参数：M={TEST_M}, efConstruction={TEST_EF_CONSTRUCTION}, topK={TOP_K}")
    print(f"查询文本：{QUERY_TEXT}")
    print(f"平均检索时间：{avg_time:.2f}ms")
    print(f"召回率：{recall_rate:.2f}%（命中{hit_count}/{TOP_K}个真实相似向量）")
