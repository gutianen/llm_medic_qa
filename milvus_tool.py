from pymilvus import MilvusClient

# 1. 连接Milvus（替换为你的连接信息）
client = MilvusClient(uri="http://172.17.0.1:19530", user = "root", password= "Milvus", db_name="default")  # 若启用认证，需加user/password

# 2. 查询指定向量字段的索引配置
index_details = client.describe_index(
    collection_name="Vector_index_63d7adb9_e67b_45d2_853b_d5e358354b7a_Node",  # 你的Collection名称
    index_name="vector"  # 向量字段名称（如content_vector）
)

# 3. 打印HNSW索引参数
print(f"HNSW索引配置(all)：{index_details}")
print(f"索引类型：{index_details['index_type']}")
print(f"距离度量：{index_details['metric_type']}")
print(f"HNSW参数（M/efConstruction）：{index_details['M']}/{index_details['efConstruction']}")


collection_info = client.describe_collection("Vector_index_63d7adb9_e67b_45d2_853b_d5e358354b7a_Node")
print(f'collection_info: {collection_info}')
