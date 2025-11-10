from pydantic import BaseModel, Field
from typing import List, Optional
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import numpy as np
from zhipuai import ZhipuAI
import os
import requests
from typing import List
import jieba
import jieba.analyse
from pymilvus import WeightedRanker


class MedicalContext(BaseModel):
    """医疗知识上下文"""
    context_id: str = Field(description="知识唯一ID（对应Milvus主键）")
    content: str = Field(description="医疗知识文本")
    source: str = Field(description="知识来源")
    relevance: float = Field(description="与查询内容的相似度（0-1，≥0.7）")
    timestamp: str = Field(description="发布时间（YYYY-MM-DD）")

class ConversationContext(BaseModel):
    """对话上下文"""
    history_ids: List[str] = Field(default_factory=list, description="历史引用的知识ID列表")
    previous_queries: List[str] = Field(default_factory=list, description="历史查询文本列表")
    # previous_embeddings: List[List[float]] = Field(default_factory=list, description="历史查询向量列表（用于话题相似度计算）")
    last_query_time: Optional[datetime] = Field(default=None, description="最后一次查询时间（ISO格式）")
    current_topic: Optional[str] = Field(default=None, description="当前对话主题")

class MedicalQueryRequestByText(BaseModel):
    """医疗查询请求（只带文本）"""
    query: str = Field(description="用户查询文本")
    conversation_context: Optional[Dict[str, Any]] = Field(default=None, description="对话上下文，包含历史信息和状态")

class MedicalQueryRequest(BaseModel):
    """医疗查询请求"""
    query: str = Field(description="用户查询文本")
    embedding: List[float] = Field(description="查询向量")
    conversation_context: Optional[Dict[str, Any]] = Field(default=None, description="对话上下文，包含历史信息和状态")

class MedicalQueryResponse(BaseModel):
    """医疗查询响应"""
    contexts: List[MedicalContext] = Field(description="结构化知识列表")
    conversation_context: ConversationContext = Field(description="更新后的对话上下文")
    is_new_topic: bool = Field(description="是否为新话题")

# 标准化提示词请求
class PromptRequest(BaseModel):
    query: str = Field(description="用户查询文本")
    contexts: List[MedicalContext] = Field(description="结构化知识列表")

# 标准化提示词响应
class PromptResponse(BaseModel):
    standard_prompt: str = Field(description="带来源约束的提示词")



class ZhipuAIService:
    """使用官方SDK的智谱AI嵌入服务"""

    def __init__(self):
        self.api_key = '2003486fa7684ffaa74ac689fd1d6d47.4Xu2onGXxbSerTi0'
        self.client = ZhipuAI(api_key=self.api_key)
        self.model = "embedding-3"

    def get_embedding(self, text: str) -> List[float]:
        """使用官方SDK获取文本嵌入向量"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )

            # 官方SDK返回的对象有data属性
            embedding_data = response.data[0]
            return embedding_data.embedding

        except Exception as e:
            print(f"智谱AI SDK调用失败: {e}")
            raise


class HybridSearchProcessor:
    def __init__(self):
        # 加载医疗领域自定义词典（如果有）
        try:
            jieba.load_userdict("dict/medical_terms.txt")
        except:
            pass  # 如果没有自定义词典就忽略

        # 配置关键词提取参数
        jieba.analyse.set_stop_words("dict/stop_words.txt")  # 停用词文件

    def extract_keywords(self, query_text, topK=5):
        """使用 Jieba 提取关键词"""
        # 预处理：去除标点等
        import re
        cleaned_text = re.sub(r'[^\w\s]', '', query_text)

        # 使用 TF-IDF 提取关键词
        keywords_tfidf = jieba.analyse.extract_tags(
            cleaned_text,
            topK=topK,
            withWeight=False
        )

        # 使用 TextRank 提取关键词（作为备选）
        keywords_textrank = jieba.analyse.textrank(
            cleaned_text,
            topK=topK,
            withWeight=False
        )

        # 合并并去重
        all_keywords = list(set(keywords_tfidf + keywords_textrank))
        return all_keywords

    def build_search_expression(self, keywords, content_field="page_content"):
        """构建 Milvus 搜索表达式"""
        if not keywords:
            return ""

        # 构建 OR 条件表达式
        conditions = []
        for keyword in keywords:
            # 使用 like 进行模糊匹配
            condition = f'{content_field} like "%{keyword}%"'
            conditions.append(condition)

        # 如果有多个条件，用 OR 连接
        if len(conditions) > 1:
            expr = " || ".join(conditions)
        else:
            expr = conditions[0] if conditions else ""

        return expr



def are_vectors_equal(vec1, vec2, epsilon=1e-6):
    """
    判断两个浮点向量是否完全一致（考虑浮点数精度误差）
    参数：
        vec1: 第一个浮点数组
        vec2: 第二个浮点数组
        epsilon: 精度容忍度，默认 1e-6（可根据场景调整）
    返回：
        bool: 若一致则返回 True，否则 False
    """
    # 第一步：检查长度是否相同
    if len(vec1) != len(vec2):
        return False

    # 第二步：逐个元素比较，允许微小误差
    for a, b in zip(vec1, vec2):
        # 计算差值的绝对值，若超过 epsilon 则不一致
        if abs(a - b) > epsilon:
            return False

    # 所有元素均满足条件，一致
    return True



def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    计算两个向量的余弦相似度

    Args:
        embedding1: 第一个向量
        embedding2: 第二个向量

    Returns:
        相似度分数 (0-1)
    """
    if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
        return 0.0

    # 转换为numpy数组
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    # 计算余弦相似度
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)



# 初始化嵌入服务
try:
    zhipu_service = ZhipuAIService()
    print("智谱AI嵌入服务初始化成功")
except Exception as e:
    print(f"智谱AI嵌入服务初始化失败: {e}")
    zhipu_service = None


# 创建全局处理器实例
hybrid_processor = HybridSearchProcessor()