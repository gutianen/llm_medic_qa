from mcp.server.fastmcp import FastMCP
from pymilvus import MilvusClient
# from mcp_protocol import (
#     MedicalContext, RetrieveRequest, MultiTurnRequest,
#     KnowledgeResponse, PromptRequest, PromptResponse
# )

from mcp_protocol import (
    ZhipuAIService, zhipu_service, hybrid_processor, MedicalContext, ConversationContext, MedicalQueryRequestByText,
    MedicalQueryRequest, MedicalQueryResponse, PromptRequest, PromptResponse
)
from typing import List
import os
import json
import re
from typing import List, Dict, Any
import numpy as np
from datetime import datetime, timezone, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import jieba.analyse
from pymilvus import WeightedRanker, AnnSearchRequest

config={
    'weight_for_semantic': 0.7,      # 语义权重
    'weight_for_keywords': 0.3,       # 关键词权重
    'search_result_count': 10,       # 向量库搜索结果保留数量
    'score_threshold': 0.65,    # 召回分数阈值
    'topk': 3,                  # 最多召回个数

}

# 初始化Milvus客户端
milvus_client = MilvusClient(
    uri=os.getenv("MILVUS_URI", "http://172.17.0.1:19530"),
    user="root",
    password="Milvus",
    db_name="default"
)
COLLECTION = "Vector_index_63d7adb9_e67b_45d2_853b_d5e358354b7a_Node"

# 初始化FastMCP服务
mcp = FastMCP(
    name="医疗科普MCP服务",
    host="0.0.0.0",
    port=8001,
    stateless_http=True,
    json_response=True
)

# 配置参数
TOPIC_SIMILARITY_THRESHOLD = 0.7  # 话题相似度阈值
CONVERSATION_TIMEOUT_MINUTES = 30  # 对话超时时间（分钟）


def parse_milvus_result(res_data: List[Dict]) -> List[Dict[str, Any]]:
    """
    解析 Milvus 返回的结果数据
    Args:
        res_data: Milvus 返回的数据列表
    Returns:
        解析后的结构化数据列表
    """
    formatted_results = []

    for result in res_data:
        try:
            # 直接提取字段
            page_content = result['entity']['page_content']

            # 简单的字符串分割方法
            fields = {}
            parts = page_content.split('";"')
            for part in parts:
                if '":"' in part:
                    key, value = part.split('":"', 1)
                    # 去除可能的引号
                    key = key.strip('"')
                    value = value.strip('"')
                    fields[key] = value

            formatted_result = {
                'id': result['id'],
                'distance': result['distance'],
                'title': fields.get('标题', ''),
                'question': fields.get('问题', ''),
                'answer': fields.get('答案', ''),
                'primary_category': fields.get('一级分类', ''),
                'secondary_category': fields.get('二级分类', '').strip(),
                'keywords': fields.get('关键词', ''),
                'knowledge_source': fields.get('知识来源', ''),
                'update_time': fields.get('更新时间', ''),
                'metadata': result['entity']['metadata']
            }

            formatted_results.append(formatted_result)

        except Exception as e:
            print(f"解析结果时出错: {e}")
            # 如果解析失败，至少返回基本信息
            formatted_results.append({
                'id': result.get('id', ''),
                'distance': result.get('distance', 0),
                'title': '解析失败',
                'question': '',
                'answer': str(result.get('entity', {}).get('page_content', ''))[:500],  # 截取前200字符
                'primary_category': '',
                'secondary_category': '',
                'keywords': '',
                'knowledge_source': '',
                'update_time': '',
                'metadata': result.get('entity', {}).get('metadata', {})
            })

    return formatted_results

class TopicManager:
    """话题管理器"""

    @staticmethod
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

    @staticmethod
    def is_same_topic(current_query: str, current_embedding: List[float], previous_queries: List[str]) -> bool:
        """
        判断当前查询是否与历史查询属于同一话题

        Args:
            current_query: 当前查询文本
            current_embedding: 当前查询向量
            previous_queries: 历史查询文本列表
            previous_embeddings: 历史查询向量列表

        Returns:
            True表示同一话题，False表示新话题
        """
        # 如果没有历史查询，视为新话题
        if not previous_queries:
            return False

        # 计算与最近几个查询的相似度
        # recent_count = min(3, len(previous_embeddings))
        # recent_embeddings = previous_embeddings[-recent_count:]
        previous_embeddings = []
        for previous_query in previous_queries:
            previous_embeddings.append(zhipu_service.get_embedding(previous_query))

        similarities = []
        for prev_embedding in previous_embeddings:
            similarity = TopicManager.calculate_similarity(current_embedding, prev_embedding)
            similarities.append(similarity)

        # 取最高相似度
        max_similarity = max(similarities) if similarities else 0

        # 判断是否超过阈值
        return max_similarity >= TOPIC_SIMILARITY_THRESHOLD

    # deprecated
    @staticmethod
    def is_same_topic_simple(current_query: str, previous_queries: List[str]) -> bool:
        """
        判断当前查询是否与历史查询属于同一话题（简化版本）

        Args:
            current_query: 当前查询文本
            previous_queries: 历史查询文本列表

        Returns:
            True表示同一话题，False表示新话题
        """
        # 如果没有历史查询，视为新话题
        if not previous_queries:
            return False

        # 计算与最近几个查询的相似度
        recent_count = min(3, len(previous_queries))
        recent_queries = previous_queries[-recent_count:]

        similarities = []
        for prev_query in recent_queries:
            similarity = TopicManager.calculate_text_similarity(current_query, prev_query)
            similarities.append(similarity)

        # 取最高相似度
        max_similarity = max(similarities) if similarities else 0

        # 判断是否超过阈值
        return max_similarity >= TOPIC_SIMILARITY_THRESHOLD

    # deprecated
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """
        计算两个文本的余弦相似度（基于TF-IDF）

        Args:
            text1: 第一个文本
            text2: 第二个文本

        Returns:
            相似度分数 (0-1)
        """
        if not text1 or not text2:
            return 0.0

        try:
            # 使用TF-IDF计算文本相似度
            vectorizer = TfidfVectorizer(tokenizer=jieba.cut, min_df=1)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except Exception:
            # 如果计算失败，使用简单的字符串包含判断
            words1 = set(jieba.cut(text1))
            words2 = set(jieba.cut(text2))
            if not words1 or not words2:
                return 0.0
            intersection = words1 & words2
            return len(intersection) / max(len(words1), len(words2))

    @staticmethod
    def is_conversation_timeout(last_query_time: datetime) -> bool:
        """
        判断对话是否超时

        Args:
            last_query_time: 最后查询时间

        Returns:
            True表示超时，False表示未超时
        """
        if not last_query_time:
            return True

        try:
            timediff = datetime.now() - last_query_time
            return timediff.total_seconds() > CONVERSATION_TIMEOUT_MINUTES * 60
        except Exception as ex:
            print(f"exception in is_conversation_timeout(): {str(ex)}")
            return True

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def query_medical_knowledge_by_text(request: MedicalQueryRequestByText) -> MedicalQueryResponse:
    """
    文本接口：智能医疗对话（内部自动处理文本转向量）
    Args:
        request: 包含query(查询文本)和conversation_context(对话上下文)
    Returns:
        医疗查询响应
    """
    # 从请求中获取文本
    query_text = request.query
    conversation_context = request.conversation_context

    # 使用智谱AI将文本转为向量
    if not zhipu_service:
        raise Exception("智谱AI嵌入服务初始化失败，无法处理文本")

    query_embedding = zhipu_service.get_embedding(query_text)

    # 调用现有的向量检索接口
    return query_medical_knowledge(MedicalQueryRequest(
        query=query_text,
        embedding=query_embedding,
        conversation_context=conversation_context
    ))


@mcp.tool()
def query_medical_knowledge(request: MedicalQueryRequest) -> MedicalQueryResponse:
    """
    检索医疗科普知识接口，支持多轮对话，支持混合搜索
    根据话题相似度和时间间隔自动判断是新话题还是追问，统一处理首次查询和后续追问逻辑。
    语义和关键词混合搜索
    Args:
        request: 医疗查询请求，包含查询文本、向量和可选上下文
    Returns:
        医疗查询响应，包含知识列表和更新后的对话上下文
    """
    # 提取或初始化对话上下文
    if request.conversation_context:
        context = ConversationContext(**request.conversation_context)
    else:
        context = ConversationContext()

    # 判断是否为新话题
    is_new_topic = True
    if context.previous_queries and context.last_query_time:
        # 检查超时
        if not TopicManager.is_conversation_timeout(context.last_query_time):
            # 检查话题相似度
            is_new_topic = not TopicManager.is_same_topic(
                current_query=request.query,
                current_embedding=request.embedding,
                previous_queries=context.previous_queries,
                # previous_embeddings=context.previous_embeddings
            )

    # # 执行Milvus检索
    # res = milvus_client.search(
    #     collection_name=COLLECTION,
    #     data=[request.embedding],
    #     anns_field="vector",
    #     search_params={"metric_type": "IP"},
    #     limit=10,
    #     output_fields=["id", "page_content", "metadata"]
    # )

    # ====== 执行混合搜索 ======
    # 1. 使用 Jieba 提取关键词
    keywords = hybrid_processor.extract_keywords(request.query)
    print(f"提取的关键词: {keywords}")

    # 2. 构建关键词搜索表达式
    keyword_expr = hybrid_processor.build_search_expression(keywords, "page_content")
    print(f"关键词搜索表达式: {keyword_expr}")

    # 3. 执行混合搜索
    try:
        # 使用加权排序器，语义权重0.7，关键词权重0.3
        ranker = WeightedRanker(config['weight_for_semantic'], config['weight_for_keywords'])


        # 混合搜索请求
        # 准备搜索请求列表
        search_requests = []

        # 添加语义搜索请求（必须有）
        semantic_search_req = AnnSearchRequest(
            data=[request.embedding],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=20
        )
        search_requests.append(semantic_search_req)

        # 如果有有效的关键词表达式，添加关键词搜索请求（可选）
        if keyword_expr:
            # 对于关键词搜索，我们需要使用 expr 参数
            keyword_search_req = AnnSearchRequest(
                data=[request.embedding],
                anns_field="vector",  # 可以是任意字段，因为我们要用表达式过滤
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=20,
                expr=keyword_expr  # 使用表达式进行过滤
            )
            search_requests.append(keyword_search_req)
            print(f"执行混合搜索：语义搜索 + 关键词搜索")
        else:
            print(f"执行纯语义搜索：无有效关键词")

        # 执行混合搜索
        if len(search_requests) > 1:
            # 真正的混合搜索
            hybrid_results = milvus_client.hybrid_search(
                collection_name=COLLECTION,
                reqs=search_requests,
                ranker=ranker,
                limit=config['search_result_count'],  # 混合搜索后保留的结果数量
                output_fields=["id", "page_content", "metadata"]
            )
            print(f"混合搜索结果类型: {type(hybrid_results)}")
            # 将混合搜索结果转换为与 parse_milvus_result 兼容的格式
            # hybrid_results_compatible = []
            # for hit in hybrid_results[0]:
            #     hybrid_results_compatible.append({
            #         'id': hit.id,
            #         'distance': hit.score,
            #         'entity': {
            #             'page_content': hit.entity.get('page_content'),
            #             'metadata': hit.entity.get('metadata')
            #         }
            #     })

            formatted_results = parse_milvus_result(hybrid_results[0])

        else:
            # 如果没有关键词，回退到纯语义搜索
            results = milvus_client.search(
                collection_name=COLLECTION,
                data=[request.embedding],
                anns_field="vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=config['search_result_count'],
                output_fields=["id", "page_content", "metadata"]
            )
            formatted_results = parse_milvus_result(results[0])

    except Exception as e:
        print(f"混合搜索失败，回退到纯语义搜索: {str(e)}")
        # 回退到纯语义搜索
        res = milvus_client.search(
            collection_name=COLLECTION,
            data=[request.embedding],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=config['search_result_count'],
            output_fields=["id", "page_content", "metadata"]
        )
        # 解析结果
        formatted_results = parse_milvus_result(res[0])

    # 过滤和结构化知识
    contexts = []
    for result in formatted_results:
        # 过滤低相似度
        if result['distance'] < config['score_threshold']:
            continue

        # 如果是追问，过滤历史知识
        if not is_new_topic and str(result['id']) in context.history_ids:
            continue

        contexts.append(MedicalContext(
            context_id=str(result['id']),
            content=result['answer'],
            source=result['knowledge_source'],
            relevance=round(result['distance'], 2),
            timestamp=result['update_time'].split()[0] if result['update_time'] else "2025-10-31"
        ))

        # 最多返回3条
        if len(contexts) >= config['topk']:
            break

    # 更新对话上下文
    new_history_ids = context.history_ids.copy()
    new_previous_queries = context.previous_queries.copy()
    # new_previous_embeddings = context.previous_embeddings.copy()

    if is_new_topic:
        # 新话题：重置历史ID，只保留当前查询的知识
        new_history_ids = [ctx.context_id for ctx in contexts]
        new_previous_queries = [request.query]  # 重置查询历史
        # new_previous_embeddings = [request.embedding]  # 重置向量历史
        current_topic = request.query
    else:
        # 追问：添加新知识ID到历史
        new_history_ids.extend([ctx.context_id for ctx in contexts])
        new_previous_queries.append(request.query)  # 添加当前查询到历史
        # new_previous_embeddings.append(request.embedding)  # 添加当前向量到历史
        current_topic = context.current_topic


    # 更新上下文
    updated_context = ConversationContext(
        history_ids=new_history_ids,
        previous_queries=new_previous_queries,
        # previous_embeddings=new_previous_embeddings,
        last_query_time=datetime.now(),
        current_topic=current_topic
    )

    return MedicalQueryResponse(
        contexts=contexts,
        conversation_context=updated_context,
        is_new_topic=is_new_topic
    )

@mcp.tool()
def generate_medical_prompt(request: PromptRequest) -> PromptResponse:
    """
    生成医疗科普标准化提示词
    Args:
        request: 提示词生成请求，包含查询文本和知识上下文
    """
    context_text = "\n".join([
        f"检索结果{i + 1}（来源：{ctx.source}）：{ctx.content}"
        for i, ctx in enumerate(request.contexts)
    ])

    standard_prompt = f"""
    【系统角色】
    你是一名专业的医疗健康咨询助手，需基于提供的检索结果为用户解答医疗相关问题。回答需严谨、准确，优先使用检索到的权威信息，不得编造未提及的医疗建议。


    【医疗科普回答规则】
    1. 若检索结果正常（存在有效内容且与问题相关）：
        - 仅参考以下检索结果进行回答，不编造内容；
        - 综合3条结果的核心信息，用通俗易懂的语言整理成连贯回答；
        - 重点突出一致结论，若结果有细节差异，以多数内容为准；
        - 结尾可补充“以上信息仅供参考，具体请遵医嘱”。
    
    2. 若检索结果异常（满足以下任一条件）：
       - 结果为空（无任何匹配内容）；
       - 所有结果与用户问题无关（如答非所问）；
       - 结果中存在明显错误或矛盾信息（如医疗建议冲突）；
       请直接回复：“无匹配检索结果，请联系医务科获取专业帮助。”
    
    
    【检索结果参考】
    以下是与用户问题相关的top3匹配检索结果（若结果为空或异常，需特殊处理）：
    {context_text}

    【用户问题】
    {request.query}
    """
    return PromptResponse(standard_prompt=standard_prompt.strip())

if __name__ == "__main__":
    mcp.run("streamable-http")