# mcp_client_sync_fixed.py
import requests
import json
from typing import List
from mcp_protocol import MedicalQueryResponse
from typing import List, Dict, Any
from datetime import datetime


class MCPClientSync:
    def __init__(self, server_url: str = "http://127.0.0.1:8001/mcp"):
        self.server_url = server_url
        self.request_id = 1
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        self.conversation_context = None

    def _get_next_id(self):
        self.request_id += 1
        return self.request_id

    def list_tools(self):
        """列出所有可用工具"""
        payload = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "tools/list",
            "params": {
                "_meta": {
                    "progressToken": self._get_next_id()
                }
            }
        }

        response = requests.post(
            self.server_url,
            json=payload,
            headers=self.headers
        )

        if response.status_code != 200:
            raise Exception(f"获取工具列表失败: {response.text}")

        return response.json()

    def call_tool(self, tool_name: str, arguments: dict):
        """调用工具"""
        payload = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
                "_meta": {
                    "progressToken": self._get_next_id()
                }
            }
        }

        response = requests.post(
            self.server_url,
            json=payload,
            headers=self.headers
        )

        if response.status_code != 200:
            raise Exception(f"工具调用失败: {response.text}")

        return response.json()

    def call_add(self, a: int, b: int) -> int:
        """专门调用 add 工具"""
        result = self.call_tool("add", {"a": a, "b": b})

        # 提取结果 - 根据实际的响应格式调整
        if "result" in result:
            if "structuredContent" in result["result"] and "result" in result["result"]["structuredContent"]:
                return result["result"]["structuredContent"]["result"]
            elif "content" in result["result"] and len(result["result"]["content"]) > 0:
                # 尝试从 content 字段提取
                content = result["result"]["content"][0]
                if "text" in content:
                    try:
                        return int(content["text"])
                    except ValueError:
                        return content["text"]
        elif "error" in result:
            raise Exception(f"工具执行错误: {result['error']}")

        # 如果以上都不匹配，打印完整响应以便调试
        print(f"未知的响应格式: {json.dumps(result, indent=2)}")
        raise Exception(f"无法解析响应格式")

    def call_query_medical_knowledge_by_text(self, query: str) -> Dict[str, Any]:
        """
        文本接口：查询医疗知识（内部自动处理文本转向量）
        Args:
            query: 用户查询文本
        Returns:
            查询结果，包含知识和更新后的对话上下文
        """
        # 构造请求
        payload = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "tools/call",
            "params": {
                "name": "query_medical_knowledge_by_text",
                "arguments": {
                    "request": {
                        "query": query,
                        "conversation_context": self.conversation_context
                    }
                },
                "_meta": {
                    "progressToken": self._get_next_id()
                }
            }
        }

        # 发送请求
        response = requests.post(self.server_url, json=payload, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"查询失败: {response.text}")

        result = response.json()

        # 处理响应（与call_query_medical_knowledge相同）
        if "result" in result:
            if "structuredContent" in result["result"]:
                response_data = result["result"]["structuredContent"]
            elif "content" in result["result"] and len(result["result"]["content"]) > 0:
                content = result["result"]["content"][0]
                if "text" in content:
                    try:
                        response_data = json.loads(content["text"])
                    except json.JSONDecodeError:
                        response_data = content["text"]
            else:
                raise Exception("无法解析响应格式")
        elif "error" in result:
            raise Exception(f"工具执行错误: {result['error']}")
        else:
            raise Exception("未知的响应格式")

        # 更新客户端对话上下文
        if "conversation_context" in response_data:
            self.conversation_context = response_data["conversation_context"]

        return response_data

    # 访问MCP服务query_medical_knowledge接口， 检索医疗科普知识（新话题 + 追问， 支持多轮对话）
    def call_query_medical_knowledge(self, query: str, query_embedding: List[float]) -> Dict[str, Any]:
        """
        查询医疗知识
        Args:
            query: 用户查询文本
            query_embedding: 查询向量
        Returns:
            查询结果，包含知识和更新后的对话上下文
        """
        # 构造请求
        payload = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "tools/call",
            "params": {
                "name": "query_medical_knowledge",
                "arguments": {
                    "request": {
                        "query": query,
                        "embedding": query_embedding,
                        "conversation_context": self.conversation_context
                    }
                },
                "_meta": {
                    "progressToken": self._get_next_id()
                }
            }
        }

        # 发送请求
        response = requests.post(self.server_url, json=payload, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"查询失败: {response.text}")

        result = response.json()

        # 处理响应
        if "result" in result:
            if "structuredContent" in result["result"]:
                response_data = result["result"]["structuredContent"]
            elif "content" in result["result"] and len(result["result"]["content"]) > 0:
                content = result["result"]["content"][0]
                if "text" in content:
                    try:
                        response_data = json.loads(content["text"])
                    except json.JSONDecodeError:
                        response_data = content["text"]
            else:
                raise Exception("无法解析响应格式")
        elif "error" in result:
            raise Exception(f"工具执行错误: {result['error']}")
        else:
            raise Exception("未知的响应格式")

        # 更新客户端对话上下文
        if "conversation_context" in response_data:
            self.conversation_context = response_data["conversation_context"]

        return response_data

    # 访问MCP服务generate_medical_prompt接口， 生成医疗科普标准化提示词
    def call_generate_medical_prompt(self, query: str, medical_contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        调用生成医疗提示词工具
        Args:
            query: 用户查询问题
            medical_contexts: 知识上下文列表，每个上下文应包含 source 和 content 字段
        Returns:
            包含标准化提示词的字典
        """
        # 构造请求参数
        request_data = {
            "request": {
                "query": query,
                "contexts": medical_contexts
            }
        }

        result = self.call_tool("generate_medical_prompt", request_data)
        print(f"生成提示词工具响应: {json.dumps(result, indent=2, ensure_ascii=False)}")

        # 处理响应
        if "result" in result:
            if "structuredContent" in result["result"]:
                return result["result"]["structuredContent"]
            elif "content" in result["result"] and len(result["result"]["content"]) > 0:
                content = result["result"]["content"][0]
                if "text" in content:
                    try:
                        return json.loads(content["text"])
                    except json.JSONDecodeError:
                        return content["text"]
        elif "error" in result:
            raise Exception(f"工具执行错误: {result['error']}")

        print(f"未知的响应格式: {json.dumps(result, indent=2)}")
        raise Exception("无法解析响应格式")



    def reset_conversation(self):
        """重置对话"""
        self.conversation_context = None

    def get_conversation_summary(self) -> Dict[str, Any]:
        """获取对话摘要"""
        if not self.conversation_context:
            return {"status": "no_active_conversation"}

        return {
            "status": "active",
            "current_topic": self.conversation_context.get("current_topic"),
            "knowledge_count": len(self.conversation_context.get("history_ids", [])),
            "query_count": len(self.conversation_context.get("previous_queries", []))
        }



# 供 Dify 使用的函数
def tool_function(inputs: dict) -> str:
    """同步调用 MCP 工具的包装函数"""
    client = MCPClientSync()

    try:
        # 假设输入格式为 {"a": 5, "b": 3}
        a = inputs.get("a", 0)
        b = inputs.get("b", 0)

        result = client.call_add(a, b)
        return json.dumps({"result": result}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# 测试同步版本
if __name__ == "__main__":
    client = MCPClientSync()

    try:
        # 测试工具列表
        print("📋 获取工具列表...")
        tools = client.list_tools()
        print("工具列表:", json.dumps(tools, indent=2, ensure_ascii=False))

        # 测试 add 工具
        print("\n🔧 测试 add 工具...")
        result = client.call_add(15, 25)
        print(f"✅ add(15, 25) = {result}")



    except Exception as e:
        print(f"❌ 错误: {e}")