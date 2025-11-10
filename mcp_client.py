# mcp_client_async_fixed.py
import aiohttp
import asyncio
import json

## old代码
class MCPClient:
    def __init__(self, server_url: str = "http://127.0.0.1:8001/mcp"):
        self.server_url = server_url
        self.request_id = 1
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    def _get_next_id(self):
        self.request_id += 1
        return self.request_id

    async def list_tools(self):
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

        async with self.session.post(self.server_url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"获取工具列表失败: {error_text}")

            result = await response.json()
            return result

    async def call_tool(self, tool_name: str, arguments: dict):
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

        async with self.session.post(self.server_url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"工具调用失败: {error_text}")

            result = await response.json()
            return result

    async def call_add(self, a: int, b: int) -> int:
        """专门调用 add 工具"""
        result = await self.call_tool("add", {"a": a, "b": b})

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


async def main():
    async with MCPClient() as client:
        try:
            # 1. 首先列出所有工具
            print("📋 获取工具列表...")
            tools_result = await client.list_tools()
            print(f"工具列表响应: {json.dumps(tools_result, indent=2, ensure_ascii=False)}")

            # 2. 调用 add 工具
            print("\n🔧 调用 add 工具...")
            result = await client.call_add(5, 3)
            print(f"✅ add(5, 3) = {result}")

            # 3. 再测试一次
            print("\n🔧 再次调用 add 工具...")
            result2 = await client.call_add(10, 20)
            print(f"✅ add(10, 20) = {result2}")

        except Exception as e:
            print(f"❌ 错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())