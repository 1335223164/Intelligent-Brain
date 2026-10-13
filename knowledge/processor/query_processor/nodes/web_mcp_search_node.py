import asyncio
import json
from typing import Dict, Any, List, Tuple, Union
import httpx
from mcp import ClientSession
# 直连原生 mcp SDK：openai-agents 的 MCPServerStreamableHttp 封装与本机 mcp 2.x 不兼容
# （mcp>=2 将 streamablehttp_client 重命名为 streamable_http_client 且返回值由 3 个变为 2 个），
# 用 agents 封装会在连接阶段报 500，故这里直接使用底层 SDK（已对当前服务端实测可用）。
from mcp.client.streamable_http import streamable_http_client
from json import JSONDecodeError
from knowledge.processor.import_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, _item_names_filter


class WebMcpSearchNode(BaseNode):
    name = "web_mcp_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState,Dict[str,Any]]:

        # 1. 参数校验
        rewritten_query, item_names = self._validate_state(state)

        # 2. 定义并且执行mcp调用
        # 调用方调用一个async修饰的方法，有且只有两种方式：方式一：继续添加await  方式二：将这个方法放到异步环境中(调用方是同步)
        # 联网搜索是可选增强：MCP 服务抖动/鉴权失败不应打断整个查询图，故此处兜底降级为空结果
        try:
            web_search_result = asyncio.run(self._execute_mcp_server(rewritten_query))
        except Exception as e:
            self.logger.warning(f"联网搜索失败，降级为无网页结果,原因:{str(e)}")
            web_search_result = []

        # 3. 判断
        if not web_search_result:
            return state
        # 4. 更新state
        # state['web_search_docs'] = web_search_result

        return {"web_search_docs": web_search_result}

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, list[str]]:
        # 1. 用户的问题 LLM重写后的
        rewritten_query = state.get("rewritten_query")

        # 2. 获取商品名称列表
        item_names = state.get("item_names")

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 4. 返回
        return rewritten_query, item_names

    async def _execute_mcp_server(self, rewritten_query: str) -> List[Dict[str, Any]]:
        """
        执行mcp服务（直连原生 mcp SDK 的 Streamable HTTP 客户端）
        一个mcp服务下可能有多个工具
        :param rewritten_query: 用户问题
        :return:
        """

        # 1. 定义鉴权请求头（联网搜索是阿里云百炼 MCP，用独立的 DashScope key，非硅基流动 key）
        headers = {"Authorization": f"Bearer {self.config.mcp_dashscope_api_key}"}

        async with httpx.AsyncClient(headers=headers, timeout=60) as http_client:
            # 2. 建立 Streamable HTTP 连接（base_url 需以 /mcp 结尾）并初始化会话
            async with streamable_http_client(self.config.mcp_dashscope_base_url, http_client=http_client) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    # 3. 协议握手
                    await session.initialize()
                    # 4. 调用工具（原生 SDK 首参为 name，非 agents 封装的 tool_name）
                    web_search_result = await session.call_tool(
                        "bailian_web_search",
                        arguments={"query": rewritten_query, "count": 3},
                    )

        # 5. 解析数据
        # 5.1 获取文本内容块
        if not web_search_result or not getattr(web_search_result, "content", None):
            return []
        text_content = web_search_result.content[0]
        if not text_content:
            return []

        # 5.2 获取文本内容块对象的内容
        text_content_text = getattr(text_content, "text", None)
        if not text_content_text:
            return []

        # 5.3 反序列化
        try:
            text_content_obj: Dict[str, Any] = json.loads(text_content_text)

            # 5.4 获取真正网页内容
            pages = text_content_obj.get('pages', [])
            if not pages:
                return []

            # 5.5 遍历
            web_search_results = []
            for page in pages:
                web_search_results.append({
                    "snippet": page.get('snippet', '').strip(),
                    "title": page.get('title', '').strip(),
                    "url": page.get('url', '').strip()
                })

            return web_search_results

        except JSONDecodeError as e:
            self.logger.error(f"web_search检索失败,原因:{str(e)} 失败内容:{e.doc} 失败的位置:{e.pos}")
            return []

if __name__ == '__main__':
    web_search_node = WebMcpSearchNode()

    mock_state = {
        "rewritten_query": "RS-12 数字万用表如何测量直流电压？",
        "item_names": ["RS-12 数字万用表"],
    }

    result = web_search_node.process(mock_state)
    docs = result.get("web_search_docs", []) if isinstance(result, dict) else []
    print(f"联网搜索命中 {len(docs)} 条结果：")
    for d in docs:
        print(" -", d.get("title"), "|", d.get("url"))