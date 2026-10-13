from typing import Dict, Any, List, Tuple, Union
from knowledge.processor.import_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, _item_names_filter


class HybridVectorSearchNode(BaseNode):
    name = "hybrid_vector_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState,Dict[str,Any]]:

        # 1. 参数校验
        rewritten_query, item_names = self._validate_state(state)

        # 2. 获取嵌入模型客户端
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"bge_m3客户端获取失败,原因{str(e)}")
            return state

        # 3. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端获取失败,原因{str(e)}")
            return state

        # 4. 嵌入,检索
        try:
            # 嵌入查询问题
            embed_query_vector = generate_bge_m3_hybrid_vectors(model=bge_m3_client,
                                                                embedding_documents=[rewritten_query])
        except Exception as e:
            self.logger.error(f"用户问题{rewritten_query}嵌入向量问题失败,原因{str(e)}")
            return state

        # 5 创建混合检索请求
        try:

            # 5.1 获取过滤表达式
            expr, expr_params = _item_names_filter(item_names)

            # 5.2 获取混合检索请求
            #     每路子请求的 limit 决定候选池上限，需与最终 limit 一致，否则最终结果被截到子请求的 5
            hybrid_search_req = create_hybrid_search_requests(
                embed_query_vector["dense"][0], embed_query_vector["sparse"][0],
                expr=expr, expr_params=expr_params,
                limit=self.config.embedding_search_limit
            )

            # 5.3 执行混合检索请求
            hybrid_search_res = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_search_req,
                ranker_weights=[0.5, 0.5],
                norm_score=True,
                limit=self.config.embedding_search_limit,
                output_fields=["chunk_id", "content", "item_name", 'title']
            )

            if not hybrid_search_res or not hybrid_search_res[0]:
                return state

            # 5.4 修改自己的并且返回修改后的
            # state['embedding_chunks'] = hybrid_search_res[0]
            # 只返回自己更新的字段
            return {"embedding_chunks": hybrid_search_res[0]}

        except Exception as e:
            self.logger.error(f"用户问题{rewritten_query}执行混合检索请求失败,原因是: {str(e)}")
            return state

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




if __name__ == "__main__":

    import json

    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS-12 数字万用表"]  # 没用的
    }

    vector_search = HybridVectorSearchNode()
    result = vector_search.process(state)

    for r in result.get('embedding_chunks'):
        print(json.dumps(r, ensure_ascii=False, indent=2))