from typing import Dict, Any, List, Tuple
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState


class RrfMergeNode(BaseNode):
    name = "rrf_merge_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 获取多路检索结果(基础向量 + HyDE + 知识图谱反查)
        embedding_chunks = state.get('embedding_chunks') or []
        hyde_embedding_chunks = state.get('hyde_embedding_chunks') or []
        kg_chunks = state.get('kg_chunks') or []

        # 2. 定义多路检索结果和对应的权重映射
        search_result_weight = {
            "embedding_search_chunks": (self._validate_state(embedding_chunks), 1.0),
            "hyde_embedding_chunks": (self._validate_state(hyde_embedding_chunks), 1.0),
            "kg_chunks": (self._validate_kg_chunks(kg_chunks), self.config.kg_rrf_weight),
        }

        # 3. 收集映射表中的搜索结果和权重
        rrf_inputs = list(search_result_weight.values())

        # 4. 利用rrf计算多路文档的分数（基础向量 + HyDE + 知识图谱反查）
        merged_rrf_results: List[Tuple[Dict[str, Any], float]] = self._merger_rrf_docs(rrf_inputs, self.config.rrf_k,
                                                                                       self.config.rrf_max_results)

        # 5. 更新state
        merged_rrf_results = [rrf for rrf, _ in merged_rrf_results]
        state['rrf_chunks'] = merged_rrf_results

        # 6. 返回
        return state

    def _validate_state(self, search_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        检索多路结果的校验
        :param search_chunks: 搜索到的结果
        :return:
        """

        # 1. 判断是否为空
        if not search_chunks:
            return []

        search_results = []

        # 2. 遍历字典对象
        for res in search_chunks:
            # 2.1 res 必须存在且为 dict
            if not res or not isinstance(res, dict):
                continue

            # 2.2 获取entity对象
            entity = res.get('entity')

            # 2.3 entity 必须存在且为 dict
            if not entity or not isinstance(entity, dict):
                continue

            search_results.append(entity)

        return search_results

    def _validate_kg_chunks(self, kg_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        校验知识图谱反查得到的 chunk。
        kg_query_node 回填的是 Milvus 原始实体 dict(含 chunk_id/content)，无需再拆 entity 层。
        :param kg_chunks: KG 回填的切片列表
        :return: 合法切片列表
        """
        if not kg_chunks:
            return []

        search_results = []
        for chunk in kg_chunks:
            if not chunk or not isinstance(chunk, dict):
                continue
            if not chunk.get('chunk_id') or not chunk.get('content'):
                continue
            search_results.append(chunk)

        return search_results

    def _merger_rrf_docs(self, rrf_inputs: List[Tuple[List[Dict[str, Any]], float]], rrf_k: int,
                         rrf_max_results: int) -> List[Tuple[Dict[str, Any], float]]:
        """
        RRF计算多路检索返回的文档切分
        :param rrf_inputs: 多路检索的文档和对应权重
        :param rrf_k: 平滑参数
        :param rrf_max_results: 返回最大个数
        :return:
        """

        chunk_score = {}
        chunk_data = {}
        # 1. 遍历所有路的检索结果
        for search_result, weight in rrf_inputs:
            # 2. 遍历某一路的检索结果
            for rank, res in enumerate(search_result, 1):

                # 2.1 计算rrf分数
                rrf_score = weight / (rrf_k + rank)

                # 2.2 获取chunk_id
                chunk_id = res.get('chunk_id')

                # 2.3 判断
                if not chunk_id:
                    continue

                # 2.4 存储chunk_id的分数
                chunk_score[chunk_id] = chunk_score.get(chunk_id, float(0)) + rrf_score

                # 2.5 存储chunk_id和chunk对象
                chunk_data.setdefault(chunk_id, res)  # 去重

        # 3. 排序以及构建chunk对象和得分结果
        final_rrf_result = sorted([(chunk_data.get(chunk_id), score) for chunk_id, score in chunk_score.items()],
                                  key=lambda x: x[1], reverse=True)

        # 4. 返回
        return final_rrf_result[:rrf_max_results] if rrf_max_results else final_rrf_result


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#3"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfMergeNode()
    result = rrf_node.process(mock_state)
