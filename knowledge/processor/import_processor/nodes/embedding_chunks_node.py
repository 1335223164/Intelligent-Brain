from pathlib import Path
from typing import List, Dict, Any
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import StateFieldError, EmbeddingError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients


class EmbeddingChunksNode(BaseNode):
    name = "embedding_chunks_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1. 校验状态
        self.log_step("step1", "检验chunks的状态")
        self._validate_state(state)

        # 2. 获取嵌入模型客户端
        self.log_step("step2", "获取BGE-M3嵌入模型客户端")
        try:
            embed_model = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型获取失败,原因: {str(e)}")
            raise EmbeddingError(message=f"BGE-M3嵌入模型获取失败,原因: {str(e)}", node_name=self.name)

        # 3.批量嵌入
        self.log_step("step3", "批量嵌入chunks")
        # 3.1 获取批量嵌入批次
        batch_size = self.config.embedding_batch_size
        # 3.2 获取chunks的总数
        total = len(state.get("chunks"))
        # 3.3 遍历
        final_chunks = []
        for index in range(0, total, batch_size):
            # 获取当前这一批
            batch = state.get("chunks")[index:index + batch_size]
            # 当前这一批的最后一个编号
            batch_end = index + len(batch)
            self.logger.info(f"批量嵌入chunks,当前批次: {index + 1}-{batch_end} / {total}")

            # 批量嵌入
            current_chunks = self._embed_chunks(batch, embed_model)
            final_chunks.extend(current_chunks)

        # 4. 更新state的chunks
        state["chunks"] = final_chunks

        # 5. 返回state
        return state

    def _validate_state(self, state: ImportGraphState) -> List[Dict[str, Any]]:
        # 1. 获取chunks
        chunks = state.get("chunks")

        # 2. 校验
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)

        # 3. 遍历检验每一个chunk类型
        for index, chunk in enumerate(chunks):
            # 检验单个chunk
            if not isinstance(chunk, dict):
                raise StateFieldError(node_name=self.name, field_name="chunks",
                                      message=f"[chunk_{index + 1}] 类型和期望的类型不匹配，实际的类型{type(chunk).__name__}")

        # 4. 返回chunks
        return chunks

    def _embed_chunks(self, batch: List[Dict[str, Any]], embed_model: BGEM3EmbeddingFunction) -> List[Dict[str, Any]]:
        """
        批量嵌入chunks
        :param batch: 批量chunks
        :param embed_model: 嵌入模型
        :return: 嵌入好的chunks
        """

        # 1. 获取要嵌入的内容
        embedding_documents = [f"{chunk.get("item_name",'')}\n{chunk.get("content",'')}" for chunk in batch]

        # 2. 嵌入chunks内容
        try:
            vector_result = embed_model.encode_documents(embedding_documents)
        except Exception as e:
            raise EmbeddingError(message=f"嵌入失败,原因:{str(e)}",node_name=self.name)

        if not vector_result:
            raise EmbeddingError(message="嵌入结果为空",node_name=self.name)

        # 3. 遍历这一批的每一个chunk
        sparse_csr = vector_result.get("sparse")
        for i,chunk in enumerate(batch):
            chunk["dense_vector"] = vector_result.get('dense')[i].tolist()
            chunk["sparse_vector"] = self._extract_sparse_vector(sparse_csr,i)

        # 4. 返回
        return batch

    def _extract_sparse_vector(self, sparse_csr, index: int):
        """
        从稀疏向量矩阵里提前当前chunk的向量
        :param sparse_csr: 稀疏向量矩阵
        :param index: 当前chunk的索引
        :return:
        """

        # 1. 从行索引中获取当前chunk的起始索引和结束索引
        start_index =  sparse_csr.indptr[index]
        end_index = sparse_csr.indptr[index + 1]

        # 2. 获取当前chunk的token_id
        token_ids = sparse_csr.indices[start_index:end_index].tolist()
        # 3. 获取当前chunk的权重
        data = sparse_csr.data[start_index:end_index].tolist()

        # 4. 构建稀疏向量矩阵
        sparse_vector = dict(zip(token_ids, data))

        return sparse_vector




if __name__ == '__main__':
    import json

    setup_logging()

    base_dir = Path(
        r""
    )
    input_path = base_dir / "chunks_item_name.json"
    output_path = base_dir / "chunks_vector.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    node = EmbeddingChunksNode()
    result_state = node.process({"chunks": chunks_data.get('chunks')})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"向量生成完成，结果已保存至:\n{output_path}")


