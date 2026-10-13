from typing import Tuple, List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pymilvus import MilvusClient, DataType
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import StateFieldError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.prompt import load_prompt, render
from knowledge.utils.client.storage_clients import StorageClients


class ItemNameRecognitionNode(BaseNode):
    name = "item_name_recognition_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        主要职责:
        1. 负责利用LLM提取商品的具体型号 (名)
        2. 嵌入商品具体型号
        3. 存储到Milvus中
        :param state: 状态对象
        :return: 状态对象
        """

        # 1. 参数校验
        file_title, chunks, item_name_chunk_k, item_name_chunk_size = self._validate_state(state)

        # 2. 构建上下文
        item_name_context = self._prepare_llm_context(chunks, item_name_chunk_k)

        # 3. 调用llm模型 自动提取商品名
        item_name = self.recognition_item_name(item_name_context, file_title)

        # 4. 向量化
        dense_vector, sparse_vector = self.embedding_item_name(item_name)

        # 5. 入库
        self._insert_milvus(dense_vector, sparse_vector, file_title, item_name)

        # 6. 回填(更新LLM提前到item_name)
        self.fill_llm_item_name(state, item_name)

        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[str, list, int, int]:
        """
        参数校验
        :param state: 状态对象
        :return: md_content, file_title, item_name_chunk_k, item_name_chunk_size
        """

        # 1. 获取文档标题 (商品具体型号): 兜底
        file_title = state.get("file_title")

        # 2. 判断
        if not file_title:
            raise StateFieldError(node_name=self.name, field_name="file_title", expected_type=str)

        # 3. 获取chunks 给LLM做上下文
        chunks = state.get("chunks")

        # 4. 判断
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name="chunks", expected_type=list)

        # 5. 获取item_name_chunk_k 以及 item_name_chunk_content
        item_name_chunk_k = self.config.item_name_chunk_k
        item_name_chunk_size = self.config.item_name_chunk_size

        # 6. 判断
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValueError(f"商品名识别的辅助切片数不合法")

        if not item_name_chunk_size or item_name_chunk_size <= 0:
            raise ValueError(f"商品名识别的辅助切片内容长度不合法")

        # 7. 返回
        return file_title, chunks, item_name_chunk_k, item_name_chunk_size

    def _prepare_llm_context(self, chunks: list, item_name_chunk_k: int) -> str:
        """
        准备商品名识别的上下文
        :param chunks: 该文档的全部切片
        :param item_name_chunk_k: 准备使用的块数
        :return: 上下文信息
        """

        final_content = []

        # 1. 遍历
        for index, chunk in enumerate(chunks[:item_name_chunk_k]):
            # 1.1. 不是字典类型过滤掉这一块
            if not isinstance(chunk, dict):
                continue

            # 1.2. 获取chunk中content
            content = chunk.get("content")

            splice_content = f"[切片] - f{index + 1} - {content}"

            final_content.append(splice_content)

        return "\n".join(final_content)

    def recognition_item_name(self, item_name_context: str, file_title: str) -> str:
        """
        商品名识别并提取
        :param item_name_context: 上下文信息
        :param file_title: 文件标题
        :return: 商品名
        """

        # 1.调用LLM客户端
        try:
            llm_client: ChatOpenAI = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"OpenAI 的LLM客户端创建失败,降级使用文件标题{file_title}作为商品名 {str(e)}")
            return file_title

        # 2. 调用LLM模型
        # 2.1 获取商品名识别的llm系统提示词
        system_prompt = load_prompt("import", "item_name_system")

        # 2.2 获取商品名识别的llm用户提示词
        user_prompt = render("import", "item_name_user", file_title=file_title, context=item_name_context)

        try:
            # 3. 调用 返回AIMessage对象
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # 4. 获取AI回复的具体内容
            # 注意：strip() 不能写空参 strip('')（空字符集等于不清理），
            # Qwen3 类模型常在正文前输出换行，不 strip 会把 "\n\n商品名" 写进库，导致对齐失败
            llm_result = " ".join(llm_response.content.split()).strip()
            if not llm_result or llm_result == 'UNKNOWN':
                self.logger.error(f"LLM提取商品名失败，降级使用文件标题{file_title}作为商品名兜底")
                return file_title

            self.logger.info(f"LLM为文档：{file_title} 提取的商品名：{llm_result}")
            return llm_result
        except Exception as e:
            self.logger.error(f"LLM提取商品名失败，降级使用文件标题{file_title}作为商品名: {str(e)}")
            return file_title

    def embedding_item_name(self, item_name: str) -> Tuple[List, Dict[str, Any]] | None:
        """
        商品名嵌入
        :param item_name: 商品名
        :return: 稠密嵌入向量, 稀疏嵌入向量
        """

        # 1. 获取嵌入模型客户端
        try:
            bge_m3_client: BGEM3EmbeddingFunction = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3 嵌入模型客户端创建失败: {str(e)}")
            return None, None
        try:
            # 2. 计算稠密和稀疏向量
            vector_result = bge_m3_client.encode_documents([item_name])

            # 3. 解析向量
            # 3.1 稠密向量
            dense_vector = vector_result.get("dense")[0].tolist()

            # 3.2 稀疏向量
            sparse_csr = vector_result.get("sparse")
            # 3.2.1 获取行索引
            start_index = sparse_csr.indptr[0]
            end_index = sparse_csr.indptr[1]
            # 3.2.2 获取token_id
            token_ids = sparse_csr.indices[start_index:end_index].tolist()
            # 3.2.3 获取权重
            weights = sparse_csr.data[start_index:end_index].tolist()
            # 3.2.4 构建稀疏向量字典
            sparse_vector = dict(zip(token_ids, weights))

            return dense_vector, sparse_vector

        except Exception as e:
            self.logger.error(f"{item_name}嵌入计算失败: {str(e)}")
            return None, None

    def _insert_milvus(self, dense_vector: List, sparse_vector: Dict[str, Any], file_title: str, item_name: str):
        """
        将llm识别的商品名,保存到Milvus里
        row行记录{"dense_vector": 值,"sparse_vector": 值, "file_title": 值, "item_name": 值}
        :param dense_vector: 稠密向量
        :param sparse_vector: 稀疏向量
        :param file_title: 文件标题
        :param item_name: 商品名
        :return:
        """

        # 1. 判断稠密向量和稀疏向量是否存在
        if not dense_vector or not sparse_vector:
            self.logger.error(f"商品名:{item_name} 稠密向量或稀疏向量不存在")
            return

        # 2. 获取milvus客户端
        try:
            milvus_client: MilvusClient = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"获取Milvus客户端失败: {e}")
            return

        # 3. Milvus三大核心概念（集合：Collection[1.集合名 2.字段约束：schema 3.索引]）
        # 3.1 集合名：表名类似于归纳数据的容器，逻辑概念
        # 3.2 约束：类似于MySQL字段的长度、字段的类型..
        # 3.3 索引：类似于MySQL中的索引【索引类型比较多 B+树 Hash】。Milvus索引类型也有很多（专门针对于稠密向量的索引类型 针对于稀疏向量的索引类型 标量字段类型、主键类型）
        # 索引：本质就是算法（图、树、hash..）目的：提高检索【查询】效率。milvus中不管稠密向量索引还是稀疏向量的索引都是为了能够快速找到和问题相似的向量。
        # 使用Milvus的流程：①：创建集合（约束、索引）②：插入数据  ③：查询/检索

        # 4. 获取集合名称
        item_name_collection_name = self.config.item_name_collection

        try:
            # 5. 创建Milvus集合
            if not milvus_client.has_collection(item_name_collection_name):
                self._create_item_name_collection(item_name_collection_name, milvus_client)
            else:
                # 5.1 同一文档重复导入时先清掉旧的商品名记录，避免同一文档出现多条近似商品名
                #（否则查询侧会把它们当歧义候选，返回"您是在查询以下产品吗"）
                try:
                    milvus_client.delete(
                        collection_name=item_name_collection_name,
                        filter=f'file_title == "{file_title}"',
                    )
                except Exception as e:
                    self.logger.warning(f"清理旧商品名失败（不阻断本次插入）: {str(e)}")

            # 6. 构建数据行
            item_name_data_row = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }

            # 7. 插入数据
            inserted_result = milvus_client.insert(item_name_collection_name, [item_name_data_row])
            # pymilvus insert() 返回 {"insert_count":N,"ids":[...]}, 主键列在 ids（不是 'pk'，取 'pk' 会得到 None 导致 len 报错）
            self.logger.info(f"向集合：{item_name_collection_name} 插入数据成功，行数：{inserted_result.get('insert_count', len(inserted_result.get('ids') or []))}")

        except Exception as e:
            self.logger.error(f"向集合：{item_name_collection_name} 插入数据失败: {str(e)}")

    def _create_item_name_collection(self, item_name_collection_name: str, milvus_client: MilvusClient):
        """
        创建商品名向量数据库的集合
        :param item_name_collection_name: 商品名向量数据库的集合名称
        :param milvus_client: Milvus客户端
        :return:
        """

        # 1. 创建schema约束
        schema = milvus_client.create_schema()
        # 1.1 创建主键字段约束
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=10)
        # 1.2 创建标量字段约束
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        # 1.3 创建向量字段约束
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 2. 创建索引(主键字段建立索引,标量建立,向量字段建立)
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        # 3. 创建集合
        milvus_client.create_collection(
            collection_name=item_name_collection_name,
            schema=schema,
            index_params=index_params,
        )
        self.logger.info(f"集合:{item_name_collection_name} 创建成功")

    def fill_llm_item_name(self, state: ImportGraphState, item_name: str):
        """
        回填item_name
        位置一: 回填给chunk 方便下游模型
        位置二: 回填给state 方便其他节点
        :param state: ImportGraphState
        :param item_name: 商品名称
        :return:
        """

        # 1. 更新chunks item_name
        chunks = state.get('chunks')
        for chunk in chunks:
            chunk["item_name"] = item_name

        # 2. state的item_name
        state["item_name"] = item_name

        return state




if __name__ == '__main__':
    import json

    setup_logging()

    # 1. 读取chunk.json
    chunk_json_path = r""
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建state
    state = {
        "file_title": "万用表的使用",
        "chunks": chunk_content
    }

    # 3. 实例化节点
    node = ItemNameRecognitionNode()

    # 4. 调用process
    result = node.process(state)

    # 5. 输出结果
    print(f"商品名: {result.get('item_name')}")
    print(f"chunks数量: {len(result.get('chunks', []))}")
    print(f"首个chunk是否含item_name: {'item_name' in result['chunks'][0]}")