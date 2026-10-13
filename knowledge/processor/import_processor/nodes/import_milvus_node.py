from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Sequence, Optional
from pymilvus import MilvusClient, DataType
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError, MilvusError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.storage_clients import StorageClients


@dataclass
class _SCALAR_FILED_SPC:
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None


_SCALAR_FILED: Sequence[_SCALAR_FILED_SPC] = (
    # Milvus VARCHAR 的 max_length 上限为 65535（按字节计），之前误写 65565 会直接建表失败
    _SCALAR_FILED_SPC(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FILED_SPC(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FILED_SPC(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FILED_SPC(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FILED_SPC(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
)


class _MilvusSchemaBuilder:
    """
    负责处理和Milvus字段约束相关逻辑
    """

    @staticmethod
    def build_schema(milvus_client: MilvusClient, dim: int):
        """
        创建schema
        enable_dynamic_field: 动态字段
        :param milvus_client: Milvus客户端
        :param dim: 向量维度
        :return:
        """

        # 1. 创建schema
        schema = milvus_client.create_schema(enable_dynamic_field=True)

        # 2. 添加字段的约束
        # 2.1 添加主键字段的约束
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        # 2.2 添加向量字段的约束
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 2.3 添加标量字段的约束 5个
        for spec in _SCALAR_FILED:
            kwargs: Dict = {
                "field_name": spec.field_name,
                "datatype": spec.datatype,
            }
            if spec.max_length:
                kwargs["max_length"] = spec.max_length

            schema.add_field(**kwargs)

        # 3. 返回schema
        return schema


class _MilvusIndexBuilder:

    @staticmethod
    def build_index_params(milvus_client: MilvusClient):
        index = milvus_client.prepare_index_params()

        # 稠密向量：AUTOINDEX
        index.add_index(field_name="dense_vector",
                        index_name="dense_vector_index",
                        index_type="AUTOINDEX",
                        metric_type="COSINE")

        # 稀疏向量：倒排索引
        index.add_index(field_name="sparse_vector",
                        index_name="sparse_vector_index",
                        index_type="SPARSE_INVERTED_INDEX",
                        metric_type="IP")

        return index


class _MilvusInsert:

    def __init__(self, milvus_client: MilvusClient, collection_name: str):
        self._milvus_client = milvus_client
        self._collection_name = collection_name

    def insert_rows(self, datas: List[Dict[str, Any]]):
        # 1. 插入数据
        inserted_result = self._milvus_client.insert(
            collection_name=self._collection_name,
            data=datas
        )
        # 2. 得到每一个chunk_id
        chunk_ids = inserted_result.get('ids')

        # 3. 回填到chunk中
        for chunk_id, chunk in zip(chunk_ids, datas):
            chunk["chunk_id"] = chunk_id


class ImportMilvusNode(BaseNode):
    """
    角色:充当门面模式
    """

    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1. 校验state
        chunks, dim = self._validate_state(state)

        # 2. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"获取Milvus客户端失败: {e}")
            raise MilvusError(message=f"获取 Milvus 客户端失败,异常原因:{str(e)}", node_name=self.name)

        # 3. 获取chunks集合名称
        chunks_collection = self.config.chunks_collection

        # 4. 创建集合
        self._create_chunks_collection(chunks_collection, milvus_client, dim)

        # 4.1 同一文档重复导入：先删除旧的切片，否则旧数据会一直参与检索
        file_title = state.get("file_title", "")
        if file_title:
            try:
                milvus_client.delete(
                    collection_name=chunks_collection,
                    filter=f'file_title == "{file_title}"',
                )
            except Exception as e:
                self.logger.warning(f"清理旧切片失败（不阻断本次插入）: {str(e)}")

        # 5. 插入数据
        _inserter = _MilvusInsert(milvus_client, chunks_collection)
        _inserter.insert_rows(chunks)

        # 6. 返回
        return state

    def _validate_state(self, state: ImportGraphState) -> Tuple[list, int]:

        self.log_step("validate", "参数校验")
        chunks = state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError("待入库的 chunks 为空或类型无效", self.name)

        validated_chunks = []
        for i, chunk in enumerate(chunks):
            # 类型不对 → 抛异常（和上游 embedding 节点保持一致）
            if not isinstance(chunk, dict):
                raise ValidationError(
                    f"chunks[{i}] 类型无效：期望 dict，实际为 {type(chunk).__name__}", self.name
                )
            # 缺少向量 → 跳过（嵌入可能部分失败，属于数据级容错）
            if chunk.get("dense_vector") and chunk.get("sparse_vector"):
                validated_chunks.append(chunk)
            else:
                self.logger.warning(f"chunks[{i}] 缺少混合向量，已跳过")

        if not validated_chunks:
            raise ValidationError("所有 chunk 均无有效向量，无法入库", self.name)

        dim = len(validated_chunks[0]["dense_vector"])
        self.logger.info(f"有效 chunks：{len(validated_chunks)}，向量维度：{dim}")
        return validated_chunks, dim

    def _create_chunks_collection(self, chunks_collection: str, milvus_client: MilvusClient, dim: int):

        # 1. 判断集合
        if milvus_client.has_collection(chunks_collection):
            self.logger.info(f"集合 {chunks_collection} 已存在,跳过创建")
            return

        # 2. 创建约束字段
        schema = _MilvusSchemaBuilder.build_schema(milvus_client, dim)
        # 3. 创建索引
        index_params = _MilvusIndexBuilder.build_index_params(milvus_client)
        # 4. 创建集合
        milvus_client.create_collection(collection_name=chunks_collection, schema=schema, index_params=index_params)


def _cli_main() -> None:
    import json
    from pathlib import Path
    setup_logging()

    temp_dir = Path(
        r""
    )
    input_path = temp_dir / "chunks_vector_bak.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    state: ImportGraphState = {"chunks": content.get('chunks')}

    node = ImportMilvusNode()
    result_state = node.process(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"结果已保存至: {output_path}")


if __name__ == '__main__':
    _cli_main()
