import logging
import os, json, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from json import JSONDecodeError
from time import sleep
from typing import Tuple, List, Dict, Any, Set
from langchain_core.messages import SystemMessage, HumanMessage
from neo4j.exceptions import Neo4jError
from pymilvus import DataType
from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import MilvusError
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.prompt import load_prompt
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils import uie_util
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.neo4j_utils import get_neo4j_driver

# ------------------------------------------------------------
ALLOWED_ENTITY_LABELS: Set[str] = {
    "Device", "Part", "Operation", "Step", "Warning", "Condition", "Tool"
}
MAX_ENTITY_NAME_LENGTH = 15
# 关系类型白名单
ALLOWED_RELATION_TYPES: Set[str] = ({
    "HAS_OPERATION", "HAS_PART", "HAS_STEP", "USES_TOOL",
    "HAS_WARNING", "NEXT_STEP", "AFFECTS", "REQUIRES",
    "MENTIONED_IN", "RELATED_TO",
})
DEFAULT_RELATION_TYPES = "RELATED_TO"

CYPHER_CLEAR_ITEM = """
    MATCH (n {item_name: $item_name}) DETACH DELETE n
"""

CYPHER_MERGE_CHUNK = """
    MERGE (c:Chunk {id: $chunk_id, item_name: $item_name})
"""

CYPHER_MERGE_ENTITY_TEMPLATE = """
    MERGE (n:Entity {{name: $name, item_name: $item_name}})
    ON CREATE SET
        n.source_chunk_id = $chunk_id,
        n.description     = $description
    ON MATCH SET
        n.description = CASE
            WHEN $description <> "" THEN $description
            ELSE coalesce(n.description, "")
        END
    SET n:`{label}`
"""

CYPHER_LINK_ENTITY_TO_CHUNK = """
    MATCH (n:Entity {name: $name, item_name: $item_name})
    MATCH (c:Chunk  {id: $chunk_id, item_name: $item_name})
    MERGE (n)-[:MENTIONED_IN]->(c)
"""

CYPHER_MERGE_RELATION_TEMPLATE = """
    MATCH (h:Entity {{name: $head, item_name: $item_name}})
    MATCH (t:Entity {{name: $tail, item_name: $item_name}})
    MERGE (h)-[:{rel_type}]->(t)
"""

# ------------------------------------------------------------

@dataclass
class ProcessingStats:
    """处理过程统计信息，用于日志和监控。"""

    total_chunks: int = 0
    processed_chunks: int = 0
    skipped_chunks: int = 0
    failed_chunks: int = 0
    total_entities: int = 0
    total_relations: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"处理完成: {self.processed_chunks}/{self.total_chunks} 切片成功, "
            f"{self.skipped_chunks} 无实体跳过, "
            f"{self.failed_chunks} 失败, "
            f"共 {self.total_entities} 实体 / {self.total_relations} 关系"
        )


class MilvusEntityWriter:
    """负责将实体向量化并写入 Milvus，仅供本模块内部使用。"""

    def __init__(self, milvus_url: str, collection_name: str):
        self.milvus_url = milvus_url
        self.collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def insert(self, milvus_client, entities: List[Dict], chunk_id: str, content: str, item_name: str) -> None:
        """对外唯一入口：将实体写入 Milvus。"""

        # 1. 判断实体是否存在
        if not entities:
            raise ValueError("参数校验失败，实体不存在")

        # 2. 获取去重后的实体名
        entities_names = list(dict.fromkeys(e["name"] for e in entities if e.get('name')))
        if not entities_names:
            raise ValueError("参数校验失败，无有效实体名")

        # 3. 获取嵌入模型
        bge_ef_model = AIClients.get_bge_m3_client()
        if bge_ef_model is None:
            raise MilvusError("嵌入模型获取失败")

        # 4. 创建集合（不存在则创建）
        try:
            self._ensure_collection(milvus_client, self.collection_name)
        except Exception as e:
            raise MilvusError(f"Milvus 创建集合失败: {e}")

        # 5. 嵌入向量化
        try:
            embedded_result = bge_ef_model.encode_documents(entities_names)
        except Exception as e:
            raise MilvusError(f"实体嵌入失败: {e}")

        # 6. 构建记录
        records = self._build_records(entities_names, embedded_result, chunk_id, content, item_name)
        if not records:
            raise MilvusError("构建 Milvus 记录为空")

        # 7. 写入 Milvus
        try:
            milvus_client.insert(collection_name=self.collection_name, data=records)
            self.logger.debug(f"Milvus 写入 {len(records)} 条实体向量")
        except Exception as e:
            raise MilvusError(f"Milvus 插入数据失败: {e}")

    def _ensure_collection(self, client, collection_name: str) -> None:
        """集合不存在则创建（schema + 索引）。"""

        # 1. 判断集合是否已存在
        if client.has_collection(collection_name):
            return

        # 2. 构建 schema
        schema = client.create_schema(enable_dynamic_field=True)
        schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("entity_name", DataType.VARCHAR, max_length=65535)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("source_chunk_id", DataType.VARCHAR, max_length=65535)
        schema.add_field("context", DataType.VARCHAR, max_length=65535)
        schema.add_field("item_name", DataType.VARCHAR, max_length=65535)

        # 3. 构建索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        # 4. 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

    @staticmethod
    def _build_records(
            entities_names: List[str], embedded_result: Dict[str, Any],
            chunk_id: str, content: str, item_name: str,
    ) -> List[Dict[str, Any]]:
        """组装插入记录。"""

        # 1. 校验嵌入结果
        if not embedded_result:
            raise ValueError("嵌入结果为空")

        # 2. 获取稠密向量和稀疏向量
        dense_vector_list = embedded_result.get("dense")
        sparse_matrix = embedded_result.get("sparse")

        # 3. 校验向量是否存在
        if not dense_vector_list or sparse_matrix is None:
            raise ValueError("参数校验失败，向量不存在")

        # 4. 获取对应块的部分内容作为上下文
        context = content[:200]
        records: List[Dict] = []

        # 5. 遍历每一个实体名，构建记录
        for idx, entity_name in enumerate(entities_names):
            # 5.1 边界检查
            if idx >= len(dense_vector_list):
                break

            # 5.2 获取稠密向量
            dense = dense_vector_list[idx]
            if hasattr(dense, "tolist"):
                dense = dense.tolist()

            # 5.3 解构稀疏向量（从 CSR 矩阵中提取当前实体的稀疏向量）
            start = sparse_matrix.indptr[idx]
            end = sparse_matrix.indptr[idx + 1]
            indices = sparse_matrix.indices[start:end].tolist()
            data = sparse_matrix.data[start:end].tolist()
            sparse_dict = dict(zip(indices, data))

            # 5.4 构建单条记录
            record = {
                "entity_name": entity_name,
                "context": context,
                "item_name": item_name,
                "source_chunk_id": chunk_id,
                "dense_vector": dense,
                "sparse_vector": sparse_dict,
            }

            records.append(record)

        return records


class Neo4jGraphWriter:
    def __init__(self, database: str = ""):
        self.database = database
        self.logger = logging.getLogger(self.__class__.__name__)

    def insert(self, driver, entities, relations, chunk_id, item_name):
        if not entities:
            raise ValueError("参数校验失败，实体列表为空")
        if not driver:
            raise Neo4jError("Neo4j 驱动获取失败")

        try:
            with self._session(driver) as session:
                session.execute_write(
                    self._write_graph_tx, entities, relations, chunk_id, item_name,
                )
            self.logger.info(f"Neo4j 写入: {len(entities)} 实体, {len(relations)} 关系")
        except Exception as e:
            raise Neo4jError(f"Neo4j 写入失败: {e}")

    def _session(self, driver):
        return driver.session(database=self.database)

    def clear(self, driver, item_name: str) -> None:
        if not driver:
            raise Neo4jError("Neo4j 驱动获取失败")

        try:
            with self._session(driver) as session:
                session.execute_write(
                    lambda tx, name: tx.run(CYPHER_CLEAR_ITEM, item_name=name),
                    item_name,
                )
            self.logger.info(f"Neo4j 旧数据已清理: {item_name}")
        except Exception as e:
            raise Neo4jError(f"Neo4j 清理失败: {e}")

    def _write_graph_tx(self, tx, entities, relations, chunk_id, item_name):
        # 1. 创建 Chunk 节点
        tx.run(CYPHER_MERGE_CHUNK, chunk_id=chunk_id, item_name=item_name)

        # 2. 创建实体节点 + 关联到 Chunk
        for entity in entities:
            name = entity.get("name")
            raw_label = entity.get("label")
            description = entity.get("description", "")

            # 动态格式化 Cypher，将安全标签注入
            cypher_query = CYPHER_MERGE_ENTITY_TEMPLATE.format(label=raw_label)
            tx.run(cypher_query, name=name, description=description,
                   chunk_id=chunk_id, item_name=item_name)

            # 关联实体到 Chunk
            tx.run(CYPHER_LINK_ENTITY_TO_CHUNK,
                   name=name, chunk_id=chunk_id, item_name=item_name)

        # 3. 创建实体间关系
        for rel in relations:
            head = rel.get("head")
            tail = rel.get("tail")
            rel_type = rel.get("type")

            cypher = CYPHER_MERGE_RELATION_TEMPLATE.format(rel_type=rel_type)
            tx.run(cypher, head=head, tail=tail, item_name=item_name)


class KnowLedgeGraphNode(BaseNode):
    name = "knowledge_graph_node"

    def __init__(self):
        super().__init__()
        self._milvus_writer = MilvusEntityWriter(self.config.milvus_url, self.config.entity_name_collection)
        self._neo4j_writer = Neo4jGraphWriter(self.config.neo4j_database)

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1. 参数校验
        validated_chunks, global_item_name = self._validate_get_inputs(state)

        # 2. 构建统计初始信息
        stats = ProcessingStats(total_chunks=len(validated_chunks))

        # 3. 获取客户端
        # try:
        milvus_client = StorageClients.get_milvus_client()
        neo4j_driver = get_neo4j_driver()
        # except ConnectionError as e:
        # self.logger.error(f"客户端获取失败,原因:{str(e)}")

        # 3. 删除已经存在的数据 (删除Milvus中存储实体名的集合下的item_name:幂等性 删除neo4j的整个库下的所有节点以及关系)
        self._clean_exist_double_data(stats, milvus_client, neo4j_driver, global_item_name)

        # 2. 批量处理(串行版本)
        self._process_all_chunks_v2(stats, validated_chunks, milvus_client, neo4j_driver)

        # 3. 简单日志观察
        self.logger.info(stats.summary())

        return state

    def _validate_get_inputs(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], str]:
        self.log_step("step1", "知识图谱构建参数校验")

        # 1. 获取基础字段
        chunks = state.get("chunks") or []
        global_item_name = str(state.get("item_name", "")).strip()

        # 2. 校验整体 chunks 是否存在
        if not chunks:
            raise ValueError("待提取图谱的切块(chunks)不存在，跳过图谱构建。")

        # 3. 逐个校验 Chunk 的有效性
        validated_chunks = []
        for i, chunk in enumerate(chunks):

            # 3.1 chunk 是否是字典
            if not isinstance(chunk, dict):
                self.logger.warning(f"第 {i} 个 chunk 不是字典类型，已抛弃。")
                continue

            # 3.2 处理 chunk_id
            raw_id = chunk.get("chunk_id")
            chunk_id = str(raw_id).strip() if raw_id is not None else f"kg_chunk_temp_{i}"

            # 3.3 获取 content 内容
            content = str(chunk.get("content", "")).strip()
            if not content:
                self.logger.warning(f"Chunk {chunk_id} 缺少 content，已抛弃。")
                continue

            # 3.4 获取 item_name（chunk 级别优先，全局兜底）
            chunk_item = str(chunk.get("item_name", "")).strip() or global_item_name
            if not chunk_item:
                self.logger.warning(f"Chunk {chunk_id} 缺少 item_name 归属，已抛弃。")
                continue

            # 3.5 更新 chunk 字段
            chunk["chunk_id"] = chunk_id
            chunk["item_name"] = chunk_item
            chunk["content"] = content

            # 3.6 加入有效列表
            validated_chunks.append(chunk)

        # 4. 校验清洗后是否还有有效数据
        if not validated_chunks:
            raise ValueError(f"经过清洗后，没有任何有效的 chunk（{len(validated_chunks)}）可用于构建图谱。")

        self.logger.info(f"参数校验完成: 原始 {len(chunks)} 块 -> 有效 {len(validated_chunks)} 块。")

        return validated_chunks, global_item_name

    def _clean_exist_double_data(self, stats, milvus_client, neo4j_driver, global_item_name):
        """
        删除milvus以及neo4j对于记录
        :param milvus_client:
        :param neo4j_driver:
        :param global_item_name:
        :return:
        """

        # 1. 删除milvus中的item_name=item_name的记录
        if not milvus_client:
            raise MilvusError("Milvus 客户端获取失败")

        collection_name = self.config.entity_name_collection
        try:
            if milvus_client.has_collection(collection_name):
                milvus_client.delete(
                    collection_name=collection_name,
                    filter=f'item_name == "{global_item_name}"',
                )
                self.logger.info(f"Milvus 旧数据已清理: item_name={global_item_name}")
        except Exception as e:
            raise MilvusError(f"Milvus 清理失败: {e}")

        # 2. 删除neo4j中的item_name=item_name的实体以及关系
        driver = neo4j_driver
        self._neo4j_writer.clear(driver, global_item_name)

    def _process_all_chunks_v2(self, stats, validated_chunks, milvus_client, neo4j_driver):
        """
        循环处理每一个chunk
        :param validated_chunks:
        :param milvus_client:
        :param neo4j_driver:
        :return:
        """

        # 1. 遍历所有chunks
        # for i, chunk in enumerate(validated_chunks):
        #
        #     if not isinstance(chunk, dict):
        #         continue
        #
        #     # 1. 获取chunk的信息
        #     chunk_id = chunk.get('chunk_id')
        #     item_name = chunk.get('item_name')
        #     content = chunk.get('content')
        #
        #     try:
        #         entities, relations_count = self._process_single_chunk(chunk_id, item_name, content,
        #                                                                milvus_client, neo4j_driver)
        #         stats.processed_chunks += 1
        #         stats.total_entities += entities
        #         stats.total_relations += relations_count
        #         self.logger.info(f"成功处理的 {chunk_id} / {len(validated_chunks)}")
        #     except Exception as e:
        #         stats.failed_chunks += 1
        #         stats.errors.append(str(e))
        #         self.logger.error(f"处理失败的 {chunk_id} / {len(validated_chunks)}")
        with ThreadPoolExecutor(max_workers=4) as pool:
            # 1. 提交所有任务
            future_to_idx = {}
            for i, chunk in enumerate(validated_chunks):
                content = chunk.get("content")
                chunk_id = str(chunk.get("chunk_id"))
                chunk_item = chunk.get("item_name")

                future = pool.submit(
                    self._process_single_chunk,
                    chunk_id, chunk_item, content, milvus_client, neo4j_driver
                )
                future_to_idx[future] = (i, chunk_id)

            # 2. 收集结果（按完成顺序）
            for future in as_completed(future_to_idx):
                idx, chunk_id = future_to_idx[future]
                try:
                    entity_count, relation_count, skipped = future.result()
                    if skipped:
                        # 切片无可抽取实体（纯表格/噪声等），属正常情况，不计入失败
                        stats.skipped_chunks += 1
                    else:
                        stats.processed_chunks += 1
                        stats.total_entities += entity_count
                        stats.total_relations += relation_count
                except Exception as e:
                    stats.failed_chunks += 1
                    msg = f"切片 {chunk_id} 处理失败: {e}"
                    stats.errors.append(msg)
                    self.logger.error(msg)

    def _process_single_chunk(self, chunk_id, item_name, content, milvus_client, neo4j_driver):
        # 1. 提取实体与关系（UIE 为主, LLM 兜底）
        graph_result = self._extract_graph(content)

        # 2. 获取清洗后的实体和关系
        final_entities = graph_result.get('entities') or []
        final_relations = graph_result.get('relations') or []

        # 3. 无有效实体：该切片没有可抽取的图谱信息（纯表格/噪声/LLM返回空entities），
        #    属正常情况，直接跳过写入（关系依附实体存在，无实体则无法建图），不记为失败
        if not final_entities:
            self.logger.info(f"切片 {chunk_id} 未抽取到有效实体，跳过图谱写入")
            return 0, 0, True

        # 4. 实体名写入Milvus
        self._milvus_writer.insert(milvus_client, final_entities, chunk_id, content, item_name)

        # 5. 写入neo4j
        self._neo4j_writer.insert(neo4j_driver, final_entities, final_relations, chunk_id, item_name)

        return len(final_entities), len(final_relations), False

    def _extract_graph(self, content):
        """
        优先使用本地 UIE 模型联合抽取实体/关系；
        UIE 不可用或抽取结果为空时，降级调用 LLM 兜底。
        :param content: 切片内容
        :return: {"entities": [...], "relations": [...]}
        """

        # 1. 尝试 UIE 本地抽取
        try:
            uie_result = uie_util.extract_triples(content)
            entities = self._clean_entities(uie_result.get("entities", []))
            unique_names = {e.get("name") for e in entities}
            relations = self._clean_relations(unique_names, uie_result.get("relations", []))
            if entities:
                self.logger.debug(f"UIE 抽取: {len(entities)} 实体 / {len(relations)} 关系")
                return {"entities": entities, "relations": relations}
        except Exception as e:
            self.logger.warning(f"UIE 抽取异常, 降级使用 LLM: {str(e)}")

        # 2. UIE 结果为空 -> LLM 兜底
        self.logger.info("知识图谱抽取降级使用 LLM (UIE 结果为空)")
        llm_response = self._extract_graph_with_retry(content)
        return self._parse_and_clean(llm_response)

    def _extract_graph_with_retry(self, content):
        """LLM 兜底抽取（带重试），仅在 UIE 无结果时调用。"""
        # 1. 获取llm客户端
        llm_client = AIClients.get_llm_client()

        if llm_client is None:
            raise ValueError(f"LLM客户端初始化失败")

        # 2. 系统提示词（UIE 兜底路径专用）
        system_prompt = load_prompt("import", "kg_extraction_system")

        # 3. 重试机制
        MAX_COUNT = self.config.kg_llm_retry_max
        for i in range(1, MAX_COUNT + 1):
            try:
                llm_response = llm_client.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"切分的信息\n\n{content}")
                ])

                result = getattr(llm_response, 'content', '').strip()

                if result:
                    return result

            except Exception as e:
                if i < MAX_COUNT:
                    # 睡一会
                    delay = 0.5 * (2 ** (i - 1))
                    self.logger.warning(f"开始第{i}次重试,间隔:{delay:.1f}s后开始")
                    sleep(delay)

        self.logger.error(f"已经进行了{MAX_COUNT}次重试,都是失败")

        return ""

    def _parse_and_clean(self, llm_response):
        """
        解析llm返回的json代码片段的围栏
        反序列化
        获取实体信息和关系信息
        清洗实体信息和关系信息
        返回实体信息和关系信息
        :param llm_response:
        :return:
        """

        # 1. 判断
        if not llm_response:
            raise ValueError(f"LLM提取chunk的图谱信息为空")

        # 2. 清洗json围栏
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response.strip())
        content = re.sub(r"\s*```$", "", cleaned.strip())

        # 3. 反序列化
        try:
            llm_response_obj = json.loads(content)

            # 4. 获取实体和关系信息
            entities = llm_response_obj.get('entities', [])
            relations = llm_response_obj.get('relations', [])

            # 5. 清洗实体
            cleaned_entities = self._clean_entities(entities)

            # 6. 获取清洗后的实体名
            cleaned_unique_entities = {entity.get('name') for entity in cleaned_entities}

            # 7. 清洗关系
            cleaned_relations = self._clean_relations(cleaned_unique_entities, relations)

            # 8. 返回结果
            return {"entities": cleaned_entities, "relations": cleaned_relations}

        except JSONDecodeError as e:
            self.logger.error(f"LLM输出的json反序列化失败, 原因{str(e)}")
            raise JSONDecodeError(msg=e.msg, doc=e.doc, pos=e.pos)

    @staticmethod
    def _normalize_name(name):
        """实体名归一化：把内部换行/制表符等空白压成单个空格并去首尾。

        LLM 降级抽取的名字可能含 "蜂鸣\n\n电池" 这类内部换行，
        仅 strip() 去不掉，落库后会污染 Neo4j 与 kb_entity_names。
        """
        return re.sub(r"\s+", " ", str(name or "")).strip()

    def _clean_entities(self, entities):
        """
        清洗实体信息
        无效的实体
        截断过长的实体名
        实体的标签是否在白名单中
        去重 同标签的实体名只能存在一份
        :param entities:
        :return:
        """

        unique_seen = set()
        clean_entities_result = []
        for entity in entities:
            # 1.1 获取实体名（含内部空白归一化）
            entity_name = self._normalize_name(entity.get('name', ''))

            # 1.2 校验是否存在
            if not entity_name:
                continue

            # 1.3 截取实体名
            if len(entity_name) >= MAX_ENTITY_NAME_LENGTH:
                entity_name = entity_name[:15]

            # 1.4 获取实体标签
            entity_label = entity.get('label', '').strip()

            # 1.5 标签是否在白名单中
            if entity_label not in ALLOWED_ENTITY_LABELS:
                continue

            # 1.6 定义去重key
            unique_key = (entity_name, entity_label)

            # 1.7 判断
            if unique_key in unique_seen:
                continue

            unique_seen.add(unique_key)

            # 1.8 构建最终字典
            clean_entities = {"name": entity_name, "label": entity_label}

            # 1.9 判断实体描述是否有
            entity_description = str(entity.get('description', '')).strip()
            if entity_description:
                clean_entities['description'] = entity_description
            clean_entities_result.append(clean_entities)

        # 返回结果
        return clean_entities_result

    def _clean_relations(self, cleaned_unique_entities, relations):
        """
        关系的头尾是否不存在
        截取头尾实体名
        校验头尾实体是否有效
        校验关系的类型是否在白名单中
        :param cleaned_unique_entities:
        :param relations:
        :return:
        """

        cleaned_relation_result = []

        # 1. 遍历所有关系
        for relation in relations:
            # 1.1 提取头实体名
            head_entity_name = self._normalize_name(relation.get('head'))
            # 1.2 提取尾实体名
            tail_entity_name = self._normalize_name(relation.get('tail'))

            # 1.3 头尾实体名是否存在
            if not head_entity_name or not tail_entity_name:
                continue

            # 1.4判断头尾实体名长度是否超过阈值
            if len(head_entity_name) >= MAX_ENTITY_NAME_LENGTH:
                head_entity_name = head_entity_name[:15]

            if len(tail_entity_name) >= MAX_ENTITY_NAME_LENGTH:
                tail_entity_name = tail_entity_name[:15]

            # 1.5 判断头尾实体名是否有效
            if head_entity_name not in cleaned_unique_entities or tail_entity_name not in cleaned_unique_entities:
                continue

            # 1.6 获取关系类型
            relation_type = str(relation.get('type') or '').strip()

            # 1.7 判断是否在白名单
            if relation_type not in ALLOWED_RELATION_TYPES:
                # 如果没有可以反哺这个类型到白名单去
                relation_type = DEFAULT_RELATION_TYPES

            # 1.8 构建最终关系链的返回字典
            cleaned_relation = {"head": head_entity_name, "tail": tail_entity_name, "type": relation_type}

            cleaned_relation_result.append(cleaned_relation)

        # 返回
        return cleaned_relation_result



def test_kg_extraction():
    """测试：模拟单个切片，跑通 LLM → 解析 → 清洗全流程。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    mock_state = {
        "item_name": "万用表",
        "chunks": [
            {
                "content": """# 电池安装
                    警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。
                    1. 把表笔与仪表断开。
                    2. 用螺丝刀拧开电池后盖上的螺母。
                    3. 正确安装电池，正负极应一致。
                    4. 盖上电池后盖并拧紧螺丝钉。
                    警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。
                    注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。""",
                "chunk_id": "chunk_test_001",
                "item_name": "万用表",
            }
        ],
    }

    KnowLedgeGraphNode().process(mock_state)


if __name__ == "__main__":
    test_kg_extraction()