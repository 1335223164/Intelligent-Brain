"""数据扩展：从 docs/ 目录的文档，产出可直接用于 RAGAS 评估的“四元组”。

设计（对齐使用方诉求）：
- 文档驱动：逐篇读取 docs/ 下已转好的 .md/.txt，每篇按目标数量产出问答对（问题 + 标准答案）。
- 一步到位产出四元组：对每个问题跑一遍线上查询链路（多路检索→RRF→rerank→答案生成），
  补齐“召回文档”和“生成答案”，连同“问题、标准答案”组装为评估所需的四元组。
- 直接返回 JSON 友好的结构（不落盘、不绑定“数据集”概念）；如需保存，自行对返回值 json.dump 即可。

四元组字段：
- question   问题
- reference  标准答案（ground truth，来自文档原文）
- contexts   召回文档（查询链路 rerank 后进入提示词的文档正文，最多 context_top_k 条）
- answer     生成答案（查询链路最终产出）
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.evaluation.config import EvalConfig, get_eval_config
from knowledge.prompt import render
from knowledge.utils.client.ai_clients import AIClients

logger = logging.getLogger("evaluation.expansion")

# 默认按这些后缀识别文档
DOC_SUFFIXES = (".md", ".txt")


@dataclass
class QAPair:
    """从文档生成的问答对（问题 + 标准答案）。"""

    question: str
    reference: str
    doc: str = ""              # 来源文档名
    chunk_index: int = -1       # 来源片段序号

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalRecord:
    """评估所需的四元组记录。"""

    question: str
    reference: str
    contexts: List[str] = field(default_factory=list)   # 召回文档
    answer: str = ""                                     # 生成答案
    doc: str = ""
    rewritten_query: str = ""
    item_names: List[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EvalRecord":
        return EvalRecord(
            question=str(data.get("question", "")).strip(),
            reference=str(data.get("reference") or data.get("ground_truth") or "").strip(),
            contexts=list(data.get("contexts") or []),
            answer=str(data.get("answer", "") or ""),
            doc=str(data.get("doc", "") or ""),
            rewritten_query=str(data.get("rewritten_query", "") or ""),
            item_names=list(data.get("item_names") or []),
            error=str(data.get("error", "") or ""),
        )


# ------------------------------------------------------------------
# 文档读取与分块
# ------------------------------------------------------------------
def iter_doc_files(docs_dir: str, suffixes: tuple = DOC_SUFFIXES) -> List[str]:
    """列出 docs_dir 下（含子目录）匹配后缀的文档路径。"""
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"文档目录不存在: {docs_dir}")

    files: List[str] = []
    for root, _dirs, names in os.walk(docs_dir):
        for name in names:
            if name.lower().endswith(tuple(s.lower() for s in suffixes)):
                files.append(os.path.join(root, name))
    return sorted(files)


def read_doc_text(path: str) -> str:
    """读取文档正文文本（UTF-8）。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_doc(text: str, chunk_chars: int, overlap: int) -> List[str]:
    """把文档切成送入 LLM 的片段。"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(200, chunk_chars),
        chunk_overlap=min(overlap, chunk_chars // 2),
        separators=["\n\n", "\n", "。", "？", "！", "；", ".", "?", "!", ";", " ", ""],
        keep_separator=True,
    )
    return [c for c in splitter.split_text(text) if len(c.strip()) >= 40]


# ------------------------------------------------------------------
# 从文件名派生权威产品名（用于出题时强制逐字引用，避免品牌别名撞闸）
# ------------------------------------------------------------------
# 文档名里常见的“文档类型”后缀词，用于从文件名截取产品名
_DOC_TYPE_TAIL = re.compile(
    r"(用户手册|使用手册|安装手册|配置手册|产品手册|用户指南|使用说明|"
    r"快速(?:开始|指南|入门)|说明书|指南|手册|的?使用|的?安装|的?配置|说明)"
)


def derive_product_name(doc_path: str) -> str:
    """从文档文件名派生一个权威产品名。

    规则：去扩展名 → 去括号内备注 → 截断到首个“文档类型”词 → 去尾部连接符。
    取不到则回退整个文件名（不含扩展名）。
    例：'Aolynk CB304n Cable网桥 用户手册-5W100-整本手册.md' -> 'Aolynk CB304n Cable网桥'。
    """
    stem = os.path.splitext(os.path.basename(doc_path))[0]
    cleaned = re.sub(r"[\(（\[].*?[\)）\]]", " ", stem)  # 去括号及其内容
    m = _DOC_TYPE_TAIL.search(cleaned)
    if m:
        cleaned = cleaned[: m.start()]
    cleaned = cleaned.strip(" -_·、,，的")
    return cleaned or stem.strip()


# 进程内缓存：file_title -> 导入时 LLM 识别并注册到 kb_item_names 的真实商品名
_REGISTERED_NAMES_CACHE: Optional[Dict[str, str]] = None


def load_registered_item_names(refresh: bool = False) -> Dict[str, str]:
    """读取 Milvus kb_item_names，得到 {file_title: item_name} 权威映射。

    出题时应优先用导入时真实注册的商品名（查询链路的前置闸正是按它识别产品），
    而不是文件名派生名——否则会出现文件名“迅饶网关”而注册名“绿米网关”这类品牌别名撞闸。
    读取失败（中间件未起/集合缺失）时返回空表，调用方回退到文件名派生。
    """
    global _REGISTERED_NAMES_CACHE
    if _REGISTERED_NAMES_CACHE is not None and not refresh:
        return _REGISTERED_NAMES_CACHE

    mapping: Dict[str, str] = {}
    try:
        from pymilvus import MilvusClient

        uri = os.getenv("MILVUS_URL", "")
        coll = os.getenv("ITEM_NAME_COLLECTION", "") or "kb_item_names"
        mc = MilvusClient(uri=uri)
        if coll in mc.list_collections():
            mc.load_collection(coll)
            rows = mc.query(collection_name=coll, filter='file_title != ""',
                            output_fields=["file_title", "item_name"], limit=16384)
            for r in rows:
                ft = str(r.get("file_title") or "").strip()
                nm = str(r.get("item_name") or "").strip()
                if ft and nm:
                    mapping[ft] = nm
        logger.info(f"[扩展] 读回注册商品名映射 {len(mapping)} 条")
    except Exception as e:
        logger.warning(f"[扩展] 读取 kb_item_names 失败，出题回退文件名派生产品名: {e}")
        mapping = {}

    _REGISTERED_NAMES_CACHE = mapping
    return mapping


def resolve_product_name(doc_path: str, registered_names: Optional[Dict[str, str]] = None) -> str:
    """确定本篇出题使用的产品名：注册名优先，文件名派生兜底。"""
    stem = os.path.splitext(os.path.basename(doc_path))[0]
    if registered_names:
        nm = registered_names.get(stem)
        if nm:
            return nm
    return derive_product_name(doc_path)


# ------------------------------------------------------------------
# LLM 生成问答对
# ------------------------------------------------------------------
def _clean_json_content(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*", "", (text or "").strip())
    return re.sub(r"\s*```$", "", cleaned.strip())


def _parse_qa(response_text: str) -> List[Dict[str, str]]:
    try:
        obj = json.loads(_clean_json_content(response_text))
    except json.JSONDecodeError as e:
        logger.error(f"问答 JSON 解析失败, 原因{e}")
        return []

    pairs = obj.get("qa_pairs") if isinstance(obj, dict) else None
    if not isinstance(pairs, list):
        return []

    result = []
    for p in pairs:
        if not isinstance(p, dict):
            continue
        q = str(p.get("question", "")).strip()
        a = str(p.get("ground_truth", "") or p.get("answer", "")).strip()
        if q and a:
            result.append({"question": q, "ground_truth": a})
    return result


def generate_qa_for_snippet(snippet: str, num: int, product: str = "") -> List[Dict[str, str]]:
    """基于单个文档片段调用 LLM 生成问答对。

    product 为本篇文档的权威产品名，会注入模板要求每个问题逐字包含它，
    以杜绝品牌别名导致查询链路实体识别失败（撞闸）。
    """
    rendered = render("eval", "gen_qa", num=num, content=snippet, product=product)
    system_prompt = rendered.split("原文片段：")[0].strip()
    user_prompt = rendered

    try:
        llm = AIClients.get_llm_client()  # JSON 模式
        resp = llm.invoke([SystemMessage(system_prompt), HumanMessage(user_prompt)])
    except Exception as e:
        logger.error(f"LLM 生成问答对失败, 原因{e}")
        return []

    return _parse_qa(getattr(resp, "content", "") or "")


def generate_qa_for_doc(doc_path: str, cfg: EvalConfig,
                        registered_names: Optional[Dict[str, str]] = None) -> List[QAPair]:
    """逐篇文档生成约 cfg.qa_per_doc 个问答对（跨片段摊开，保证多样性）。"""
    doc_name = os.path.basename(doc_path)
    try:
        text = read_doc_text(doc_path)
    except Exception as e:
        logger.error(f"读取文档失败: {doc_path}, 原因{e}")
        return []

    chunks = split_doc(text, cfg.doc_chunk_chars, cfg.doc_chunk_overlap)
    if not chunks:
        logger.warning(f"文档无有效片段: {doc_name}")
        return []

    product = resolve_product_name(doc_path, registered_names)
    logger.info(f"[扩展] {doc_name} -> 产品名='{product}'")

    per_chunk = max(1, cfg.qa_per_chunk)
    items: List[QAPair] = []
    for idx, chunk in enumerate(chunks):
        if len(items) >= cfg.qa_per_doc:
            break
        need = min(per_chunk, cfg.qa_per_doc - len(items))
        for p in generate_qa_for_snippet(chunk, need, product):
            items.append(QAPair(question=p["question"], reference=p["ground_truth"], doc=doc_name, chunk_index=idx))
            if len(items) >= cfg.qa_per_doc:
                break
    return items


# ------------------------------------------------------------------
# 复用查询链路补齐（召回文档 + 生成答案）
# ------------------------------------------------------------------
def _extract_contexts(result: Dict[str, Any], top_k: int) -> List[str]:
    docs = result.get("reranked_docs") or []
    contents: List[str] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        content = str(doc.get("content", "")).strip()
        if content:
            contents.append(content)
        if len(contents) >= top_k:
            break
    return contents


def run_query(question: str, context_top_k: int = 5) -> Dict[str, Any]:
    """对单个问题执行查询图，返回召回文档与生成答案（异常不抛出，记入 error）。"""
    from knowledge.processor.query_processor.main_graph import query_app

    out: Dict[str, Any] = {"contexts": [], "answer": "", "rewritten_query": question, "item_names": [], "error": ""}
    init_state = {
        "session_id": f"eval_{uuid.uuid4()}",
        "task_id": uuid.uuid4().hex[:12],
        "original_query": question,
        "is_stream": False,
    }
    try:
        result = query_app.invoke(init_state)
    except Exception as e:
        logger.error(f"查询图执行失败, 问题='{question[:40]}', 原因{e}")
        out["error"] = str(e)
        return out

    out["contexts"] = _extract_contexts(result, context_top_k)
    out["answer"] = str(result.get("answer", "") or "")
    out["rewritten_query"] = str(result.get("rewritten_query", "") or question)
    out["item_names"] = list(result.get("item_names") or [])
    return out


# ------------------------------------------------------------------
# 顶层：文档目录 -> 四元组列表
# ------------------------------------------------------------------
def expand_documents(
    docs_dir: Optional[str] = None,
    cfg: Optional[EvalConfig] = None,
    doc_paths: Optional[List[str]] = None,
    run_chain: bool = True,
) -> List[EvalRecord]:
    """一步式：出题(+可选跑链路)。已拆为 build_dataset 与 run_chain_for_records，便于冻结复用做配对 A/B。"""
    records = build_dataset(docs_dir=docs_dir, cfg=cfg, doc_paths=doc_paths)
    if run_chain:
        run_chain_for_records(records, cfg)
    logger.info(f"[扩展] 共产出 {len(records)} 条四元组")
    return records


def build_dataset(
    docs_dir: Optional[str] = None,
    cfg: Optional[EvalConfig] = None,
    doc_paths: Optional[List[str]] = None,
) -> List[EvalRecord]:
    """仅从文档出题，产出“冻结评测集”（只含 question/reference/doc，不含 contexts/answer）。

    这是做配对 A/B 的“尺子”：生成一次并保存复用，之后每次实验只重跑查询链路，
    问题/标准答案不变，分数变化即可归因到链路改动。
    """
    cfg = cfg or get_eval_config()
    paths = doc_paths or iter_doc_files(docs_dir or cfg.docs_dir)
    if not paths:
        logger.warning("没有找到任何文档可处理")
        return []

    # 读回导入时注册的真实商品名（按 file_title），出题时优先用它避免品牌别名撞闸
    registered_names = load_registered_item_names()

    dataset: List[EvalRecord] = []
    for path in paths:
        qa_items = generate_qa_for_doc(path, cfg, registered_names=registered_names)
        logger.info(f"[出题] {os.path.basename(path)}: 生成 {len(qa_items)} 个问答对")
        for qa in qa_items:
            dataset.append(EvalRecord(question=qa.question, reference=qa.reference, doc=qa.doc))

    logger.info(f"[出题] 评测集共 {len(dataset)} 条问题")
    return dataset


def run_chain_for_records(records: List[EvalRecord], cfg: Optional[EvalConfig] = None) -> List[EvalRecord]:
    """对既有评测集逐条跑查询链路，就地补齐 contexts/answer 等（用于冻结集的配对重跑）。"""
    cfg = cfg or get_eval_config()
    for rec in records:
        chain = run_query(rec.question, context_top_k=cfg.eval_context_top_k)
        rec.contexts = chain["contexts"]
        rec.answer = chain["answer"]
        rec.rewritten_query = chain["rewritten_query"]
        rec.item_names = chain["item_names"]
        rec.error = chain["error"]
    return records


def records_to_json(records: List[EvalRecord], path: Optional[str] = None, indent: int = 2) -> str:
    """把四元组序列化为 JSON 字符串；给出 path 时同时落盘。"""
    payload = [r.to_dict() for r in records]
    text = json.dumps(payload, ensure_ascii=False, indent=indent)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"四元组已保存: {path} (共 {len(payload)} 条)")
    return text


def dataset_fingerprint(records: List[EvalRecord]) -> str:
    """按问题集内容算一个短指纹，用于冻结集自动命名/去重。"""
    h = hashlib.sha1()
    for r in records:
        h.update((r.question or "").strip().encode("utf-8"))
    return h.hexdigest()[:8]


def save_dataset(records: List[EvalRecord], path: Optional[str] = None,
                 cfg: Optional[EvalConfig] = None) -> str:
    """保存“冻结评测集”：仅保留 question/reference/doc（跳链路的结果不入库，保证可配对复用）。

    path 缺省时按指纹+条数自动命名到 cfg.dataset_dir。返回实际路径。
    """
    if not path:
        cfg = cfg or get_eval_config()
        path = os.path.join(cfg.dataset_dir, f"qa_{dataset_fingerprint(records)}_{len(records)}.json")
    payload = [{"question": r.question, "reference": r.reference, "doc": r.doc} for r in records]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info(f"评测集已冻结: {path} (共 {len(payload)} 条)")
    return path


def load_dataset(path: str) -> List[EvalRecord]:
    """读取冻结评测集，返回仅含 question/reference/doc 的 EvalRecord 列表（contexts/answer 待跑链路补齐）。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("评测集文件应为 JSON 数组")
    records = [EvalRecord.from_dict(d) for d in raw
               if isinstance(d, dict) and str(d.get("question", "")).strip()]
    logger.info(f"评测集已加载: {path} (共 {len(records)} 条)")
    return records
