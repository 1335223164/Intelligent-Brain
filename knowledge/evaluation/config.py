"""RAGAS 评测配置。

集中管理评测相关可调项，全部支持环境变量覆盖。
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from knowledge.core.paths import KNOWLEDGE_ROOT

load_dotenv()

# 项目根目录（knowledge 的上一级），docs 目录默认在根目录下
PROJECT_ROOT = os.path.abspath(os.path.join(KNOWLEDGE_ROOT, ".."))
DEFAULT_DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

# 评测报告默认目录
EVAL_REPORT_DIR = os.path.join(KNOWLEDGE_ROOT, "evaluation", "reports")

# 冻结评测集（问题+标准答案）默认存放目录
EVAL_DATASET_DIR = os.path.join(KNOWLEDGE_ROOT, "evaluation", "datasets")

# 内置可选指标名（对外暴露的稳定别名，内部会映射到 ragas 的真实 metric.name）
AVAILABLE_METRICS = (
    "faithfulness",        # 忠诚度：答案是否忠于检索到的上下文
    "answer_relevancy",    # 答案相关性：答案是否切题（需嵌入模型）
    "context_precision",   # 上下文精确率：检索上下文中有用内容排前的比例（需 reference）
    "context_recall",      # 上下文召回率：reference 被上下文覆盖的比例（需 reference）
)


def _split_csv(raw: str) -> List[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


@dataclass
class EvalConfig:
    """评测流程配置。"""

    # 参与评估的指标（逗号分隔，默认四维全开）
    metrics: List[str] = field(
        default_factory=lambda: _split_csv(os.getenv("EVAL_METRICS", "")) or list(AVAILABLE_METRICS)
    )

    # ragas 并发度（单条样本内多个断言/问题的并行 LLM 调用）
    max_workers: int = field(
        default_factory=lambda: int(os.getenv("EVAL_MAX_WORKERS", "4"))
    )

    # ── 数据扩展（文档驱动） ──
    # 文档目录：默认项目根下的 docs/（环境变量留空则回退默认，避免空串覆盖）
    docs_dir: str = field(default_factory=lambda: os.getenv("EVAL_DOCS_DIR", "").strip() or DEFAULT_DOCS_DIR)
    # 每篇文档目标产出的问答对数量（样本量；A/B 建议 >=15）
    qa_per_doc: int = field(
        default_factory=lambda: int(os.getenv("EVAL_QA_PER_DOC", "15"))
    )
    # 送入 LLM 生成问答的文档片段长度（字符）
    doc_chunk_chars: int = field(
        default_factory=lambda: int(os.getenv("EVAL_DOC_CHUNK_CHARS", "1200"))
    )
    # 文档片段重叠长度（字符）
    doc_chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("EVAL_DOC_CHUNK_OVERLAP", "150"))
    )
    # 每个文档片段最多生成的问答对数（用于把 qa_per_doc 摊到多个片段上）
    qa_per_chunk: int = field(
        default_factory=lambda: int(os.getenv("EVAL_QA_PER_CHUNK", "2"))
    )

    # 评测运行时：每条问答送入 ragas 的上下文条数上限（来自 rerank 后的文档）
    eval_context_top_k: int = field(
        default_factory=lambda: int(os.getenv("EVAL_CONTEXT_TOP_K", "5"))
    )

    # ── 评估模型（与链路模型解耦） ──
    # 评判用模型名；留空则回退到链路模型 LLM_DEFAULT_MODEL
    eval_llm_model: str = field(default_factory=lambda: os.getenv("EVAL_LLM_MODEL", "").strip())
    # 默认 base_url / key 复用链路（同一硅基流动账号）；如需彻底隔离可单独配置
    eval_llm_api_base: str = field(
        default_factory=lambda: os.getenv("EVAL_LLM_API_BASE", "") or os.getenv("OPENAI_API_BASE", "")
    )
    eval_llm_api_key: str = field(
        default_factory=lambda: os.getenv("EVAL_LLM_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    )
    eval_llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("EVAL_LLM_TEMPERATURE", "0"))
    )

    # 报告目录
    report_dir: str = field(default_factory=lambda: os.getenv("EVAL_REPORT_DIR", EVAL_REPORT_DIR))

    # 冻结评测集目录（环境变量留空则回退默认，避免空串覆盖）
    dataset_dir: str = field(
        default_factory=lambda: os.getenv("EVAL_DATASET_DIR", "").strip() or EVAL_DATASET_DIR
    )

    def __post_init__(self):
        # 过滤非法指标名
        self.metrics = [m for m in self.metrics if m in AVAILABLE_METRICS]
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)


@lru_cache(maxsize=1)
def get_eval_config() -> EvalConfig:
    """获取评测配置单例。"""
    return EvalConfig()
