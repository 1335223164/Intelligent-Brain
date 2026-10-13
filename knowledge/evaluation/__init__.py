"""RAGAS 评测包。

分工（按使用方诉求重构，使用方自行编排“扩展 → 评估”两步）：
- compat:     修复 ragas 0.2.x 与 langchain-community 0.4.x 的静态导入不兼容（vertexai 被移除）。
- config:     评测配置（docs 目录、每文档问答数、文档分块、指标、并发、独立评估模型）。
- clients:    评判 LLM 走独立模型(EVAL_LLM_MODEL, 复用硅基流动 base/key)，嵌入复用本地 BGE-M3。
- expansion:  数据扩展——读 docs/ 下 .md/.txt 逐篇生成(问题+标准答案)，再跑查询链路补齐(召回文档+生成答案)，
             一步产出评估所需的“四元组”，直接返回 JSON 友好结构（不落盘、不绑定“数据集”概念）。
- evaluator:  纯评估——只吃四元组(问题/召回文档/生成答案/标准答案)做四维指标打分，输出报告。

任何子模块在 import ragas 前都必须先调用 compat.ensure_ragas_compat()，本包 __init__ 已统一处理。
"""

from knowledge.evaluation.compat import ensure_ragas_compat

# 在导入任何 ragas 符号之前安装兼容 shim
ensure_ragas_compat()

from knowledge.evaluation.evaluator import evaluate_samples, EvaluationReport
from knowledge.evaluation.expansion import (
    EvalRecord,
    QAPair,
    expand_documents,
    generate_qa_for_doc,
    run_query,
    records_to_json,
)

__all__ = [
    "ensure_ragas_compat",
    "evaluate_samples",
    "EvaluationReport",
    "EvalRecord",
    "QAPair",
    "expand_documents",
    "generate_qa_for_doc",
    "run_query",
    "records_to_json",
]
