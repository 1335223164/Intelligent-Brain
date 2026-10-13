"""RAGAS 评测核心：把“四元组”喂给四维指标打分。

评估只需四元组（问题 / 召回文档 / 生成答案 / 标准答案），不关心它从哪来：
- 由数据扩展(expansion)产出的 EvalRecord；或
- 你自己构造/加载的等价 dict。

指标（默认全开）：
- faithfulness            忠诚度        需 question + contexts + answer
- answer_relevancy        答案相关性    需 question + answer (+ 嵌入模型)
- context_precision       上下文精确率  需 question + contexts + reference
- context_recall          上下文召回率  需 question + contexts + reference

评判 LLM 与链路模型解耦（见 clients.build_ragas_llm），嵌入复用本地 BGE-M3。
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from knowledge.evaluation.compat import ensure_ragas_compat

ensure_ragas_compat()

from ragas import evaluate  # noqa: E402
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # noqa: E402
from ragas.metrics import (  # noqa: E402
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.run_config import RunConfig  # noqa: E402

from knowledge.evaluation.config import EvalConfig, get_eval_config  # noqa: E402

logger = logging.getLogger("evaluation.evaluator")

# 指标别名 -> ragas 指标类（别名对外稳定，内部真实 metric.name 由 name_alias 映射）
_METRIC_REGISTRY = {
    "faithfulness": Faithfulness,
    "answer_relevancy": AnswerRelevancy,
    "context_precision": LLMContextPrecisionWithReference,
    "context_recall": LLMContextRecall,
}

# 一条待评测样本：既接受 EvalRecord，也接受等价 dict
SampleLike = Union[Any, Dict[str, Any]]


@dataclass
class EvaluationReport:
    """评测结果汇总。"""

    aggregate: Dict[str, Optional[float]] = field(default_factory=dict)   # 各指标均值
    per_item: List[Dict[str, Any]] = field(default_factory=list)            # 逐条明细
    metrics: List[str] = field(default_factory=list)
    sample_count: int = 0
    generated_at: str = ""
    report_path: str = ""

    def to_markdown(self) -> str:
        lines = [
            "# RAGAS 评测报告",
            f"- 生成时间: {self.generated_at}",
            f"- 样本数: {self.sample_count}",
            f"- 指标: {', '.join(self.metrics)}",
            "",
            "## 指标均值",
            "",
            "| 指标 | 得分 |",
            "| --- | --- |",
        ]
        for m in self.metrics:
            score = self.aggregate.get(m)
            lines.append(f"| {m} | {('N/A' if score is None else f'{score:.4f}')} |")
        lines += ["", "## 逐条明细", ""]
        header_written = False
        for row in self.per_item:
            if not header_written:
                cols = list(row.keys())
                lines.append("| " + " | ".join(cols) + " |")
                lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
                header_written = True
            vals = [_cell_str(row[c]) for c in row.keys()]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)


def _cell_str(v: Any) -> str:
    s = "" if v is None else str(v)
    return s.replace("\n", " ").replace("|", "/")[:80]


def _norm(sample: SampleLike) -> Dict[str, Any]:
    """把 EvalRecord 或 dict 统一成 dict。"""
    if isinstance(sample, dict):
        return sample
    to_dict = getattr(sample, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return {k: getattr(sample, k) for k in ("question", "reference", "contexts", "answer") if hasattr(sample, k)}


def build_samples(samples: List[SampleLike]) -> List[SingleTurnSample]:
    """把四元组样本转成 ragas SingleTurnSample。"""
    out: List[SingleTurnSample] = []
    for s in samples:
        d = _norm(s)
        out.append(
            SingleTurnSample(
                user_input=str(d.get("question", "") or ""),
                retrieved_contexts=list(d.get("contexts") or []),
                response=str(d.get("answer", "") or ""),
                reference=str(d.get("reference", "") or ""),
            )
        )
    return out


def _make_dataset(samples: List[SingleTurnSample]) -> EvaluationDataset:
    try:
        return EvaluationDataset.from_single_turn_samples(samples)
    except AttributeError:
        return EvaluationDataset(samples=samples)


def _mean_ignore_none(values: List[Any]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return round(sum(nums) / len(nums), 4) if nums else None


def evaluate_samples(
    samples: List[SampleLike],
    cfg: Optional[EvalConfig] = None,
    save_report: bool = True,
) -> EvaluationReport:
    """对四元组样本执行 RAGAS 打分。

    Args:
        samples: [{question, contexts, answer, reference}, ...] 或 EvalRecord 列表
        cfg: 评测配置（指标、并发、评估模型等）
        save_report: 是否把 JSON/Markdown 报告落盘
    """
    cfg = cfg or get_eval_config()
    report = EvaluationReport(metrics=list(cfg.metrics))

    if not samples:
        logger.warning("没有可评测的样本")
        return report

    ragas_samples = build_samples(samples)
    if not ragas_samples:
        return report

    dataset = _make_dataset(ragas_samples)

    metric_objs = [_METRIC_REGISTRY[name]() for name in cfg.metrics if name in _METRIC_REGISTRY]
    if not metric_objs:
        logger.error(f"未匹配到任何有效指标: {cfg.metrics}")
        return report

    # cfg 里的指标别名 -> ragas 内部 metric.name(二者不一定相同,
    # 如 context_precision 对应 LLMContextPrecisionWithReference.name="llm_context_precision_with_reference")
    name_alias: Dict[str, str] = {}
    for name in cfg.metrics:
        if name in _METRIC_REGISTRY:
            name_alias[name] = getattr(_METRIC_REGISTRY[name](), "name", name) or name

    # 延迟导入，避免无谓的模型加载
    from knowledge.evaluation.clients import build_ragas_embeddings, build_ragas_llm

    llm = build_ragas_llm()
    embeddings = build_ragas_embeddings()

    run_config = RunConfig(max_workers=cfg.max_workers)

    logger.info(f"开始 RAGAS 评测：样本={len(ragas_samples)}, 指标={cfg.metrics}, 并发={cfg.max_workers}")
    result = evaluate(
        dataset=dataset,
        metrics=metric_objs,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    # 解析结果：ragas 0.2.x 的 EvaluationResult.scores 为“逐样本 dict 列表”，
    # 每个 dict 形如 {真实metric.name: 分数}；据此转成 指标名->分数列表。
    metric_scores: Dict[str, List[Any]] = {}
    try:
        for score_row in (result.scores or []):
            if not isinstance(score_row, dict):
                continue
            for k, v in score_row.items():
                metric_scores.setdefault(k, []).append(v)
    except Exception as e:  # pragma: no cover
        logger.error(f"读取 ragas 结果 scores 失败, 原因{e}")

    def _scores_for(cfg_name: str) -> Optional[List[Any]]:
        real_key = name_alias.get(cfg_name, cfg_name)
        if real_key in metric_scores:
            return metric_scores[real_key]
        if cfg_name in metric_scores:
            return metric_scores[cfg_name]
        for k, v in metric_scores.items():  # 大小写兜底
            if k.lower() == real_key.lower():
                return v
        return None

    for cfg_name in cfg.metrics:
        scores = _scores_for(cfg_name)
        report.aggregate[cfg_name] = _mean_ignore_none(scores) if scores else None

    # 逐条明细
    for idx, s in enumerate(samples):
        d = _norm(s)
        row: Dict[str, Any] = {
            "question": str(d.get("question", ""))[:60],
            "context_count": len(d.get("contexts") or []),
            "answer": str(d.get("answer", ""))[:60],
        }
        for cfg_name in cfg.metrics:
            scores = _scores_for(cfg_name) or []
            row[cfg_name] = round(float(scores[idx]), 4) if idx < len(scores) and scores[idx] is not None else None
        report.per_item.append(row)

    report.sample_count = len(ragas_samples)
    report.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if save_report:
        _save_report(report, cfg)

    return report


def _save_report(report: EvaluationReport, cfg: EvalConfig) -> str:
    os.makedirs(cfg.report_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(cfg.report_dir, f"ragas_report_{stamp}.json")
    md_path = os.path.join(cfg.report_dir, f"ragas_report_{stamp}.md")

    payload = asdict(report)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report.to_markdown())

    report.report_path = json_path
    logger.info(f"评测报告已保存: {json_path} 与 {md_path}")
    return json_path
