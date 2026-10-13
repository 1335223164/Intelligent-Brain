r"""RAGAS 评测命令行工具（两步独立，使用方自行编排）。

  # 1) 数据扩展：读 docs/ 下文档 → 逐篇生成问答对 → 跑查询链路补齐 → 产出四元组 JSON
  python knowledge/evaluation/run_ragas_eval.py expand --per-doc 10 --out eval_records.json

  # 2) 评估：把四元组 JSON 喂给 RAGAS 四维打分
  python knowledge/evaluation/run_ragas_eval.py evaluate --input eval_records.json --metrics faithfulness,context_precision,context_recall

说明：
- expand 默认从配置 docs 目录(项目根/docs)读取 .md/.txt；也可 --docs-dir / --files 指定。
- expand 需要中间件与已导入文档在跑（因为要复用查询链路补齐召回文档/生成答案）；
  若只想先拿到(问题+标准答案)，加 --no-chain（此时召回文档/生成答案为空，仅便于离线出题）。
- 评估模型通过环境变量 EVAL_LLM_MODEL 指定（默认复用硅基流动 base/key，仅换模型名）。

配对 A/B（每次用同一数据，推荐）：
  # a) 生成一次并冻结问题集（只存 问题/标准答案）
  python knowledge/evaluation/run_ragas_eval.py expand --save-dataset --out temp\eval_records.json
  # b) 之后每次改动链路，复用同一集只重跑链路：
  python knowledge/evaluation/run_ragas_eval.py expand --dataset knowledge\evaluation\datasets\qa_xxx.json --out temp\eval_records.json
  # c) 打分可 --repeat N 看裁判不稳度：
  python knowledge/evaluation/run_ragas_eval.py evaluate --input temp\eval_records.json --metrics faithfulness --repeat 3
"""

import argparse
import json
import logging
import os
import sys

# 保证能以 `knowledge.*` 导入项目包（无论从何处运行本脚本）
# 本文件位于 <项目根>/knowledge/evaluation/，向上三级即项目根
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_ragas_eval")


def _split_csv(raw):
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def cmd_expand(args):
    from knowledge.evaluation.config import get_eval_config
    from knowledge.evaluation.expansion import (
        build_dataset, run_chain_for_records, records_to_json, save_dataset, load_dataset)

    cfg = get_eval_config()
    if args.per_doc:
        cfg.qa_per_doc = args.per_doc
    run_chain = not args.no_chain

    if args.dataset:
        # 复用冻结评测集：只重跑查询链路（配对 A/B：问题/标准答案不变）
        records = load_dataset(args.dataset)
        print(f"  已复用冻结评测集: {args.dataset} ({len(records)} 条)")
        if run_chain:
            run_chain_for_records(records, cfg)
    else:
        doc_paths = _split_csv(args.files) or None
        records = build_dataset(docs_dir=args.docs_dir or cfg.docs_dir, cfg=cfg, doc_paths=doc_paths)
        if args.save_dataset is not None:
            ds_path = None if args.save_dataset == "__auto__" else args.save_dataset
            saved = save_dataset(records, ds_path, cfg)
            print(f"  评测集已冻结: {saved}")
        if run_chain:
            run_chain_for_records(records, cfg)

    text = records_to_json(records, path=args.out)
    print(f"\n=== 数据扩展完成：{len(records)} 条四元组 ===")
    if args.out:
        print(f"  已保存: {args.out}")
    else:
        # 未指定输出文件则打印到 stdout
        print(text)
    return 0 if records else 1


def cmd_evaluate(args):
    from knowledge.evaluation.config import get_eval_config
    from knowledge.evaluation.evaluator import evaluate_samples

    cfg = get_eval_config()
    if args.metrics:
        cfg.metrics = _split_csv(args.metrics)
    if args.max_workers:
        cfg.max_workers = args.max_workers

    with open(args.input, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("输入四元组文件应为 JSON 数组")

    repeat = max(1, args.repeat or 1)
    reports = []
    for i in range(repeat):
        rep = evaluate_samples(raw, cfg=cfg, save_report=(not args.no_save) and i == 0)
        reports.append(rep)
        if repeat > 1:
            detail = ", ".join(f"{m}={rep.aggregate.get(m)}" for m in rep.metrics)
            print(f"  [第 {i + 1}/{repeat} 次] {detail}")
    _print_report(reports[-1])
    if repeat > 1:
        _print_variance(reports)
    return 0


def _print_variance(reports):
    """多次重复打分的均值/标准差，用于看裁判本身的不稳度。"""
    import statistics
    metrics = reports[0].metrics
    print(f"\n=== {len(reports)} 次重复稳定性（均值±标准差）===")
    for m in metrics:
        vals = [r.aggregate.get(m) for r in reports if r.aggregate.get(m) is not None]
        if not vals:
            print(f"    - {m:<18}: N/A")
            continue
        mean = sum(vals) / len(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"    - {m:<18}: {mean:.4f} ± {sd:.4f}   (n={len(vals)})")


def _print_report(report):
    print("\n=== RAGAS 评测报告 ===")
    print(f"  样本数: {report.sample_count}")
    print(f"  生成时间: {report.generated_at}")
    if report.report_path:
        print(f"  报告文件: {report.report_path}")
    print("  指标均值:")
    for m in report.metrics:
        v = report.aggregate.get(m)
        print(f"    - {m:<18}: {'N/A' if v is None else f'{v:.4f}'}")


def build_parser():
    p = argparse.ArgumentParser(description="RAGAS 评测工具（扩展/评估两步独立）")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("expand", help="读 docs 文档 → 产出评估四元组（支持冻结/复用评测集）")
    g.add_argument("--docs-dir", type=str, help="文档目录（默认配置 docs_dir）")
    g.add_argument("--files", type=str, help="指定文档路径（逗号分隔，优先于 --docs-dir）")
    g.add_argument("--per-doc", type=int, help="每篇文档生成问答对数量")
    g.add_argument("--no-chain", action="store_true", help="不跑查询链路（仅出题，召回/答案留空）")
    g.add_argument("--dataset", type=str, help="复用已冻结评测集（只重跑查询链路，做配对 A/B；优先于 --docs-dir/--files/--per-doc）")
    g.add_argument("--save-dataset", nargs="?", const="__auto__", default=None,
                   help="把本次生成的问题冻结为可复用评测集（只存 question/reference/doc）；可带路径，省略则按指纹自动命名到 dataset 目录")
    g.add_argument("--out", type=str, help="四元组输出 JSON 路径（省略则打印到 stdout）")
    g.set_defaults(func=cmd_expand)

    e = sub.add_parser("evaluate", help="对四元组 JSON 执行 RAGAS 打分")
    e.add_argument("--input", type=str, required=True, help="四元组 JSON 路径")
    e.add_argument("--metrics", type=str, help="指标(逗号分隔): faithfulness,answer_relevancy,context_precision,context_recall")
    e.add_argument("--max-workers", type=int, help="ragas 并发度")
    e.add_argument("--repeat", type=int, help="重复打分次数（>1 时额外输出均值±标准差，用于衡量裁判不稳度）")
    e.add_argument("--no-save", action="store_true", help="不落盘报告（仅打印）")
    e.set_defaults(func=cmd_evaluate)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
