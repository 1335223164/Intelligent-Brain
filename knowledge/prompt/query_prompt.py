"""查询流程相关提示词模板（兼容层）

模板已迁移至 templates/query/*.txt，请优先使用 knowledge.prompt.render 加载。
本模块仅保留旧代码引用的常量与函数入口，勿在此新增提示词。
"""

from typing import List
from knowledge.prompt.loader import load_prompt, render

# 商品名提取（用户提示词模板，占位符: history_text / query）
ITEM_NAME_USER_EXTRACT_TEMPLATE = load_prompt("query", "item_name_extract")

# 商品名提取（系统提示词，静态）
ITEM_NAME_SYSTEM_PROMPT = load_prompt("query", "item_name_system")

# HyDE 假设文档生成
HYDE_SYSTEM_TEMPLATE = load_prompt("query", "hyde_system")
HYDE_USER_PROMPT_TEMPLATE = load_prompt("query", "hyde_user")

# 最终答案生成
ANSWER_PROMPT = load_prompt("query", "answer")


def item_name_extract_prompt(history_text: str, query: str) -> str:
    """商品名提取用户提示词（带占位符渲染）。"""
    return render("query", "item_name_extract", history_text=history_text, query=query)


def answer_prompt(context: str, history: str, item_names: List[str], question: str,
                  kg_triples: List[str]) -> str:
    """最终答案提示词（带占位符渲染）。"""
    return render(
        "query", "answer",
        context=context,
        history=history,
        item_names=",".join(item_names or []),
        question=question,
        kg_triples="\n".join(kg_triples or []) or "暂无图谱关联信息",
    )
