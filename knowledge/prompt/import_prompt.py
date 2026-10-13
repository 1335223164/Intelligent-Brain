"""
导入相关的提示词模版管理（兼容层）

模板已迁移至 templates/import/*.txt，请优先使用 knowledge.prompt.render 加载。
本模块仅保留旧代码引用的常量与函数入口，勿在此新增提示词。
"""

from knowledge.prompt.loader import load_prompt, render

# 商品名系统提示词（静态模板）
ITEM_NAME_SYSTEM_PROMPT = load_prompt("import", "item_name_system")

# 知识图谱抽取系统提示词（UIE 抽取失败时的 LLM 兜底路径使用）
KNOWLEDGE_GRAPH_SYSTEM_PROMPT = load_prompt("import", "kg_extraction_system")


def item_name_user_prompt(file_title: str, context: str) -> str:
    """商品名用户提示词（带占位符渲染）。"""
    return render("import", "item_name_user", file_title=file_title, context=context)
