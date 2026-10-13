"""提示词加载器

统一从 templates/ 目录加载提示词模板文件（UTF-8 .txt）。

规范：
- 模板路径: templates/{category}/{name}.txt
- 命名: {场景}_{角色}.txt，如 item_name_system.txt / kg_extraction_system.txt
- 占位符: 单花括号 {var}，渲染时仅替换显式传入的变量，
  因此模板中的 JSON 示例可以直接使用单花括号书写，无需转义
"""

from functools import lru_cache
from pathlib import Path

# 模板根目录
TEMPLATES_DIR = Path(__file__).parent / "templates"


class PromptNotFoundError(FileNotFoundError):
    """提示词模板不存在时抛出。"""

    def __init__(self, category: str, name: str):
        super().__init__(f"提示词模板不存在: templates/{category}/{name}.txt")
        self.category = category
        self.name = name


@lru_cache(maxsize=None)
def load_prompt(category: str, name: str) -> str:
    """
    加载指定分类下的提示词模板原文（带缓存）。

    Args:
        category: 模板分类目录名（import / query）
        name: 模板名（不含 .txt 后缀）

    Returns:
        模板原文（去除首尾空白）
    """
    template_path = TEMPLATES_DIR / category / f"{name}.txt"
    if not template_path.is_file():
        raise PromptNotFoundError(category, name)
    return template_path.read_text(encoding="utf-8").strip()


def render(category: str, name: str, **variables) -> str:
    """
    加载模板并渲染占位符。

    仅替换以 {key} 形式显式传入的变量，模板中其余花括号原样保留，
    因此 JSON 输出示例无需写成 {{ }}。

    Args:
        category: 模板分类目录名
        name: 模板名（不含 .txt 后缀）
        **variables: 占位符变量

    Returns:
        渲染后的提示词文本
    """
    text = load_prompt(category, name)
    for key, value in variables.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def clear_cache() -> None:
    """清空模板缓存（模板文件热更新时可用）。"""
    load_prompt.cache_clear()
