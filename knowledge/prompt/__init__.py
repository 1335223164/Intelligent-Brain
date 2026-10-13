"""提示词统一管理包

- 模板文件存放于 templates/{import|query}/*.txt
- 通过 load_prompt / render 加载与渲染
- import_prompt / query_prompt 为兼容旧引用的薄层，勿新增引用
"""

from knowledge.prompt.loader import load_prompt, render, clear_cache, PromptNotFoundError

__all__ = ["load_prompt", "render", "clear_cache", "PromptNotFoundError"]
