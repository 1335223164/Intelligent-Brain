"""
ragas 0.2.x 与当前依赖栈的兼容性补丁。
"""

import importlib
import sys
import types

_SHIM_MODULE = "langchain_community.chat_models.vertexai"
_installed = False


def ensure_ragas_compat() -> None:
    """注册 langchain_community.chat_models.vertexai 兼容占位模块（幂等）。"""
    global _installed
    if _installed:
        return

    # 若原生模块仍在（旧版 langchain-community），无需处理
    try:
        importlib.import_module(_SHIM_MODULE)
        _installed = True
        return
    except ModuleNotFoundError:
        pass

    if _SHIM_MODULE in sys.modules:
        _installed = True
        return

    shim = types.ModuleType(_SHIM_MODULE)

    # 尽量复用仍存在的 VertexAI 作为基类；缺失则退化为普通 object 基类。
    # 用 importlib 以字符串方式取模块：langchain-community 已被官方逐步弃用
    # （IDE 会对静态引用画删除线），动态取模块可避免把弃用引用写进静态代码，行为等价。
    try:
        _VertexAI = importlib.import_module("langchain_community.llms").VertexAI
    except Exception:  # pragma: no cover - 极端环境兜底
        _VertexAI = object

    class ChatVertexAI(_VertexAI):  # noqa: D401 - 仅作占位
        """占位类：满足 ragas 顶层静态导入，评测链路不会实际使用 Vertex。"""
        pass

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[_SHIM_MODULE] = shim

    # 让 `import langchain_community.chat_models.vertexai` 的属性访问也能命中
    try:
        _chat_models = importlib.import_module("langchain_community.chat_models")
        _chat_models.vertexai = shim
    except Exception:  # pragma: no cover
        pass

    _installed = True
