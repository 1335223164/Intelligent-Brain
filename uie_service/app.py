# -*- coding: utf-8 -*-
"""UIE 抽取微服务（FastAPI）。

独立 venv 运行，避开 paddle/pymilvus 的 protobuf 冲突。主项目通过 HTTP 调用本服务。

启动（在 uie_service/.venv 下）：
    uvicorn app:app --host 127.0.0.1 --port 8001
或：
    python -m uvicorn app:app --host 127.0.0.1 --port 8001

接口：
    GET  /health            -> {"status":"ok","ready":bool}
    POST /extract_triples   -> {"text": str}  => {"entities":[...],"relations":[...]}
    POST /extract_entities  -> {"text": str}  => {"entities":[str, ...]}
    DEBUG GET /raw?text=... -> UIE 原始输出（用于核对结构）
"""
import os
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 读取主项目配置（UIE_MODEL_PATH / UIE_PRECISION 与 .env 统一管理）；
# 已存在的环境变量优先于 .env（override=False）
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "knowledge", ".env"), override=False)

_engine = None
_engine_lock = threading.Lock()
_engine_error = None


def _precision() -> float:
    try:
        return float(os.getenv("UIE_PRECISION", "0.5"))
    except (TypeError, ValueError):
        return 0.5


def _model() -> str:
    # 本地路径优先，留空则用 paddlenlp 内置 uie-base（首次自动下载）
    return os.getenv("UIE_MODEL_PATH", "").strip() or "uie-base"


def get_engine():
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            try:
                from uie_engine import UieEngine

                _engine = UieEngine(precision=_precision(), position=_model())
                _engine_error = None  # 加载成功后清除历史错误，避免 /health 残留
            except Exception as e:  # noqa: BLE001
                _engine_error = str(e)
                raise
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动即加载模型，避免首个请求超时；失败不阻断服务（health 会显示未就绪）
    try:
        get_engine()
    except Exception as e:  # noqa: BLE001
        print(f"[UIE] 模型预热失败：{e}")
    yield


app = FastAPI(title="UIE Extraction Service", version="1.0", lifespan=lifespan)


class TextIn(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok", "ready": _engine is not None, "error": _engine_error}


@app.post("/extract_triples")
def extract_triples(inp: TextIn):
    try:
        eng = get_engine()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"UIE 引擎不可用: {e}") from e
    return eng.extract_triples(inp.text)


@app.post("/extract_entities")
def extract_entities(inp: TextIn):
    try:
        eng = get_engine()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"UIE 引擎不可用: {e}") from e
    return {"entities": eng.extract_entities(inp.text)}


@app.get("/raw")
def raw(text: str):
    """调试：返回 UIE 联合抽取原始结构，用于核对版本相关格式。"""
    try:
        eng = get_engine()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"UIE 引擎不可用: {e}") from e
    with eng._infer_lock:
        return {"raw": eng._joint_ie(text)[0]}
