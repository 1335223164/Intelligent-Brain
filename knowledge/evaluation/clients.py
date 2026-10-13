"""把主项目的 LLM / 嵌入客户端适配为 RAGAS 所需的包装对象。

- 评判 LLM：与链路模型解耦。默认走同一硅基流动 base_url/key，仅用 EVAL_LLM_MODEL 换模型名
  （如 DeepSeek 系列），避免“自己评自己”的偏好偏差；留空则回退链路模型。
  直接用原生 ChatOpenAI（不套 NoThink 包装，避免向非 Qwen 模型追加 /no_think 造成干扰）。
- Embeddings：AnswerRelevancy 需要嵌入模型；直接复用本地 BGE-M3 的稠密向量，
  既与检索空间一致（评测更贴合真实链路），又无需额外引入远程 embedding 依赖。
"""

from __future__ import annotations

import os
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

from knowledge.evaluation.compat import ensure_ragas_compat

ensure_ragas_compat()

from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402

from knowledge.utils.client.ai_clients import AIClients  # noqa: E402


class BGEEmbeddings(Embeddings):
    """将本地 BGE-M3 稠密向量适配为 langchain Embeddings 接口。"""

    def __init__(self):
        self._model = AIClients.get_bge_m3_client()

    @staticmethod
    def _dense_to_list(dense) -> List[List[float]]:
        # BGEM3EmbeddingFunction 的 dense 可能是 numpy 数组或列表
        return [list(map(float, vec)) for vec in dense]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        result = self._model.encode_documents(texts)
        return self._dense_to_list(result["dense"])

    def embed_query(self, text: str) -> List[float]:
        result = self._model.encode_queries([text])
        dense = self._dense_to_list(result["dense"])
        return dense[0] if dense else []


def build_ragas_llm() -> LangchainLLMWrapper:
    """构造 ragas 评判用 LLM 包装（与链路模型解耦，默认仅换模型名）。"""
    from knowledge.evaluation.config import get_eval_config

    cfg = get_eval_config()
    model = cfg.eval_llm_model or os.getenv("LLM_DEFAULT_MODEL", "")
    if not model:
        raise EnvironmentError("未配置评判模型：请设置 EVAL_LLM_MODEL 或 LLM_DEFAULT_MODEL")

    llm_client = ChatOpenAI(
        model=model,
        temperature=cfg.eval_llm_temperature,
        api_key=cfg.eval_llm_api_key,
        base_url=cfg.eval_llm_api_base,
    )
    return LangchainLLMWrapper(llm_client)


def build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    """构造 ragas 用嵌入包装（本地 BGE-M3 稠密向量）。"""
    return LangchainEmbeddingsWrapper(BGEEmbeddings())
