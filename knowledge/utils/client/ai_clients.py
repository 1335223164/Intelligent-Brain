import threading
from typing import Optional
from langchain_openai import ChatOpenAI
from openai import OpenAI
from FlagEmbedding import FlagReranker
from dotenv import load_dotenv
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from knowledge.utils.client.base import BaseClientManager, logger

load_dotenv()


class NoThinkChatOpenAI(ChatOpenAI):
    """硅基流动 Qwen3 系列默认开启思考模式：实测 JSON 模式输出空对象且耗时 26s+。

    官方关闭开关是在最后一条消息尾部追加 /no_think，这里统一包装，
    避免逐个污染 prompt 模板；对非 Qwen3 模型无副作用（普通模型忽略该标记）。
    """

    @staticmethod
    def _suppress_thinking(messages):
        last = messages[-1] if messages else None
        if last is not None and last.type == "human" and isinstance(last.content, str) \
                and "/no_think" not in last.content:
            messages = list(messages)
            messages[-1] = last.model_copy(update={"content": last.content + "\n/no_think"})
        return messages

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return super()._generate(self._suppress_thinking(messages), stop, run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return await super()._agenerate(self._suppress_thinking(messages), stop, run_manager, **kwargs)


class AIClients(BaseClientManager):
    """AI 模型类客户端"""

    # ── VLM ──
    _openai_client: Optional[OpenAI] = None
    _openai_lock = threading.Lock()

    # ── LLM ──
    _openai_llm_response_json_client: Optional[ChatOpenAI] = None
    _openai_llm_response_json_lock = threading.Lock()

    _openai_llm_response_text_client: Optional[ChatOpenAI] = None
    _openai_llm_response_text_lock = threading.Lock()

    # ── Embedding ──
    _bge_m3_client: Optional[BGEM3EmbeddingFunction] = None
    _bge_m3_lock = threading.Lock()

    # ── rerank ──
    _bge_m3_rerank_client: Optional[FlagReranker] = None
    _bge_m3_rerank_lock = threading.Lock()

    # ── VLM ──
    @classmethod
    def get_vlm_client(cls) -> OpenAI:
        return cls._get_or_create("_openai_client", cls._openai_lock, cls._create_vlm_client)

    @classmethod
    def _create_vlm_client(cls) -> OpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")

            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"OpenAI 客户端初始化成功 (base_url={base_url})")
            return client

        except EnvironmentError:
            raise
        except Exception as e:
            logger.error(f"OpenAI 客户端创建失败: {e}")
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e

    # ── LLM ──
    @classmethod
    def get_llm_client(cls, response_format: bool = True) -> ChatOpenAI:
        if response_format:
            return cls._get_or_create("_openai_llm_json_client", cls._openai_llm_response_json_lock,
                                      lambda: cls._create_llm_client(response_format))
        else:
            return cls._get_or_create("_openai_llm_text_client", cls._openai_llm_response_text_lock,
                                      lambda: cls._create_llm_client(response_format))

    @classmethod
    def _create_llm_client(cls, response_format: bool = True) -> ChatOpenAI:
        try:
            api_key = cls._require_env("OPENAI_API_KEY")
            base_url = cls._require_env("OPENAI_API_BASE")
            model_name = cls._require_env('LLM_DEFAULT_MODEL')

            model_kwargs = {}
            if response_format:
                model_kwargs['response_format'] = {"type": "json_object"}

            llm_client = NoThinkChatOpenAI(
                model=model_name,
                temperature=0,
                api_key=api_key,
                base_url=base_url,
                model_kwargs=model_kwargs
            )
            logger.info(f"OpenAI LLM 客户端初始化成功")
            return llm_client

        except EnvironmentError:
            raise
        except Exception as e:
            raise ConnectionError(f"OpenAI 连接失败: {e}") from e


    # ── Embedding ──
    @classmethod
    def get_bge_m3_client(cls):
        return cls._get_or_create("_bge_m3_client", cls._bge_m3_lock, cls._create_bge_m3_client)

    @classmethod
    def _create_bge_m3_client(cls):
        """
        创建bge_m3 客户端
        Returns:
        """

        try:
            # 1. 获取环境变量
            model_name = cls._require_env('BGE_M3_PATH')
            device = cls._require_env('BGE_DEVICE')
            fp16 = cls._require_env('BGE_FP16')
            fp16 = fp16.lower() in ("true","1")

            # 2. 创建
            bge_m3_ef = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=fp16
            )
            return bge_m3_ef
        except EnvironmentError as e:
            raise

        except Exception as e:
            raise ConnectionError(f"BGE_M3嵌入模型客户端创建失败: {e}") from e


    # ── BGE-M3重排序模型客户端 ──
    @classmethod
    def get_bge_m3_rerank_client(cls):
        return cls._get_or_create("_bge_m3_rerank_client",
                                  cls._bge_m3_rerank_lock,
                                  cls._create_bge_m3_rerank_client)

    @classmethod
    def _create_bge_m3_rerank_client(cls):
        """
        创建bge_m3 重排序模型客户端
        Returns:
        """

        try:
            # 1. 获取环境变量
            model_name_or_path = cls._require_env('BGE_RERANKER_LARGE')
            device = cls._require_env('BGE_DEVICE')
            fp16_str = cls._require_env('BGE_FP16')
            fp16 = fp16_str.lower() in ("true", "1")

            # 2. 创建
            reranker = FlagReranker(
                model_name_or_path=model_name_or_path,
                device=device,
                use_fp16=fp16
            )

            return reranker
        except EnvironmentError as e:
            raise

        except Exception as e:
            raise ConnectionError(f"BGE-M3重排序模型客户端创建失败: {e}") from e