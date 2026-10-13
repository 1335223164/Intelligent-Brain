"""知识图谱实体/关系抽取客户端（HTTP 调用独立 UIE 微服务）。

背景：UIE 基于 PaddleNLP，而 paddlenlp 在 Windows 强制 protobuf==3.20.2，与主项目的
pymilvus（需 protobuf>=5）不可共存于同一解释器。因此 UIE 抽取被拆到独立 venv 的
HTTP 微服务（见 uie_service/），本模块只是它的薄客户端，保持与历史实现完全一致的
对外契约，两个 KG 节点（kg_graph_node / kg_query_node）无需改动。

对外入口：
- get_uie_client(): 获取全局单例
- extract_triples(text): 联合抽取实体+关系，返回
  {"entities": [{"name","label","description"}], "relations": [{"head","tail","type"}]}
- extract_entities(text): 仅抽取实体名列表 list[str]

降级策略：服务不可用/超时/非 2xx 时，记录告警并返回空结构（entities/relations 为空），
使 kg_graph_node 自动降级走远程 LLM 兜底抽取，主流程不崩。

环境变量：
- UIE_SERVICE_URL:  UIE 微服务地址（默认 http://127.0.0.1:8001）
- UIE_TIMEOUT:      单次请求超时秒数（默认 30）
- UIE_ENABLED:      是否启用（默认 True；置 False 时直接返回空，强制走 LLM 兜底）
"""

import logging
import os
import threading
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

EMPTY_TRIPLES: Dict[str, List[Dict[str, Any]]] = {"entities": [], "relations": []}


def _to_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


class UieClient:
    """UIE 微服务 HTTP 客户端（全局单例，无本地模型）。"""

    _instance: Optional["UieClient"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._base_url = httpx.URL(
            os.getenv("UIE_SERVICE_URL", "http://127.0.0.1:8001").strip().rstrip("/")
        )
        try:
            self._timeout = float(os.getenv("UIE_TIMEOUT", "30") or 30)
        except (TypeError, ValueError):
            self._timeout = 30.0
        self._enabled = _to_bool(os.getenv("UIE_ENABLED", "True"), default=True)

    @classmethod
    def get_instance(cls) -> "UieClient":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ==================== 对外接口 ====================

    def extract_triples(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """联合抽取实体 + 关系（导入侧使用）。失败/降级返回空结构。"""
        if not self._enabled or not text or not text.strip():
            return {"entities": [], "relations": []}
        data = self._post("/extract_triples", {"text": text})
        if not isinstance(data, dict):
            return {"entities": [], "relations": []}
        entities = data.get("entities") or []
        relations = data.get("relations") or []
        return {"entities": entities, "relations": relations}

    def extract_entities(self, text: str) -> List[str]:
        """仅抽取实体名（查询侧使用）。失败/降级返回空列表。"""
        if not self._enabled or not text or not text.strip():
            return []
        data = self._post("/extract_entities", {"text": text})
        if not isinstance(data, dict):
            return []
        names = data.get("entities") or []
        # 去重保序
        result: List[str] = []
        seen = set()
        for name in names:
            name = str(name).strip()
            if name and name not in seen:
                seen.add(name)
                result.append(name)
        return result

    def health(self) -> Dict[str, Any]:
        """探测服务健康状态（供自测/监控使用）。"""
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(self._base_url.join("health"))
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            logger.warning(f"UIE 服务健康检查失败: {e}")
            return {"status": "unavailable", "error": str(e)}

    # ==================== 内部 ====================

    def _post(self, path: str, payload: Dict[str, Any]) -> Optional[Any]:
        url = self._base_url.join(path.lstrip("/"))
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(str(url), json=payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.warning(f"UIE 服务返回错误状态 {e.response.status_code} ({path}): {e}")
        except (httpx.RequestError, Exception) as e:  # noqa: BLE001
            logger.warning(f"UIE 服务调用失败({path})，本次返回空结果以触发降级: {e}")
        return None


# ==================== 模块级快捷函数（保持历史接口） ====================

def get_uie_client() -> UieClient:
    """获取 UIE 客户端全局单例。"""
    return UieClient.get_instance()


def extract_triples(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """联合抽取实体+关系（导入侧快捷入口）。"""
    return get_uie_client().extract_triples(text)


def extract_entities(text: str) -> List[str]:
    """仅抽取实体名（查询侧快捷入口）。"""
    return get_uie_client().extract_entities(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    sample = """# 电池安装
警告: 为防触电, 打开电池后盖前后，请勿操作仪表并把表笔与电源断开。
1. 把表笔与仪表断开。
2. 用螺丝刀拧开电池后盖上的螺母。
3. 正确安装电池，正负极应一致。
4. 盖上电池后盖并拧紧螺丝钉。
警告: 为防触电,在电池后盖安装和固定之前，请勿操作仪表。
注意: 若仪表出现工作不正常，请检测保险丝和电池是否完好以及是否放在正确的位置。"""

    print("服务健康:", get_uie_client().health())
    print("=" * 60)
    print("【三元组抽取 extract_triples】")
    result = extract_triples(sample)
    for entity in result["entities"]:
        print(f"  实体: {entity}")
    for relation in result["relations"]:
        print(f"  关系: {relation}")

    print("=" * 60)
    print("【问题实体抽取 extract_entities】")
    question = "更换电池时需要用到螺丝刀吗？表笔要怎么断开？"
    print(f"  问题: {question}")
    print(f"  实体: {extract_entities(question)}")
