# -*- coding: utf-8 -*-
"""UIE 抽取引擎（PaddleNLP Taskflow, uie-base, 联合实体+关系抽取）。

仅在独立 venv 内运行，输出直接对齐主项目 uie_util 的对外契约：
  entities: [{"name","label","description"}]   label 为英文白名单
  relations: [{"head","tail","type"}]           type  为英文白名单
"""
import os
import sys
import threading
from typing import Any, Dict, List

# Windows + paddlepaddle-gpu：pip wheel 不含 CUDA/cuDNN 动态库，paddle 的 dynload 只按
# 进程搜索目录找 cudnn64_8.dll。把 nvidia-* pip 包带来的 bin 目录注册给加载器（
# 必须在 import paddle 之前执行）；非 Windows/无这些目录时静默跳过。
if os.name == "nt":
    _site_nvidia = os.path.join(
        os.path.dirname(sys.executable), "..", "Lib", "site-packages", "nvidia"
    )
    _dll_dirs = [
        os.path.join(os.path.dirname(sys.executable)),  # .venv\Scripts（亦拷有 DLL）
        *(os.path.join(_site_nvidia, s, "bin") for s in ("cudnn", "cublas", "cuda_nvrtc", "cuda_runtime")),
    ]
    _dll_dirs = [d for d in _dll_dirs if os.path.isdir(d)]
    for _p in _dll_dirs:
        try:
            os.add_dll_directory(_p)
        except OSError:
            pass
    # paddle 的 dynamic_loader 按进程 PATH 搜 DLL，仅 add_dll_directory 不够，需同时前置 PATH
    os.environ["PATH"] = os.pathsep.join(_dll_dirs) + os.pathsep + os.environ.get("PATH", "")

from paddlenlp import Taskflow

# ---------------- 白名单映射（与主项目保持一致，中文 schema -> 英文枚举） ----------------
ENTITY_LABEL_MAP: Dict[str, str] = {
    "设备": "Device", "部件": "Part", "操作": "Operation", "步骤": "Step",
    "警告": "Warning", "条件": "Condition", "工具": "Tool",
}
RELATION_TYPE_MAP: Dict[str, str] = {
    "包含部件": "HAS_PART", "执行操作": "HAS_OPERATION", "包含步骤": "HAS_STEP",
    "关联警告": "HAS_WARNING", "使用工具": "USES_TOOL", "需要条件": "REQUIRES",
    "下一步": "NEXT_STEP", "影响部件": "AFFECTS",
}
DEFAULT_RELATION_TYPE = "RELATED_TO"
ALLOWED_ENTITY_LABELS = {"Device", "Part", "Operation", "Step", "Warning", "Condition", "Tool"}

MAX_ENTITY_NAME_LENGTH = 15

# 联合抽取 schema：key 为实体主体，value 为其作为 head 的关系类型列表
JOINT_SCHEMA: Dict[str, List[str]] = {
    "设备": ["包含部件", "执行操作", "包含步骤", "需要条件", "关联警告", "影响部件"],
    "操作": ["包含步骤", "使用工具", "关联警告", "需要条件", "下一步", "影响部件"],
    "步骤": ["使用工具", "关联警告", "下一步", "需要条件"],
    "部件": ["需要条件", "关联警告", "影响部件"],
    "条件": [],
    "工具": [],
    "警告": [],
}
# 仅实体抽取 schema（查询侧更快）
ENTITY_ONLY_SCHEMA: List[str] = list(ENTITY_LABEL_MAP.keys())


class UieEngine:
    """封装 Taskflow，进程内单例、推理加锁（Taskflow 非线程安全）。"""

    def __init__(self, precision: float = 0.5, position: str = "uie-base", device: str = "gpu"):
        self.precision = precision
        self._infer_lock = threading.Lock()
        # position 传模型名或本地路径；Taskflow 会自动检测可用设备（装了 paddlepaddle-gpu
        # 即走 GPU，无 GPU 自动回落 CPU），fp32 最稳。CPU 推理慢数倍且首次含图编译，
        # 故内置下方预热推理，客户端 UIE_TIMEOUT 需留足预算。
        # 注意：paddlenlp 2.6.1 的 Taskflow 强制 model 名在白名单内，本地目录需另走
        # task_path= 参数（权重/词表从该目录读，不再下载）；本地目录应为 uie-base 拷贝。
        kwargs = dict(precision="fp32")
        if position and os.path.isdir(position):
            kwargs.update({"model": "uie-base", "task_path": position})
        else:
            kwargs.update({"model": position})
        self._joint_ie = Taskflow("information_extraction", schema=JOINT_SCHEMA, **kwargs)
        self._entity_ie = Taskflow("information_extraction", schema=ENTITY_ONLY_SCHEMA, **kwargs)
        # 预热推理：首次推理含图编译/内核初始化，可耗数十秒；主动跑一遍，
        # 避免业务首个真实请求超时被客户端误判降级。
        try:
            self._entity_ie("预热：电池安装。")
            self._joint_ie("预热：电池安装。")
        except Exception:  # noqa: BLE001
            pass

    # ------------------------- 对外两个抽取方法 -------------------------
    def extract_triples(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        text = (text or "").strip()
        if not text:
            return {"entities": [], "relations": []}
        with self._infer_lock:
            raw = self._joint_ie(text)[0]
        return self._convert_joint(raw, text)

    def extract_entities(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        with self._infer_lock:
            raw = self._entity_ie(text)[0]
        names: List[str] = []
        seen = set()
        for cn_label, items in raw.items():
            for it in items or []:
                if float(it.get("probability", 0)) < self.precision:
                    continue
                name = str(it.get("text", "")).strip()
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
        return names

    # ------------------------- 结构转换 -------------------------
    def _convert_joint(self, raw: Dict[str, List[Dict]], text: str) -> Dict[str, List[Dict[str, Any]]]:
        """解析 paddlenlp 2.6.x Taskflow 的嵌套结构输出：
        {
          "<实体类型>": [
             {"text","start","end","probability",
              "relations": {"<关系类型>": [{"text","start","end","probability"}, ...], ...}},
             ...
          ], ...
        }
        实体：各实体类型键下、概率达阈值的 item["text"]；
        关系：实体项(head) 的 relations 子字典中，每项 tail 概率达阈值；
              head/tail 只要以任何实体类型达标即可（同一文本可能同时出现在多个 schema 键下，
              关系常挂在低概率身份下，故不在收集期卡 head 门槛，统一由 present_names 终筛）。
        """
        entities: List[Dict[str, Any]] = []
        seen_ent = set()
        rel_candidates: List[tuple] = []  # (head_name, tail_name, rel_type)

        for schema_key, items in raw.items():
            en_label = ENTITY_LABEL_MAP.get(schema_key)
            for it in items or []:
                if not isinstance(it, dict):
                    continue
                prob = float(it.get("probability", 0) or 0)
                name = str(it.get("text", "")).strip()
                head_ok_prob = prob >= self.precision and name and len(name) <= MAX_ENTITY_NAME_LENGTH
                # 1) 收集实体
                if head_ok_prob and en_label in ALLOWED_ENTITY_LABELS:
                    key = (name, en_label)
                    if key not in seen_ent:
                        seen_ent.add(key)
                        entities.append({"name": name, "label": en_label, "description": ""})
                # 2) 收集该 head 下的嵌套关系（head 达标与否在此不卡，终筛看 present_names）
                relations_map = it.get("relations") or {}
                if not name or not isinstance(relations_map, dict):
                    continue
                for cn_rel, r_items in relations_map.items():
                    rel_type = RELATION_TYPE_MAP.get(cn_rel, DEFAULT_RELATION_TYPE)
                    for ri in r_items or []:
                        if not isinstance(ri, dict):
                            continue
                        rprob = float(ri.get("probability", 0) or 0)
                        tail = str(ri.get("text", "")).strip()
                        if rprob >= self.precision and tail and tail != name:
                            rel_candidates.append((name, tail, rel_type))

        present_names = {e["name"] for e in entities}
        relations: List[Dict[str, Any]] = []
        seen_rel = set()
        for head, tail, rel_type in rel_candidates:
            # head/tail 必须都在已达阈值的实体集合中（对齐主项目硬约束）
            if head not in present_names or tail not in present_names:
                continue
            key = (head, tail, rel_type)
            if key in seen_rel:
                continue
            seen_rel.add(key)
            relations.append({"head": head, "tail": tail, "type": rel_type})

        return {"entities": entities, "relations": relations}
