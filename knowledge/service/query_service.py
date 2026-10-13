import shutil
import time
from datetime import datetime
import os
import uuid
import logging
from typing import List, Dict, Any

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_processor.exceptions import FileProcessingError
from knowledge.processor.import_processor.main_graph import import_app
from knowledge.processor.query_processor.main_graph import query_app
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.mongo_history_util import get_recent_messages, clear_history
from knowledge.utils.task_util import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED, add_running_task, add_done_task, add_node_duration, get_task_result

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryService:

    @staticmethod
    def generate_session_id():
        return str(uuid.uuid4())

    @staticmethod
    def generate_task_id():
        return str(uuid.uuid4().hex[:12])

    def run_query_graph(self, session_id: str, task_id: str, query: str, is_stream: bool):
        """
        查询流程的pineline
        :param session_id: 会话id
        :param task_id: 任务id
        :param query: 问题
        :param is_stream: 是否是流式
        :return:
        """

        update_task_status(task_id,TASK_STATUS_PROCESSING)

        # 1. 构建查询初始化的状态
        query_init_state = {
            "session_id": session_id,
            "task_id": task_id,
            "original_query": query,
            "is_stream": is_stream,
        }

        # 2. 运行查询的pineline
        try:
            query_app.invoke(query_init_state)
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            update_task_status(task_id, TASK_STATUS_FAILED)
            logger.error(f"运行查询流程出现异常,原因:{str(e)}")


    def get_task_result(self,task_id):
        answer = get_task_result(task_id,"answer")
        return answer

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:

        # 1. 根据session_id获取最近的指定条数的历史对话
        records = get_recent_messages(session_id, limit=limit)
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        return clear_history(session_id)








