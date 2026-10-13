import shutil
import time
from datetime import datetime
import os
import uuid
import logging
from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_processor.exceptions import FileProcessingError
from knowledge.processor.import_processor.main_graph import import_app
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED, add_running_task, add_done_task, add_node_duration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UploadService:
    """
    处理文件上传相关逻辑
    """

    def _get_base_url(self):
        return os.path.join(get_local_base_dir(), datetime.now().strftime("%Y%m%d"))

    def run_import_graph(self, task_id: str, import_file_path: str, file_dir: str, is_kg_enabled: bool = True):
        """
        运行图谱流程
        :param task_id:
        :param import_file_path:
        :param file_dir:
        :param is_kg_enabled: 是否构建知识图谱
        :return:
        """

        # 1. 更新任务状态为processing
        update_task_status(task_id,TASK_STATUS_PROCESSING)

        # 2. 定义运行流程的状态
        graph_state = {
            "task_id": task_id,
            "import_file_path": import_file_path,
            "file_dir": file_dir,
            "is_kg_enabled": is_kg_enabled,
        }

        # stream:迭代整个graph图状态每一个节点的事件 节点名称 节点调用后的状态
        # 3. 运行整个图谱
        try:
            for event in import_app.stream(graph_state):
                for node_name, state in event.items():
                    # 逐节点日志：节点级进度已由 BaseNode 通过 add_running_task/add_done_task 上报，这里只打印
                    logger.info(f"{node_name} 节点调用后的状态: {state}")

            # 全部节点执行完毕后，才把整体任务标记为完成（不能放在循环内，否则首个节点跑完就误报完成）
            update_task_status(task_id, TASK_STATUS_COMPLETED)

        except Exception as e:
            logger.error(f"任务{task_id}执行失败,原因{str(e)}")

            update_task_status(task_id, TASK_STATUS_FAILED)

    def process_upload_file(self, file):
        """
        处理文件上传
        :param file:
        :return:
        """

        # 1. 生成任务id
        task_id = str(uuid.uuid4().hex[:8])  # 真正的随机
        add_running_task(task_id,"upload_file")
        start_time = time.time()

        # 2. 生成日期目录和临时目录,并且拼接到一起
        base_file_dir = self._get_base_url()

        # 3. 构建完整文件归属目录
        file_dir = os.path.join(base_file_dir, task_id)

        # 4. 保存文件到临时目录
        import_file_path = self._save_upload_file_to_local(file, file_dir)

        # 5. 保存文件到minio
        self._save_upload_file_to_minio(import_file_path, file.filename)

        # 6. 返回
        add_done_task(task_id,"upload_file")
        add_node_duration(task_id,"upload_file",time.time() - start_time)
        return import_file_path, file_dir, task_id

    def _save_upload_file_to_local(self, file, file_dir):
        """
        保存上传文件到本地
        :param file: 上传的文件
        :param file_dir: 文件归属目录
        :return: 保存后的文件路径
        """

        # 1. 创建文件归属目录
        os.makedirs(file_dir, exist_ok=True)

        # 2. 构建导入文件的path路径
        import_file_path = os.path.join(file_dir, file.filename)

        # 3. 写入
        try:
            with open(import_file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        except Exception as e:
            logger.info(f"写入的文件{file.filename}写入失败")
            raise FileProcessingError(f"写入的文件{file.filename}写入失败,原因{str(e)}")

        # 4. 返回文件路径
        return import_file_path

    def _save_upload_file_to_minio(self, import_file_path, filename):
        """
        上传文件到minio
        :param import_file_path: 上传文件的路径
        :param filename: 文件名
        :return:
        """

        # 1. 获取minio客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.error(f"Minio客户端获取失败{str(e)}")
            return

            # 2. 获取minio相关信息
        bucket_name = os.getenv("MINIO_BUCKET_NAME")
        object_name = f"origin_files/{datetime.now().strftime("%Y%m%d")}/{filename}"

        # 3. 上传
        try:
            minio_client.fput_object(bucket_name, object_name, file_path=import_file_path)
        except Exception as e:
            logger.error(f"{filename}文件上传到minio失败,原因{str(e)}")
            return
