import os
import uvicorn
from fastapi import FastAPI, UploadFile, Depends, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from starlette.staticfiles import StaticFiles
from knowledge.core.deps import get_upload_file_service
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.service.upload_service import UploadService
from knowledge.utils.task_util import get_task_info


# 1. 创建fastapi实例
# 2. 注册路由
# 3. 利用uvicorn服务器启动fastapi

def create_app():
    """
    创建fastapi实例
    :return: fastapi实例
    """

    # 1. 实例化
    app = FastAPI(description="掌柜智库导入的应用", version="v1.0")

    # 2. 跨域配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许的源
        allow_credentials=False,  # 允许携带cookie
        allow_methods=["*"],  # 允许的请求方法
        allow_headers=["*"],  # 允许的请求头
    )

    # 3. 挂载静态文件
    page_dir = get_front_page_dir()

    if page_dir and os.path.exists(page_dir):
        app.mount("/front", StaticFiles(directory=page_dir))

    # 4. 注册路由
    register_router(app)

    # 5. 返回fastapi实例
    return app


def register_router(app: FastAPI):
    @app.get("/")
    def hello_world():
        return {"flag": "success"}

    # 1. 上传请求
    @app.post("/upload", response_model=UploadResponse)
    def upload_endpoint(file: UploadFile,
                        background_tasks: BackgroundTasks,
                        is_kg_enabled: bool = Form(True),
                        upload_service: UploadService = Depends(get_upload_file_service)):
        """
        处理文件上传
        :param is_kg_enabled: 是否构建知识图谱（默认开启）
        :return:
        """

        # 1. 将上传文件写入本地临时目录,和远程minio
        import_file_path, file_dir, task_id = upload_service.process_upload_file(file)

        # 2. 运行整个导入的图谱（后台任务，传入知识图谱开关）
        background_tasks.add_task(upload_service.run_import_graph, task_id, import_file_path, file_dir, is_kg_enabled)

        # 3. 返回上传后的响应
        return UploadResponse(message=f"{file.filename}文件上传成功", task_id=task_id)

    # 2. 查询上传任务状态的请求
    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    def get_task_status_endpoint(task_id: str):
        """
        查询上传任务: 前端会轮询调用查询上传任务状态结构
        :return:
        """

        task_info = get_task_info(task_id)

        return TaskStatusResponse(**task_info)


if __name__ == '__main__':
    uvicorn.run(app=create_app(), host="0.0.0.0", port=18000, log_level="info")
