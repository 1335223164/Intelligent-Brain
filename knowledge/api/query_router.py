import asyncio
import os
from typing import Union
from fastapi import Request, HTTPException
import uvicorn
from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.params import Depends
from starlette.responses import StreamingResponse
from starlette.staticfiles import StaticFiles
from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from knowledge.schema.query_schema import QueryRequest, StreamSubmitResponse, QueryResponse
from knowledge.service.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator


# 1. 创建fastapi实例
# 2. 注册路由
# 3. 利用uvicorn服务器启动fastapi


def create_app():
    """
    创建fastapi实例
    :return: fastapi实例
    """

    # 1. 实例化
    app = FastAPI(description="掌柜智库查询的应用", version="v1.0")

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

    @app.post("/query")
    async def query(request: QueryRequest,
                    background_tasks: BackgroundTasks,
                    query_service: QueryService = Depends(get_query_service)
                    ) -> Union[QueryResponse, StreamSubmitResponse]:
        """
        处理查询请求
        :param background_tasks: 后台任务对象
        :param query_service:
        :param request: 前端发送的请求对象
        :return:
        """

        # 1. 获取session_id
        session_id = request.session_id or query_service.generate_session_id()

        # 2. 获取task_id
        task_id = query_service.generate_task_id()

        # 3. 判断是否是流式调用
        if request.is_stream:
            # sse队列创建出来
            create_sse_queue(task_id)

            # 利用fastapi的background_tasks启动线程执行查询任务
            background_tasks.add_task(query_service.run_query_graph, session_id, task_id, request.query,
                                      request.is_stream)

            return StreamSubmitResponse(
                message="查询请求已经提交",
                session_id=session_id,
                task_id=task_id
            )

        else:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                query_service.run_query_graph,
                session_id, task_id, request.query, request.is_stream,
            )

            return QueryResponse(
                message="查询请求已经处理完成",
                session_id=session_id,
                answer=query_service.get_task_result(task_id)
            )

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request) -> StreamingResponse:
        """
        返回sse协议需要的数据包: 通过yield返回 流式 + yield
        :param request:
        :param task_id:
        :return:
        """

        return StreamingResponse(
            content=sse_generator(task_id, request),
            media_type="text/event-stream",
        )

    @app.get("/history/{session_id}")
    async def get_history(
            session_id: str, limit: int = 50,
            service: QueryService = Depends(get_query_service),
    ):
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")


    @app.delete("/history/{session_id}")
    async def clear_chat_history(
            session_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}




if __name__ == '__main__':
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8000, log_level="info")
