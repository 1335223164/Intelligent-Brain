from typing import Dict, Any, List, Tuple
from langchain_openai import ChatOpenAI
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompt import render
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.utils.task_util import set_task_result


class AnswerOutputNode(BaseNode):
    name = "answer_output_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        核心逻辑
        1. 从state中获取answer
        1.1 如果获取到answer: 不是模糊的是精确的不用多路检索[不用在生成答案,直接返回]
        1.2 如果没有获取到answer: 模糊的进行多路检索(混合向量/HyDE/知识图谱/联网)[需要调用LLM模型对多路检索内容生成答案]
        :param state:
        :return:
        """

        # 1. 获取是否是流式
        is_stream = state.get('is_stream')
        # 2. 获取任务id
        task_id = state.get('task_id')

        # 3. 判断state是否有answer
        if state.get('answer'):
            # 将答案推送出去
            self._push_exist_answer(task_id, is_stream, state)
            is_streamed = False
        else:
            # 组装提示词
            prompt = self._build_prompt(state)
            state['prompt'] = prompt

            # 调用LLM生成答案
            self._generate_answer(prompt, state, task_id)

            is_streamed = is_stream

        # 4. 保存历史对话
        self._save_history(state)

        # 5. 告诉前端sse通过可以关闭 (更新sse状态) 前提is_stream是True
        if is_stream:
            # 5.1 已经流过（LLM生成的答案）
            if is_streamed:
                push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={})
            # 5.2 没有流过(自己生成的答案)
            else:
                push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={"answer": state.get('answer')})

            # 6. 返回
        return state

    def _push_exist_answer(self, task_id: str, is_stream: bool, state: QueryGraphState):
        """
        :param task_id:
        :param is_stream:
        :param state:
        :return:
        """

        # 1. 判断是非流式 [普通任务的队列:任务结果队列:]
        if not is_stream:
            set_task_result(task_id, 'answer', state.get('answer'))
        # else:
        #     # 2. 判断是流式 [sse队列]
        #     push_sse_event(task_id, SSEEvent.FINAL, {"answer": state.get('answer')})

    def _build_prompt(self, state) -> str:

        max_context_chars = self.config.max_context_chars
        # 1. 获取用户问题
        user_query = state.get('rewritten_query')

        # 2. 获取商品名列表
        item_names = state.get('item_names') or []

        # 3. 构建检索上下文
        retrieval_context = state.get('reranked_docs') or []
        formatted_context, usage_chars = self._format_retrieval_context(retrieval_context, max_context_chars)

        # 4. 构建历史上下文
        chat_history = state.get('history') or []  # 从内存获取历史对话
        formatted_history = self._format_chat_history(chat_history, usage_chars)

        # 4. 获取知识图谱关系三元组
        kg_triples = state.get('kg_triples') or []

        # 5. 格式化提示词模板
        return render(
            "query", "answer",
            context=formatted_context or "暂无检索到上下文",
            kg_triples="\n".join(kg_triples) or "暂无图谱关联信息",
            history=formatted_history or "暂无历史上下文",
            item_names=",".join(item_names),
            question=user_query
        )

    def _format_retrieval_context(self, retrieval_context: List[Dict[str, Any]], max_context_chars: int):
        """
        格式化检索到的上下文
        拼接一些元数据,让LLM学习
        :param retrieval_context: 检索到的上下文
        :param max_context_chars: 最大上下文长度
        :return:
        """

        # 1. 遍历
        formatted_lines = []
        usage = 0
        for index, context in enumerate(retrieval_context, 1):
            # 1.1 获取内容
            content = context.get('content', '')

            # 1.2 判断内容
            if not content:
                continue

            # 1.3 获取元数据
            metadata_content = [f"[文档:{index}]"]

            # 1.4 定义其他元数据
            for mete_field, template in [("chunk_id", "[chunk_id={}]"), ("title", "[title={}]"),
                                         ("source", "[source={}]"), ("url", "[url={}]")]:
                # 获取各个元数据字段的值
                filed_value = str(context.get(mete_field, '')).strip()

                # 格式化模板中的占位符
                if filed_value:
                    meta_content = template.format(filed_value)
                    metadata_content.append(meta_content)

            # 1.5 获取得分
            doc_score = context.get('score')

            # 1.6 判断得分
            if doc_score is not None:
                metadata_content.append(f"[score={float(doc_score):.6f}]")

            # 1.7 构建完整行数据 元数据 + 获取内容
            formatted_line = " ".join(metadata_content) + "\n" + content

            # 1.8 计算行与行的字符数 \n\n
            sep_chars = 2 if formatted_lines else 0

            line_length = len(formatted_line) + sep_chars

            # 1.9 计算总长度
            if usage + line_length > max_context_chars:
                break
            else:
                formatted_lines.append(formatted_line)
                usage += line_length

        return "\n\n".join(formatted_lines), max_context_chars - usage

    def _generate_answer(self, prompt: str, state: QueryGraphState, task_id: str):
        """
        调用llm生成答案,更新到state中
        :param prompt: 提示词
        :param state:
        :return:
        """

        # 1. 获取LLM客户端（答案是自然语言，必须用文本模式；默认的 JSON 模式会让 Qwen3 返回空内容）
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"LLM客户端获取失败,原因:{str(e)}")
            state['answer'] = "LLM暂无回答任何问题"
            return

        # 2. 判断是否是流式
        if state['is_stream']:
            # 2.1 流式调用
            # 获取llm的结果
            state['answer'] = self._stream_llm(prompt, llm_client, task_id)

        else:
            # 3. 非流式
            llm_result = self._invoke_llm(prompt, llm_client)
            state['answer'] = llm_result

            # 4. 写入到任务结果队列中 非流式
            set_task_result(task_id, 'answer', llm_result)

    def _invoke_llm(self, prompt: str, llm_client: ChatOpenAI) -> str:
        """
        非流式调用
        :param prompt:
        :return:
        """

        # 1. 同步调用
        try:
            llm_res = llm_client.invoke(prompt)
        except Exception as e:
            return "LLM暂无回答任何问题"

        # 2. 获取内容
        llm_content = getattr(llm_res, 'content', '') or ""

        # 3. 判断
        if not llm_content:
            return "LLM暂无回答任何问题"

        return llm_content

    def _stream_llm(self, prompt, llm_client, task_id):
        """
        流式调用
        :param prompt:
        :param llm_client:
        :return:
        """

        # 流式调用
        accelerate_data = ""
        try:
            for chunk in llm_client.stream(prompt):

                dela = getattr(chunk, 'content', '') or ""
                if dela:
                    push_sse_event(task_id, SSEEvent.DELTA, {"delta": dela})
                    accelerate_data += dela
        except Exception as e:
            return "LLM暂无回答任何问题"

        return accelerate_data

    def _save_history(self, state):
        """
        保存历史会话
        :param state:
        :return:
        """

        # 1. 获取session_id
        session_id = state.get('session_id')

        # 2. 获取用户查询问题
        user_query = state.get('original_query')

        # 3. 获取改写后的查询问题
        rewritten_query = state.get('rewritten_query')

        # 4. 获取商品名列表
        item_names = state.get('item_names') or []

        try:
            # 5 保存用户的信息
            save_chat_message(
                session_id=session_id,
                role="user",
                text=user_query,
                rewritten_query=rewritten_query,
                item_names=item_names
            )

            # 保存ai角色的信息
            save_chat_message(
                session_id=session_id,
                role="assistant",
                text=state.get('answer'),
                rewritten_query=rewritten_query,
                item_names=item_names
            )
        except Exception as e:
            self.logger.error(f"保存历史对话到mongDB失败,原因:{str(e)}")

    def _format_chat_history(self, chat_history: List[Dict[str, Any]], usage_chars: int):
        """
        格式化历史上下文
        :param chat_history: 历史会话
        :param usage_chars: 最大长度
        :return:
        """

        formatted_lines = []
        used_chars = 0
        # 1. 遍历格式化后的文档
        role_map = {"user": "用户", "assistant": "ai助手"}
        for msg in chat_history:

            # 1.1 获取消息角色
            role = msg.get('role', '')

            # 1.2 获取消息内容
            text = msg.get('text', '')

            # 1.3 获取格式化后的行
            if not text or role not in role_map:
                continue

            formatted_line = f"{role_map[role]}: {text}"

            # 1.4 计算分割符长度
            seperator_usage = 1 if formatted_lines else 0

            # 1.5 计算总长度
            total_usage = seperator_usage + len(formatted_line)

            if used_chars + total_usage > usage_chars:
                break

            formatted_lines.append(formatted_line)
            used_chars += total_usage

        return "\n".join(formatted_lines)



