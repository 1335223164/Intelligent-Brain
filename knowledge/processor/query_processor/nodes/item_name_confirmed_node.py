import re, json
from json import JSONDecodeError
from typing import Dict, Any, List, Tuple
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_processor.config import get_config
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.prompt import load_prompt, render
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages


# 商品名归一化：仅用于“逐字相等 shortcut”的容错比较（在高置信候选集内），
# 刻意保守：只做小写化、去分隔符、去尾部虚词，不重排词序——避免把不同型号误并。
_ITEM_NAME_NOISE_SUFFIX = re.compile(r"(系列|型号|款型|款|版本|版|产品|设备|机型)+$")


def _normalize_item_name(name: str) -> str:
    """保守归一化商品名（仅用于对齐比较，不改变实际写库/展示的名称）。"""
    s = (name or "").strip().lower()
    s = re.sub(r"[\s\-_·、,，/]+", "", s)          # 去空白与常见分隔符
    s = _ITEM_NAME_NOISE_SUFFIX.sub("", s)          # 去尾部虚词（系列/型号/款…）
    return s


class _ItemNameExtractor:

    def __init__(self, logger):
        self.logger = logger

    def extractor_item_name(self, original_query: str, history_context: str) -> Dict[str, Any]:
        """
        提取商品名
        :param original_query: 用户原始查询
        :param history_context: 历史上下文
        :return:
        """

        # 默认结果
        llm_result = {"item_names": [], "rewritten_query": original_query}

        # 1. 获取LLM客户端
        try:
            llm_client = AIClients.get_llm_client()
        except ConnectionError as e:
            self.logger.error(f"LLM客户端获取失败,原因{str(e)}")
            return llm_result

        # 2. 获取商品名提取的提示词
        # 2.1 系统提示词
        item_name_system_prompt = load_prompt("query", "item_name_system")
        # 2.2 用户提示词
        item_name_user_prompt = render(
            "query", "item_name_extract",
            history_text=history_context.strip() if history_context else "暂无历史上下文",
            query=original_query
        )

        # 3. 调用LLM
        try:
            llm_response = llm_client.invoke([
                SystemMessage(item_name_system_prompt),
                HumanMessage(item_name_user_prompt)
            ])
        except Exception as e:
            self.logger.error(f"LLM调用失败, 原因{str(e)}")
            return llm_result

        # 4. 获取LLM输出
        llm_response_context = llm_response.content

        # 5. 判断LLM输出
        if not llm_response_context:
            return llm_result

        # 6. 清洗(判断输出内容的类型以及空格)和解析(反序列化)
        parsed_result: Dict[str, Any] = self._clean_and_parse(llm_response_context)

        # 8. 最终返回
        llm_result["item_names"] = parsed_result.get("item_names")
        llm_result["rewritten_query"] = parsed_result.get('rewritten_query') \
            if parsed_result.get('rewritten_query') else original_query

        return llm_result

    def _clean_and_parse(self, llm_response_context: str) -> Dict[str, Any]:
        """
        请求以及解析LLM的结果
        :param llm_response_context: llm的输出
        :return:
        """

        # 1. 去除json代码快围栏标记```{}```  防御性编程
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response_context.strip())
        content = re.sub(r"\s*```$", "", cleaned.strip())

        # 2. 解析
        try:
            # 2.1 反序列化
            llm_response_obj = json.loads(content)

            # 2.2 获取item_names
            raw_item_name = llm_response_obj.get("item_names")

            # 2.3 判断类型
            if not isinstance(raw_item_name, list):
                item_names = []
            else:
                item_names = [item_name.strip() for item_name in raw_item_name if
                              item_name.strip() and isinstance(item_name, str)]

            # 2.4 获取rewritten_query
            raw_rewritten_query = llm_response_obj.get("rewritten_query")

            # 2.5 判断类型
            if not isinstance(raw_rewritten_query, str):
                rewritten_query = ""
            else:
                rewritten_query = raw_rewritten_query.strip()

            # 2.6 返回
            return {"item_names": item_names, "rewritten_query": rewritten_query}


        except JSONDecodeError as e:
            self.logger.error(f"LLM输出的json反序列化失败, 原因{str(e)}")
            raise JSONDecodeError(msg=e.msg, doc=e.doc, pos=e.pos)


class _ItemNameAligner:

    def __init__(self, logger):
        self.logger = logger
        self._config = get_config()

    def search_and_align(self, item_names: list[str]) -> Tuple[List[str], List[str]]:
        """
        检索向量数据库,并且和向量数据库中的商品对齐 最终返回确定的商品名列表或者模糊的商品名列表
        :param item_names: LLM提取到的商品名称列表
        :return:
        """

        # 1. 混合检索向量数据库
        search_result: List[Dict[str, Any]] = self._search_vector(item_names)
        if not search_result:
            return [], []
        # 2. 根据混合向量检索的结果做对齐
        confirmed, options = self._align(search_result)

        # 3. 返回确定的confirmed和options
        # 差异化过滤
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)
        return confirmed, options

    def _search_vector(self, item_names: list[str]) -> List[Dict[str, Any]]:
        """
        对LLM提取到的所以商品名进行向量检索
        :param item_names: LLM提取到的商品名
        :return:
        """

        final_search_result = []

        # 1. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端获取失败,原因{str(e)}")
            return final_search_result

        # 2. 获取BGE-M3嵌入模型
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3客户端获取失败,原因{str(e)}")
            return final_search_result

        # 2. 商品名批量向量化 (混合向量)
        try:
            hybrid_vector_result = generate_bge_m3_hybrid_vectors(model=bge_m3_client, embedding_documents=item_names)
        except Exception as e:
            self.logger.error(f"商品名列表{item_names}向量化失败,原因{str(e)}")
            return final_search_result

        # 4. 混合向量检索
        for index, item_name in enumerate(item_names):
            # 4.1 构建稠密以及稀疏混合向量的检索请求
            hybrid_requests = create_hybrid_search_requests(
                hybrid_vector_result["dense"][index],
                hybrid_vector_result["sparse"][index])
            # 4.2 执行混合检索

            hybrid_search_result = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self._config.item_name_collection,
                search_requests=hybrid_requests,
            )

            # 4.3 解析结果
            matches = [{"score": item_search_res["distance"], "item_name": item_search_res["entity"]["item_name"]}
                       for item_search_res in (hybrid_search_result[0] if hybrid_search_result else [])]

            # 4.4 封装
            final_search_result.append({
                "extracted_name": item_name,
                "matches": matches
            })

        # 5. 返回
        return final_search_result

    def _align(self, search_result: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        主要职责：对齐
        制定规则：
        两个规则：1.如果从向量数据库中查到的商品名分数比如大于0.7 放到confirmed
        两个规则：2.如果从向量数据库中查到的商品名分数比如小于等于0.7大于0.45 放到options
                 3.如果从向量数据库中查到的商品名分数小于等于0.45 两个容器都不放
        像options中添加的商品名满足三个条件 条件1：小于high阈值且大于options 条件2：该商品名不在options 条件3：不在confirmed中
        像confirmed中添加商品名的:条件1：阈值大于high(3个小条件) 条件2：该商品名不在confirmed中 条件3：不用考虑options的，哪怕这个商品名已经在options.只要能进入不到
        confirmed中添加


        Args:
            search_result:  向量数据库检索到的结果

        Returns:
          最终两个容器confirmed、options列表中的商品名
        """

        # 1. 定义两个容器
        confirmed = []
        options = []

        # 2. 遍历检索到的所有商品名从Milvus中搜索结果
        for item_sea_res in search_result:
            # 2.1 获取extracted_name LLM提取的商品名
            llm_extract_item_name = item_sea_res.get("extracted_name")

            # 2.2 获取matches(某一个商品名下的搜索结果)
            item_name_matches = item_sea_res.get("matches")

            # 2.3 (可选)按分数降序排序
            item_name_matches_sorted = sorted(item_name_matches, key=lambda x: x["score"], reverse=True)

            # 2.3.1 前导命中：用户把某个注册商品名逐字写在最前（其后接定语/普通名词，
            #        如“华为擎云L420的蓝牙鼠标”“绿米网关信息查询界面”）。在 mid 及以上候选里，
            #        取“归一化后是抽取名前缀且最长”的注册名，直接进 confirmed（精确相等也属于此情形）。
            #        证据很强（注册名确实出现在句首），误并风险极低。
            llm_norm = _normalize_item_name(llm_extract_item_name)
            prefix_hits = [m for m in item_name_matches_sorted
                           if m["score"] >= self._config.item_name_mid_confidence
                           and llm_norm
                           and _normalize_item_name(m.get("item_name", ""))
                           and llm_norm.startswith(_normalize_item_name(m.get("item_name", "")))]
            if prefix_hits:
                picked = max(prefix_hits,
                             key=lambda m: len(_normalize_item_name(m.get("item_name", "")))).get("item_name")
                if picked and picked not in confirmed:
                    confirmed.append(picked)
                continue

            # 2.4 (收集---->分数值大于0.7的结果: 高置信度)
            high = [h for h in item_name_matches_sorted if h["score"] >= self._config.item_name_high_confidence]

            # 2.5 如果高置信度有
            if high:
                # 2.5.1 高置信候选里是否有与 llm 提取名“归一化后相等”的商品名（如 B530系列↔B530），
                #        命中即视为最准，直接进 confirmed，避免因近同名分差不足被误判为歧义。
                llm_norm = _normalize_item_name(llm_extract_item_name)
                extract = next((h for h in high
                                if llm_norm and _normalize_item_name(h.get("item_name", "")) == llm_norm), None)

                # 2.5.2 添加到confirmed
                if extract:  # 条件很难除法
                    picked = extract.get("item_name")
                    if picked not in confirmed:
                        confirmed.append(picked)
                elif len(high) == 1:
                    picked = high[0].get("item_name")
                    if picked not in confirmed:
                        confirmed.append(picked)
                else:
                    # 有多个分数较高的商品名
                    top_score = high[0]['score']
                    if top_score - high[1]['score'] >= self._config.item_name_score_gap:
                        picked = high[0].get("item_name")
                        if picked not in confirmed:
                            confirmed.append(picked)
                    else:
                        # 高置信度多个且分差不够大 → 属于歧义，把候选（最多 max_options 个）放进 options 让用户选
                        # 注意：这里必须是切片，写成 high[N] 会在候选不足时 IndexError
                        for h in high[:self._config.item_name_max_options]:
                            picked = h.get("item_name")
                            if picked not in confirmed and picked not in options:
                                options.append(picked)
            # 2.5 不是高置信度,有可能是中等置信,也可能不是
            else:
                mid = [m for m in item_name_matches_sorted
                       if m['score'] >= self._config.item_name_mid_confidence
                       and m.get('item_name') not in confirmed
                       and m.get('item_name') not in options]
                if mid:
                    for m in mid[:self._config.item_name_max_options]:
                        options.append(m.get('item_name'))

        # 3. 返回
        return confirmed, options[:self._config.item_name_max_options]

    def _item_name_score_filter(self, confirmed: List[str], search_result: List[Dict[str, Any]]):
        """
        # RS12数字万用表和RS13数字万用表
        item_name:RS12数字万用表:0.90
        item_name:万用表:0.71
        item_name:RS13数字万用表:0.88
        Args:
            confirmed:
            search_result:[]

        Returns:

        """

        # 1. 构建 商品名 → 最高分数 的映射
        item_name_score = {}
        for search_result in search_result:
            matches = search_result.get('matches', [])
            for m in matches:
                score = m.get('score', 0)
                item_name = m.get('item_name')
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name, 0), score)

        # 2. 防御性检查：如果没有收集到任何分数，直接返回原始 confirmed
        if not item_name_score:
            return confirmed

        # 3. 取出分数值最大的作为基准
        max_score = max(item_name_score.values())
        return [name for name, score in item_name_score.items() if
                max_score - score <= self._config.item_name_score_gap]


class ItemNameConfirmedNode(BaseNode):
    name = "item_name_confirmed_node"

    def __init__(self):
        # 初始化父类
        super().__init__()
        self._extractor = _ItemNameExtractor(self.logger)
        self._aligner = _ItemNameAligner(self.logger)

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        主要职责：
        1. 利用LLM从用户原始查询中提取商品名以及改写原始问查询（我喜欢你）
        1.1 如果LLM提取到了商品名，才进行第2步 去milvus对齐
        1.2 如果LLM没有提取到商品名，直接返回
        2. 根据Milvus中存储的商品名进行对齐（目的：检索更加的准确：多路检索都会利用该节点提取到的商品名，因此直接用LLM提取到商品名的话 下游多路检索在过滤的时候，过滤条件极其不准确。导致检索到的噪音很多 LLM最终输出的幻觉很高）
        最终不是要LLM的商品名 而是要Milvus中存储的商品名：因为milvus中每一个chunk都会关联milvus自己的商品名
        3. 决策（该走下去，还是回头）

        利用两个容器，产生三个分支：第一个分支去检索  第二个分支：给用户确认  第三个分支：抱歉
        1. confirmed:如果是精确的商品名--->给confirmed添加精确的商品名
        2. options:商品名不是精确，可是找到多个相似的---->给options中添加找到的多个不精确的商品名。

        state['answer']不要给，进行多路检索
        获取到多路检索结果
        把多路检索到的结果(RRF  RERANKER)给LLM
        LLM生成答案,在state['answer']
        state['answer']:就返回：
        1. 返回候选商品名【不精确】，给用户下一步确认使用
        2. 没有任何商品名，返回抱歉，没有找到您询问的关于任何商品的名字
        Args:
            state:
        Returns:
        """

        # 1. 获取用户原始问题
        original_query = state.get("original_query")
        session_id = state.get('session_id')

        # 2. 获取历史对话
        history_context = get_recent_messages(session_id=session_id,limit=10)

        formatted_history = []
        role_map = {"user": "用户", "assistant": "ai助手"}
        for history in history_context:
            role = history.get('role','')
            text = history.get('text','')
            formatted_context = f"角色:{role_map[role]} 内容:{text}"
            formatted_history.append(formatted_context)
        formatted_history_str = " ".join(formatted_history)


        # 3. 利用LLM进行商品名提取和查询重写
        llm_result: Dict[str, Any] = self._extractor.extractor_item_name(original_query, formatted_history_str)

        # 3.1 获取LLM结果信息
        item_names = llm_result.get("item_names")
        rewritten_query = llm_result.get("rewritten_query")

        # 4. 根据item_names做判断
        if item_names:
            confirmed, options = self._aligner.search_and_align(item_names)
        else:
            confirmed, options = [], []

        # 5. 决策
        self._decide(confirmed, options, state, rewritten_query, item_names)

        # 6.返回
        state['history'] = history_context
        return state

    def _decide(self, confirmed: List[str], options: List[str], state: QueryGraphState, rewritten_query: str,
                item_names: List[str]):
        """
        根据confirmed以及options判断是继续检索还是返回用户信息
        :param confirmed: 确认的商品名列表
        :param options: 模糊的商品名列表
        :param state: 查询状态
        :param rewritten_query: 重写后的问题
        :param item_names: LLM提取到的商品名列表
        :return:
        """

        if confirmed:
            state["item_names"] = confirmed
            state["rewritten_query"] = rewritten_query
        elif options:
            state["answer"] = (
                f"我不确定您指的的哪款产品。"
                f"您是在查询以下产品吗: {"、".join(options)}？"
            )
        else:
            state["answer"] = f"抱歉，我无法识别您询问的具体产品名称,请提供更准确的产品名称或型号。"


if __name__ == '__main__':
    item_name_confirmed_node = ItemNameConfirmedNode()
    init_state = {
        # "original_query": "RS-12数字万用表和H3C LA2608 室内无线网关的操作区别是什么?"
        # "original_query": "RS-12数字万用表和RS-13数字万用表的区别?"
        "original_query": "RS-12数字万用表如何测量电压以及HAK180的介质规格有哪些?"
        # "original_query": "RS-12数字万用表如何测量电压"  # 单个商品询问
    }
    llm_result = item_name_confirmed_node.process(init_state)

    print(llm_result)
