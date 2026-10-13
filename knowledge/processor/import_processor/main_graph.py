import json
from typing import Sequence, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from knowledge.processor.import_processor.base import setup_logging
from knowledge.processor.import_processor.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_processor.nodes.md_to_img_node import MarkDownToImgNode
from knowledge.processor.import_processor.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_processor.nodes.embedding_chunks_node import EmbeddingChunksNode
from knowledge.processor.import_processor.nodes.import_milvus_node import ImportMilvusNode
from knowledge.processor.import_processor.nodes.kg_graph_node import KnowLedgeGraphNode
from knowledge.processor.import_processor.state import ImportGraphState
"""
编排节点

定义节点
定义条件边
定义顺序边
运行整个pineline图谱的各个节点

"""


def import_router(state: ImportGraphState) -> Sequence[Literal["pdf_to_md_node", "md_to_img_node", END]]:
    """
    根据state中的 is_pdf_read_enabled 和 is_md_read_enabled 来确定运行流程
    :param state: 运行时状态
    :return: 下一个运行节点
    """
    # 1. 获取上传的文件属于pdf还是md
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"
    elif state.get("is_md_read_enabled"):
        return "md_to_img_node"
    else:
        return END


def kg_router(state: ImportGraphState) -> Literal["kg_graph_node", END]:
    """
    向量入库后，根据 is_kg_enabled 决定是否构建知识图谱。
    :param state: 运行时状态
    :return: 下一个运行节点
    """
    if state.get("is_kg_enabled"):
        return "kg_graph_node"
    return END


def import_graph() -> CompiledStateGraph:
    """
    1. 定义运行时图状态
    2. 定义节点 (入口节点,业务节点)
    3. 定义边 (条件边,普通边)
    4. 返回运行时状态
    :return:
    """

    # 1. 定义运行时图状态workflow
    work_flow = StateGraph(ImportGraphState)

    # 2. 定义入口节点
    work_flow.set_entry_point("entry_node")

    # 2. 定义其他节点名和节点映射表
    node_name_instance = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "md_to_img_node": MarkDownToImgNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_recognition_node": ItemNameRecognitionNode(),
        "embedding_chunks_node": EmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
        "kg_graph_node": KnowLedgeGraphNode(),
    }

    # 3. 添加节点
    for node_name, node_instance in node_name_instance.items():
        work_flow.add_node(node_name, node_instance)

    # 5. 定义边
    # 5.1 定义条件边
    work_flow.add_conditional_edges("entry_node", import_router, {
        "pdf_to_md_node": "pdf_to_md_node",
        "md_to_img_node": "md_to_img_node",
        END: END,
    })

    # 5.2 添加普通边
    work_flow.add_edge("pdf_to_md_node", "md_to_img_node")
    work_flow.add_edge("md_to_img_node", "document_split_node")
    work_flow.add_edge("document_split_node", "item_name_recognition_node")
    work_flow.add_edge("item_name_recognition_node", "embedding_chunks_node")
    work_flow.add_edge("embedding_chunks_node", "import_milvus_node")

    # 5.3 向量入库后根据开关决定是否构建知识图谱
    work_flow.add_conditional_edges("import_milvus_node", kg_router, {
        "kg_graph_node": "kg_graph_node",
        END: END,
    })
    work_flow.add_edge("kg_graph_node", END)

    # 5.4 编译
    compiled_state_graph = work_flow.compile()
    return compiled_state_graph


import_app = import_graph()


########################################
# 测试
########################################
def run_import_graph() -> ImportGraphState:
    # 1. 定义运行流程的状态
    graph_state = {
        "import_file_path": "",
        "file_dir": "",
        "is_kg_enabled": False,
    }

    # stream:迭代整个graph图状态每一个节点的事件 节点名称 节点调用后的状态
    for event in import_app.stream(graph_state):
        final_state = {}
        for node_name, state in event.items():
            print(f"{node_name} 节点调用后的状态: {state}")
            final_state = state

    return final_state


if __name__ == '__main__':
    setup_logging()

    final_state = run_import_graph()

    json.dumps(final_state, indent=4, ensure_ascii=False)

    # 观测执行状态图
    print(import_app.get_graph().print_ascii())
