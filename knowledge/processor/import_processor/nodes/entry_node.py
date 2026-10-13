import json
from pathlib import Path
from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import *
from knowledge.processor.import_processor.state import ImportGraphState


class EntryNode(BaseNode):
    name = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        根据上传文件后缀 修改state 中 is_md_read_enabled is_pdf_read_enabled   属性
        :param state: ImportGraphState  全局状态
        :return:
        """

        # 1. 获取上传文件
        import_file_path = state.get("import_file_path", "")
        file_dir = state.get("file_dir", "")

        # 2. 判断
        if not import_file_path:
            raise StateFieldError(node_name=self.name, field_name="import_file_path", expected_type=str)

        if not file_dir:
            raise StateFieldError(node_name=self.name, field_name="file_dir", expected_type=str)

        # 3. 标准化
        import_file_path_obj = Path(import_file_path)
        file_dir_obj = Path(file_dir)

        # 4. 判断路径是否存在
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name="import_file_path", expected_type=str)

        if not file_dir_obj.exists():
            raise StateFieldError(node_name=self.name, field_name="file_dir", expected_type=str)

        # 5. 获取文件后缀
        # 注意：state 运行时就是普通 dict（TypedDict 仅声明期），必须用键赋值，不能写属性
        if import_file_path_obj.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = str(import_file_path_obj)
        elif import_file_path_obj.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = str(import_file_path_obj)
        else:
            self.logger.error("文件格式不支持")
            raise StateFieldError(node_name=self.name, field_name="import_file_path", message="文件格式错误")

        # 6. 获取上传文件标题,更新状态
        state["file_title"] = import_file_path_obj.stem

        # 7. 返回sate
        return state


if __name__ == '__main__':
    entry_node = EntryNode()
    init_state = {
        "import_file_path": r"",
        "file_dir": r""
    }

    result = entry_node.process(init_state)

    json.dump(result, ensure_ascii=False, indent=4)
