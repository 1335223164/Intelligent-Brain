import subprocess,time,json,shutil,sys,os
from pathlib import Path
from typing import Tuple
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import *
from knowledge.processor.import_processor.state import ImportGraphState, create_default_state


class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点处理逻辑的入口
        接收pdf文件path 利用mineru解析工具将pdf解析成md
        :param state:
        :return: ImportGraphState
        """

        # 核心逻辑 (接收pdf文件path 利用mineru解析工具将pdf解析成md)

        # 1. 获取导入文件的路径以及输出目录
        import_file_path_obj, file_dir_obj = self._validate_state(state)

        # 2. 执行mineru解析 (命令 mineru -p input_path -o output_dir --source=local)
        processed_code = self._execute_mineru_parse(import_file_path_obj, file_dir_obj)

        if processed_code != 0:
            raise PdfConversionError(node_name=self.name, message="Mineru解析失败")

        # 3. 获取解析后的md文件路径
        md_path = self._get_md_path(import_file_path_obj, file_dir_obj)

        # 4. 更新state['md_path']
        state["md_path"] = md_path

        # 5. 返回状态
        return state

    # 获取导入文件的路径以及输出目录
    def _validate_state(self, state: ImportGraphState) -> Tuple[Path, Path]:
        """
        获取导入文件的路径以及输出目录
        :param state: 导入图谱节点状态
        :return: 导入文件的路径以及输出目录
        Tuple[Path, Path]
        """
        self.log_step("step1", "准备校验和获取解析文件路径和输出目录")

        # 1. 获取解析的文件Path
        import_file_path = Path(state.get("import_file_path", ""))

        # 2. 判断是否为空
        if not import_file_path:
            raise StateFieldError(node_name=self.name, field_name="import_file_path", expected_type=str,
                                  message="导入文件路径为空")

        # 3. 标准化解析文件路径
        import_file_path_obj = Path(import_file_path)

        # 4. 判断是否是一个有效的路径
        if not import_file_path_obj.exists():
            raise StateFieldError(node_name=self.name, field_name="import_file_path", expected_type=str,
                                  message="导入文件路径不存在")

        # 5. 获取输出文件目录
        file_dir = Path(state.get("file_dir", ""))

        # 6. 判断是否为空
        if not file_dir:
            # 6.1 如果为空，则使用导入文件的父目录作为输出目录
            file_dir = import_file_path_obj.parent

        # 7. 判断是否是有效的目录
        if not file_dir.is_dir():
            raise StateFieldError(node_name=self.name, field_name="file_dir", expected_type=str,
                                  message="输出路径目录不存在")

        # 8. 标准化输出目录
        file_dir_obj = Path(file_dir)

        self.logger.info(f"解析文件路径: {import_file_path_obj}")
        self.logger.info(f"输出目录: {file_dir_obj}")

        # 返回校验通过
        return import_file_path_obj, file_dir_obj

    # 执行mineru解析
    def _execute_mineru_parse(self, import_file_path_obj: Path, file_dir_obj: Path) -> int:
        """
        执行mineru解析 (命令 mineru1 -p input_path -o output_dir --source=local)
        :param import_file_path_obj: 文档输入路径
        :param file_dir_obj: 解析结果输出目录
        :return: processed_code 成功状态码 0或非0
        0: 成功   非0: 失败
        """

        # 子进程执行cmd命令

        # 1. 定义cmd（mineru 3.x CLI：-p 输入 -o 输出，默认 hybrid-engine 后端，
        # 产物落在 {output}/{文件名}/hybrid_auto/；服务进程 PATH 可能不含 venv\Scripts，
        # 故用 which 解析失败后回退到当前解释器同目录的 mineru.exe）
        mineru_exe = shutil.which("mineru")
        if not mineru_exe:
            fallback = Path(sys.executable).parent / ("mineru.exe" if os.name == "nt" else "mineru")
            if fallback.exists():
                mineru_exe = str(fallback)
        if not mineru_exe:
            raise PdfConversionError(node_name=self.name, message="未找到 mineru 可执行文件，请确认主 venv 已安装 mineru")
        cmd = [
            mineru_exe,
            "-p",
            str(import_file_path_obj),
            "-o",
            str(file_dir_obj),
        ]
        # mineru 3.x 默认 backend 是 hybrid-engine（需 VLM 模型 + Linux 推理引擎，Windows 本地跑不了），
        # 默认改用 pipeline 传统模型；如需切回可设环境变量 MINERU_BACKEND
        backend = os.getenv("MINERU_BACKEND", "pipeline")
        if backend:
            cmd += ["--backend", backend]

        # 2. 利用子进程执行命令
        # 模型下载源：HF 直连不通，默认 modelscope（可用环境变量 MINERU_MODEL_SOURCE 覆盖）
        child_env = {**os.environ, "MINERU_MODEL_SOURCE": os.getenv("MINERU_MODEL_SOURCE", "modelscope")}
        # 定义开始时间
        start_time = time.time()
        # 子进程(执行命令产生日志) ------- 外部
        proc = subprocess.Popen(
            cmd,  # cmd命令
            stdout=subprocess.PIPE,  # 正常日志
            stderr=subprocess.STDOUT,  # 错误日志
            text=True,  # 输出二级制字符流 -> 字符串
            errors="replace",  # 特殊字符替换?,菱形
            encoding="utf-8",  # 设置编码
            bufsize=1,  # 实时输出 按行输出 遇到\n就产生出日志
            env=child_env,  # 传入模型源等环境变量
        )

        # 3. 打印日志
        for line in proc.stdout:
            self.logger.info(f"Mineru解析产生的日志:{line}")

        # 4. 主线程等待子进程做完 (状态0:成功, 反之失败)
        processed_result = proc.wait()

        end_time = time.time()

        if processed_result == 0:
            self.logger.info(f"Mineru解析成功,耗时:{end_time - start_time:.2f}s")
        else:
            self.logger.error(f"Mineru解析失败")

        return processed_result

    # 获取解析后的md文件路径
    def _get_md_path(self, import_file_path_obj, file_dir_obj: Path) -> str:
        """
        获取解析后的md文件路径
        :param: 输入文件路径, 输出文件目录
        :return: md文件路径
        """
        file_name = import_file_path_obj.stem

        # mineru 3.x 产物目录随 backend 不同：hybrid-engine -> hybrid_auto/，pipeline -> auto/
        # （旧版 2.x 为 pdfs/）；按优先级探测，兜底在结果目录内 glob 找同名 md
        out_root = file_dir_obj / file_name
        for sub in ("hybrid_auto", "auto", "pdfs"):
            candidate = out_root / sub / f"{file_name}.md"
            if candidate.exists():
                return str(candidate)

        matches = sorted(out_root.rglob(f"{file_name}.md"))
        if matches:
            return str(matches[0])

        raise PdfConversionError(
            node_name=self.name,
            message=f"mineru 解析产物中未找到 md 文件: {out_root}",
        )


# 测试
if __name__ == '__main__':
    # 打印日志
    setup_logging()

    # 1. 构建节点实例
    pdf_to_md_node = PdfToMdNode()

    # 2. 构建状态
    init_state = {
        "import_file_path": "",
        "file_dir": ""
    }
    state = create_default_state(init_state)

    # 3. 调用process方法
    result = pdf_to_md_node(state)

    # 4. 序列化(对象转成字符串)   反序列化(字符串转成对象)
    result_str = json.dumps(result, ensure_ascii=False)
    print(result_str)