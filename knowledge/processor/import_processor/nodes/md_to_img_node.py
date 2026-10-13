import base64
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List, Deque
from openai import OpenAI
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import *
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


@dataclass
class ImageContext:
    """
    图片上下文
    """
    head: str  # 上文标题内容
    pre_text: str  # 上文内容
    post_text: str  # 下文内容


@dataclass
class ImageInfo:
    """
    一张图片的完整信息
    图片的名称: 作为存储图片摘要字典容器的key
    图片的地址: 1. vlm要用     2. minio要用
    图片上下文信息: 作为vlm使用
    """
    imag_context: ImageContext  # 图片上下文
    name: str  # 图片名称(全名)
    path: str  # 图片路径


class _MdFileHandler:
    """
    主要职责:
    1. 读取md内容,md_path,图片目录
    2. 备份新的md_content (方便测试观察)
    """

    def __init__(self, logger, node_name: str):
        self.logger = logger
        self.node_name = node_name

    def validate_and_read_md(self, state) -> Tuple[str, Path, Path]:
        """
        核心逻辑:
        1. 读取md内容
        2. 读取md的路径
        3. 读取图片目录
        :param state: 上一个节点处理后的state
        :return: 返回md内容,md路径,图片目录的元组形式
        """

        # 1. 从state中获取md_path
        md_path = state.get("md_path", "")

        # 2. 非空判断
        if not md_path:
            raise StateFieldError(node_name=self.node_name, field_name="md_path", expected_type=str)

        # 3. 标准化
        md_path_obj = Path(md_path)

        # 4. 判断路径是否存在
        if not md_path_obj.exists():
            raise StateFieldError(node_name=self.node_name, field_name="md_path", expected_type=Path,
                                  message="md文件路径不存在")

        # 5. 读取md_context
        try:
            with open(md_path_obj, "r", encoding="utf-8") as f:
                md_context = f.read()
        except Exception as e:
            logging.error(f"{md_path_obj.name} 读取md文件失败")
            raise FileProcessingError(node_name=self.node_name, message="md文件读取失败")

        # 6. 获取图片目录
        img_dir_obj = md_path_obj.parent / "images"

        # 7. 返回
        return md_context, md_path_obj, img_dir_obj

    def backup(self, md_path_obj: Path, new_md_content: str) -> str:
        self.logger.info("【step_5】备份新文件")

        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )
        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise FileProcessingError(
                f"文件写入失败: {e}", node_name="md_img_node"
            )
        return str(new_file_path)


class _ImageScanner:
    """
    主要职责:
    1. 根据图片目录得到改目录下有效的图片文件
    2. 去到md文件中定位图片的位置
    3. 获取该图片在md中上下文内容 (给VLM模型提供上下文,让模型更加准确)
    4. 最终组装全部图片上下文内容 (list)
    """

    def __init__(self, logger):
        self.logger = logger

    def scan_imgs_dir(self, image_dir_obj: Path, md_content: str, image_extensions: set[str],
                      img_content_length: int) -> List[ImageInfo]:
        """
        核心逻辑:
        1. 扫描指定图片目录下的所有图片文件
        2. 遍历每一个图片文件去md文件中获取到位置 (上下文)
         - 2.1 上文包含的是标题加上文内容
         - 2.2 下文是下文内容
        3. 将每一个图片的上下文(ImageContext)放到最终封装ImageInfo类中
        # 4. 返回
        :param image_dir_obj: 图片目录
        :param md_content: md文本内容
        :param image_extensions: 允许的图片后缀
        :param img_content_length: 上下文长度
        :return: List[ImageInfo]
        """

        img_info_list = []

        # 1. 遍历图片目录
        for img_path in image_dir_obj.iterdir():
            # 1.1 过滤掉子目录
            if not img_path.is_file():
                self.logger.info(f"{img_path.name} 不是文件")
                continue

            # 1.2 过滤掉不合法图片后缀名的图片
            if not img_path.suffix in image_extensions:
                self.logger.info(f"{img_path.name} 无法识别的图片后缀名")
                continue

            # 1.3 找该图片的上下文
            ctx = self._find_context(img_path.name, md_content, img_content_length)

            if not ctx:
                self.logger.info(f"MD中未找到该图片应用{img_path.name}")
                continue

            # 1.4 封装ImageInfo对象并放到容器中
            img_info_list.append(
                ImageInfo(
                    imag_context=ctx,
                    name=img_path.name,
                    path=str(img_path)
                )
            )

        self.logger.info(f"扫描图片目录完成,共找到{len(img_info_list)}张图片")

        # 2. 返回
        return img_info_list

    def _find_context(self, img_name: str, md_content: str, img_content_length: int) -> ImageContext | None:
        """
        查找图片上下文
        :param img_name: 图片名称
        :param md_content: md文本
        :param img_content_length: 上下文长度
        :return: 找到了 -> 上下文对象   没找到 -> 空值
        """

        # 1. 预编译正则规则 (主要目的: 从md很多行中抓取到整个图片)
        pattern = re.compile(r"!\[.*?\]\(.*" + re.escape(img_name) + r".*?\)")

        # 2. 按行切割md_context
        md_lines = md_content.split("\n")

        # 3. 遍历每一行以及对应行索引
        for i, line in enumerate(md_lines):
            # 3.1 当前行不是当前图片
            if not pattern.search(line):
                continue

            # 3.2 当前行包含当前图片
            # 上文
            # 上文标题的索引作为起始索引
            head, prev_index = self._find_heading_up(md_lines, i)
            pre_lines = md_lines[prev_index + 1:i]
            pre_context = self._extract_limited_context(pre_lines, img_content_length, direction="front")

            # 下文
            # 下文标题的索引作为结束索引
            next_index = self._find_heading_down(md_lines, i)
            next_lines = md_lines[i + 1:next_index]
            post_context = self._extract_limited_context(next_lines, img_content_length, direction="back")

            return ImageContext(
                head=head,
                pre_text=pre_context,
                post_text=post_context,
            )
        return None

    def _find_heading_up(self, md_lines: List[str], from_idx: int) -> Tuple[str, int]:
        """
        获取当前图片上文的标题内容以及索引
        :param md_lines: 整个md内容
        :param from_idx: 图片索引
        :return: 当前图片最近的上文标题内容 + 索引
        """
        for i in range(from_idx - 1, -1, -1):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return md_lines[i], i
        return "", -1

    def _find_heading_down(self, md_lines: List[str], from_idx: int) -> int:
        """
        获取当前图片下文的标题索引
        :param md_lines: 整个md内容
        :param from_idx: 图片索引
        :return: 当前图片最近的下文标题索引
        """
        for i in range(from_idx + 1, len(md_lines)):
            if re.match(r"^#{1,6}\s+", md_lines[i]):
                return i
        return len(md_lines) + 1

    def _extract_limited_context(self, lines: List[str], img_content_length: int, direction: str) -> str:
        """
        根据上下文长度截取上下文 (按段落(\n )截取或图片上/下一段落)
        :param lines: md全部内容
        :param img_content_length: 上下文长度
        :param direction: 截取方向
        :return: 截取内容
        """
        current_paragraph = []
        paragraphs = []
        # 1. 遍历截取的行内容
        for line in lines:
            # 1.1 定义自然的段落规则
            is_blank_line = not line.strip()
            # 1.2 定义图片段落规则
            is_ather_image = re.match(r"!\[.*?\]\(.*?\)$", line.strip())
            # 1.3 当前行是空行或其他图片行
            if is_blank_line or is_ather_image:
                if current_paragraph:
                    paragraphs.append("\n".join(current_paragraph))
                    current_paragraph = []
                continue
            # 1.4 当前行不是空行或图片行,把当前行添加临时容器中
            current_paragraph.append(line)

        # 2. 处理最后的行
        if current_paragraph:
            paragraphs.append("\n".join(current_paragraph))

        # 反转
        if direction == "front":
            paragraphs.reverse()

        # 3. 遍历段落列表 (判断段落长度,以及最终选择留下哪些段落)
        total = 0
        selected = []  # 最终收集到的段落
        for paragraph in paragraphs:
            if total + len(paragraph) > img_content_length and selected:
                break
            selected = selected + [paragraph]
            total += len(paragraph)

        # 4. 将段落列表中的段落转成一个字符串
        # 反转回来
        if direction == "front":
            selected.reverse()
        return "\n\n".join(selected)


class _VLMSummarizer:
    """
    最主要职责:
    1. 主要根据每一张图片信息以及每一张图片的上下文信息生成图片摘要
    """

    def __init__(self, logger, requests_per_minute: int):
        self.logger = logger
        self.requests_per_minute = requests_per_minute

    def summary_all(self, document_name: str, image_info_list: List[ImageInfo], vl_model: str) -> Dict[str, str]:
        """
        主要职责: 为所有图片生成摘要
        :param document_name: 文档名称
        :param image_info_list: 图片信息列表
        :param vl_model: 模型名称
        :return: 图片摘要
        """
        summaries = {}
        request_timestamps: Deque[float] = deque()

        # 1. 获取VLM客户端
        try:
            vlm_client = AIClients.get_vlm_client()
        except Exception as e:
            # 兼容
            for image_info in image_info_list:
                summaries[image_info.name] = "暂无图片摘要"
            return summaries

        # 2. 调用vlm 为每一张图片生成摘要
        for image_info in image_info_list:
            # 测试一下
            self._enforce_rate_limit(request_timestamps, self.requests_per_minute)
            summaries[image_info.name] = self._summary_one(image_info, vlm_client, vl_model, document_name)

        self.logger.info(f"生成{len(summaries)} 个图片摘要")
        return summaries

    def _summary_one(self, image_info: ImageInfo, vlm_client: OpenAI, vl_model: str, document_name: str):
        """
        为一张图片生成摘要
        :param image_info: 图片信息
        :param vlm_client: VLM客户端
        :param vl_model: 模型名称
        :return: 图片摘要
        """

        # 1. 构建VLM需要的上下文
        parts = [p for p in
                 (image_info.imag_context.head, image_info.imag_context.pre_text, image_info.imag_context.post_text)
                 if p]

        # 2. 构建最终的上下文
        final_context = "\n".join(parts) if parts else "暂无上下文"

        # 3. 根据图片地址获取到图片的内容  base64 -> 解码utf8 -> 字符串
        try:
            with open(image_info.path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            self.logger.error(f"图片{image_info.name} 获取内容失败: {e}")
            return "暂无图片摘要"

        # 4. 利用vlm客户端调用VLM模型
        try:
            resp = vlm_client.chat.completions.create(
                model=vl_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"任务：为Markdown文档中的图片生成一个简短的中文标题。\n"
                                f"背景信息：\n"
                                f"  1. 所属文档标题：\"{document_name}\"\n"
                                f"  2. 图片上下文：{final_context}\n"
                                f"请结合图片内容和上述上下文信息，"
                                f"用中文简要总结这张图片的内容，"
                                f"生成一个精准的中文标题摘要（不要包含图片二字）。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_data}"
                            },
                        },
                    ],
                }],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"图片摘要生成失败 {image_info.path}: {e}")
            return "暂无图片描述"

    def _enforce_rate_limit(
            self, timestamps: Deque[float],
            max_requests: int,
            window: int = 60,
    ):
        now = time.time()
        while timestamps and now - timestamps[0] >= window:
            timestamps.popleft()

        if len(timestamps) >= max_requests:
            sleep_dur = window - (now - timestamps[0])
            if sleep_dur > 0:
                self.logger.info(
                    f"达到速率限制，暂停 {sleep_dur:.2f} 秒..."
                )
                time.sleep(sleep_dur)
            now = time.time()
            while timestamps and now - timestamps[0] >= window:
                timestamps.popleft()

        timestamps.append(now)


class _ImageUploader:
    """
    主要职责:
    1. 上传图片到minio,得到远程可用得到图片的url地址
    2. 替换md中的摘要和图片地址
    """

    def __init__(self, logger):
        self.logger = logger

    def upload_and_replace(self, md_name: str, md_content: str, image_info_list: List[ImageInfo],
                           summaries: Dict[str, str], minio_base_url: str, minio_bucket: str) -> str:
        """
        将文件图片上传到Minio,并且更新md中的图片地址以及摘要
        :param md_name: md文件名称
        :param md_content: md文件内容
        :param image_info_list: 图片信息列表
        :param summaries: 图片摘要
        :param minio_base_url: minio的url
        :param minio_bucket: minio的bucket
        :return: 更新后md文件内容
        """

        # 1. 上传
        remote_urls = self._upload_all(md_name, image_info_list, minio_base_url, minio_bucket)

        # 2. 更新
        md_content = self._update_md(md_content, remote_urls, summaries)

        # 3. 返回
        return md_content

    def _upload_all(self, md_name: str, image_info_list: List[ImageInfo],
                    minio_base_url: str, minio_bucket: str) -> Dict[str, str]:
        """
        上传所有图片
        :param md_name: md文件名称
        :param image_info_list: 图片信息列表
        :param minio_base_url: minio的url
        :param minio_bucket: minio的bucket
        :return: 封装图片的远程url到字典
        """

        remote_urls = {}

        # 1. 得到Minio远程客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except Exception as e:
            self.logger.error(f"获取MinIO客户端失败: {e}")
            for image_info in image_info_list:
                remote_urls[image_info.name] = image_info.path
            return remote_urls

        # 2. 上传每一个
        for image_info in image_info_list:
            # 2.1 设置上传的文件夹和图片文件名
            object_name = f"{md_name}/{image_info.name}"
            try:
                # 2.2 上传图片
                minio_client.fput_object(bucket_name=minio_bucket, object_name=object_name, file_path=image_info.path)

                # 2.3 自己拼装远程图片地址
                remote_urls[image_info.name] = f"{minio_base_url}/{minio_bucket}/{object_name}"
                self.logger.info(f"图片 {image_info.name} 上传成功, 远程地址: {remote_urls[image_info.name]}")
            except Exception as e:
                self.logger.error(f"上传图片 {image_info.name} 失败: {e}")
                remote_urls[image_info.name] = image_info.path

        # 返回图片远程地址
        self.logger.info(f"图片上传完成, 共 {len(remote_urls)} 张图片")
        return remote_urls

    def _update_md(self, md_content: str, remote_urls: str, summaries: Dict[str, str]) -> str:
        """
        更新md文件中的图片描述和图片地址
        :param md_content: md文件内容
        :param remote_urls: 图片远程地址
        :param summaries: 图片摘要
        :return: 更新后的md文件内容
        """

        # 利用正则查找
        pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        def replacer(match: re.Match) -> str:
            """
            ![摘要]()远程url地址
            :param match: 匹配结果
            :return: 替换后的字符串
            """
            for img_name,img_summary in summaries.items():
                img_path = match.group(2)
                md_img_name = Path(img_path).name

                if md_img_name == img_name:
                    return f"![{img_summary}]({remote_urls[img_name]})"
            return match.group(0)

        return pattern.sub(replacer, md_content)


class MarkDownToImgNode(BaseNode):
    """
    主要逻辑:
    1. 得到四个类的实例对象
    2. 分别调用四个实例对象的方法

    """

    name = "markdown_to_img_node"

    # 实例对象初始化
    def __init__(self):
        super().__init__()  # 显示调用父类的初始化构造方法
        self._md_file_handler = _MdFileHandler(self.logger, node_name=self.name)
        self._image_scanner = _ImageScanner(self.logger)
        self._vlm_summarizer = _VLMSummarizer(self.logger, self.config.requests_per_minute)
        self._image_uploader = _ImageUploader(self.logger)

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        入口逻辑
        :param state:
        :return:
        """

        config = self.config

        # 1. 操作md_file_handler
        self.log_step("step1", "读取md内容,路径以及图片目录")

        md_content, md_path_obj, image_dir_obj = self._md_file_handler.validate_and_read_md(state)
        # 1.1 判断图片目录是否存在
        if not image_dir_obj.exists():
            state["md_content"] = md_content
            return state

        # 2. 操作_image_scanner
        self.log_step("step2", "扫描图片目录,获取上下文和图片名")

        image_info_list: List[ImageInfo] = self._image_scanner.scan_imgs_dir(image_dir_obj, md_content,
                                                                             config.image_extensions,
                                                                             config.img_content_length)

        # 3. 操作_vlm_summarizer
        self.log_step("step3", "生成图片摘要")

        summaries: Dict[str, str] = self._vlm_summarizer.summary_all(md_path_obj.stem, image_info_list, config.vl_model)

        # 4. 操作_image_uploader
        self.log_step("step4", "上传图片到minio,替换图片地址和摘要")

        new_md_content = self._image_uploader.upload_and_replace(md_path_obj.stem, md_content, image_info_list,
                                                                 summaries, config.get_minio_base_url(),
                                                                 config.minio_bucket)

        # 5. 备份调试
        self._md_file_handler.backup(md_path_obj, new_md_content)

        # 6. 更新md文本内容到状态
        state["md_content"] = new_md_content

        return state


if __name__ == '__main__':
    setup_logging()
    md_img_node = MarkDownToImgNode()
    init_state = {
        "md_path": r""
    }
    md_img_node.process(init_state)
