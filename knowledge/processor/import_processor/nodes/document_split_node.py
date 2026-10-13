import json
import os
import re
from typing import Tuple, List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.utils.markdown_util import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    name = "document_split_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        文档切分核心逻辑入口
        (源文档打散) --- 组装
        :param state: 状态对象
        :return:
        """

        # 1. 参数校验（配置用 BaseNode 注入的 self.config 实例，非模块）
        md_content, file_title, max_content_length, min_content_length = self._validate_state(state, self.config)

        # 2. 切分 (一级策略: 根据md文档内标题切分)
        sections: List[Dict[str, Any]] = self._split_by_headings(md_content, file_title)

        # 3. 二次切分或者合并
        final_sections = self._split_and_merge(sections, max_content_length, min_content_length)

        # 4. 组装成chunk对象
        final_chunks = self._assemble_chunks(final_sections)

        # 5. 备份
        self._back_up(final_chunks, state)

        # 6. 更新state (chunks)
        state["chunks"] = final_chunks

        # 7. 返回
        return state

    def _validate_state(self, state: ImportGraphState, config) -> Tuple[str, str, int, int]:
        self.log_step("step1", "切分文档的参数校验以及获取...")

        # 1. 获取md_content
        md_content = state.get('md_content')

        # 2. 统一换行符
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 3. 获取文件标题
        file_title = state.get('file_title')

        # 4. 校验最大最小值
        if config.max_content_length <= 0 or config.min_content_length <= 0 \
                or config.max_content_length <= config.min_content_length:
            raise ValueError(f"切片长度参数校验失败")

        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content, file_title) -> List[Dict[str, Any]]:
        """
        根据标题做切分 1-6级都有可能切分
        :param md_content: md内容
        :param file_title: 文件标题
        :return: 切分后的结果,封装每个标题以下内容的列表
        """

        in_fence = False  # 是否在代码块内
        body_liens = []  # 存储所有行
        sections = []  # 最终收集到的章节对象
        current_title = ""  # 当前标题
        hierarchy = [""] * 7  # （数组）存储所有标题内容（作为section的父标题使用） 标题层级追踪数组
        current_level = 0

        def _flush() -> List[Dict[str, Any]]:
            """
            打包section
            {
            "body": "收集到的所有行"
            “title”:"当前内容的标题"
            "parent_title":当前内容的父标题（最麻烦）
            "file_title":文档标题（最简单）
            }
            Returns:
            如果current_title没有，body有 能进入打包成section,【也有意义】
            如果current_title有,body没有。也打包成section:在合并阶段可以保留上【可选：建议留下来】在后续合并阶段没有任何影响
            如果current_title有，body也有 能进入打包成section【一定留】
            如果current_title没有 body也没有 不会进入（不能打包）
            """

            # 1. 处理内容行
            body = "\n".join(body_liens)

            if current_title or body:
                parent_title = ""
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:  # 找父标题的时候 排除某一个位置的空值
                        parent_title = hierarchy[i]  # 读取操作
                        break

                if not parent_title:
                    parent_title = current_title if current_title else file_title

                sections.append({
                    "body": body,
                    "title": current_title if current_title else file_title,  # 内容标题
                    "parent_title": parent_title,  # 内容父标题
                    "file_title": file_title,
                })
            body_liens.clear()  # 新增，清空当前缓存行

        # 1. 根据\n切分md_content
        md_lines = md_content.split("\n")

        # 2. 定义正则（正则的规则是从MD中找标题#{1,6}）():捕获组:产生三个group(0) group(1):#(1)#(6) group(2)标题的内容
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        # 3. 遍历切分后md_lines
        for md_line in md_lines:

            # 3.1 检测代码块边界（``` 或 ~~~）代码块要留下来
            if md_line.strip().startswith("```") or md_line.strip().startswith("~~~"):
                in_fence = not in_fence  # 不要用固定true  or false

            # 3.2 判读是否要走正则
            match = heading_re.match(md_line) if not in_fence else None

            # 3.3 判断math 是否有
            # 代表匹配到了标题而且一定是非代码块中的# 标题
            if match:

                # 将 body_liens中收集到的行封装到section对象
                _flush()

                current_title = md_line  # 当前标题
                level = len(match.group(1))  # 当前标题的层级（# {1,6}）
                current_level = level
                hierarchy[level] = current_title  # 写入操作

                for i in range(level + 1, 7):
                    hierarchy[i] = ""  # 下面的清空
                # 没有匹配到标题[普通行] 或者是代码块（加入）
            else:
                body_liens.append(md_line)

        _flush()
        return sections

    def _split_and_merge(self, sections: List[Dict[str, Any]], max_content_length: int, min_content_length: int) -> \
    List[Dict[str, Any]]:
        """
        切分较大的章节以及合并较小的章节
        :param sections: 一级切分后的章节对象
        :param max_content_length: 最大内容长度   (content:title +\n\n+body)
        :param min_content_length: 最小内容长度
        :return: 二次切分后的章节对象
        """
        current_sections = []

        # 1. 切分
        for section in sections:
            current_section = self._split_long_section(section, max_content_length)
            current_sections.extend(current_section)

        # 2. 合并
        final_sections = self._merger_short_section(current_sections, min_content_length, max_content_length)

        return final_sections

    def _split_long_section(self, section: Dict[str, Any], max_content_length: int) -> List[Dict[str, Any]]:
        """
        拆分过长的章节
        :param section: 一级切分后的章节对象
        :param max_content_length: 最大内容长度
        :return: 二次切分后的章节对象
        """

        # 1. 获取section对象的属性
        body = section.get('body')  # 行内容
        title = section.get('title')  # 标题
        parent_title = section.get('parent_title')  # 父标题
        file_title = section.get('file_title')  # 文档标题

        if len(title) > 80:
            title = title[:80]  # 防御性编程：title 本身就超长的极端情况

        if "<table>" in body:
            self.logger.info("检查到section中有表格")
            body = MarkdownTableLinearizer.process(body)
            section["body"] = body

        # 4.5 Q&A 型表格（多行「【问题】：解答」）：每一行是一个自包含的问答单元，
        # 拆成独立原子切片（标题带行主语），提升细粒度召回与精排；
        # 否则整表塞进一个切片会导致问某一行时命中相邻行（如问“Cable指示灯”答“所有灯不亮”）。
        exploded = self._explode_qa_table(section)
        if exploded:
            return exploded

        # 2. 获取标题前缀
        title_prefix = f"{title}\n\n"

        # 3. 获取总长度【标题(前缀)+body】
        total_length = len(title_prefix) + len(body)

        # 4. 判断总长度是否超过阈值
        if total_length <= max_content_length:
            return [section]

        # 5. 能切分的内容长度计算出来（body）
        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]  # 防御性编程：title 本身就超长的极端情况

        # 6. 切分(6.1 用谁切:LangChain的切分器:递归切分器 6. 去切谁:body)
        # 6.1 切分器对象  # 1. chunk_size:chunk块的大小  2.chunk_overlap块与块之间的重叠的字符数  3. separators：分割符 【"\n\n",'\n',' ',''】
        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "？", "！", "；", ".", "?", "!", ';', " ", ""],
            keep_separator=True)

        # 6.2 切分器对象切分
        sections = text_spliter.split_text(body)

        # 6.3 判断
        if len(sections) == 1:
            return [section]

        # 6.4 遍历
        sub_sections = []
        for index, section in enumerate(sections):
            sub_sections.append({
                "body": section,
                "title": f"{title}_{index + 1}",
                "parent_title": parent_title,
                "file_title": file_title
            })

        # 7. 返回
        return sub_sections

    def _explode_qa_table(self, section: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将 Q&A 型表格按行拆成独立原子子章节（保留夹杂的散文段）。
    
        适用场景：一个 section 内混有「散文小标题/表名行」与「- 【问题】：解答」表格行。
        策略：扫描内容行，连续的 bullet 行（≥ MIN_ROWS 条）逐条原子化（标题带行主语）；
        非 bullet 的散文/表名行归为普通段落（保留，后续走正常长度切分）。
        若整个 section 不足 MIN_ROWS 条 bullet，则不拆（返回空列表），避免把普通短列表拆碎。
        行格式与 MarkdownTableLinearizer 输出对齐。
        """
        MIN_ROWS = 3
        body = section.get("body") or ""
        title = section.get("title") or ""
        parent_title = section.get("parent_title")
        file_title = section.get("file_title")
    
        # 按换行（含空行）逐行拆开：表格行之间为单 \n，散文/表名与表格间可能为 \n\n，统一按一个或多个换行分割
        lines = [ln.strip() for ln in re.split(r"\n+", body) if ln.strip()]
        # 过滤表格转换后可能残留的 HTML 标签裸行（<table>/<tr>/<td>/</table> 等）噪声
        content_lines = []
        for ln in lines:
            if ln.startswith("<") or ln.startswith("</") or ln in (">", "/>"):
                continue
            content_lines.append(ln)
    
        total_bullets = sum(1 for ln in content_lines if ln.startswith("- 【"))
        if total_bullets < MIN_ROWS:
            return []
    
        base_title = self._clean_heading_title(title)
    
        def _row_title(bullet_line: str) -> str:
            subject = bullet_line.lstrip("- ").strip()
            if subject.startswith("【") and "】" in subject:
                subject = subject[1:subject.index("】")].strip()
            return f"{base_title}-{subject}"[:80]
    
        # 按连续 bullet 段 / 散文段分段
        result: List[Dict[str, Any]] = []
        prose_buf: List[str] = []
    
        def _flush_prose():
            if prose_buf:
                result.append({
                    "body": "\n\n".join(prose_buf),
                    "title": title,
                    "parent_title": parent_title,
                    "file_title": file_title,
                })
                prose_buf.clear()
    
        for ln in content_lines:
            if ln.startswith("- 【"):
                # 连续 bullet：逐条原子化（散文段先落盘）
                _flush_prose()
                result.append({
                    "body": ln,  # 保留原始整行（含“【主语】：”前缀，主语在标题+正文各出现一次，利检索）
                    "title": _row_title(ln),
                    "parent_title": parent_title,
                    "file_title": file_title,
                    "atomic": True,  # 原子切片，禁止在合并阶段被粘回大块
                })
            else:
                prose_buf.append(ln)
        _flush_prose()
    
        atomic_count = sum(1 for s in result if s.get("atomic"))
        if atomic_count < MIN_ROWS:
            return []  # 拆不出足够原子行，保持原样
        self.logger.info(f"Q&A 表格 '{base_title}' 按行拆分为 {atomic_count} 个原子切片")
        return result
    
    @staticmethod
    def _clean_heading_title(title: str) -> str:
        """去除标题前导的 Markdown # 与空白。"""
        return (title or "").lstrip("#").strip()

    def _merger_short_section(self, current_sections: List[Dict[str, Any]], min_content_length: int,
                              max_content_length):
        """
        合并短的章节：
        短章节来源：
        来源1：原本根据一级标题切分之后可能内容就很短
        来源2：（LangChain递归切分器）二次切分之后可能有很短的内容
        合并策略：
        条件1：section很短比最小阈值还小
        条件2：同源（父标题相同）
        :param current_sections: 二次切分后的所有section对象
        :param min_content_length: 每一个section的内容最小长度
        :return: 合并之后的section对象 贪心累加算法
        """

        current_section = current_sections[0]
        final_sections = []

        # 2. 遍历合并
        for next_section in current_sections[1:]:
            same_parent = (current_section['parent_title'] == next_section['parent_title'])
            # 原子切片（Q&A 表格拆出的单行问答）禁止被合并粘回大块：
            # 既不能做合并的起点（current 是原子），也不能被吸入相邻短段落（next 是原子）。
            if (same_parent and len(current_section.get('body')) < min_content_length
                    and not current_section.get('atomic') and not next_section.get('atomic')):
                # 防止合并后过长
                new_body = current_section.get('body').rstrip() + "\n\n" + next_section.get('body').lstrip()
                if len(new_body) > max_content_length:
                    # 合并后超长，禁止合并
                    final_sections.append(current_section)
                    current_section = next_section
                    continue

                # 合并 body
                current_section['body'] = (
                        current_section.get('body').rstrip() + "\n\n" + next_section.get('body').lstrip()
                )
                # 标题回退为父标题
                current_section['title'] = current_section['parent_title']
            else:
                # 封箱
                final_sections.append(current_section)
                current_section = next_section  # 更新指针
        # 最后一个封箱
        final_sections.append(current_section)
        return final_sections

    def _assemble_chunks(self, final_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        组装最后的chunks对象
        :param final_sections: 最终的section对象
        :return: chunks对象
        """

        final_chunks = []

        for section in final_sections:
            body = section.get('body')
            title = section.get('title')
            file_title = section.get('file_title')
            parent_title = section.get('parent_title')

            content = f"{title}\n\n{body}"

            final_chunks.append({
                "content": content,
                "title": title,
                "file_title": file_title,
                "parent_title": parent_title
            })

        self.logger.info(f"最终的chunks数量：{len(final_chunks)}")
        return final_chunks

    def _back_up(self, final_chunks: List[Dict[str, Any]], state: ImportGraphState):
        """将切分结果备份到 JSON 文件"""
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return
        try:
            os.makedirs(local_dir, exist_ok=True)  # 如果目录存在 不报错
            output_path = os.path.join(local_dir, "chunks.json")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_chunks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == '__main__':
    document_split_node = DocumentSplitNode()

    md_path = r""

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    init_state = {
        "md_content": md_content,
        "file_title": "万用表的使用",
        "file_dir": r""
    }
    document_split_node.process(init_state)
