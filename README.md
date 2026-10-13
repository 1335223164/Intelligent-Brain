# 智慧大脑 · Intelligent-Brain

> 面向设备/产品说明书的 **RAG 智能知识库问答系统**：文档一键导入即自动完成「PDF 解析 → 切片 → 向量化 → 商品名注册 → 知识图谱抽取」，查询侧以「商品名前置闸 + 混合检索 + HyDE + 联网兜底 + RRF 融合 + Rerank + 图谱反查 + 落地作答」的 LangGraph 编排给出可溯源答案，并内置基于 **RAGAS** 的四维自动化评测。

---

## 一、项目介绍

### 1.1 解决什么问题
电商/售前/运维场景下，产品手册、规格说明、FAQ 分散且篇幅长，人工检索慢、通用大模型又缺乏这些私有知识、易幻觉。本系统把文档沉淀为**向量 + 知识图谱**双通道知识，提供**准确、可溯源、可评测**的问答能力。

### 1.2 核心特性
- **双链路 LangGraph 编排**：导入链路（`import_processor`）与查询链路（`query_processor`）均为可插拔节点图。
- **商品名前置闸**：查询先抽取商品名并与注册表（Milvus `kb_item_names`）向量对齐，只有确认命中具体商品才放行检索，避免"张冠李戴"式误答。
- **混合检索 + HyDE + 联网兜底**：稠密+稀疏向量、假设文档嵌入（HyDE）、MCP Web 搜索多路召回，RRF 融合后 BGE-Reranker 精排动态截断。
- **知识图谱增强**：UIE 微服务做实体/关系抽取入 Neo4j，查询侧图谱反查参与融合；UIE 未启动时自动降级远程 LLM 抽取。
- **图片可溯源**：文档图片经视觉模型转述并存 MinIO，答案可回指原文档/切片。
- **内置 RAGAS 评测**：两步式解耦（出题+跑链路 → 打分），支持**冻结评测集做配对 A/B** 与**重复打分看裁判方差**。

### 1.3 技术栈总览
| 层次 | 选型 |
|---|---|
| Web / API | FastAPI + Uvicorn（查询 `:8000`、导入 `:18000`） |
| 编排 | LangGraph（导入图 / 查询图） |
| 向量库 | Milvus 2.5（切片 `kb_chunks`、商品名 `kb_item_names`、实体名 `kb_entity_names`） |
| 图数据库 | Neo4j（知识图谱） |
| 文档/会话库 | MongoDB |
| 对象存储 | MinIO（图片、原始文件） |
| 嵌入 / 精排 | BGE-M3（1024 维，COSINE + 稀疏）、BGE-Reranker-Large |
| LLM | 硅基流动（OpenAI 兼容），默认 `Qwen/Qwen3-8B`，视觉 `Qwen3-VL-8B-Instruct` |
| 评测裁判 | 与链路解耦的 `deepseek-ai/DeepSeek-V3`（temp=0） |
| 评测框架 | RAGAS `0.2.15` |
| 图谱抽取 | PaddleNLP UIE（独立微服务 `:8001`，规避 protobuf 冲突） |
| PDF 解析 | MinerU |

---

## 二、评估结果

> 评测口径：文档驱动自动生成问答四元组（`问题 / 标准答案 / 召回上下文 / 模型答案`），由 RAGAS 四维打分；裁判模型与链路模型解耦（`deepseek-ai/DeepSeek-V3`、温度 0），重复打分标准差约 ±0.005，故分数差异可视为真实信号。

### 2.1 优化前 → 优化后

| 指标 | 含义 |     优化前 | 优化后（当前） | 提升 |
|---|---|--------:|---:|---:|
| **faithfulness（忠诚度）** | 答案对检索上下文的忠实度，越低越易幻觉 |   0.537 | **0.771** | **+0.234** |
| **answer_relevancy（答案相关性）** | 答案与问题的相关程度 |   0.610 | **0.747** | +0.137 |
| **context_precision（上下文精确率）** | 检索上下文中相关内容的占比与排序 |   0.677 | **0.800** | +0.123 |
| **context_recall（上下文召回率）** | 标准答案所需信息被检索覆盖的比例 |   0.718 | **0.859** | +0.141 |

### 2.2 做了哪些优化（对应上面的提升）
1. **提示词优化**：商品名抽取约束"逐字保留、不重排词序、不加'系列/型号'后缀"，并补充"品牌/型号即便作定语或藏长短语里也原样抽出、不漏抽"；修正阈值误拦、以及句式隐蔽时 `item_names=[]` 的空抽取。
2. **商品名归一化**： 商品名识别时的"逐字相等"放宽为"归一化相等"（小写/去分隔符/去尾部虚词），并新增"注册名逐字写在句首即命中"的前导命中。这两项直接拆掉了商品名前置闸的误拦截，是 faith / recall 提升的主因。
3. **文档表格切分**：PDF 解析阶段对表格做转译/降维、切片按语义边界切分，使规格/参数类结构化内容更易被检索命中（主要贡献在 `context_precision` / `context_recall`）。

> ℹ️ 提示：以上为**单篇闭卷、LLM 生成标准答案**的自动化评测，参考与出题同源，绝对分数宜看**趋势与配对对比**而非当作线上真实满意度。复现方式见 §4.3。

---

## 三、快速开始

### 环境要求
- Windows / Linux，Python **3.12**
- Docker + Docker Compose（跑中间件）
- 建议 NVIDIA GPU（BGE-M3 / Reranker / UIE 走 `cuda:0`；无 GPU 可改 `*_DEVICE=cpu`，速度下降）
- 硅基流动 API Key（`sk-` 开头）；可选阿里云百炼 DashScope Key（联网搜索 MCP）

### 3.1 安装依赖
> ⚠️ **安装顺序关键**（避免 CPU 版 torch 误装、protobuf 冲突）：
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -U pip
# 1) 先按官方索引装 GPU 版 torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
# 2) 文档解析
pip install "mineru[all]"
# 3) 嵌入/精排
pip install FlagEmbedding
# 4) 其余依赖（含 ragas==0.2.15 锁定版本）
pip install -r requirements.txt
```
UIE 微服务使用**独立环境**（与主项目 protobuf 冲突）：
```powershell
cd uie_service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
```

### 3.2 配置环境变量
```powershell
Copy-Item knowledge\.env.example knowledge\.env
```
编辑 `knowledge/.env`，至少填：
- `OPENAI_API_KEY`（硅基流动）、`OPENAI_API_BASE`
- `MILVUS_URL`、`MONGO_URL`、`MINIO_*`、`NEO4J_*`（**用户名/密码须与 `docker-compose.yml` 一致**）
- `BGE_M3_PATH` / `BGE_RERANKER_LARGE` / `UIE_MODEL_PATH`（本地模型路径）
- `EVAL_LLM_MODEL=deepseek-ai/DeepSeek-V3`（评测裁判，与链路模型解耦）

### 3.3 启动中间件
```powershell
docker compose up -d          # 在项目根执行
docker compose ps             # 全部 healthy 再继续
```
| 服务 | 端口 |
|---|---|
| Milvus standalone | 19530（SDK）/ 9091（健康检查） |
| Attu（Milvus GUI） | 7000 |
| MinIO | 9000（API）/ 9001（控制台） |
| Neo4j | 7474（HTTP）/ 7687（Bolt） |
| MongoDB | 27017 |
| etcd | 2379（内部） |

### 3.4 启动应用服务
```powershell
# 导入服务（:18000）
python -m knowledge.api.import_router
# 查询服务（:8000）
python -m knowledge.api.query_router
```
可选：启动 UIE 图谱抽取微服务（`:8001`，用于知识图谱；不启动则查询链路自动降级）：
```powershell
cd uie_service
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001
# 健康检查
curl.exe http://127.0.0.1:8001/health   # 期望 status:ok, ready:true
```

### 3.5 访问前端
- 导入页：http://localhost:18000/front/import.html
- 问答页：http://localhost:8000/front/chat.html

---

## 四、使用指南

### 4.1 导入文档
1. 打开导入页，上传 PDF/文档，可选择是否构建知识图谱（`is_kg_enabled`，默认开启）。
2. 后台链路：`entry → pdf_to_md → md_to_img → document_split → embedding_chunks → import_milvus → item_name_recognition → kg_graph`。
3. 轮询任务状态：`GET http://localhost:18000/status/{task_id}`。

对应 API：`POST /upload`（multipart：`file`、`is_kg_enabled`）。

### 4.2 提问
- 问答页输入问题即可；支持流式（SSE）返回。
- 查询链路：`item_name_confirmed（前置闸）→ 混合检索 / HyDE / 联网兜底 → rrf_merge → reranker →（kg_query 图谱反查）→ answer_output`。
- API：
  - `POST http://localhost:8000/query`，body：`{"query": "...", "session_id": "...", "is_stream": true}`
  - 流式订阅：`GET http://localhost:8000/stream/{task_id}`
  - 历史：`GET /history/{session_id}`、`DELETE /history/{session_id}`

### 4.3 运行 RAGAS 评测
评测分**两步**、可自由编排（脚本：`knowledge/evaluation/run_ragas_eval.py`）。

**① 出题 + 跑链路 → 产出四元组，并冻结成"尺子"（推荐做一次）**
```powershell
python knowledge/evaluation/run_ragas_eval.py expand --save-dataset --out temp\eval_records.json
```
- `--per-doc` 每篇问答数（默认 15）；`--no-chain` 仅出题不跑链路。
- 冻结集只存 `问题/标准答案/文档`，落在 `knowledge/evaluation/datasets/qa_<指纹>_<条数>.json`。

**② 改动链路后复用同一集做配对 A/B（只重跑链路）**
```powershell
python knowledge/evaluation/run_ragas_eval.py expand --dataset knowledge\evaluation\datasets\qa_xxx.json --out temp\eval_records_ab.json
```

**③ 打分（可 `--repeat` 看裁判方差）**
```powershell
python knowledge/evaluation/run_ragas_eval.py evaluate --input temp\eval_records.json
python knowledge/evaluation/run_ragas_eval.py evaluate --input temp\eval_records.json --metrics faithfulness --repeat 3
```
- 指标：`faithfulness / answer_relevancy / context_precision / context_recall`。
- 报告写入 `knowledge/evaluation/reports/`（JSON + Markdown）。

---

## 五、目录结构
```
Intelligent-Brain/
├─ knowledge/
│  ├─ api/                # query_router(:8000) / import_router(:18000)
│  ├─ core/              # 依赖注入、路径
│  ├─ processor/
│  │  ├─ import_processor/  # 导入 LangGraph（nodes/）
│  │  └─ query_processor/   # 查询 LangGraph（nodes/）
│  ├─ evaluation/        # RAGAS：config/expansion/evaluator/CLI
│  │   ├─ datasets/      # 冻结评测集（不入库）
│  │   └─ reports/       # 评测报告（不入库）
│  ├─ prompt/templates/  # 各节点提示词（import/query/eval）
│  ├─ schema/ service/   # 请求响应模型、业务服务
│  ├─ utils/client/      # Milvus/Neo4j/Mongo/MinIO/LLM/UIE 客户端与工具
│  ├─ front/             # chat.html / import.html
│  └─ .env.example       # 环境变量模板（.env 不入库）
├─ uie_service/          # UIE 图谱抽取独立微服务(:8001，自带 venv)
├─ docs/                 # 原始文档（PDF/md，不入库，仅保留空目录）
├─ docker-compose.yml    # Milvus/Neo4j/Mongo/MinIO/Attu
└─ requirements.txt
```

---

## 六、注意事项
- `knowledge/.env`、`docs/*`、评测数据**均不纳入版本库**，仅保留空目录占位；克隆后需自行准备文档与 `.env`。
- 前端跨域已放开（`allow_origins=["*"]`），生产环境请按需收紧。
- UIE 与主项目 protobuf 版本冲突，务必用 `uie_service/` 的独立虚拟环境启动。
- 无 GPU 时把 `BGE_DEVICE / BGE_RERANKER_DEVICE / UIE` 相关改为 `cpu` 并接受速度下降。
