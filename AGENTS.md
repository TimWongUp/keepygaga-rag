# Keepygaga RAG Agent Entry

## Scope and authority

- Keepygaga RAG 是 Keepygaga 产品家族中独立安装、运行与发布的 Knowledge/RAG sibling 产品；`keepygaga` sibling repo 只负责核心记忆。两者仅由 Agent Host 并列注册，不互相 import、打包或建立运行依赖。
- 原始 Markdown/TXT source 永远是真源；SQLite、FTS 与 LanceDB 是可重建派生层。当前行为由代码、schema、配置消费代码和测试裁决，运行状态由配置、Doctor、数据库、进程和日志现场裁决。
- `README.md` 面向首次使用者，不是 agent 默认项目真源。

## Repo-native context

- 讨论 source、chunk、generation、Retrieval Profile 等领域词时读 `CONTEXT.md`。
- 修改索引、授权、查询、存储、协调器或 Dashboard 稳定边界时读 `docs/architecture.md`。
- 修改验证、Doctor、Indexer 运行检查、失败分流或证据路由时读 `docs/operations.md`。
- 修改 chunk/metadata/identity/rebuild 语义时读 `docs/knowledge-chunks.md`。
- 追溯多 table 与未来 PDF/图片候选理由时读 `docs/design-history.md`；候选设计不自动成为开放任务。
- 追溯拆仓或短 MCP 名理由时读 `docs/adr/`。长期上下文只在本 repo 维护；Vault 旧项目上下文只作历史归档。

## Project constraints

- Python `3.12+`；优先使用 repo `.venv/`。
- 产品与发行包使用 Keepygaga RAG / `keepygaga-rag`，Python 包使用 `keepygaga_rag`，CLI 使用 `keepygaga-rag`。MCP 客户端配置约定使用注册 key `keepygaga_rag`，它不等同于 MCP `serverInfo.name`。
- 公开 raw MCP Tool 必须且只能是 `search`；客户端使用约定 key 时完整宿主名为 `mcp__keepygaga_rag__search`。Doctor、Indexer 和 Dashboard 不注册为 MCP Tool。
- 原文件始终是真源。SQLite 负责 registry、source、manifest、chunk、FTS、任务和 generation；LanceDB 只保存向量及必要对账字段。
- `agents-memory/`、`_context-backups/` 与开发依赖在 source root 和任意扫描深度硬排除；source roots 不得重叠，扫描不跟随 symlink。
- source/provider/model/范围首次绑定或变化必须重新取得在线数据传输授权；授权过期时索引、向量查询和 Rerank fail closed。
- 新 generation 的 SQLite/FTS/vector 全部就绪后才能切换 active generation；失败保留 last-good，pending/orphan 不可见。
- 只有独立 `keepygaga-rag indexer` 进程取得单实例锁后写 schema、队列和派生删除；Dashboard 与 MCP 查询进程只读或排队，不启动第二个扫描器。
- `search` 在 table 内执行 FTS/vector recall、RRF、Reranker 和单 source file 分块限额；结果只定位 source，不能替代原文件。
- `keepygaga-rag.toml`、`.env`、`.keepygaga/` 与 `.venv/` 是本机产物，不提交；密钥不得进入日志、Doctor、文档或测试 fixture。
- 普通验证不得调用真实全量重建、外部 Embedding/Rerank API、eval 或 benchmark。

## Commands and verification

- 安装：`uv sync --extra dashboard`
- 默认验收：`uv run python scripts/smoke_mcp_server.py`
- 测试：`uv run pytest -q`
- 静态检查：`uv run ruff check . && uv run pyright`
- 诊断：`uv run keepygaga-rag doctor --json`
- Dashboard：`uv run keepygaga-rag` 或 `uv run keepygaga-rag dashboard`
- Indexer：`uv run keepygaga-rag indexer`；只处理当前队列一次使用 `--once`。
- 必需检查无法运行时，报告命令、阻塞和剩余不确定性。

## Git and delivery

- 写入前运行 `git status --short`；存在 `HEAD` 时记录 `git rev-parse HEAD`。任务前资产不得为了 clean 自动 commit、stash、reset、restore、checkout、clean、删除或移动。
- 所有写入从最新集成 baseline 创建 `codex/<task-slug>` 短期分支，不直接修改 `main`。dirty `main` 的无关变化使用独立 worktree 隔离。
- 有 GitHub 远端时以 `origin/main` 为集成真源；缺少远端时以任务开始时未移动的本地 `main` 为临时 baseline，完成后不合回 `main`。
- coherent change 通过必要验证后，只 stage 本任务文件或 hunks，检查 staged diff，再按 `type(scope): 中文摘要` 创建 Conventional Commits Lite commit。
- PR 前同步最新主线并按 R0–R3 风险路由只读 Review。配置远端、push、创建或合并 PR、发布、改写历史均需仓库维护者明确授权。
- 交接时报告本任务 commit、验证和全部剩余 dirty 路径，并区分任务前、任务中未提交与任务外变化。

## Context change gate

- Run 不进入长期上下文；Artifact 留在代码、schema、配置和测试；Evidence 保留原始输出或可复现命令。
- 稳定领域词更新 `CONTEXT.md`；系统边界更新 `docs/architecture.md`；验证策略更新 `docs/operations.md`；chunk 稳定语义更新 `docs/knowledge-chunks.md`；符合门槛的难改取舍写入 ADR。
- 当前 source、授权、索引数量、进程、端口、模型绑定和任务状态只从 live 系统刷新，不复制进长期上下文。
