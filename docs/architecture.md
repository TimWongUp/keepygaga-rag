# Keepygaga RAG Architecture

## Boundary

Keepygaga RAG 负责授权本地 Markdown/TXT source 的登记、扫描、chunk、FTS/vector 派生、table 内混合检索、在线 Rerank、独立 Indexer、Doctor 与 Dashboard。它与核心记忆产品 Keepygaga 同属一个产品家族，但两者是独立安装、运行、发布和演进的 sibling 产品。核心记忆格式和 mutation 只属于 `keepygaga` 仓库；两仓仅通过 Agent Host 并列注册，各自拥有配置与存储合同，也不 import、打包或依赖对方的 Python package。

原始 source files 永远是真源。SQLite 与 LanceDB 都是可重建派生层：SQLite 保存 registry、manifest、chunk 正文、四字段 FTS、任务、运行和 generation；LanceDB 保存固定 Embedding identity 下的向量与必要过滤/对账字段。

## Source and consent

- source 可绑定整个目录、子目录或单个文件；目录范围覆盖后续新增的 Markdown/TXT。
- roots 不得互相重叠；扫描使用 `lstat` 身份快照，不跟随 symlink，隐藏树和硬排除目录不进入候选。
- `agents-memory/` 与 `_context-backups/` 在任意深度硬排除。
- source/provider/model/范围首次绑定或变化必须重新授权；授权过期时正文、查询和候选不得发送给 Provider。
- 范围收缩先进入 cleanup gate，越界 SQLite 和 LanceDB 派生数据清理完成前不能重新授权或查询绕过。

## Index lifecycle

- 扫描以 `mtime_ns + size` 筛候选、SHA-256 确认内容变化，并要求稳定期和连续缺失确认。
- UTF-8 是输入合同；遍历、权限、I/O、身份或编码失败都可定位并 fail closed，不把不完整扫描当作删除。
- 新 generation 的 chunk/FTS/vector 全部就绪后，才能在 SQLite transaction 中切换 active generation；失败保留 last-good。
- source/file 删除和 generation 切换先持久化待删 vector ID；Coordinator 恢复删除并清理 inactive generation。
- 只有独立 `keepygaga-rag indexer` 进程在单实例锁内执行可写 schema、队列和派生恢复。Dashboard 与 MCP 查询进程不启动扫描器。
- schema 未知、缺失识别信息或高于当前代码时，在任何 DDL 前 fail closed；旧 schema 只读查询可走显式兼容路径。

## Search contract

raw MCP Tool 只有 `search`。MCP 客户端配置约定使用注册 key `keepygaga_rag`，因此按该 key 接线时完整名为 `mcp__keepygaga_rag__search`；MCP `serverInfo.name` 为 `Keepygaga RAG`，repo smoke 验证 2026-07-28 stdio 协议路径，但不验证客户端配置 key。

Phase 1 只有 `text_chunks_v1`：四字段 FTS 与 vector recall 在 table 内经 RRF 融合，再由该 table 的 Reranker 排序，最后按 `source_id + relative_path` 限制同一文件的 chunk 数并返回 `top_k`。授权过滤与 active generation 映射发生在融合前，崩溃遗留向量不可见。

结果返回命中 chunk 与 source provenance，只负责定位；调用方必须读取原文件后才能把内容当作 Authority。Embedding 失败可以降级到 FTS，Reranker 失败可以降级到 table 内 RRF并返回 warning；授权过期不允许降级。

## Dashboard boundary

Dashboard 是本地控制面：配置 source、范围、授权、查询参数、队列和诊断；它不成为事实源，不直接写向量，也不注册 MCP Tool。Indexer 重启能力只接受固定服务入口与本机确认，不接受任意命令、PID 或 service label 输入。

## Non-goals

当前不处理会话历史、核心记忆、跨 table 全局排名、PDF/Office/图片/音视频解析、外部数据库服务、eval 或 benchmark。未来候选见 `docs/design-history.md`，未经仓库维护者接受不属于开放任务。
