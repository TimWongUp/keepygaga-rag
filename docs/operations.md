# Keepygaga RAG Operations

## Verification

按改动风险依次使用：

1. 最接近的 chunk、DB、Indexer、search、Dashboard 或 schema 定向测试。
2. `uv run python scripts/smoke_mcp_server.py` 验证 MCP 2026-07-28 stdio、唯一 raw Tool `search`、参数拒绝与 Knowledge-only Doctor。
3. `uv run pytest -q`。
4. `uv run ruff check . && uv run pyright`。

普通验证不调用真实 Provider、不做真实全量重建、不消费 live 队列。当前状态从 `keepygaga-rag.toml`、Doctor、SQLite/LanceDB、Coordinator 进程和日志现场刷新。

## Doctor semantics

- `ok`：配置、数据库、active chunks、授权、Provider 凭据、Coordinator、开放任务和向量完整性均满足适用检查。
- `warning`：存在可读兼容状态或需关注项，例如旧 schema 待迁移；必须读取具体 check。
- `error`：配置、授权、schema、索引完整性或 Coordinator 等直接检查失败。

schema 低于当前代码时只报告 migration required；升级必须先停止旧 Coordinator，再由当前版本在单实例锁内完成。schema 高于代码或不可识别时禁止写入。

## Failure routing

- 无命中：核对 source enabled、Consent Identity、active generation、过滤参数与查询表达，不能推断原文件没有相关内容。
- source 不同步：核对授权、cleanup gate、队列、Coordinator 锁、最近 run 与 file error，不在 MCP/Dashboard 临时启动第二个扫描器。
- 扫描失败：恢复权限、I/O、身份或 UTF-8 合同后重试；不完整扫描不推进 missing reconciliation。
- 向量完整性失败：核对 active chunk、Embedding identity 和 Coordinator 日志；通过隔离新表重建后切换，不手工删除 live table。
- Provider 失败：保留 last-good；按正式降级语义返回 warning。HTTP 401/402/403 视为需要人工处理的凭据、余额、配额或模型权限问题，自动同步退避一小时，Dashboard 手动重试不受退避限制；成功后恢复正常周期。授权过期必须重新确认，不能降级绕过。
- 范围清理卡住：保持 source 禁用并让 Coordinator 恢复持久化删除，gate 清除后再授权。

## Evidence

测试、smoke、Doctor、SQLite/LanceDB 对账、日志和人工授权属于 Evidence；当前进程、端口、计数和任务属于 Run。稳定合同更新 repo-native context，实时状态不写成长期事实。
