# Keepygaga RAG

[English](README.md) | [简体中文](README.zh-CN.md)

Keepygaga RAG 是从 Keepygaga 拆出的可独立运行 Knowledge/RAG 子项目。
它索引经用户授权的本地 Markdown 与纯文本资料，通过 SQLite FTS5 和 LanceDB
向量召回、RRF 融合与在线 Reranker 提供可追溯到原文坐标的混合检索。

> **项目状态：开发中的个人软件（Pre-Alpha）。** 本仓库公开用于早期试用与协作，
> 接口、数据格式和行为可能随时变化；不承诺支持、可用性或向后兼容，Issue 与 PR
> 由维护者视精力处理。

公开 MCP raw Tool 只有 `search`。宿主注册 ID 使用 `keepygaga_rag` 时，完整 Tool
名为 `mcp__keepygaga_rag__search`；MCP Server 显示名为 `Keepygaga RAG`。

## 环境要求

- Python 3.12+
- 一个包含 Markdown 或纯文本资料的本地目录
- 已配置且获得用户授权的 Embedding 与 Reranker API
- 推荐使用 [`uv`](https://docs.astral.sh/uv/)

## 安装与使用

```bash
git clone https://github.com/TimWongUp/keepygaga-rag.git
cd keepygaga-rag
uv sync --extra dashboard
cp keepygaga-rag.example.toml keepygaga-rag.toml
uv run keepygaga-rag doctor
uv run python mcp_server.py
uv run keepygaga-rag dashboard
uv run keepygaga-rag indexer
```

Dashboard 默认监听 `127.0.0.1:8765`。运行时覆盖项为
`KEEPYGAGA_RAG_CONFIG`、`KEEPYGAGA_RAG_DASHBOARD_PORT` 与
`KEEPYGAGA_RAG_DASHBOARD_AUTO_CLOSE`。

## 安全边界

- 原文件始终是真源，索引只是可重建的派生产物。
- `agents-memory/**` 与 `_context-backups/**` 在任意深度始终排除。
- 索引正文只发送给已授权的 Embedding Provider；查询文本与候选正文只发送给
  已授权的 Reranker。
- Provider、模型或数据源范围变化后必须重新取得用户授权。
- 混合检索结果只用于定位，定论前仍需读取返回的原文件。

本地与外部数据边界、凭据处理方式和 Provider 责任详见
[隐私与数据流说明](PRIVACY.zh-CN.md)。

## 贡献与安全

欢迎在项目当前的 Pre-Alpha 范围内参与贡献。提交 Issue 或 PR 前，请先阅读
[贡献指南](CONTRIBUTING.md)。

请勿在公开 Issue 中报告安全漏洞，私密报告方式见 [安全政策](SECURITY.md)。

## 许可证

[MIT](LICENSE)
