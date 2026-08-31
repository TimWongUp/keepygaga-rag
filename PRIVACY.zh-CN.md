# 隐私与数据流说明

[English](PRIVACY.md) | [简体中文](PRIVACY.zh-CN.md)

Keepygaga RAG 是开发中的 local-first 个人软件。本文说明当前技术实现的
数据流，不构成法律意义上的隐私政策，也不代表第三方 Provider 的承诺。

## 留在本机的数据

- Markdown 与纯文本原文件保留在原位置。
- SQLite、FTS 与 LanceDB 索引保存在配置指定的本地目录。
- 运行配置与 API Key 从本地配置和环境变量读取，不应提交到本仓库。
- Dashboard 默认只监听 `127.0.0.1`。
- 当前应用没有独立的遥测、分析或自动更新上报客户端。

`agents-memory/**` 与 `_context-backups/**` 在任意深度均由代码硬排除。默认配置还会
排除 source root 以下的隐藏路径、`.obsidian/`、`.git/`、`.keepygaga/`、`.venv/`
和 `node_modules/`。但 source root 本身可以是隐藏目录，自定义索引目录也不会自动
加入扫描排除项。

本地 `.env`、`keepygaga-rag.toml`、默认 `.keepygaga/` 索引目录与 `.venv/`
默认由 Git 排除；自定义索引目录不会自动加入 `.gitignore`。

## 发送给已配置 Provider 的数据

在用户授权数据源范围、Provider 与模型绑定之前，外部传输保持关闭。授权后：

- 构建索引时，授权范围内的 source chunk 会发送给配置的 Embedding Provider；
  Embedding 输入可能包括正文、文档标题、标题路径和文件名；
- 搜索时，查询文本会发送给配置的 Embedding Provider；
- 查询文本与召回的候选 chunk（含上述元数据）会发送给配置的 Rerank Provider；
- 对应 API Key 仅作为认证信息发送到所配置的 Provider endpoint。

数据源范围、Provider 或模型变化后必须重新授权。第三方 Provider 的服务条款、
日志、保留时间和数据处理地域不受本项目控制。

## 通过 MCP 返回的数据

`search` Tool 会向所连接的 MCP host 或 client 返回命中 chunk 正文、标题路径、分数和
原文件绝对路径。应把该 host 或 client 视为数据接收方，并据此配置访问权限。

## 用户责任

索引前应核对所选 Provider、endpoint、模型和数据源范围。不要处理无权使用的资料；
不应传给外部 Provider 的敏感内容应事先删除或脱敏。请使用可信 HTTPS endpoint；
自定义 `base_url` 不会被强制要求使用 HTTPS。若自定义索引目录，应按需同时把它加入
数据源扫描排除项与 Git 忽略项，并按原资料的敏感程度保护本地配置、环境变量和
索引文件。
