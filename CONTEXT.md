# Keepygaga RAG Context

Keepygaga RAG 是面向 Agent 的 local-first Knowledge/RAG 系统。这里定义项目特有领域词；行为与精确 schema 以代码和测试为准。

## Language

**Source**：
用户明确登记并授权索引的本地目录或文件边界；原文件始终是真源。
_Avoid_: 索引、数据库副本

**Knowledge Table**：
绑定一种内容合同、Embedding space 与 table 内检索流程的逻辑检索单元。
_Avoid_: 任意数据库表、跨模态总榜

**Retrieval Profile**：
Knowledge Table 的 Embedding、Reranker、chunker、FTS 与召回限制合同。

**Chunk**：
从 source file 派生、带 title、heading、filename 与 provenance 的可检索文本单元。
_Avoid_: 原文件全文、搜索结果摘要

**Generation**：
同一 source file 一次完整派生结果的版本集合；只有验证完成的 active generation 对查询可见。
_Avoid_: Git revision、数据库 schema version

**Consent Identity**：
source 范围与当前 Provider/模型身份的授权绑定；任一受控维度变化都会使旧授权失效。
_Avoid_: API key、永久授权

**Derived Index**：
可由原 source 和当前 Retrieval Profile 重建的 SQLite/FTS/LanceDB 状态。
_Avoid_: 事实真源

**Coordinator**：
持有单实例锁并串行执行 schema、索引队列、generation 切换和派生删除的 Indexer 进程。
_Avoid_: Dashboard 线程、MCP 查询进程
