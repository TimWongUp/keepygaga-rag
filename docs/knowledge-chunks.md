# Text Chunk Contract

## Input and representation

`text_chunks_v1` 接受严格 UTF-8 Markdown/TXT。BOM 与 CRLF 在 frontmatter 识别前规范化；非法编码成为可定位文件错误。检索字段固定为 `text`、`title`、`heading_path`、`filename`：SQLite FTS 分字段索引并加权，Embedding 与 Reranker 使用同一份带字段标签的表示。

## Chunking

- Markdown frontmatter 只用于 title 推导，不进入正文 chunk。
- parser 按 fenced code、heading 与普通 block 建立结构；超长 block 再按字符边界拆分。
- `chunk_target_chars` 是跨 heading 合并的软目标，`chunk_max_chars` 是硬上限；`chunk_mode=structure|length` 控制结构优先或长度切分。
- overlap 默认关闭；调整 target/max/mode/overlap 会标记受影响文件 rechunk，但由独立 Coordinator 消费。
- chunk 保存索引时原文件的 `start_line` / `end_line`（从 1 开始、两端包含）；行范围覆盖命中正文，包括 overlap，不把剥离 frontmatter、规范化空行后的行号当作原文行号。只定位到行，不保存字符位置。

## Identity and generation

chunk identity 必须绑定 source file、规范化内容、结构位置、chunker/preprocessor 与 Embedding input hash。内容不变的 chunk 可以复用向量；字段表示、preprocessor 或 Embedding identity 变化必须重建不兼容派生数据。

SQLite 保存 chunk 正文、FTS、metadata、generation 与 active pointer；LanceDB 保存 chunk ID、Embedding space/hash、vector、generation 和必要过滤字段。新 generation 只有在两层全部就绪后可见，last-good 在失败时继续服务。

## Retrieval provenance

查询先把 FTS/vector 候选映射到当前授权 source 与 active generation，再做 table 内 RRF、Rerank 和单文件限额。结果正文是命中 chunk，不是全文；需要精确引用、表格或高风险符号时必须回读原文件。

行范围描述索引时的原文件版本；原文件后续编辑可能使位置偏移，调用方仍须回读核对。旧 chunk 的行号返回 `null`，由 Coordinator 迁移并重新切块后补齐；检索字段和 Embedding identity 不变时复用已有向量，查询不额外读取原文件。

精确字段、默认值、schema version 和当前行数以 `keepygaga_rag/knowledge/`、配置、测试和 live SQLite/LanceDB 为准。
