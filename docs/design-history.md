# Knowledge retrieval design history

Phase 1 选择 SQLite + LanceDB，是为了把原文件、控制面/FTS 与向量层分离：原文件保留 Authority，SQLite 管理 source、manifest、chunk、任务与 generation，LanceDB 专注固定 Embedding space 的向量。周期扫描与 manifest 对账避免把平台文件事件作为唯一正确性来源。

未来候选可以按独立 Knowledge Table 扩展：文本 table 使用文本 Embedding/Reranker，图片 table 使用视觉模型；不同 table 的分数不直接比较，也不跨 table 合并 Reranker 排名。PDF 候选路线是保留原 PDF，并把可靠文本层或 OCR 结果规范化为带页码/bbox provenance 的文本 chunk，而不是直接对整页图像建立统一视觉排名。

这些内容记录设计方向，不是开放任务。只有仓库维护者明确接受范围、数据传输边界、依赖、验证和发布影响后，候选才进入实现。
