# AI-005 实施记录：RAGRetriever — Wiki 索引与检索

## 任务目标
实现 CSM Wiki 增量 embedding + 检索，支持本地/线上 embedding 切换。

## 实施内容

### 1. 模块实现 (`scripts/rag_retriever.py`)
- **RAGRetriever**: 管理 wiki_collection 和 reply_collection 两个向量索引
- **sync_wiki(force=False)**: MD5 哈希比对增量更新，只处理变更文件
- **retrieve(query, k=3, threshold=0.72)**: reply_index top-2 优先 + wiki 补充
- **index_human_reply(question, reply, article_id, thread_id)**: 高权重 (weight=high) 写入
- **_chunk_markdown()**: 按 # 标题边界分块，保留来源溯源

### 2. 遇到的问题
- **chromadb import 位置**: 初始将 `import chromadb` 放在方法内部（local import），导致 mock patch 失败
  - 解决方案：改为模块级别 import，并 patch `chromadb.PersistentClient`
- **embedding 模式测试**: 需要同时 mock local 和 online 两个方法，否则实际调用会触发依赖安装
  - 解决方案：在测试中同时 patch 两个方法

## 测试结果
```
tests/test_rag_retriever.py — 20 项测试全部通过 ✅
- TestChunkMarkdown: 5 项（标题分块、来源保留、空文本、无标题、多级标题）
- TestSyncWiki: 5 项（新文件、未变更跳过、变更更新、强制重建、多文件）
- TestRetrieve: 4 项（正常检索、reply优先、阈值过滤、空索引）
- TestIndexHumanReply: 4 项（高权重、文档格式、元数据字段、ID格式）
- TestEmbeddingMode: 2 项（本地模式、线上模式）
```

## 验收状态
✅ 单元测试全部通过
