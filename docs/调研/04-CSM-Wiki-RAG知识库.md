# CSM Wiki RAG 知识库调研

> **调研结论摘要**
>
> 对比 RAG 与 Skill（Function Calling）两种方案后，**推荐使用 RAG**：
> CSM Wiki 是相对稳定的领域知识文档，不需要实时数据或外部 API 操作，RAG 精准且成本低。
> Skill 适合"需要执行动作或获取实时数据"的场景，不适合本项目的静态知识检索需求。
> 向量库选 **ChromaDB**（本地持久化，无服务器），Embedding 选 **BAAI/bge-small-zh-v1.5**（免费，中文优秀）。
> 通过 MD5 哈希增量更新，只对变更文件重新 embedding，最大化节省 API 费用。

## 1. 问题定义

| 需求 | 挑战 |
|------|------|
| 每次回答都能参考最新 CSM Wiki | Wiki 更新后需同步到检索库 |
| 不全量灌入 Prompt | Token 消耗巨大，且超出上下文窗口 |
| 回答质量高 | 检索要精准，避免引入无关内容 |

## 2. RAG 架构总览

```
CSM Wiki (Markdown 文件，存放于 csm-wiki/remote/)
    │ 增量同步（python -m csm_qa.sync_wiki）
    ▼
文档分块（Chunking）
    │
    ▼
Embedding 生成（BAAI/bge-small-zh-v1.5 本地 或 OpenAI text-embedding-*）
    │
    ▼
向量存储（ChromaDB）← 保存在 .csm_qa/vector_store/
    │
    ▼
查询时：用户问题 → Embedding → Top-K 相似片段 → 注入 Prompt
```

## 3. 向量库选型

| 库 | 持久化 | 无服务器 | GitHub Actions 友好 | 元数据查询 | 推荐度 |
|----|--------|----------|---------------------|------------|--------|
| **ChromaDB** | 内置（本地文件） | ✅ | ✅ | ✅ | ⭐⭐⭐⭐⭐ |
| **FAISS** | 手动（save/load） | ✅ | ✅ | ❌（需额外处理） | ⭐⭐⭐⭐ |
| Pinecone | 云服务 | ❌（需联网） | 需 API Key | ✅ | ⭐⭐（复杂度高） |

**推荐：ChromaDB**，开箱即用本地持久化，适合 Git 管理小型向量库。

```bash
pip install chromadb sentence-transformers
```

参考：[DocRAG with FAISS/ChromaDB](https://github.com/EMoetez/DocRAG-with-FAISS)

## 4. Embedding 模型选型

| 模型 | 方式 | 成本 | 中文支持 |
|------|------|------|----------|
| `text-embedding-3-small` (OpenAI) | API 调用 | 极低（$0.02/1M tokens） | 良好 |
| `text-embedding-3-large` (OpenAI) | API 调用 | 稍贵 | 更好 |
| `BAAI/bge-small-zh-v1.5` | 本地运行 | 免费 | 专为中文优化 |

> 推荐：优先使用 `BAAI/bge-small-zh-v1.5`（本地，零成本，中文优秀）  
> Wiki 内容多为中文时尤其适合，可通过 `sentence-transformers` 加载。

## 5. 文档分块策略

```python
# 按 Markdown 标题分块，保留语义完整性
# 每块约 300~500 tokens，带标题前缀用于上下文溯源
def chunk_markdown(text: str, source: str) -> list[dict]:
    sections = re.split(r'\n(?=#{1,3} )', text)
    return [{"text": s.strip(), "source": source} for s in sections if s.strip()]
```

- 按 `#` 标题边界分割，比固定字符截断保留更完整的语义单元
- 每块附带来源文件名，便于回复时引用出处

## 6. 增量更新机制

避免每次全量重建向量库（消耗 embedding API 费用）：

```python
# 维护 data/wiki_hash.json：{文件路径: MD5哈希}
def sync_wiki(wiki_dir: str, vectorstore: Chroma):
    old_hashes = load_hashes("data/wiki_hash.json")
    new_hashes = {}
    for md_file in Path(wiki_dir).glob("**/*.md"):
        content = md_file.read_text()
        h = md5(content)
        new_hashes[str(md_file)] = h
        if old_hashes.get(str(md_file)) != h:
            # 删除旧向量，插入新向量
            vectorstore.delete(where={"source": str(md_file)})
            chunks = chunk_markdown(content, str(md_file))
            vectorstore.add_documents(chunks)
    save_hashes(new_hashes, "data/wiki_hash.json")
```

关键点：
- 只对**变更文件**重新 embedding（节省 API 费用）
- 新增/删除文件均处理

参考：[How to Update RAG Knowledge Base Without Rebuilding Everything](https://particula.tech/blog/update-rag-knowledge-without-rebuilding)

## 7. 检索与注入

```python
def retrieve_context(query: str, k: int = 3) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]
```

- `k=3` 约 900–1500 tokens，足够提供上下文又不过载
- 可加 **reranker**（如 `cross-encoder/ms-marco-MiniLM-L-6-v2`）提升精度，但需权衡延迟
- 默认相似度阈值 `0.72`，过滤低相关性片段

## 8. Wiki 更新方式

```bash
python -m csm_qa.sync_wiki                 # 增量同步
python -m csm_qa.sync_wiki --force         # 强制重建
python -m csm_qa.sync_wiki --remote        # 检查远程并按需拉取
```

或在代码中：

```python
qa.sync_wiki(force=False)
```

## 9. RAG vs Skill（Function Calling）方案对比

### 9.1 概念定义

| 方案 | 原理 | 典型场景 |
|------|------|----------|
| **RAG** | 查询时向量检索相关文档片段，注入 Prompt | 静态/半静态知识库问答 |
| **Skill（Function Calling）** | LLM 决定调用哪个函数（工具），系统执行后返回结果 | 实时数据、外部 API、执行操作 |

### 9.2 详细对比

| 维度 | RAG | Skill / Function Calling |
|------|-----|--------------------------|
| **知识更新** | 需重新 embedding（可增量） | 函数逻辑更新即生效 |
| **实时性** | 取决于索引更新频率 | 天然实时（直接调用 API） |
| **精确性** | 向量相似度，可能检索到不相关内容 | 确定性高，函数输出可控 |
| **Token 消耗** | 注入文档片段，消耗较多输入 token | 只返回精确结果，消耗少 |
| **适合内容类型** | 文档、FAQ、规范、指南 | 数据库查询、外部服务、实时信息 |
| **工程复杂度** | 中（向量库 + embedding） | 高（函数设计、错误处理、安全） |
| **CSM Wiki 适用性** | ✅ 高度适合 | ❌ 过度设计（知识是静态文档） |

### 9.3 结论

**CSM Wiki 是相对稳定的规范文档**，不需要实时查询，RAG 完全满足需求。  
未来如需扩展实时查询能力，可叠加 Function Calling，不影响 RAG 层。

> 参考：[RAG vs Function Calling (getstream.io)](https://getstream.io/blog/rag-function-calling/) | [RAG, Tool Calling, Function Calling: Boundaries & Patterns](https://jit.pro/blog/rag-tool-calling-function-calling-boundaries-patterns)

## 10. 参考资源

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [BAAI/bge 中文 Embedding 模型](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- [Context-Aware RAG (Microsoft)](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/context-aware-rag-system-with-azure-ai-search-to-cut-token-costs-and-boost-accur/4456810)
- [RAG vs Function Calling](https://getstream.io/blog/rag-function-calling/)
