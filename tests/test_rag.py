"""RAG 检索器测试（使用 fake embedding，避开真实模型加载）。"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from csm_qa.rag import EmbeddingFunction, RAGRetriever


class FakeEmbedding(EmbeddingFunction):
    """词袋式 fake embedding：稳定、可重现、无需外部模型。"""

    DIM = 32

    def __init__(self) -> None:
        # 不调用 super().__init__ 以避免 sentence_transformers 触发
        self.provider = "local"
        self.model = "fake"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.DIM
            for token in text.split():
                token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                vec[token_hash % self.DIM] += 1.0
            # L2 归一化
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out

    def __call__(self, input):  # noqa: A002
        return self.embed(input)


def _write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


@pytest.fixture
def retriever(tmp_dir: Path) -> RAGRetriever:
    wiki = tmp_dir / "wiki"
    store = tmp_dir / "store"
    wiki.mkdir()
    return RAGRetriever(
        wiki_dir=wiki,
        vector_store_dir=store,
        embedding_fn=FakeEmbedding(),
    )


def test_initial_state_is_empty(retriever: RAGRetriever):
    assert retriever.is_empty()
    # 空向量库：检索返回空列表
    assert retriever.retrieve("anything") == []


def test_chunk_markdown_splits_by_heading():
    content = "# Title\nintro\n\n## Section A\nbody A\n\n## Section B\nbody B"
    chunks = RAGRetriever._chunk_markdown(content, "f.md")
    headings = [c["heading"] for c in chunks]
    assert headings == ["Title", "Section A", "Section B"]


def test_chunk_markdown_no_heading_returns_single_chunk():
    chunks = RAGRetriever._chunk_markdown("plain text only", "f.md")
    assert len(chunks) == 1
    # 没有 #/##/### 标题时，使用 "Untitled" 占位
    assert chunks[0]["heading"] == "Untitled"
    assert chunks[0]["text"] == "plain text only"


def test_sync_wiki_indexes_files(retriever: RAGRetriever):
    _write(retriever.wiki_dir / "intro.md", "# CSM\nCSM 是一种通信状态机框架。")
    _write(retriever.wiki_dir / "guide.md", "# 使用指南\n如何配置 CSM 的状态。")

    result = retriever.sync_wiki()
    assert result["updated"] == 2
    assert result["skipped"] == 0
    assert not retriever.is_empty()


def test_sync_wiki_incremental_skips_unchanged(retriever: RAGRetriever):
    f = retriever.wiki_dir / "a.md"
    _write(f, "# A\ncontent A")
    retriever.sync_wiki()

    # 第二次同步：未修改 → 全部跳过
    result = retriever.sync_wiki()
    assert result["updated"] == 0
    assert result["skipped"] == 1


def test_sync_wiki_force_rebuilds(retriever: RAGRetriever):
    _write(retriever.wiki_dir / "a.md", "# A\ncontent A")
    retriever.sync_wiki()

    result = retriever.sync_wiki(force=True)
    assert result["updated"] == 1
    assert result["skipped"] == 0


def test_sync_wiki_removes_deleted_files(retriever: RAGRetriever):
    a = retriever.wiki_dir / "a.md"
    b = retriever.wiki_dir / "b.md"
    _write(a, "# A\ncontent A")
    _write(b, "# B\ncontent B")
    retriever.sync_wiki()

    a.unlink()
    result = retriever.sync_wiki()
    assert result["removed"] == 1


def test_retrieve_returns_relevant_chunks(retriever: RAGRetriever):
    _write(
        retriever.wiki_dir / "csm.md",
        "# CSM 状态机\nCSM 状态机 切换 通过 消息 通信",
    )
    _write(
        retriever.wiki_dir / "other.md",
        "# 无关\n这是 完全 无关 的 内容",
    )
    retriever.sync_wiki()

    # threshold=0 以确保 fake embedding 返回结果
    results = retriever.retrieve("CSM 状态机 切换", k=2, threshold=0.0)
    assert len(results) >= 1
    # 应至少包含相关的 csm.md 内容
    assert any("CSM" in r and "状态机" in r for r in results)


def test_retrieve_empty_query_returns_empty(retriever: RAGRetriever):
    _write(retriever.wiki_dir / "a.md", "# A\ncontent")
    retriever.sync_wiki()
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_retrieve_embedding_failure_returns_empty(retriever: RAGRetriever):
    """embedding 调用抛出异常时 retrieve 应返回空列表而非向上传播。"""
    _write(retriever.wiki_dir / "a.md", "# A\ncontent")
    retriever.sync_wiki()

    class BrokenEmbedding(FakeEmbedding):
        def embed(self, texts):
            raise RuntimeError("模型加载失败")

    retriever.embedding_fn = BrokenEmbedding()
    result = retriever.retrieve("content")
    assert result == []
