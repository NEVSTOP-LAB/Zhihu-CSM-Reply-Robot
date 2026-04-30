"""RAG 检索器测试（使用 fake embedding，避开真实模型加载）。"""

from __future__ import annotations

import hashlib
import math
import os
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
    retriever = RAGRetriever(
        wiki_dir=wiki,
        vector_store_dir=store,
        embedding_fn=FakeEmbedding(),
    )
    yield retriever
    retriever.close()


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


def test_retrieve_with_meta_returns_source_and_heading(retriever: RAGRetriever):
    _write(
        retriever.wiki_dir / "csm.md",
        "# CSM 状态机\nCSM 状态机 切换 通过 消息 通信",
    )
    retriever.sync_wiki()

    hits = retriever.retrieve_with_meta("CSM 状态机 切换", k=1, threshold=0.0)
    assert len(hits) == 1
    hit = hits[0]
    assert set(hit.keys()) >= {"text", "source", "heading", "similarity"}
    assert hit["source"] == "csm.md"
    assert hit["heading"] == "CSM 状态机"
    assert "CSM" in hit["text"]


def test_retrieve_logs_hit_details(retriever: RAGRetriever, caplog):
    _write(
        retriever.wiki_dir / "csm.md",
        "# CSM 状态机\nCSM 状态机 切换 通过 消息 通信",
    )
    retriever.sync_wiki()

    with caplog.at_level("INFO", logger="csm_qa.rag"):
        results = retriever.retrieve("CSM 状态机 切换", k=1, threshold=0.0)

    assert len(results) == 1
    assert "RAG 命中 1 条" in caplog.text
    assert "source=csm.md" in caplog.text
    assert "heading=CSM 状态机" in caplog.text


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


def test_local_embedding_sets_default_hf_endpoint_before_loading(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    embedding = EmbeddingFunction()
    observed: dict[str, str | None] = {}

    class FakeEncoded:
        def tolist(self):
            return [[1.0, 0.0]]

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            return FakeEncoded()

    def fake_create_local_model(self):
        # 记录首次尝试时的 endpoint（第一次即应为国内镜像）
        if "endpoint" not in observed:
            observed["endpoint"] = os.environ.get("HF_ENDPOINT")
        return FakeModel()

    monkeypatch.setattr(
        EmbeddingFunction,
        "_create_local_model",
        fake_create_local_model,
    )

    result = embedding.embed(["hello"])
    assert result == [[1.0, 0.0]]
    assert observed["endpoint"] == "https://hf-mirror.com"


def test_local_embedding_fallback_tries_all_endpoints(monkeypatch):
    """首个端点失败时应依次尝试回落候选，最终成功返回。"""
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    embedding = EmbeddingFunction()
    attempted_endpoints: list[str] = []

    class FakeEncoded:
        def tolist(self):
            return [[0.5, 0.5]]

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            return FakeEncoded()

    call_count = {"n": 0}

    def fake_create_local_model(self):
        call_count["n"] += 1
        attempted_endpoints.append(os.environ.get("HF_ENDPOINT", ""))
        if call_count["n"] == 1:
            # 第一次（hf-mirror.com）失败
            raise ConnectionError("mirror unreachable")
        # 第二次（huggingface.co）成功
        return FakeModel()

    monkeypatch.setattr(EmbeddingFunction, "_create_local_model", fake_create_local_model)

    result = embedding.embed(["test"])
    assert result == [[0.5, 0.5]]
    assert call_count["n"] == 2
    assert attempted_endpoints[0] == "https://hf-mirror.com"
    assert attempted_endpoints[1] == "https://huggingface.co"


def test_local_embedding_load_failure_is_not_retried(monkeypatch):
    """所有端点均失败后，应停止重试并在后续调用中直接抛出。"""
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    embedding = EmbeddingFunction()
    calls = {"count": 0}

    def fake_create_local_model(self):
        calls["count"] += 1
        raise RuntimeError("ssl failed")

    monkeypatch.setattr(
        EmbeddingFunction,
        "_create_local_model",
        fake_create_local_model,
    )

    with pytest.raises(RuntimeError, match="本地 Embedding 模型初始化失败"):
        embedding.embed(["first"])

    # 所有候选端点（hf-mirror.com + huggingface.co = 2 次）均应被尝试
    from csm_qa.rag import _HF_FALLBACK_ENDPOINTS
    assert calls["count"] == len(_HF_FALLBACK_ENDPOINTS)

    # 后续调用不应再重试
    with pytest.raises(RuntimeError, match="已停止后续重试"):
        embedding.embed(["second"])

    assert calls["count"] == len(_HF_FALLBACK_ENDPOINTS)


def test_local_embedding_user_endpoint_appended_as_last_fallback(monkeypatch):
    """用户配置的 HF_ENDPOINT（非默认值）应追加到回落列表末尾。"""
    user_url = "https://my-custom-mirror.example.com"
    monkeypatch.setenv("HF_ENDPOINT", user_url)
    embedding = EmbeddingFunction()
    candidates = embedding._build_hf_endpoint_candidates()

    from csm_qa.rag import _HF_FALLBACK_ENDPOINTS
    assert candidates[: len(_HF_FALLBACK_ENDPOINTS)] == list(_HF_FALLBACK_ENDPOINTS)
    assert candidates[-1] == user_url


def test_local_embedding_user_endpoint_not_duplicated(monkeypatch):
    """用户配置的 HF_ENDPOINT 若与内置候选重复，不应重复追加。"""
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
    embedding = EmbeddingFunction()
    candidates = embedding._build_hf_endpoint_candidates()
    assert candidates.count("https://hf-mirror.com") == 1


def test_local_embedding_restores_hf_endpoint_after_success(monkeypatch):
    """模型加载成功后，HF_ENDPOINT 环境变量应恢复到调用前的值。"""
    original = "https://original-endpoint.example.com"
    monkeypatch.setenv("HF_ENDPOINT", original)
    embedding = EmbeddingFunction()

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            class R:
                def tolist(self):
                    return [[1.0]]
            return R()

    monkeypatch.setattr(EmbeddingFunction, "_create_local_model", lambda self: FakeModel())
    embedding.embed(["test"])
    assert os.environ.get("HF_ENDPOINT") == original


def test_local_embedding_restores_hf_endpoint_after_failure(monkeypatch):
    """所有端点失败后，HF_ENDPOINT 环境变量应恢复到调用前的值。"""
    original = "https://original-endpoint.example.com"
    monkeypatch.setenv("HF_ENDPOINT", original)
    embedding = EmbeddingFunction()

    monkeypatch.setattr(
        EmbeddingFunction, "_create_local_model", lambda self: (_ for _ in ()).throw(RuntimeError("fail"))
    )

    with pytest.raises(RuntimeError):
        embedding.embed(["test"])

    assert os.environ.get("HF_ENDPOINT") == original


def test_local_embedding_removes_hf_endpoint_env_when_originally_absent(monkeypatch):
    """HF_ENDPOINT 原本不存在时，加载后应将其从环境变量中移除（而非留下最后一次尝试的值）。"""
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    embedding = EmbeddingFunction()

    class FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            class R:
                def tolist(self):
                    return [[1.0]]
            return R()

    monkeypatch.setattr(EmbeddingFunction, "_create_local_model", lambda self: FakeModel())
    embedding.embed(["test"])
    assert "HF_ENDPOINT" not in os.environ
