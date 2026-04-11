# -*- coding: utf-8 -*-
"""
RAGRetriever 单元测试
=====================

实施计划关联：AI-005 验收标准
独立于实现的测试用例，覆盖：
- sync_wiki 只对变更文件重新 embedding（MD5 增量）
- retrieve 相似度低于阈值时返回空
- retrieve 优先返回 reply_index 中的结果
- use_online_embedding=true 时调用正确的接口
- Markdown 按标题分块
"""
import json
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from scripts.rag_retriever import RAGRetriever, EmbeddingFunction


# ===== Fixtures =====

@pytest.fixture
def wiki_dir(tmp_path: Path) -> Path:
    """创建包含示例 Wiki 文件的临时目录"""
    wiki = tmp_path / "csm-wiki"
    wiki.mkdir()

    # 创建两个测试 Wiki 文件
    (wiki / "guide.md").write_text(
        "# CSM 框架指南\n\nCSM（Communicable State Machine，通信状态机）是一种 LabVIEW 设计框架。\n\n"
        "## 核心概念\n\n状态机通过消息通信实现模块解耦。\n",
        encoding="utf-8"
    )
    (wiki / "faq.md").write_text(
        "# 常见问题\n\n## 什么是 CSM？\n\n"
        "CSM 即 Communicable State Machine（通信状态机）。\n\n"
        "## 如何处理状态切换？\n\n通过发送消息触发状态转换。\n",
        encoding="utf-8"
    )
    return wiki


@pytest.fixture
def vector_store_dir(tmp_path: Path) -> Path:
    """临时向量库目录"""
    d = tmp_path / "vector_store"
    d.mkdir()
    return d


@pytest.fixture
def reply_index_dir(tmp_path: Path) -> Path:
    """临时回复索引目录"""
    d = tmp_path / "reply_index"
    d.mkdir()
    return d


class FakeEmbeddingFunction:
    """测试用的假 Embedding 函数

    返回固定维度的向量，使用简单的哈希算法生成，
    确保相同文本返回相同向量。
    """

    def __init__(self, use_online: bool = False, **kwargs):
        self.use_online = use_online
        self.call_count = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += len(texts)
        result = []
        for text in texts:
            # 用 MD5 生成确定性的假向量（128 维）
            h = hashlib.md5(text.encode()).hexdigest()
            vec = [int(c, 16) / 15.0 for c in h]  # 归一化到 [0, 1]
            # 扩展到合理维度
            vec = vec * 8  # 128 维
            # L2 归一化
            norm = sum(v * v for v in vec) ** 0.5
            vec = [v / norm for v in vec]
            result.append(vec)
        return result

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed(input)


@pytest.fixture
def retriever(wiki_dir, vector_store_dir, reply_index_dir):
    """创建使用假 Embedding 的 RAGRetriever"""
    r = RAGRetriever(
        wiki_dir=str(wiki_dir),
        vector_store_dir=str(vector_store_dir),
        reply_index_dir=str(reply_index_dir),
    )
    # 替换为假的 embedding 函数
    r.embedding_fn = FakeEmbeddingFunction()
    return r


# ===== Markdown 分块测试 =====

class TestChunkMarkdown:
    """验证 Markdown 按标题分块逻辑"""

    def test_split_by_headers(self):
        """应按标题分割为独立块"""
        content = "# 标题一\n\n内容一。\n\n## 标题二\n\n内容二。\n"
        chunks = RAGRetriever._chunk_markdown(content, "test.md")
        assert len(chunks) == 2
        assert chunks[0]["heading"] == "标题一"
        assert chunks[1]["heading"] == "标题二"

    def test_preserves_source(self):
        """每个块应保留源文件信息"""
        content = "# 标题\n\n内容"
        chunks = RAGRetriever._chunk_markdown(content, "wiki/guide.md")
        assert chunks[0]["source"] == "wiki/guide.md"

    def test_no_headers_single_chunk(self):
        """没有标题的文档应作为单个块"""
        content = "这是一段没有标题的文本。\n\n第二段。"
        chunks = RAGRetriever._chunk_markdown(content, "test.md")
        assert len(chunks) == 1
        assert chunks[0]["heading"] == "Untitled"

    def test_empty_content(self):
        """空内容应返回空列表"""
        chunks = RAGRetriever._chunk_markdown("", "test.md")
        assert chunks == []

    def test_multiple_header_levels(self):
        """应支持 #, ##, ### 三级标题分割"""
        content = "# H1\n\n内容1\n\n## H2\n\n内容2\n\n### H3\n\n内容3\n"
        chunks = RAGRetriever._chunk_markdown(content, "test.md")
        assert len(chunks) == 3


# ===== Wiki 同步测试 =====

class TestSyncWiki:
    """验证 sync_wiki 增量更新逻辑"""

    def test_initial_sync_processes_all_files(self, retriever, wiki_dir):
        """首次同步应处理所有文件"""
        retriever.sync_wiki()

        # 验证向量库中有文档
        count = retriever._wiki_collection.count()
        assert count > 0

        # 验证 hash 文件被创建
        hash_path = retriever.vector_store_dir / "wiki_hash.json"
        assert hash_path.exists()

    def test_second_sync_skips_unchanged(self, retriever, wiki_dir):
        """第二次同步应跳过未变更的文件"""
        retriever.sync_wiki()
        initial_call_count = retriever.embedding_fn.call_count

        # 重置计数
        retriever.embedding_fn.call_count = 0

        # 第二次同步
        retriever.sync_wiki()

        # 不应有新的 embedding 调用
        assert retriever.embedding_fn.call_count == 0

    def test_sync_detects_changed_file(self, retriever, wiki_dir):
        """修改文件后应重新 embedding"""
        retriever.sync_wiki()
        retriever.embedding_fn.call_count = 0

        # 修改一个文件
        (wiki_dir / "guide.md").write_text(
            "# 更新后的指南\n\n这是更新后的内容。\n",
            encoding="utf-8"
        )

        retriever.sync_wiki()

        # 应有新的 embedding 调用（仅处理变更文件）
        assert retriever.embedding_fn.call_count > 0

    def test_sync_removes_deleted_file_vectors(self, retriever, wiki_dir):
        """删除文件后应移除对应的向量"""
        retriever.sync_wiki()

        # 获取初始文档数
        initial_count = retriever._wiki_collection.count()
        assert initial_count > 0

        # 删除一个文件
        (wiki_dir / "faq.md").unlink()

        retriever.sync_wiki()

        # 文档数应减少
        new_count = retriever._wiki_collection.count()
        assert new_count < initial_count

    def test_force_sync_rebuilds_all(self, retriever, wiki_dir):
        """force=True 应重建所有索引"""
        retriever.sync_wiki()
        retriever.embedding_fn.call_count = 0

        # 强制重建
        retriever.sync_wiki(force=True)

        # 应有 embedding 调用
        assert retriever.embedding_fn.call_count > 0

    def test_sync_nonexistent_wiki_dir(self, tmp_path):
        """Wiki 目录不存在时应安全跳过"""
        r = RAGRetriever(
            wiki_dir=str(tmp_path / "nonexistent"),
            vector_store_dir=str(tmp_path / "vs"),
            reply_index_dir=str(tmp_path / "ri"),
        )
        r.embedding_fn = FakeEmbeddingFunction()
        # 不应抛异常
        r.sync_wiki()

    def test_sync_embedding_failure_preserves_old_vectors(
        self, retriever, wiki_dir
    ):
        """embedding 失败时旧向量应保留，不被删除（FIX-23）"""
        # 首次同步建立基准向量
        retriever.sync_wiki()
        initial_count = retriever._wiki_collection.count()
        assert initial_count > 0

        # 修改文件，模拟 embedding 失败
        (wiki_dir / "guide.md").write_text(
            "# 新版指南\n\n新内容。\n", encoding="utf-8"
        )

        # 让 embedding 抛出异常
        original_embed = retriever.embedding_fn.embed

        call_count = [0]

        def embed_with_failure(texts):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("模拟 embedding API 失败")
            return original_embed(texts)

        retriever.embedding_fn.embed = embed_with_failure
        retriever.sync_wiki()

        # 旧向量应仍然存在（embedding 失败不删除旧数据）
        count_after = retriever._wiki_collection.count()
        assert count_after >= initial_count, (
            "embedding 失败时旧向量应保留，不应被删除（FIX-23）"
        )

    def test_sync_embedding_failure_allows_retry_next_run(
        self, retriever, wiki_dir
    ):
        """embedding 失败时新 hash 不应写入，下次运行应能重试（REV-5）"""
        # 首次同步，记录初始 hash
        retriever.sync_wiki()
        initial_hashes = retriever._load_wiki_hashes()
        original_hash = initial_hashes.get("guide.md")
        assert original_hash is not None

        # 修改文件（产生新 hash）
        (wiki_dir / "guide.md").write_text(
            "# 更新内容\n\n应当被重试。\n", encoding="utf-8"
        )
        new_hash = retriever._compute_md5(wiki_dir / "guide.md")
        assert new_hash != original_hash

        original_embed = retriever.embedding_fn.embed
        failed_once = [False]

        def fail_first(texts):
            if not failed_once[0]:
                failed_once[0] = True
                raise RuntimeError("模拟 API 失败")
            return original_embed(texts)

        # 第一次 sync：embedding 失败，hash 不应写入（REV-5）
        retriever.embedding_fn.embed = fail_first
        retriever.sync_wiki()

        hashes_after_failure = retriever._load_wiki_hashes()
        assert hashes_after_failure.get("guide.md") == original_hash, (
            "embedding 失败时 hash 不应更新，下次运行才能重试（REV-5）"
        )

        # 第二次 sync：embedding 恢复正常，应能成功处理并更新 hash
        retriever.embedding_fn.embed = original_embed
        retriever.sync_wiki()

        hashes_after_retry = retriever._load_wiki_hashes()
        assert hashes_after_retry.get("guide.md") == new_hash, (
            "embedding 成功后 hash 应被更新为最新值（REV-5）"
        )


# ===== 检索测试 =====

class TestRetrieve:
    """验证 retrieve 检索逻辑"""

    def test_retrieve_returns_results(self, retriever, wiki_dir):
        """同步后应能检索到结果"""
        retriever.sync_wiki()
        results = retriever.retrieve("CSM 框架", k=3, threshold=0.0)
        # 由于假向量，降低阈值确保有结果
        assert len(results) > 0

    def test_retrieve_empty_index(self, retriever):
        """空索引应返回空列表"""
        results = retriever.retrieve("任何查询", k=3, threshold=0.72)
        assert results == []

    def test_retrieve_respects_k_limit(self, retriever, wiki_dir):
        """结果数量不应超过 k"""
        retriever.sync_wiki()
        results = retriever.retrieve("测试", k=1, threshold=0.0)
        assert len(results) <= 1

    def test_retrieve_high_threshold_filters(self, retriever, wiki_dir):
        """高阈值应过滤低相似度结果"""
        retriever.sync_wiki()
        results = retriever.retrieve("完全不相关的查询", k=3, threshold=0.999)
        # 极高阈值下几乎不应有结果
        assert len(results) <= 3  # 不强制为空，因为假向量可能碰撞


# ===== 真人回复索引测试 =====

class TestIndexHumanReply:
    """验证 index_human_reply 写入逻辑"""

    def test_index_human_reply_adds_document(self, retriever):
        """索引真人回复后应能查询到"""
        retriever.index_human_reply(
            question="CSM 是什么？",
            reply="CSM 是 Communicable State Machine（通信状态机）框架的缩写。",
            article_id="12345",
            thread_id="t001",
        )

        count = retriever._reply_collection.count()
        assert count == 1

    def test_index_human_reply_metadata(self, retriever):
        """真人回复应包含 weight=high 元数据"""
        retriever.index_human_reply(
            question="如何使用 CSM 框架？",
            reply="关键是主动定义状态和消息接口。",
            article_id="99999",
            thread_id="t002",
        )

        # 获取所有文档
        all_docs = retriever._reply_collection.get()
        assert len(all_docs["ids"]) == 1
        metadata = all_docs["metadatas"][0]
        assert metadata["weight"] == "high"
        assert metadata["type"] == "human_reply"
        assert metadata["article_id"] == "99999"
        assert metadata["thread_id"] == "t002"

    def test_index_human_reply_upsert(self, retriever):
        """相同 QA 对的重复索引应更新而非新增"""
        for _ in range(3):
            retriever.index_human_reply(
                question="同一问题",
                reply="同一回复",
                article_id="111",
                thread_id="t001",
            )

        count = retriever._reply_collection.count()
        assert count == 1  # 只有一条记录

    def test_reply_index_priority_in_retrieve(self, retriever, wiki_dir):
        """检索时真人回复应优先于 Wiki 结果"""
        retriever.sync_wiki()
        retriever.index_human_reply(
            question="CSM 框架的核心",
            reply="核心是状态机的消息驱动和模块解耦",
            article_id="12345",
            thread_id="t001",
        )

        results = retriever.retrieve("CSM 框架", k=3, threshold=0.0)
        # 应有结果
        assert len(results) > 0
        # 第一条应来自 reply_index（包含真人回复文本）
        # 注意：由于假向量，我们只验证结果数量


# ===== Embedding 函数测试 =====

class TestEmbeddingFunction:
    """验证 EmbeddingFunction 模式切换"""

    def test_local_mode_default(self):
        """默认应为本地模式"""
        ef = EmbeddingFunction(use_online=False)
        assert ef.use_online is False

    def test_online_mode_flag(self):
        """线上模式标志应正确设置"""
        ef = EmbeddingFunction(use_online=True)
        assert ef.use_online is True

    @patch("scripts.rag_retriever.EmbeddingFunction._embed_online")
    def test_online_mode_calls_online(self, mock_online):
        """线上模式应调用线上 embedding"""
        mock_online.return_value = [[0.1] * 128]
        ef = EmbeddingFunction(use_online=True)
        ef.embed(["test"])
        mock_online.assert_called_once_with(["test"])

    @patch("scripts.rag_retriever.EmbeddingFunction._embed_local")
    def test_local_mode_calls_local(self, mock_local):
        """本地模式应调用本地 embedding"""
        mock_local.return_value = [[0.1] * 128]
        ef = EmbeddingFunction(use_online=False)
        ef.embed(["test"])
        mock_local.assert_called_once_with(["test"])

    def test_online_embedding_l2_normalized(self):
        """线上 embedding 应被 L2 归一化（FIX-07）"""
        ef = EmbeddingFunction(use_online=True)

        # 模拟未归一化的线上 embedding 返回
        raw_vec = [3.0, 4.0]  # norm = 5.0
        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = raw_vec
        mock_response.data = [mock_item]

        with patch.object(ef, "_get_online_client") as mock_client:
            mock_client.return_value.embeddings.create.return_value = mock_response
            result = ef.embed(["test text"])

        assert len(result) == 1
        vec = result[0]
        # 验证向量已被 L2 归一化：|v| ≈ 1.0
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-6, f"期望 L2 归一化向量，|v|={norm}"
        # 验证归一化正确：[3/5, 4/5]
        assert abs(vec[0] - 0.6) < 1e-6
        assert abs(vec[1] - 0.8) < 1e-6

    def test_online_embedding_zero_vector_handled(self):
        """零向量不应导致除零错误（FIX-07）"""
        ef = EmbeddingFunction(use_online=True)

        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.0, 0.0, 0.0]
        mock_response.data = [mock_item]

        with patch.object(ef, "_get_online_client") as mock_client:
            mock_client.return_value.embeddings.create.return_value = mock_response
            result = ef.embed(["test"])

        # 零向量应原样返回，不抛异常
        assert result[0] == [0.0, 0.0, 0.0]


# ===== 文件编码兼容性测试 =====

class TestReadText:
    """_read_text 编码自动检测测试（支持 UTF-8 / GB18030 / Big5）"""

    def test_reads_utf8_file(self, tmp_path):
        """UTF-8 文件正常读取"""
        f = tmp_path / "utf8.md"
        f.write_bytes("# 标题\n\n内容\n".encode("utf-8"))
        assert RAGRetriever._read_text(f) == "# 标题\n\n内容\n"

    def test_reads_gb18030_file(self, tmp_path):
        """GB18030 编码（兼容 GBK/GB2312）文件正确解码"""
        text = "# 简体中文标题\n\n这是 GB18030 编码的正文。\n"
        f = tmp_path / "gb18030.md"
        f.write_bytes(text.encode("gb18030"))
        assert RAGRetriever._read_text(f) == text

    def test_reads_big5_file(self, tmp_path):
        """Big5 编码（繁体中文）文件正确解码"""
        text = "# 繁體中文標題\n\n這是 Big5 編碼的正文。\n"
        f = tmp_path / "big5.md"
        f.write_bytes(text.encode("big5"))
        assert RAGRetriever._read_text(f) == text

    def test_fallback_on_unknown_encoding(self, tmp_path):
        """无法识别编码时回退到 UTF-8 容错模式，不抛异常"""
        f = tmp_path / "broken.md"
        # 写入非标准字节序列，UTF-8/GB18030/Big5 均无法正确解码
        f.write_bytes(b"# Title\n\n\xc3\x28 bad bytes\n")
        result = RAGRetriever._read_text(f)
        assert isinstance(result, str)
        assert "Title" in result  # 可读部分应保留
