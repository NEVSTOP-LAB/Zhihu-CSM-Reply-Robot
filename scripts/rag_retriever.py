# -*- coding: utf-8 -*-
"""
RAG 检索器 — ChromaDB + BGE Embedding
======================================

实施计划关联：AI-005 RAGRetriever — Wiki 索引与检索
参考文档：docs/调研/04-CSM-Wiki-RAG知识库.md

功能：
- CSM Wiki 文档增量 embedding（MD5 比对）
- 混合检索（真人回复优先 + Wiki 补充）
- 本地/线上 embedding 双模式支持

使用方式：
    retriever = RAGRetriever(
        wiki_dir="csm-wiki/",
        vector_store_dir="data/vector_store/",
        reply_index_dir="data/reply_index/",
    )
    retriever.sync_wiki()
    results = retriever.retrieve("如何处理客户投诉？")
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import chromadb

logger = logging.getLogger(__name__)


class EmbeddingFunction:
    """Embedding 函数封装

    实施计划关联：AI-005 任务 1

    支持两种模式：
    - 本地模式：BAAI/bge-small-zh-v1.5（免费，需 sentence-transformers）
    - 线上模式：text-embedding-3-small（需 OPENAI_API_KEY）

    Args:
        use_online: 是否使用线上 embedding
        model_name: 本地模型名称
        online_model: 线上模型名称
    """

    def __init__(
        self,
        use_online: bool = False,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        online_model: str = "text-embedding-3-small",
    ):
        self.use_online = use_online
        self._local_model = None
        self._online_client = None
        self.model_name = model_name
        self.online_model = online_model

    def _get_local_model(self):
        """延迟加载本地 Embedding 模型"""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"加载本地 Embedding 模型: {self.model_name}")
            self._local_model = SentenceTransformer(self.model_name)
        return self._local_model

    def _get_online_client(self):
        """延迟加载线上 Embedding 客户端"""
        if self._online_client is None:
            from openai import OpenAI
            self._online_client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            )
        return self._online_client

    def embed(self, texts: list[str]) -> list[list[float]]:
        """生成文本向量

        Args:
            texts: 待 embedding 的文本列表

        Returns:
            向量列表（每个向量为 float 列表）
        """
        if self.use_online:
            return self._embed_online(texts)
        return self._embed_local(texts)

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """使用本地模型生成向量"""
        model = self._get_local_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def _embed_online(self, texts: list[str]) -> list[list[float]]:
        """使用线上 API 生成向量，并显式 L2 归一化（FIX-07）"""
        client = self._get_online_client()
        response = client.embeddings.create(
            input=texts,
            model=self.online_model,
        )
        raw = [item.embedding for item in response.data]
        # 显式 L2 归一化，确保与本地 normalize_embeddings=True 一致
        # 参考 FIX-07：在线 embedding 未归一化时余弦相似度公式失效
        normalized = []
        for vec in raw:
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                normalized.append([x / norm for x in vec])
            else:
                normalized.append(vec)
        return normalized

    def __call__(self, input: list[str]) -> list[list[float]]:
        """ChromaDB 兼容的调用接口"""
        return self.embed(input)


class RAGRetriever:
    """RAG 检索器

    实施计划关联：AI-005 RAGRetriever — Wiki 索引与检索

    结合 CSM Wiki 知识库和历史真人回复索引，为 LLM 提供
    相关上下文。真人回复索引优先级高于 Wiki（自学习闭环）。

    Args:
        wiki_dir: CSM Wiki Markdown 文件目录
        vector_store_dir: ChromaDB Wiki 向量库存储路径
        reply_index_dir: ChromaDB 真人回复索引存储路径
        use_online_embedding: 是否使用线上 embedding
        embedding_model: 本地 Embedding 模型名称
        embedding_fallback: 线上兜底 Embedding 模型名称
    """

    # Wiki hash 文件路径（用于增量更新检测）
    WIKI_HASH_FILENAME = "wiki_hash.json"

    def __init__(
        self,
        wiki_dir: str,
        vector_store_dir: str,
        reply_index_dir: str,
        use_online_embedding: bool = False,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        embedding_fallback: str = "text-embedding-3-small",
    ):
        self.wiki_dir = Path(wiki_dir)
        self.vector_store_dir = Path(vector_store_dir)
        self.reply_index_dir = Path(reply_index_dir)

        # 初始化 Embedding 函数
        self.embedding_fn = EmbeddingFunction(
            use_online=use_online_embedding,
            model_name=embedding_model,
            online_model=embedding_fallback,
        )

        # 确保目录存在
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.reply_index_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 ChromaDB 客户端
        # 参考 AI-005 任务 5：向量库路径可配置
        self._wiki_client = chromadb.PersistentClient(
            path=str(self.vector_store_dir)
        )
        self._reply_client = chromadb.PersistentClient(
            path=str(self.reply_index_dir)
        )

        # 获取或创建集合
        self._wiki_collection = self._wiki_client.get_or_create_collection(
            name="csm_wiki",
            metadata={"description": "CSM Wiki 知识库向量索引"},
        )
        self._reply_collection = self._reply_client.get_or_create_collection(
            name="human_replies",
            metadata={"description": "真人回复高权重索引"},
        )

    def _load_wiki_hashes(self) -> dict[str, str]:
        """加载 Wiki 文件 MD5 哈希记录

        返回 {文件路径: MD5 哈希} 的字典，
        用于增量更新检测（只处理变更文件）。
        """
        hash_path = self.vector_store_dir / self.WIKI_HASH_FILENAME
        if hash_path.exists():
            with open(hash_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_wiki_hashes(self, hashes: dict[str, str]):
        """保存 Wiki 文件 MD5 哈希记录"""
        hash_path = self.vector_store_dir / self.WIKI_HASH_FILENAME
        with open(hash_path, "w", encoding="utf-8") as f:
            json.dump(hashes, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _compute_md5(file_path: Path) -> str:
        """计算文件 MD5 哈希（二进制模式，兼容非 UTF-8 文件）"""
        return hashlib.md5(file_path.read_bytes()).hexdigest()

    @staticmethod
    def _read_text(file_path: Path) -> str:
        """读取文本文件，自动检测编码，支持 UTF-8 / GB18030 / Big5 等中文编码。

        检测顺序：
        1. 尝试 UTF-8（最常见，速度最快）
        2. 使用 charset-normalizer 自动检测编码（支持 GB18030/GBK/Big5 等）
        3. 最终回退：以 UTF-8 容错模式读取
        """
        raw = file_path.read_bytes()

        # 1. 优先尝试 UTF-8
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            pass

        # 2. 使用 charset-normalizer 检测编码（随 requests 一起安装）
        try:
            from charset_normalizer import from_bytes
            result = from_bytes(raw).best()
            if result is not None:
                return str(result)
        except Exception:
            pass

        # 3. 最终回退：UTF-8 容错模式
        logger.warning(f"文件编码无法识别，以 UTF-8 容错模式读取: {file_path}")
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _chunk_markdown(content: str, source: str) -> list[dict]:
        """按 Markdown 标题分块

        实施计划关联：AI-005 任务 2（按标题分块）
        参考：docs/调研/04-CSM-Wiki-RAG知识库.md

        将 Markdown 文档按 # 标题分割为独立的语义块，
        每个块保留标题和内容，约 300-500 tokens。

        Args:
            content: Markdown 文本
            source: 源文件路径（用于元数据追溯）

        Returns:
            分块列表，每个块包含 text, source, heading
        """
        chunks = []
        # 按一级/二级/三级标题分割
        sections = re.split(r'\n(?=#{1,3}\s)', content)

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # 提取标题
            heading_match = re.match(r'^(#{1,3})\s+(.+)', section)
            heading = heading_match.group(2) if heading_match else "Untitled"

            chunks.append({
                "text": section,
                "source": source,
                "heading": heading,
            })

        # 如果没有标题分割，整个文档作为一个块
        if not chunks and content.strip():
            chunks.append({
                "text": content.strip(),
                "source": source,
                "heading": "Document",
            })

        return chunks

    def sync_wiki(self, force: bool = False):
        """同步 CSM Wiki 到向量库

        实施计划关联：AI-005 任务 2

        使用 MD5 哈希比对实现增量更新：
        - force=True 时强制重建所有索引
        - 只处理内容有变化的文件
        - 删除已移除的文件对应的向量

        Args:
            force: 是否强制重建（忽略 MD5 缓存）
        """
        if not self.wiki_dir.exists():
            logger.warning(f"Wiki 目录不存在: {self.wiki_dir}")
            return

        old_hashes = {} if force else self._load_wiki_hashes()
        new_hashes: dict[str, str] = {}
        updated_count = 0
        skipped_count = 0

        # 遍历所有 Markdown 文件
        md_files = list(self.wiki_dir.glob("**/*.md"))

        for md_file in md_files:
            rel_path = str(md_file.relative_to(self.wiki_dir))
            new_hash = self._compute_md5(md_file)

            # 跳过未变更的文件
            if not force and old_hashes.get(rel_path) == new_hash:
                # 文件未变更：new_hash 与旧 hash 相同，写入 new_hashes 供最终保存
                new_hashes[rel_path] = new_hash
                skipped_count += 1
                continue

            # 分块并 embedding
            content = self._read_text(md_file)
            chunks = self._chunk_markdown(content, rel_path)

            if chunks:
                texts = [c["text"] for c in chunks]

                # FIX-23/REV-5：先 embedding，成功后再更新 new_hashes 和删旧向量。
                # embedding 失败时不更新 new_hashes，保证下次运行可以重试。
                try:
                    embeddings = self.embedding_fn.embed(texts)
                except Exception as e:
                    logger.warning(f"embedding 失败，跳过 {rel_path}: {e}")
                    # 失败时保留旧哈希（如有），确保下次还能重试
                    if rel_path in old_hashes:
                        new_hashes[rel_path] = old_hashes[rel_path]
                    continue

                # embedding 成功后删除旧向量
                try:
                    existing = self._wiki_collection.get(
                        where={"source": rel_path}
                    )
                    if existing and existing["ids"]:
                        self._wiki_collection.delete(ids=existing["ids"])
                except Exception as e:
                    logger.debug(f"删除旧向量时出错（可能不存在）: {e}")

                ids = [f"{rel_path}#{i}" for i in range(len(chunks))]
                metadatas = [
                    {"source": c["source"], "heading": c["heading"]}
                    for c in chunks
                ]

                self._wiki_collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                # embedding 和写入均成功，才记录新哈希（REV-5）
                new_hashes[rel_path] = new_hash
                updated_count += 1

        # 删除已移除的文件对应的向量
        removed_files = set(old_hashes.keys()) - set(new_hashes.keys())
        for removed in removed_files:
            try:
                existing = self._wiki_collection.get(
                    where={"source": removed}
                )
                if existing and existing["ids"]:
                    self._wiki_collection.delete(ids=existing["ids"])
                    logger.info(f"已删除移除文件的向量: {removed}")
            except Exception as e:
                logger.debug(f"删除移除文件向量时出错: {e}")

        self._save_wiki_hashes(new_hashes)
        logger.info(
            f"Wiki 同步完成: 更新 {updated_count} 文件，"
            f"跳过 {skipped_count} 文件，"
            f"删除 {len(removed_files)} 文件"
        )

    def retrieve(
        self,
        query: str,
        k: int = 3,
        threshold: float = 0.72,
    ) -> list[str]:
        """检索相关文档片段

        实施计划关联：AI-005 任务 3

        混合检索策略：先从 reply_index 取 top-2 真人回复，
        再从 wiki 补足 top-(k-2) 条，确保总量不超过 k。
        真人回复优先级高于 Wiki（自学习闭环）。

        Args:
            query: 查询文本
            k: 返回结果数量
            threshold: 相似度阈值（低于此值不返回）

        Returns:
            相关文档片段列表
        """
        results: list[str] = []
        query_embedding = self.embedding_fn.embed([query])[0]

        # 1. 先从 reply_index 检索（真人回复优先）
        # 参考 AI-005 任务 3：reply_index top-2
        reply_k = min(2, k)
        try:
            reply_count = self._reply_collection.count()
            if reply_count > 0:
                actual_reply_k = min(reply_k, reply_count)
                reply_results = self._reply_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=actual_reply_k,
                )
                if reply_results and reply_results["documents"]:
                    for docs, distances in zip(
                        reply_results["documents"],
                        reply_results["distances"],
                    ):
                        for doc, dist in zip(docs, distances):
                            # ChromaDB 返回 L2 距离，对于 L2 归一化向量:
                            # dist^2 = 2 - 2*cosine，因此 cosine = 1 - dist^2/2
                            similarity = 1 - (dist ** 2) / 2
                            if similarity >= threshold:
                                results.append(doc)
        except Exception as e:
            logger.debug(f"reply_index 检索出错: {e}")

        # 2. 从 wiki 补足剩余
        wiki_k = k - len(results)
        if wiki_k > 0:
            try:
                wiki_count = self._wiki_collection.count()
                if wiki_count > 0:
                    actual_wiki_k = min(wiki_k, wiki_count)
                    wiki_results = self._wiki_collection.query(
                        query_embeddings=[query_embedding],
                        n_results=actual_wiki_k,
                    )
                    if wiki_results and wiki_results["documents"]:
                        for docs, distances in zip(
                            wiki_results["documents"],
                            wiki_results["distances"],
                        ):
                            for doc, dist in zip(docs, distances):
                                # cosine = 1 - dist^2/2（L2 归一化向量）
                                similarity = 1 - (dist ** 2) / 2
                                if similarity >= threshold:
                                    results.append(doc)
            except Exception as e:
                logger.debug(f"wiki 检索出错: {e}")

        return results[:k]

    def index_human_reply(
        self,
        question: str,
        reply: str,
        article_id: str,
        thread_id: str,
    ):
        """索引真人回复

        实施计划关联：AI-005 任务 4, AI-013

        将作者的真人回复写入 reply_index，标记 weight=high，
        后续检索时优先返回。形成"自学习闭环"。

        Args:
            question: 用户的原始问题
            reply: 作者的回复内容
            article_id: 所属文章 ID
            thread_id: 所属线程 ID
        """
        # 组合 QA 对作为索引文本
        qa_text = f"问题：{question}\n回复：{reply}"
        embedding = self.embedding_fn.embed([qa_text])[0]

        doc_id = f"reply_{article_id}_{thread_id}_{hashlib.md5(qa_text.encode()).hexdigest()[:8]}"

        self._reply_collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[qa_text],
            metadatas=[{
                "article_id": article_id,
                "thread_id": thread_id,
                "weight": "high",
                "type": "human_reply",
            }],
        )

        logger.info(
            f"已索引真人回复: article={article_id}, thread={thread_id}"
        )
