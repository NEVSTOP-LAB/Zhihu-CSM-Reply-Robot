"""RAG 检索器：单一 wiki 向量库 + 增量同步。

相对老版本 ``scripts/rag_retriever.py``，本版本：

- 移除 ``human_replies`` 双库（库语义不再针对"评论自动回复"场景）。
- 仅保留 wiki 增量索引 + 检索两个能力。
- ``EmbeddingFunction`` 支持 ``local``（默认 BAAI/bge-small-zh-v1.5）与 ``openai`` 两种。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Literal, Optional

import chromadb

logger = logging.getLogger(__name__)

EmbeddingProvider = Literal["local", "openai"]

# HuggingFace 镜像回落顺序：国内镜像 → 官方 → 用户配置（运行时拼入）。
_HF_FALLBACK_ENDPOINTS = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]


def _preview_text(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


class EmbeddingFunction:
    """Embedding 抽象，支持本地 sentence-transformers 与线上 OpenAI。"""

    def __init__(
        self,
        provider: EmbeddingProvider = "local",
        model: str = "BAAI/bge-small-zh-v1.5",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_folder: Optional[str] = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.cache_folder = cache_folder or os.environ.get(
            "SENTENCE_TRANSFORMERS_HOME"
        )
        self._local_model = None
        self._local_model_error: Optional[Exception] = None
        self._online_client = None

    # ─── 内部加载 ────────────────────────────────────────────────

    @staticmethod
    def _apply_huggingface_endpoint(endpoint: str) -> None:
        """将 ``endpoint`` 写入环境变量并同步 huggingface_hub 内部常量。"""
        os.environ["HF_ENDPOINT"] = endpoint
        try:
            import huggingface_hub.constants as _hf_constants

            _hf_constants.ENDPOINT = endpoint
        except Exception:
            pass

    def _build_hf_endpoint_candidates(self) -> list[str]:
        """构造 HuggingFace 镜像回落候选列表。

        顺序：国内镜像站 → 官方 → 用户通过 ``HF_ENDPOINT`` 配置的地址。
        用户配置地址若与前两者重复则跳过。
        """
        candidates = list(_HF_FALLBACK_ENDPOINTS)
        user_endpoint = os.environ.get("HF_ENDPOINT", "").strip()
        if user_endpoint and user_endpoint not in candidates:
            candidates.append(user_endpoint)
        return candidates

    def _create_local_model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model, cache_folder=self.cache_folder)

    def _get_local_model(self):
        if self._local_model_error is not None:
            raise RuntimeError("本地 Embedding 模型初始化失败，已停止后续重试") from self._local_model_error

        if self._local_model is not None:
            return self._local_model

        # 保存原始 HF 端点状态，确保修改不永久影响全局配置
        original_hf_env: Optional[str] = os.environ.get("HF_ENDPOINT")
        original_hf_const: Optional[str] = None
        try:
            import huggingface_hub.constants as _hf_const_mod

            original_hf_const = _hf_const_mod.ENDPOINT
        except (ImportError, AttributeError):
            _hf_const_mod = None  # type: ignore[assignment]

        candidates = self._build_hf_endpoint_candidates()
        last_exc: Optional[Exception] = None
        try:
            for endpoint in candidates:
                self._apply_huggingface_endpoint(endpoint)
                logger.info("尝试从 %s 加载本地 Embedding 模型: %s", endpoint, self.model)
                try:
                    self._local_model = self._create_local_model()
                    return self._local_model
                except Exception as exc:
                    logger.warning("从 %s 加载本地 Embedding 模型失败: %s", endpoint, exc)
                    last_exc = exc
        finally:
            # 无论成功还是失败，恢复调用前的端点配置（避免永久污染全局状态）
            if original_hf_env is None:
                os.environ.pop("HF_ENDPOINT", None)
            else:
                os.environ["HF_ENDPOINT"] = original_hf_env
            if _hf_const_mod is not None and original_hf_const is not None:
                try:
                    _hf_const_mod.ENDPOINT = original_hf_const
                except Exception:
                    logger.warning("恢复 huggingface_hub.constants.ENDPOINT 失败（已忽略）")

        self._local_model_error = last_exc
        logger.warning("本地 Embedding 模型初始化失败，所有端点均不可用，已停止后续重试")
        raise RuntimeError(
            f"本地 Embedding 模型初始化失败，所有端点均不可用: {candidates}"
        ) from last_exc

    def _get_online_client(self):
        if self._online_client is None:
            from openai import OpenAI

            self._online_client = OpenAI(
                api_key=self.api_key or os.environ.get("OPENAI_API_KEY", ""),
                base_url=self.base_url,
            )
        return self._online_client

    # ─── 公共接口 ────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]]:
        """生成文本向量（已 L2 归一化）。"""
        if self.provider == "openai":
            return self._embed_online(texts)
        return self._embed_local(texts)

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        model = self._get_local_model()
        emb = model.encode(texts, normalize_embeddings=True)
        return emb.tolist()

    def _embed_online(self, texts: list[str]) -> list[list[float]]:
        client = self._get_online_client()
        resp = client.embeddings.create(input=texts, model=self.model)
        out: list[list[float]] = []
        for item in resp.data:
            vec = list(item.embedding)
            norm = sum(x * x for x in vec) ** 0.5
            if norm > 0:
                out.append([x / norm for x in vec])
            else:
                out.append(vec)
        return out

    # ChromaDB 兼容签名（保留以备直接挂载到 collection）
    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self.embed(input)


class RAGRetriever:
    """简易 RAG 检索器：wiki 目录 → ChromaDB 持久化向量库。"""

    HASH_FILENAME = "wiki_hash.json"
    COLLECTION_NAME = "wiki"

    def __init__(
        self,
        wiki_dir: str | Path,
        vector_store_dir: str | Path,
        embedding_fn: EmbeddingFunction,
    ) -> None:
        self.wiki_dir = Path(wiki_dir)
        self.vector_store_dir = Path(vector_store_dir)
        self.embedding_fn = embedding_fn

        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.vector_store_dir)
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "csm_qa wiki vector store"},
        )

    # ─── 哈希管理（增量更新）─────────────────────────────────────

    def _hash_path(self) -> Path:
        return self.vector_store_dir / self.HASH_FILENAME

    def _load_hashes(self) -> dict[str, str]:
        p = self._hash_path()
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_hashes(self, hashes: dict[str, str]) -> None:
        with open(self._hash_path(), "w", encoding="utf-8") as f:
            json.dump(hashes, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _md5(file_path: Path) -> str:
        return hashlib.md5(file_path.read_bytes()).hexdigest()

    @staticmethod
    def _read_text(file_path: Path) -> str:
        """读取文本文件，自动处理 UTF-8 / GBK / Big5 等中文编码。"""
        raw = file_path.read_bytes()
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            pass
        try:
            from charset_normalizer import from_bytes

            result = from_bytes(raw).best()
            if result is not None:
                return str(result)
        except Exception:
            pass
        logger.warning("文件编码无法识别，UTF-8 容错: %s", file_path)
        return raw.decode("utf-8", errors="replace")

    @staticmethod
    def _chunk_markdown(content: str, source: str) -> list[dict]:
        """按 Markdown 1-3 级标题分块。"""
        chunks: list[dict] = []
        for section in re.split(r"\n(?=#{1,3}\s)", content):
            section = section.strip()
            if not section:
                continue
            m = re.match(r"^(#{1,3})\s+(.+)", section)
            heading = m.group(2) if m else "Untitled"
            chunks.append({"text": section, "source": source, "heading": heading})
        if not chunks and content.strip():
            chunks.append(
                {"text": content.strip(), "source": source, "heading": "Untitled"}
            )
        return chunks

    # ─── 公共 API ────────────────────────────────────────────────

    def is_empty(self) -> bool:
        """向量库是否为空。"""
        try:
            return self._collection.count() == 0
        except Exception:
            return True

    def close(self) -> None:
        """释放向量库底层资源，避免 Windows 下测试清理临时目录失败。"""
        try:
            self._client.close()
        except Exception:
            pass

    def sync_wiki(self, force: bool = False) -> dict:
        """同步 wiki 目录到向量库。

        Args:
            force: 为 ``True`` 时忽略 MD5 缓存，全部重建。

        Returns:
            ``{"updated": int, "skipped": int, "removed": int}``。
        """
        if not self.wiki_dir.exists():
            logger.warning("wiki 目录不存在: %s", self.wiki_dir)
            return {"updated": 0, "skipped": 0, "removed": 0}

        old_hashes = {} if force else self._load_hashes()
        new_hashes: dict[str, str] = {}
        updated = 0
        skipped = 0

        for md_file in sorted(self.wiki_dir.glob("**/*.md")):
            rel = str(md_file.relative_to(self.wiki_dir))
            new_hash = self._md5(md_file)
            if not force and old_hashes.get(rel) == new_hash:
                new_hashes[rel] = new_hash
                skipped += 1
                continue

            content = self._read_text(md_file)
            chunks = self._chunk_markdown(content, rel)
            if not chunks:
                # 文件存在但内容为空/无法分块：仍记录 hash 以避免重复处理，
                # 并清理该 source 在向量库中的旧片段（如果有）。
                try:
                    existing = self._collection.get(where={"source": rel})
                    if existing and existing["ids"]:
                        self._collection.delete(ids=existing["ids"])
                except Exception as exc:
                    logger.debug("删除空文档旧向量出错（可能不存在）: %s", exc)
                new_hashes[rel] = new_hash
                updated += 1
                continue

            texts = [c["text"] for c in chunks]
            try:
                embeddings = self.embedding_fn.embed(texts)
            except Exception as exc:
                logger.warning("embedding 失败，保留旧索引并跳过 %s: %s", rel, exc)
                if rel in old_hashes:
                    new_hashes[rel] = old_hashes[rel]
                continue

            # 先删旧片段再写入新片段
            try:
                existing = self._collection.get(where={"source": rel})
                if existing and existing["ids"]:
                    self._collection.delete(ids=existing["ids"])
            except Exception as exc:
                logger.debug("删除旧向量出错（可能不存在）: %s", exc)

            ids = [f"{rel}#{i}" for i in range(len(chunks))]
            metadatas = [
                {"source": c["source"], "heading": c["heading"]} for c in chunks
            ]
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            new_hashes[rel] = new_hash
            updated += 1

        # 删除已被移除的文件对应的向量
        removed_files = set(old_hashes.keys()) - set(new_hashes.keys())
        for removed in removed_files:
            try:
                existing = self._collection.get(where={"source": removed})
                if existing and existing["ids"]:
                    self._collection.delete(ids=existing["ids"])
            except Exception as exc:
                logger.debug("删除移除文件向量出错: %s", exc)

        self._save_hashes(new_hashes)
        result = {
            "updated": updated,
            "skipped": skipped,
            "removed": len(removed_files),
        }
        logger.info("wiki 同步完成: %s", result)
        return result

    def retrieve(
        self,
        query: str,
        k: int = 3,
        threshold: float = 0.72,
    ) -> list[str]:
        """检索 top-k 个相关片段，仅返回文本（向后兼容）。

        Args:
            query: 查询文本。
            k: 最多返回片段数。
            threshold: 余弦相似度阈值，低于该阈值的片段被过滤。

        Returns:
            按相关度排序的文档片段文本列表。
        """
        return [hit["text"] for hit in self.retrieve_with_meta(query, k=k, threshold=threshold)]

    def retrieve_with_meta(
        self,
        query: str,
        k: int = 3,
        threshold: float = 0.72,
    ) -> list[dict]:
        """检索 top-k 个相关片段，并附带元数据。

        Args:
            query: 查询文本。
            k: 最多返回片段数。
            threshold: 余弦相似度阈值，低于该阈值的片段被过滤。

        Returns:
            按相关度排序的字典列表，每项包含 ``text``/``source``/``heading``/``similarity``。
        """
        if not query or not query.strip():
            return []
        try:
            count = self._collection.count()
        except Exception:
            count = 0
        if count == 0:
            return []

        try:
            query_embedding = self.embedding_fn.embed([query])[0]
        except Exception as exc:
            logger.warning("embedding 查询向量生成失败，跳过 RAG 检索: %s", exc)
            return []
        actual_k = min(k, count)
        try:
            res = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_k,
            )
        except Exception as exc:
            logger.warning("向量检索失败: %s", exc)
            return []

        out: list[dict] = []
        hit_logs: list[str] = []
        if res and res.get("documents"):
            metadatas_list = res.get("metadatas") or []
            for docs, distances, metadatas in zip(
                res["documents"],
                res["distances"],
                metadatas_list or ([[]] * len(res["documents"])),
            ):
                for index, (doc, dist) in enumerate(zip(docs, distances), start=1):
                    # ChromaDB 默认返回 L2 距离，归一化向量下：
                    # cosine = 1 - dist^2 / 2
                    similarity = 1 - (dist ** 2) / 2
                    if similarity >= threshold:
                        metadata = (
                            metadatas[index - 1]
                            if index - 1 < len(metadatas)
                            else {}
                        ) or {}
                        source = metadata.get("source", "(unknown)")
                        heading = metadata.get("heading", "Untitled")
                        out.append(
                            {
                                "text": doc,
                                "source": source,
                                "heading": heading,
                                "similarity": similarity,
                            }
                        )
                        hit_logs.append(
                            f"#{index} source={source} heading={heading} "
                            f"similarity={similarity:.3f} preview={_preview_text(doc)}"
                        )
        if hit_logs:
            logger.info(
                "RAG 命中 %d 条: %s",
                len(hit_logs),
                " | ".join(hit_logs[:k]),
            )
        else:
            logger.info(
                "RAG 未命中: threshold=%.2f query=%s",
                threshold,
                _preview_text(query, limit=60),
            )
        return out[:k]
