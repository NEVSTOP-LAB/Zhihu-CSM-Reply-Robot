"""``CSM_QA`` 主入口类。

外部使用示例::

    from csm_qa import CSM_QA, Message

    qa = CSM_QA(api_key="sk-xxx")               # 默认 deepseek + 本地 embedding
    answer = qa.ask("CSM 的状态机如何切换？")

    history = [
        Message(role="user", content="CSM 是什么？"),
        Message(role="assistant", content="CSM 是 Communicable State Machine ..."),
    ]
    answer = qa.ask("那它和 JKI SM 的区别？", history=history)
"""

from __future__ import annotations

import configparser
import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Union

from csm_qa.llm import LLMClient
from csm_qa.prompts import DEFAULT_SYSTEM_PROMPT, build_system_message
from csm_qa.providers import resolve_endpoint
from csm_qa.rag import EmbeddingFunction, EmbeddingProvider, RAGRetriever
from csm_qa.types import AnswerResult, Message
from csm_qa.wiki_updater import check_and_update_wiki

logger = logging.getLogger(__name__)

# 用户可传入 ``Message`` 或裸 dict。
HistoryItem = Union[Message, dict]


class CSM_QA:
    """RAG 问答主类。

    构造后即可 :meth:`ask` 单轮问答或 :meth:`ask_detailed` 带元信息问答。

    Args:
        api_key: LLM API Key（必填）。
        provider: ``"deepseek"`` 或 ``"openai_compatible"``，默认 ``"deepseek"``。
        model: 模型名；``None`` 时取 provider 预设。``openai_compatible`` 必填。
        base_url: API base URL；``None`` 时取 provider 预设。``openai_compatible`` 必填。
        temperature: 生成温度。
        max_tokens: 单次回复的 token 上限。
        max_retries: LLM 调用重试次数。
        request_timeout: 单次 LLM 请求超时（秒）。
        wiki_dir: 知识库目录，默认 ``"csm-wiki/remote"``。
        vector_store_dir: 向量库持久化目录，默认 ``".csm_qa/vector_store"``。
        embedding_provider: ``"local"`` 或 ``"openai"``。
        embedding_model: embedding 模型名（local 默认 BAAI/bge-small-zh-v1.5）。
        embedding_api_key: ``embedding_provider="openai"`` 时使用。
        embedding_base_url: embedding 线上服务 base_url。
        top_k: RAG 返回片段数。
        similarity_threshold: 相似度阈值。
        system_prompt: 自定义 system prompt；``None`` 用内置默认（CSM/LabVIEW 场景）。
        auto_sync_wiki: 构造时若向量库为空则自动同步 wiki，默认 ``True``。
    """

    def __init__(
        self,
        api_key: str,
        *,
        provider: str = "deepseek",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 512,
        max_retries: int = 3,
        request_timeout: float = 60.0,
        wiki_dir: str | Path = "csm-wiki/remote",
        vector_store_dir: str | Path = ".csm_qa/vector_store",
        embedding_provider: EmbeddingProvider = "local",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        top_k: int = 3,
        similarity_threshold: float = 0.72,
        system_prompt: Optional[str] = None,
        auto_sync_wiki: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("api_key 不可为空")

        # ─── LLM ─────────────────────────────────────────────────
        normalized_provider = provider.strip().lower()
        resolved_base, resolved_model = resolve_endpoint(
            normalized_provider, base_url, model
        )
        self.provider = normalized_provider
        self.model = resolved_model
        self.base_url = resolved_base
        self._llm = LLMClient(
            api_key=api_key,
            base_url=resolved_base,
            model=resolved_model,
            max_tokens=max_tokens,
            temperature=temperature,
            max_retries=max_retries,
            timeout=request_timeout,
        )

        # ─── Embedding & RAG ─────────────────────────────────────
        self._embedding_fn = EmbeddingFunction(
            provider=embedding_provider,
            model=embedding_model,
            api_key=embedding_api_key,
            base_url=embedding_base_url,
        )
        self._rag = RAGRetriever(
            wiki_dir=wiki_dir,
            vector_store_dir=vector_store_dir,
            embedding_fn=self._embedding_fn,
        )

        # ─── Prompt ──────────────────────────────────────────────
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        if auto_sync_wiki and self._rag.is_empty():
            wiki_path = Path(wiki_dir)
            source_file = wiki_path.parent / "wiki_source.json"
            if not wiki_path.exists() and source_file.exists():
                # wiki 目录尚未克隆，但存在 wiki_source.json —— 触发远程同步。
                # 强制 force_sync=True 确保即使 commit_id 已与远端一致也执行 clone，
                # 避免本地目录缺失时 check_and_update_wiki 跳过拉取、RAG 仍为空。
                try:
                    check_and_update_wiki(
                        source_file=source_file,
                        local_dir=wiki_path,
                        retriever=self._rag,
                        force_sync=True,
                    )
                except Exception as exc:
                    logger.warning("auto_sync_wiki (remote) 失败（已忽略）: %s", exc)
            else:
                try:
                    self._rag.sync_wiki()
                except Exception as exc:
                    logger.warning("auto_sync_wiki 失败（已忽略）: %s", exc)

    # ─── 工厂方法 ────────────────────────────────────────────────

    @classmethod
    def from_env(cls, **overrides) -> "CSM_QA":
        """从环境变量构造。

        识别的环境变量（统一以 ``LLM_*`` 前缀，所有旧别名已移除）：

        - ``LLM_API_KEY``：LLM 服务商（DeepSeek 等）的 API Key
        - ``LLM_PROVIDER``：provider，默认 ``deepseek``
        - ``LLM_MODEL``：模型名（省略时使用 provider 预设）
        - ``LLM_BASE_URL``：base URL（省略时使用 provider 预设）

        其余参数可通过 ``**overrides`` 直接覆盖。
        """
        env = os.environ
        kwargs: dict = {
            "api_key": env.get("LLM_API_KEY", ""),
            "provider": env.get("LLM_PROVIDER", "deepseek"),
            "model": env.get("LLM_MODEL"),
            "base_url": env.get("LLM_BASE_URL"),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def from_ini(cls, config_path: str | Path, **overrides) -> "CSM_QA":
        """从 INI 配置文件构造实例。

        配置文件格式（各节均可省略，省略时使用默认值）::

            [llm]
            api_key   = sk-xxx
            provider  = deepseek
            model     = deepseek-chat
            base_url  =
            temperature     = 0.5
            max_tokens      = 512
            max_retries     = 3
            request_timeout = 60.0

            [rag]
            wiki_dir          = csm-wiki/remote
            vector_store_dir  = .csm_qa/vector_store
            top_k             = 3
            similarity_threshold = 0.72
            auto_sync_wiki    = true

            [embedding]
            provider  = local
            model     = BAAI/bge-small-zh-v1.5
            api_key   =
            base_url  =

            [prompt]
            system_prompt =

        Args:
            config_path: INI 文件路径（相对路径或绝对路径均可）。
            **overrides: 可覆盖文件中任意参数。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: api_key 未在文件中配置且未通过 overrides 提供。
        """
        path = Path(config_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")

        cp = configparser.ConfigParser(interpolation=None)
        cp.read(path, encoding="utf-8")

        def _get(section: str, key: str, fallback=None):
            return cp.get(section, key, fallback=fallback)

        def _getfloat(section: str, key: str, fallback: float) -> float:
            try:
                return cp.getfloat(section, key)
            except (configparser.NoSectionError, configparser.NoOptionError):
                return fallback

        def _getint(section: str, key: str, fallback: int) -> int:
            try:
                return cp.getint(section, key)
            except (configparser.NoSectionError, configparser.NoOptionError):
                return fallback

        def _getbool(section: str, key: str, fallback: bool) -> bool:
            try:
                return cp.getboolean(section, key)
            except (configparser.NoSectionError, configparser.NoOptionError):
                return fallback

        kwargs: dict = {
            "api_key": _get("llm", "api_key", None),
            "provider": _get("llm", "provider", "deepseek"),
            "model": _get("llm", "model"),
            "base_url": _get("llm", "base_url"),
            "temperature": _getfloat("llm", "temperature", 0.5),
            "max_tokens": _getint("llm", "max_tokens", 512),
            "max_retries": _getint("llm", "max_retries", 3),
            "request_timeout": _getfloat("llm", "request_timeout", 60.0),
            "wiki_dir": _get("rag", "wiki_dir", "csm-wiki/remote"),
            "vector_store_dir": _get("rag", "vector_store_dir", ".csm_qa/vector_store"),
            "top_k": _getint("rag", "top_k", 3),
            "similarity_threshold": _getfloat("rag", "similarity_threshold", 0.72),
            "auto_sync_wiki": _getbool("rag", "auto_sync_wiki", True),
            "embedding_provider": _get("embedding", "provider", "local"),
            "embedding_model": _get(
                "embedding", "model", "BAAI/bge-small-zh-v1.5"
            ),
            "embedding_api_key": _get("embedding", "api_key"),
            "embedding_base_url": _get("embedding", "base_url"),
            "system_prompt": _get("prompt", "system_prompt"),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    # ─── 知识库管理 ──────────────────────────────────────────────

    def sync_wiki(self, force: bool = False) -> dict:
        """手动触发 wiki 增量同步。

        Args:
            force: ``True`` 时全量重建。

        Returns:
            ``{"updated": int, "skipped": int, "removed": int}``。
        """
        return self._rag.sync_wiki(force=force)

    # ─── 问答主入口 ──────────────────────────────────────────────

    def ask(
        self,
        question: str,
        history: Optional[Iterable[HistoryItem]] = None,
    ) -> str:
        """根据问题（含历史）生成回答，仅返回文本。

        Args:
            question: 用户当前问题。
            history: 历史对话；元素可以是 :class:`Message` 或
                ``{"role": ..., "content": ...}`` 字典。

        Returns:
            模型生成的回答文本。
        """
        return self.ask_detailed(question, history=history).answer

    def ask_detailed(
        self,
        question: str,
        history: Optional[Iterable[HistoryItem]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> AnswerResult:
        """与 :meth:`ask` 一致，但返回完整的 :class:`AnswerResult`。

        Args:
            question: 用户当前问题。
            history: 历史对话。
            top_k: 覆盖默认 ``top_k``。
            similarity_threshold: 覆盖默认 ``similarity_threshold``。
        """
        if not question or not question.strip():
            raise ValueError("question 不可为空")

        # 1) 拼装检索 query：把最近一轮 user 提问也并入，提升追问场景检索质量
        retrieval_query = self._build_retrieval_query(question, history)
        contexts = self._rag.retrieve(
            retrieval_query,
            k=top_k if top_k is not None else self.top_k,
            threshold=(
                similarity_threshold
                if similarity_threshold is not None
                else self.similarity_threshold
            ),
        )

        # 2) 组装 messages
        system_content = build_system_message(self.system_prompt, contexts)
        messages: list[dict] = [{"role": "system", "content": system_content}]
        messages.extend(self._normalize_history(history))
        messages.append({"role": "user", "content": question})

        # 3) 调用 LLM
        text, usage = self._llm.chat(messages)

        return AnswerResult(
            answer=text,
            contexts=contexts,
            usage=usage,
            model=self.model,
            prompt_messages=messages,
        )

    # ─── 内部辅助 ────────────────────────────────────────────────

    @staticmethod
    def _normalize_history(
        history: Optional[Iterable[HistoryItem]],
    ) -> list[dict]:
        if not history:
            return []
        out: list[dict] = []
        for item in history:
            if isinstance(item, Message):
                out.append(item.to_openai())
            elif isinstance(item, dict) and "role" in item and "content" in item:
                role = item["role"]
                if role not in ("user", "assistant", "system"):
                    raise ValueError(f"非法 role: {role!r}")
                out.append({"role": role, "content": str(item["content"])})
            else:
                raise TypeError(
                    f"history 元素必须是 Message 或 {{role,content}} 字典，"
                    f"收到 {type(item).__name__}"
                )
        return out

    @staticmethod
    def _build_retrieval_query(
        question: str,
        history: Optional[Iterable[HistoryItem]],
    ) -> str:
        """拼接检索 query：附带最近一条 user 消息以补全追问场景的语义。"""
        if not history:
            return question
        last_user: Optional[str] = None
        for item in history:
            if isinstance(item, Message):
                role, content = item.role, item.content
            elif isinstance(item, dict):
                role, content = item.get("role"), item.get("content", "")
            else:
                continue
            if role == "user" and content:
                last_user = str(content)
        if last_user and last_user.strip() != question.strip():
            return f"{last_user}\n{question}"
        return question
