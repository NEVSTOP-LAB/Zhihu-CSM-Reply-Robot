# -*- coding: utf-8 -*-
"""
主入口 — CSM RAG/LLM 问答机器人
=================================

串联所有模块的主流程：
1. 加载配置，初始化模块
2. 每日处理量检查
3. 读取收件箱 → 过滤 → RAG → LLM → pending/
4. 检测专家回复 → 索引
5. 异常告警（预算）

使用方式：
    python scripts/run_bot.py
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# 确保可以导入 scripts 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.rag_retriever import RAGRetriever
from scripts.llm_client import LLMClient, BudgetExceededError
from scripts.thread_manager import ThreadManager
from scripts.comment_filter import CommentFilter
from scripts.alerting import AlertManager
from scripts.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


# ─── 数据模型 ──────────────────────────────────────────────────

@dataclass
class Message:
    """通用消息数据模型

    Attributes:
        id: 消息 ID
        content: 消息内容
        author: 消息作者
        created_time: 创建时间戳（Unix 秒）
        parent_id: 父消息 ID（追问时非空）
        is_author_reply: 是否为专家/维护者的权威回复（True 时仅索引，不生成 AI 回复）
    """
    id: str
    content: str
    author: str
    created_time: int
    parent_id: Optional[str] = None
    is_author_reply: bool = False


class RecentLogsHandler(logging.Handler):
    """内存日志缓冲处理器，保存最近 N 条格式化日志记录

    实施计划关联：AI-010 告警模块

    在告警 Issue 创建时附带最近日志，方便快速定位问题。

    Args:
        maxlen: 最多保留的日志条数
    """

    _LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    def __init__(self, maxlen: int = 100):
        super().__init__()
        self._records: deque[str] = deque(maxlen=maxlen)
        self.setFormatter(logging.Formatter(self._LOG_FORMAT))

    def emit(self, record: logging.LogRecord):
        try:
            self._records.append(self.format(record))
        except Exception:
            self.handleError(record)

    def format_markdown(self, n: int = 50) -> str:
        """返回 Markdown 代码块格式的最近 n 条日志"""
        logs = list(self._records)[-n:]
        if not logs:
            return ""
        return "```\n" + "\n".join(logs) + "\n```"


class BotRunner:
    """主流程控制器

    串联所有模块，实现 pending/ 人工审核流程。

    Args:
        project_root: 项目根目录路径
    """

    def __init__(self, project_root: str | Path = "."):
        self.root = Path(project_root)
        self.settings: dict = {}

        # 模块实例（延迟初始化）
        self.rag_retriever: Optional[RAGRetriever] = None
        self.llm_client: Optional[LLMClient] = None
        self.thread_manager: Optional[ThreadManager] = None
        self.comment_filter: Optional[CommentFilter] = None
        self.alert_manager: Optional[AlertManager] = None
        self.cost_tracker: Optional[CostTracker] = None

        # 运行状态
        self._processed_count = 0
        self._consecutive_failures = 0
        self._seen_ids: set[str] = set()
        self._seen_ids_path = self.root / "data" / "seen_ids.json"

        # 评论 ID → 线程根 ID 映射（Issue #3：修复多级回复的线程归档逻辑）
        # 用于在多级嵌套回复中正确追溯到最顶层的根评论，将同一对话归入同一线程文件
        self._comment_thread_map: dict[str, str] = {}
        self._comment_thread_map_path = self.root / "data" / "comment_thread_map.json"

        # 内存日志缓冲（用于告警 Issue 附带最近日志，方便快速定位问题）
        self._log_handler = RecentLogsHandler(maxlen=100)
        logging.getLogger().addHandler(self._log_handler)

    def load_config(self):
        """加载配置文件

        从 config/settings.yaml 读取配置。
        """
        settings_path = self.root / "config" / "settings.yaml"

        with open(settings_path, "r", encoding="utf-8") as f:
            self.settings = yaml.safe_load(f)

        logger.info(
            f"配置加载完成: "
            f"每日上限 {self.settings['bot']['max_new_comments_per_day']}"
        )

    def init_modules(self):
        """初始化所有模块

        从环境变量和配置文件创建各模块实例。
        """
        # RAG 检索器
        rag_cfg = self.settings.get("rag", {})
        self.rag_retriever = RAGRetriever(
            wiki_dir=str(self.root / "csm-wiki"),
            vector_store_dir=str(self.root / "data" / "vector_store"),
            reply_index_dir=str(self.root / "data" / "reply_index"),
            use_online_embedding=rag_cfg.get("use_online_embedding", False),
            embedding_model=rag_cfg.get(
                "embedding_model", "BAAI/bge-small-zh-v1.5"
            ),
        )

        # LLM 客户端
        llm_cfg = self.settings.get("llm", {})
        bot_cfg = self.settings.get("bot", {})
        self.llm_client = LLMClient(
            api_key=os.environ.get("LLM_API_KEY", ""),
            base_url=os.environ.get(
                "LLM_BASE_URL", llm_cfg.get("base_url", "")
            ),
            model=os.environ.get("LLM_MODEL", llm_cfg.get("model", "")),
            max_tokens=llm_cfg.get("max_tokens", 250),
            temperature=llm_cfg.get("temperature", 0.7),
            budget_usd_per_day=bot_cfg.get("llm_budget_usd_per_day", 0.50),
        )

        # 线程管理器
        self.thread_manager = ThreadManager(
            archive_dir=str(self.root / "archive")
        )

        # 消息过滤器（data_dir 用于持久化 dedup 缓存）
        self.comment_filter = CommentFilter(
            settings=self.settings,
            data_dir=str(self.root / "data"),
        )

        # 告警管理器
        self.alert_manager = AlertManager(
            github_token=os.environ.get("GITHUB_TOKEN", ""),
            repo=os.environ.get("GITHUB_REPOSITORY", ""),
            health_file=str(self.root / "data" / "health.json"),
        )

        # 费用追踪器
        self.cost_tracker = CostTracker(
            data_dir=str(self.root / "data")
        )

        logger.info("所有模块初始化完成")

    def load_seen_ids(self):
        """加载已处理的评论 ID 集合，支持结构校验和迁移"""
        self._seen_ids = set()

        if self._seen_ids_path.exists():
            needs_migration = False
            with open(self._seen_ids_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                self._seen_ids = {
                    item for item in data if isinstance(item, str)
                }
            elif isinstance(data, dict) and isinstance(data.get("seen_ids"), list):
                self._seen_ids = {
                    item for item in data["seen_ids"] if isinstance(item, str)
                }
                needs_migration = True
            elif isinstance(data, dict):
                # 旧格式 dict（如 {"articles": {}, "last_run": ...}），忽略并重置
                logger.warning(
                    "seen_ids 文件格式无效，已重置为空: %s",
                    self._seen_ids_path,
                )
                needs_migration = True
            else:
                logger.warning(
                    "seen_ids 文件格式无效，已重置为空: %s",
                    self._seen_ids_path,
                )

            if needs_migration:
                self.save_seen_ids()
                logger.info("已将 seen_ids 文件迁移为标准列表格式: %s", self._seen_ids_path)

        logger.info(f"已加载 {len(self._seen_ids)} 个已处理消息 ID")

    def save_seen_ids(self):
        """保存已处理的消息 ID 集合"""
        self._seen_ids_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._seen_ids_path, "w", encoding="utf-8") as f:
            json.dump(sorted(self._seen_ids), f)

    def load_comment_thread_map(self):
        """加载评论 ID → 线程根 ID 映射（Issue #3）"""
        self._comment_thread_map = {}
        if self._comment_thread_map_path.exists():
            with open(self._comment_thread_map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._comment_thread_map = {
                    str(k): str(v) for k, v in data.items()
                }
        logger.info("已加载 %d 条评论线程映射", len(self._comment_thread_map))

    def save_comment_thread_map(self):
        """保存评论 ID → 线程根 ID 映射（Issue #3）"""
        self._comment_thread_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._comment_thread_map_path, "w", encoding="utf-8") as f:
            json.dump(self._comment_thread_map, f, ensure_ascii=False)

    def _get_thread_root_id(self, message) -> str:
        """确定消息所属对话线程的根 ID

        沿映射链迭代追溯到真正的根消息，并做路径压缩以提升后续查询效率。
        内置循环检测防止坏数据导致死循环。

        Args:
            message: Message 对象

        Returns:
            线程根消息 ID（str）
        """
        if message.parent_id is None:
            # 顶级消息，自身即为线程根
            return message.id

        # 从 parent_id 开始向上迭代追溯
        current = message.parent_id
        visited: set[str] = set()

        while True:
            if current in visited:
                # 坏数据导致循环，退出并以当前节点作为根
                logger.warning("消息映射出现循环，message_id=%s，以 %s 作为线程根", message.id, current)
                break
            visited.add(current)

            parent = self._comment_thread_map.get(current)
            if parent is None or parent == current:
                # 已到达链的顶端
                break
            current = parent

        # 路径压缩：将所有经过的中间节点直接指向最终根，加速后续同链查询
        for node in visited:
            if self._comment_thread_map.get(node) != current:
                self._comment_thread_map[node] = current

        return current

    def _make_root_comment_info(self, message) -> dict:
        """构建 get_or_create_thread 所需的 root_comment 字典

        当前消息是顶级消息时，用其实际作者信息；
        当前消息是回复时，线程应已由顶级消息创建，仅传入 ID 即可。

        Args:
            message: Message 对象

        Returns:
            包含 id 和 author 的字典
        """
        thread_root_id = self._get_thread_root_id(message)
        # 仅顶级消息时才用真实作者，避免用回复者作者覆盖已有线程的元信息
        author = message.author if message.parent_id is None else "unknown"
        return {"id": thread_root_id, "author": author}

    def _check_daily_limit(self) -> bool:
        """检查是否超过每日处理上限

        Returns:
            True 如果未超限，False 如果已达上限
        """
        limit = self.settings["bot"]["max_new_comments_per_day"]
        if self._processed_count >= limit:
            logger.warning(
                f"已达每日上限 ({limit})，跳过剩余评论"
            )
            return False
        return True

    def _write_pending(
        self,
        topic: dict,
        comment_content: str,
        reply_content: str,
        comment_id: str,
        risk_reason: str = "",
    ):
        """将待审核回复写入 pending/ 目录

        Args:
            topic: 话题信息（id, title 等）
            comment_content: 原始消息内容
            reply_content: 生成的回复
            comment_id: 消息 ID
            risk_reason: 备注理由
        """
        pending_dir = self.root / "data" / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)

        topic_id = re.sub(r"[^\w\-]", "_", topic.get("id", "inbox"))
        safe_comment_id = re.sub(r"[^\w\-]", "_", str(comment_id))
        filename = f"{topic_id}_{safe_comment_id}.md"
        filepath = pending_dir / filename

        # 使用 yaml.dump 生成 frontmatter，避免特殊字符导致 YAML 解析错误
        meta = {
            "article_id": topic_id,
            "article_title": topic.get("title", ""),
            "comment_id": comment_id,
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "pending",
            "risk_reason": risk_reason,
        }
        meta_yaml = yaml.dump(meta, allow_unicode=True, default_flow_style=False)

        content = (
            f"---\n"
            f"{meta_yaml}"
            f"---\n\n"
            f"## 原始消息\n\n"
            f"> {comment_content}\n\n"
            f"## 生成的回复\n\n"
            f"{reply_content}\n"
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"待审核回复已写入: {filepath.name}")

    def process_messages(self, messages: list[Message], topic: dict = None):
        """处理一批消息，通过 RAG/LLM 生成回复并写入 pending/ 目录

        流程：过滤已处理 → RAG → LLM → pending/

        Args:
            messages: 消息列表
            topic: 可选的话题元信息（id, title, url, content 等）
        """
        if topic is None:
            topic = {}

        topic_id = topic.get("id", "inbox")
        topic_meta = {
            "title": topic.get("title", ""),
            "url": topic.get("url", ""),
        }

        logger.info(f"处理话题: {topic.get('title', topic_id)}")

        # 过滤已处理的消息
        new_messages = [
            m for m in messages if m.id not in self._seen_ids
        ]

        if not new_messages:
            logger.info(f"话题 {topic_id}: 无新消息")
            return

        logger.info(f"话题 {topic_id}: 发现 {len(new_messages)} 条新消息")

        # 获取话题摘要（缓存）— 使用 LLM 生成简短摘要，便于 AI 理解上下文
        topic_summary = ""
        if self.llm_client:
            try:
                # 无正文时直接用标题，不浪费 API 调用
                topic_content = topic.get("content", "")
                topic_title = topic.get("title", "")
                if not topic_content:
                    topic_summary = topic_title
                else:
                    topic_summary = self.llm_client.summarize_article(
                        title=topic_title,
                        content=topic_content,
                    )
            except Exception as e:
                logger.warning(f"话题摘要生成失败: {e}")

        # 将摘要写入 topic_meta，供线程管理器使用
        topic_meta["summary"] = topic_summary

        # 本次 run 处理计数
        max_per_run = self.settings.get("bot", {}).get("max_new_comments_per_run", 0)
        run_processed = 0

        for message in new_messages:
            if not self._check_daily_limit():
                break

            # 单次运行上限检查
            if max_per_run > 0 and run_processed >= max_per_run:
                logger.info(
                    "已达单次运行上限 %d 条，停止处理", max_per_run
                )
                break

            try:
                self._process_single_message(
                    topic=topic,
                    topic_meta=topic_meta,
                    message=message,
                    topic_summary=topic_summary,
                )
                self._consecutive_failures = 0
                run_processed += 1
            except BudgetExceededError as e:
                logger.warning(f"预算超限: {e}")
                raise  # 向上传播，由 run_inbox 处理告警
            except Exception as e:
                self._consecutive_failures += 1
                logger.error(f"处理消息失败: {e}")

                fail_limit = self.settings.get("alerting", {}).get(
                    "consecutive_fail_limit", 3
                )
                if self._consecutive_failures >= fail_limit:
                    logger.error(f"连续失败 {self._consecutive_failures} 次，暂停")
                    if self.alert_manager:
                        self.alert_manager.alert_consecutive_failures(
                            self._consecutive_failures,
                            recent_logs=self._log_handler.format_markdown(50),
                        )
                    break

    def _process_single_message(
        self,
        topic: dict,
        topic_meta: dict,
        message: Message,
        topic_summary: str,
    ):
        """处理单条消息

        Args:
            topic: 话题配置
            topic_meta: 话题元信息
            message: Message 对象
            topic_summary: 话题摘要
        """
        # 尽早更新消息线程映射：无论该消息是否被过滤，都应记录其所属线程根，
        # 确保后续的子消息/嵌套回复能通过映射找到正确的根消息。
        thread_root_id = self._get_thread_root_id(message)
        self._comment_thread_map[message.id] = thread_root_id

        # 检测专家回复 → 索引
        if message.is_author_reply and self.rag_retriever:
            self._handle_expert_reply(topic, topic_meta, message)
            self._seen_ids.add(message.id)
            return

        # 白名单用户：仅记录到线程，不做 AI 处理
        whitelist = self.settings.get("bot", {}).get("whitelist_users", [])
        if message.author in whitelist:
            self._handle_whitelist_message(topic, topic_meta, message)
            self._seen_ids.add(message.id)
            return

        # 前置过滤
        comment_dict = {
            "content": message.content,
            "author": message.author,
            "created_time": message.created_time,
        }

        if self.comment_filter:
            skip, reason = self.comment_filter.should_skip(comment_dict)
            if skip:
                logger.info(f"跳过消息 {message.id}: {reason}")
                # 即使被过滤，也记录到对话线程以备后期学习
                if self.thread_manager:
                    _skip_thread_path = self.thread_manager.get_or_create_thread(
                        article_id=topic.get("id", "inbox"),
                        root_comment=self._make_root_comment_info(message),
                        article_meta=topic_meta,
                    )
                    self.thread_manager.append_turn(
                        thread_path=_skip_thread_path,
                        author=message.author,
                        content=message.content,
                        comment_id=message.id,
                        is_followup=message.parent_id is not None,
                    )
                self._seen_ids.add(message.id)
                return

            # 超长截断
            comment_dict["content"] = (
                self.comment_filter.truncate_if_needed(comment_dict["content"])
            )

        # 按消息内容检索 Wiki 上下文（每条消息独立检索）
        context_chunks: list[str] = []
        if self.rag_retriever:
            context_chunks = self.rag_retriever.retrieve(
                query=comment_dict["content"],
                k=self.settings.get("rag", {}).get("top_k", 3),
                threshold=self.settings.get("rag", {}).get(
                    "similarity_threshold", 0.72
                ),
            )

        # 构建线程和历史上下文
        history_messages = []
        thread_path = None
        if self.thread_manager:
            thread_path = self.thread_manager.get_or_create_thread(
                article_id=topic.get("id", "inbox"),
                root_comment=self._make_root_comment_info(message),
                article_meta=topic_meta,
            )

            # 追加用户消息
            self.thread_manager.append_turn(
                thread_path=thread_path,
                author=message.author,
                content=message.content,
                comment_id=message.id,
                is_followup=message.parent_id is not None,
            )

            # 获取历史上下文
            max_turns = self.settings.get("rag", {}).get("history_turns", 6)
            history_messages = self.thread_manager.build_context_messages(
                thread_path, max_turns=max_turns
            )

        # LLM 生成回复
        reply_content = ""
        total_tokens = 0
        if self.llm_client:
            prev_prompt = self.llm_client.total_prompt_tokens
            prev_completion = self.llm_client.total_completion_tokens
            prev_cache = self.llm_client.total_cache_hit_tokens
            prev_cost = self.llm_client.total_cost_usd

            reply_content, total_tokens = self.llm_client.generate_reply(
                comment=comment_dict["content"],
                context_chunks=context_chunks,
                article_summary=topic_summary,
                history_messages=history_messages[:-1] if history_messages else None,
            )

            # 添加 [rob]: 回复前缀
            reply_prefix = self.settings.get("bot", {}).get("reply_prefix", "[rob]")
            if reply_content and reply_prefix:
                reply_content = f"{reply_prefix}: {reply_content}"

            if self.cost_tracker:
                self.cost_tracker.record(
                    model=self.llm_client.model,
                    prompt_tokens=self.llm_client.total_prompt_tokens - prev_prompt,
                    completion_tokens=self.llm_client.total_completion_tokens - prev_completion,
                    cache_hit_tokens=self.llm_client.total_cache_hit_tokens - prev_cache,
                    usd_cost=self.llm_client.total_cost_usd - prev_cost,
                )

        # 追加 Bot 回复到线程
        if self.thread_manager and reply_content and thread_path:
            self.thread_manager.append_turn(
                thread_path=thread_path,
                author="Bot",
                content=reply_content,
                model=self.llm_client.model if self.llm_client else "unknown",
                tokens=total_tokens,
            )

        # 写入 pending/ 等待人工审核，并索引到 RAG
        if reply_content:
            self._write_pending(
                topic=topic,
                comment_content=message.content,
                reply_content=reply_content,
                comment_id=message.id,
            )
            if self.rag_retriever:
                try:
                    self.rag_retriever.index_human_reply(
                        question=comment_dict["content"],
                        reply=reply_content,
                        article_id=topic.get("id", "inbox"),
                        thread_id=message.parent_id or message.id,
                    )
                except Exception as e:
                    logger.warning("Bot 回复索引到 RAG 失败: %s", e)

        self._seen_ids.add(message.id)
        self._processed_count += 1

    def _handle_expert_reply(self, topic: dict, topic_meta: dict, message: Message):
        """处理专家/维护者的权威回复

        检测到 is_author_reply=True 的消息时，
        找到对应 thread，提取 QA 对，索引到 reply_index。
        """
        logger.info(f"检测到专家回复: message_id={message.id}")

        topic_id = topic.get("id", "inbox")

        if self.thread_manager:
            thread_path = self.thread_manager.get_or_create_thread(
                article_id=topic_id,
                root_comment=self._make_root_comment_info(message),
                article_meta=topic_meta,
            )

            # 从历史中找最近一条 role=="user" 的内容作为 question
            question_for_rag = "用户消息"  # 默认值
            if self.rag_retriever:
                messages = self.thread_manager.build_context_messages(
                    thread_path, max_turns=6
                )
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        question_for_rag = msg.get("content", question_for_rag)
                        break

            # 追加专家回复（带 ⭐ 标记）
            self.thread_manager.append_turn(
                thread_path=thread_path,
                author=message.author,
                content=message.content,
                comment_id=message.id,
                is_human=True,
            )

        # 索引到 reply_index
        if self.rag_retriever:
            if not self.thread_manager:
                question_for_rag = "用户消息"

            self.rag_retriever.index_human_reply(
                question=question_for_rag,
                reply=message.content,
                article_id=topic_id,
                thread_id=self._get_thread_root_id(message),
            )

    def _handle_whitelist_message(self, topic: dict, topic_meta: dict, message: Message):
        """处理白名单用户的消息

        白名单用户（维护者等）的消息仅记录到对话线程和 RAG，
        不触发 AI 生成回复，节省 token。

        Args:
            topic: 话题配置
            topic_meta: 话题元信息
            message: Message 对象
        """
        logger.info(
            "白名单用户 %s 的消息，仅记录: message_id=%s",
            message.author, message.id,
        )

        # 记录到对话线程
        topic_id = topic.get("id", "inbox")
        if self.thread_manager:
            thread_path = self.thread_manager.get_or_create_thread(
                article_id=topic_id,
                root_comment=self._make_root_comment_info(message),
                article_meta=topic_meta,
            )
            self.thread_manager.append_turn(
                thread_path=thread_path,
                author=message.author,
                content=message.content,
                comment_id=message.id,
            )

        # 索引到 RAG 供后续检索
        if self.rag_retriever:
            self.rag_retriever.index_human_reply(
                question=message.content,
                reply=message.content,
                article_id=topic_id,
                thread_id=self._get_thread_root_id(message),
            )

    def _read_inbox(self) -> list[Message]:
        """读取收件箱目录中的待处理消息

        从 data/inbox/ 目录读取 JSON 格式的消息文件。
        每个文件应包含字段: id, content, author, created_time, parent_id, is_author_reply。

        Returns:
            消息列表（按文件名排序）
        """
        inbox_dir = self.root / "data" / "inbox"
        if not inbox_dir.exists():
            logger.info("收件箱目录不存在: %s", inbox_dir)
            return []

        messages: list[Message] = []
        for json_file in sorted(inbox_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw_parent = data.get("parent_id")
                message = Message(
                    id=str(data["id"]),
                    content=str(data["content"]),
                    author=str(data.get("author", "anonymous")),
                    created_time=int(data.get("created_time", 0)),
                    parent_id=str(raw_parent) if raw_parent is not None else None,
                    is_author_reply=bool(data.get("is_author_reply", False)),
                )
                messages.append(message)
            except Exception as e:
                logger.warning("读取收件箱文件失败: %s - %s", json_file.name, e)

        logger.info("从收件箱读取 %d 条消息", len(messages))
        return messages

    def run_inbox(self):
        """处理收件箱消息（可单独调用，便于测试）

        读取 data/inbox/ 目录中的消息，通过 RAG/LLM 生成回复并写入 pending/。
        捕获预算超限异常并触发告警。
        """
        messages = self._read_inbox()

        if not messages:
            logger.info("收件箱为空，无需处理")
            return

        logger.info("共 %d 条收件箱消息待处理", len(messages))

        try:
            self.process_messages(messages)
        except BudgetExceededError as e:
            logger.warning(f"预算超限，终止处理: {e}")
            if self.alert_manager:
                self.alert_manager.alert_budget_exceeded(
                    cost=self.llm_client.total_cost_usd if self.llm_client else 0,
                    budget=self.settings["bot"]["llm_budget_usd_per_day"],
                )

    def run(self):
        """执行主流程

        完整流程：
        1. 加载配置
        2. 初始化模块
        3. 同步 Wiki
        4. 加载已处理 ID
        5. 处理收件箱消息
        6. 保存状态
        7. 输出费用报告
        """
        logger.info("=== CSM RAG/LLM 问答机器人启动 ===")

        try:
            self.load_config()
            self.init_modules()

            # 同步 Wiki（增量更新）
            if self.rag_retriever:
                try:
                    self.rag_retriever.sync_wiki()
                except Exception as e:
                    logger.warning(f"Wiki 同步失败（不影响主流程）: {e}")

            self.load_seen_ids()
            self.load_comment_thread_map()

            # 处理收件箱消息
            self.run_inbox()

            # 保存状态
            self.save_seen_ids()
            self.save_comment_thread_map()
            if self.comment_filter:
                self.comment_filter.save_dedup_cache()

            # 费用报告
            if self.cost_tracker:
                self.cost_tracker.print_daily_report()
                self.cost_tracker.update_monthly_summary()

            # 记录健康状态
            if self.alert_manager:
                self.alert_manager.record_health("ok")

            logger.info(
                f"=== 运行完成: 处理 {self._processed_count} 条消息 ==="
            )

        except Exception as e:
            logger.error(f"主流程异常: {e}", exc_info=True)
            if self.alert_manager:
                self.alert_manager.record_health("error", {"error": str(e)})
            raise


def main():
    """入口函数"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    runner = BotRunner(project_root=Path(__file__).parent.parent)
    runner.run()


if __name__ == "__main__":
    main()
