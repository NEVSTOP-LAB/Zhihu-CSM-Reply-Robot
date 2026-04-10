# -*- coding: utf-8 -*-
"""
主入口 — 知乎 CSM 自动回复机器人
==================================

实施计划关联：AI-009 主流程 run_bot.py — MVP 版
参考文档：docs/plan/README.md 一、系统逻辑流程图

串联所有模块的主流程：
1. 加载配置，初始化模块
2. 每日处理量检查
3. 遍历文章 → 拉取评论 → 过滤 → RAG → LLM → pending/
4. 检测真人回复 → 索引
5. 异常告警（认证/预算）
6. 退出前 git commit

使用方式：
    python scripts/run_bot.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# 确保可以导入 scripts 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.zhihu_client import ZhihuClient, ZhihuAuthError, ZhihuRateLimitError
from scripts.rag_retriever import RAGRetriever
from scripts.llm_client import LLMClient, BudgetExceededError
from scripts.thread_manager import ThreadManager
from scripts.comment_filter import CommentFilter
from scripts.alerting import AlertManager
from scripts.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class BotRunner:
    """主流程控制器

    实施计划关联：AI-009

    串联所有模块，实现 pending/ 人工审核 MVP。

    Args:
        project_root: 项目根目录路径
    """

    def __init__(self, project_root: str | Path = "."):
        self.root = Path(project_root)
        self.settings: dict = {}
        self.articles: list[dict] = []

        # 模块实例（延迟初始化）
        self.zhihu_client: Optional[ZhihuClient] = None
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

    def load_config(self):
        """加载配置文件

        从 config/settings.yaml 和 config/articles.yaml 读取配置。
        """
        settings_path = self.root / "config" / "settings.yaml"
        articles_path = self.root / "config" / "articles.yaml"

        with open(settings_path, "r", encoding="utf-8") as f:
            self.settings = yaml.safe_load(f)

        with open(articles_path, "r", encoding="utf-8") as f:
            articles_data = yaml.safe_load(f)
            self.articles = articles_data.get("articles", [])

        logger.info(
            f"配置加载完成: {len(self.articles)} 篇文章, "
            f"每日上限 {self.settings['bot']['max_new_comments_per_day']}"
        )

    def init_modules(self):
        """初始化所有模块

        从环境变量和配置文件创建各模块实例。
        """
        # 知乎客户端
        cookie = os.environ.get("ZHIHU_COOKIE", "")
        if cookie:
            self.zhihu_client = ZhihuClient(cookie=cookie)
        else:
            logger.warning("未设置 ZHIHU_COOKIE，跳过知乎客户端初始化")

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

        # 评论过滤器
        self.comment_filter = CommentFilter(settings=self.settings)

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

        logger.info(f"已加载 {len(self._seen_ids)} 个已处理评论 ID")

    def save_seen_ids(self):
        """保存已处理的评论 ID 集合"""
        self._seen_ids_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._seen_ids_path, "w", encoding="utf-8") as f:
            json.dump(sorted(self._seen_ids), f)

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
        article: dict,
        comment_content: str,
        reply_content: str,
        comment_id: str,
        risk_reason: str = "",
    ):
        """将待审核回复写入 pending/ 目录

        Args:
            article: 文章信息
            comment_content: 原始评论
            reply_content: 生成的回复
            comment_id: 评论 ID
            risk_reason: AI 风险评估理由
        """
        pending_dir = self.root / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{article['id']}_{comment_id}.md"
        filepath = pending_dir / filename

        object_type = article.get("type", "article")

        content = (
            f"---\n"
            f"article_id: \"{article['id']}\"\n"
            f"article_title: \"{article.get('title', '')}\"\n"
            f"object_type: \"{object_type}\"\n"
            f"comment_id: \"{comment_id}\"\n"
            f"generated_at: \"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}\"\n"
            f"status: pending\n"
            f"risk_reason: \"{risk_reason}\"\n"
            f"---\n\n"
            f"## 原始评论\n\n"
            f"> {comment_content}\n\n"
            f"## 生成的回复\n\n"
            f"{reply_content}\n"
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"待审核回复已写入: {filepath.name}")

    def process_article(self, article: dict):
        """处理单篇文章的评论

        实施计划关联：AI-009 任务 3

        流程：拉取评论 → 前置过滤 → RAG → LLM → pending/

        Args:
            article: 文章配置信息
        """
        if not self.zhihu_client:
            logger.error("知乎客户端未初始化")
            return

        article_id = article["id"]
        object_type = article.get("type", "article")
        article_meta = {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
        }

        logger.info(f"处理文章: {article.get('title', article_id)}")

        # 拉取评论
        comments = self.zhihu_client.get_comments(
            object_id=article_id,
            object_type=object_type,
        )

        # 过滤已处理的评论
        new_comments = [
            c for c in comments if c.id not in self._seen_ids
        ]

        if not new_comments:
            logger.info(f"文章 {article_id}: 无新评论")
            return

        logger.info(f"文章 {article_id}: 发现 {len(new_comments)} 条新评论")

        # 获取文章摘要（缓存）— 使用 LLM 生成简短摘要，便于 AI 理解上下文
        # 不记录文章全文，只保留摘要
        article_summary = ""
        if self.llm_client:
            try:
                article_summary = self.llm_client.summarize_article(
                    title=article.get("title", ""),
                    content=article.get("title", ""),
                )
            except Exception as e:
                logger.warning(f"文章摘要生成失败: {e}")

        # 将摘要写入 article_meta，供线程管理器使用
        article_meta["summary"] = article_summary

        # 检索文章相关 Wiki 上下文（批量复用）
        context_chunks = []
        if self.rag_retriever:
            context_chunks = self.rag_retriever.retrieve(
                query=article.get("title", ""),
                k=self.settings.get("rag", {}).get("top_k", 3),
                threshold=self.settings.get("rag", {}).get(
                    "similarity_threshold", 0.72
                ),
            )

        for comment in new_comments:
            if not self._check_daily_limit():
                break

            try:
                self._process_single_comment(
                    article=article,
                    article_meta=article_meta,
                    comment=comment,
                    context_chunks=context_chunks,
                    article_summary=article_summary,
                )
                self._consecutive_failures = 0
            except BudgetExceededError as e:
                logger.warning(f"预算超限: {e}")
                raise  # 向上传播，由 run_articles 处理告警
            except Exception as e:
                self._consecutive_failures += 1
                logger.error(f"处理评论失败: {e}")

                fail_limit = self.settings.get("alerting", {}).get(
                    "consecutive_fail_limit", 3
                )
                if self._consecutive_failures >= fail_limit:
                    logger.error(f"连续失败 {self._consecutive_failures} 次，暂停")
                    if self.alert_manager:
                        self.alert_manager.alert_consecutive_failures(
                            self._consecutive_failures
                        )
                    break

    def _process_single_comment(
        self,
        article: dict,
        article_meta: dict,
        comment,
        context_chunks: list[str],
        article_summary: str,
    ):
        """处理单条评论

        Args:
            article: 文章配置
            article_meta: 文章元信息
            comment: Comment 对象
            context_chunks: RAG 检索结果
            article_summary: 文章摘要
        """
        # 检测真人回复 → 索引
        # 参考 AI-013: 真人回复高权重索引
        if comment.is_author_reply and self.rag_retriever:
            self._handle_human_reply(article, article_meta, comment)
            self._seen_ids.add(comment.id)
            return

        # 白名单用户：仅记录到线程，不做 AI 处理
        whitelist = self.settings.get("bot", {}).get("whitelist_users", [])
        if comment.author in whitelist:
            self._handle_whitelist_comment(article, article_meta, comment)
            self._seen_ids.add(comment.id)
            return

        # 前置过滤
        comment_dict = {
            "content": comment.content,
            "author": comment.author,
            "created_time": comment.created_time,
        }

        if self.comment_filter:
            skip, reason = self.comment_filter.should_skip(comment_dict)
            if skip:
                logger.info(f"跳过评论 {comment.id}: {reason}")
                self._seen_ids.add(comment.id)
                return

            # 超长截断
            comment_dict["content"] = (
                self.comment_filter.truncate_if_needed(comment_dict["content"])
            )

        # 构建线程和历史上下文
        history_messages = []
        if self.thread_manager:
            root_comment = {
                "id": comment.parent_id or comment.id,
                "author": comment.author,
                "content": comment.content,
                "created_time": comment.created_time,
            }
            thread_path = self.thread_manager.get_or_create_thread(
                article_id=article["id"],
                root_comment=root_comment,
                article_meta=article_meta,
            )

            # 追加用户评论
            self.thread_manager.append_turn(
                thread_path=thread_path,
                author=comment.author,
                content=comment.content,
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
            # 记录调用前的累计值，用于计算增量
            prev_prompt = self.llm_client.total_prompt_tokens
            prev_completion = self.llm_client.total_completion_tokens
            prev_cache = self.llm_client.total_cache_hit_tokens
            prev_cost = self.llm_client.total_cost_usd

            reply_content, total_tokens = self.llm_client.generate_reply(
                comment=comment_dict["content"],
                context_chunks=context_chunks,
                article_summary=article_summary,
                history_messages=history_messages[:-1] if history_messages else None,
            )

            # 添加 [rob]: 回复前缀
            reply_prefix = self.settings.get("bot", {}).get("reply_prefix", "[rob]")
            if reply_content and reply_prefix:
                reply_content = f"{reply_prefix}: {reply_content}"

            # 记录增量费用（而非累计值）
            if self.cost_tracker:
                self.cost_tracker.record(
                    model=self.llm_client.model,
                    prompt_tokens=self.llm_client.total_prompt_tokens - prev_prompt,
                    completion_tokens=self.llm_client.total_completion_tokens - prev_completion,
                    cache_hit_tokens=self.llm_client.total_cache_hit_tokens - prev_cache,
                    usd_cost=self.llm_client.total_cost_usd - prev_cost,
                )

        # 追加 Bot 回复到线程
        if self.thread_manager and reply_content:
            thread_path = self.thread_manager.get_or_create_thread(
                article_id=article["id"],
                root_comment={
                    "id": comment.parent_id or comment.id,
                    "author": comment.author,
                },
                article_meta=article_meta,
            )
            self.thread_manager.append_turn(
                thread_path=thread_path,
                author="Bot",
                content=reply_content,
                model=self.llm_client.model if self.llm_client else "unknown",
                tokens=total_tokens,
            )

        # 将 QA 对（用户评论 + bot 回复）索引到 RAG，用于后续检索学习
        if self.rag_retriever and reply_content:
            try:
                self.rag_retriever.index_human_reply(
                    question=comment_dict["content"],
                    reply=reply_content,
                    article_id=article["id"],
                    thread_id=comment.parent_id or comment.id if hasattr(comment, 'parent_id') else comment.id,
                )
            except Exception as e:
                logger.warning("Bot 回复索引到 RAG 失败: %s", e)

        # AI 风险评估：决定自动发布或写入 pending/
        if self.llm_client and reply_content and self.zhihu_client:
            risk_level, risk_reason = self.llm_client.assess_risk(
                comment=comment_dict["content"],
                reply=reply_content,
            )

            if risk_level == "safe":
                # CSM/LabVIEW 相关明确回复 → 自动发布
                object_type = article.get("type", "article")
                # ZhihuClient.post_comment() 约定 article/answer；
                # 配置中的 question 在发布前映射为 answer
                post_type = "answer" if object_type == "question" else object_type

                success = self.zhihu_client.post_comment(
                    object_id=article["id"],
                    object_type=post_type,
                    content=reply_content,
                    parent_id=comment.id,
                )
                if not success:
                    # 发布失败，回退到 pending/
                    self._write_pending(
                        article=article,
                        comment_content=comment.content,
                        reply_content=reply_content,
                        comment_id=comment.id,
                        risk_reason="发布失败，自动转入人工审核",
                    )
            else:
                # 高危回复 → 写入 pending/ 等待人工审核
                self._write_pending(
                    article=article,
                    comment_content=comment.content,
                    reply_content=reply_content,
                    comment_id=comment.id,
                    risk_reason=risk_reason,
                )
        elif reply_content:
            # 无 zhihu_client（如测试环境）→ 写入 pending/
            self._write_pending(
                article=article,
                comment_content=comment.content,
                reply_content=reply_content,
                comment_id=comment.id,
            )

        self._seen_ids.add(comment.id)
        self._processed_count += 1

    def _handle_human_reply(self, article: dict, article_meta: dict, comment):
        """处理作者真人回复

        实施计划关联：AI-013 真人回复高权重索引

        检测到 is_author_reply=True 的评论时，
        找到对应 thread，提取 QA 对，索引到 reply_index。
        """
        logger.info(f"检测到真人回复: comment_id={comment.id}")

        if self.thread_manager:
            root_comment = {
                "id": comment.parent_id or comment.id,
                "author": comment.author,
            }
            thread_path = self.thread_manager.get_or_create_thread(
                article_id=article["id"],
                root_comment=root_comment,
                article_meta={
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                },
            )

            # 追加真人回复（带 ⭐ 标记）
            self.thread_manager.append_turn(
                thread_path=thread_path,
                author=comment.author,
                content=comment.content,
                is_human=True,
            )

        # 索引到 reply_index
        if self.rag_retriever:
            # 获取上一条用户评论作为 question
            question = "用户评论"  # 简化处理
            if self.thread_manager:
                messages = self.thread_manager.build_context_messages(
                    thread_path, max_turns=2
                )
                if messages:
                    question = messages[0].get("content", question)

            self.rag_retriever.index_human_reply(
                question=question,
                reply=comment.content,
                article_id=article["id"],
                thread_id=comment.parent_id or comment.id,
            )

    def _handle_whitelist_comment(self, article: dict, article_meta: dict, comment):
        """处理白名单用户的评论

        白名单用户（维护者等）的回复仅记录到对话线程和 RAG，
        不触发 AI 生成回复，节省 token。

        Args:
            article: 文章配置
            article_meta: 文章元信息
            comment: Comment 对象
        """
        logger.info(
            "白名单用户 %s 的评论，仅记录: comment_id=%s",
            comment.author, comment.id,
        )

        # 记录到对话线程
        if self.thread_manager:
            root_comment = {
                "id": comment.parent_id or comment.id,
                "author": comment.author,
            }
            thread_path = self.thread_manager.get_or_create_thread(
                article_id=article["id"],
                root_comment=root_comment,
                article_meta=article_meta,
            )
            self.thread_manager.append_turn(
                thread_path=thread_path,
                author=comment.author,
                content=comment.content,
            )

        # 索引到 RAG 供后续检索
        if self.rag_retriever:
            self.rag_retriever.index_human_reply(
                question=comment.content,
                reply=comment.content,
                article_id=article["id"],
                thread_id=comment.parent_id or comment.id,
            )

    def _expand_articles(self) -> list[dict]:
        """展开 column/user_answers 类型为具体的文章/回答列表

        将 articles 配置中的 column 和 user_answers 类型自动展开
        为对应的 article/question 类型条目。

        Returns:
            展开后的文章列表
        """
        expanded = []
        for article in self.articles:
            article_type = article.get("type", "article")

            if article_type in ("article", "question"):
                expanded.append(article)
            elif article_type == "column" and self.zhihu_client:
                # 展开专栏：获取专栏下全部文章
                try:
                    column_articles = self.zhihu_client.get_column_articles(
                        article["id"]
                    )
                    expanded.extend(column_articles)
                    logger.info(
                        "专栏 %s 展开为 %d 篇文章",
                        article["id"], len(column_articles),
                    )
                except Exception as e:
                    logger.warning("展开专栏 %s 失败: %s", article["id"], e)
            elif article_type == "user_answers" and self.zhihu_client:
                # 展开用户回答：获取某人全部回答
                try:
                    user_answers = self.zhihu_client.get_user_answers(
                        article["id"]
                    )
                    expanded.extend(user_answers)
                    logger.info(
                        "用户 %s 展开为 %d 个回答",
                        article["id"], len(user_answers),
                    )
                except Exception as e:
                    logger.warning("展开用户 %s 回答失败: %s", article["id"], e)
            else:
                expanded.append(article)

        return expanded

    def run_articles(self):
        """处理所有文章（可单独调用，便于测试）

        先展开 column/user_answers 类型，再逐篇处理评论。
        捕获认证/限流/预算异常并触发告警。
        """
        # 展开 column/user_answers 为具体的文章/回答
        articles_to_process = self._expand_articles()
        logger.info("展开后共 %d 个监控目标", len(articles_to_process))

        for article in articles_to_process:
            if not self._check_daily_limit():
                break

            try:
                self.process_article(article)
            except ZhihuAuthError as e:
                logger.error(f"知乎认证失败: {e}")
                if self.alert_manager:
                    self.alert_manager.alert_cookie_expired(401)
                break
            except ZhihuRateLimitError as e:
                logger.error(f"知乎限流: {e}")
                if self.alert_manager:
                    self.alert_manager.alert_rate_limited()
                break
            except BudgetExceededError as e:
                logger.warning(f"预算超限，终止处理: {e}")
                if self.alert_manager:
                    self.alert_manager.alert_budget_exceeded(
                        cost=self.llm_client.total_cost_usd if self.llm_client else 0,
                        budget=self.settings["bot"]["llm_budget_usd_per_day"],
                    )
                break

    def run(self):
        """执行主流程

        实施计划关联：AI-009

        完整流程：
        1. 加载配置
        2. 初始化模块
        3. 同步 Wiki
        4. 加载已处理 ID
        5. 处理每篇文章
        6. 保存状态
        7. 输出费用报告
        """
        logger.info("=== 知乎 CSM 自动回复机器人启动 ===")

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

            # 处理每篇文章
            self.run_articles()

            # 保存状态
            self.save_seen_ids()

            # 费用报告
            if self.cost_tracker:
                self.cost_tracker.print_daily_report()
                self.cost_tracker.update_monthly_summary()

            # 记录健康状态
            if self.alert_manager:
                self.alert_manager.record_health("ok")

            logger.info(
                f"=== 运行完成: 处理 {self._processed_count} 条评论 ==="
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
