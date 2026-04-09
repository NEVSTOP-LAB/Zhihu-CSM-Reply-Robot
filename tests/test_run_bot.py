# -*- coding: utf-8 -*-
"""
BotRunner 单元测试
==================

实施计划关联：AI-009 验收标准
独立于实现的测试用例，覆盖：
- 文章循环、评论获取、过滤、LLM 回复、写入 pending/
- 已处理 ID 正确持久化
- ZhihuAuthError → alert + 终止
- BudgetExceededError → alert + 终止
- 真人回复 → index_human_reply
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from scripts.run_bot import BotRunner
from scripts.zhihu_client import Comment, ZhihuAuthError, ZhihuRateLimitError
from scripts.llm_client import BudgetExceededError


# ===== Fixtures =====

@pytest.fixture
def bot_root(tmp_path: Path) -> Path:
    """创建完整的项目目录结构"""
    root = tmp_path / "project"
    root.mkdir()

    # 创建配置
    config_dir = root / "config"
    config_dir.mkdir()

    settings = {
        "bot": {
            "check_interval_hours": 6,
            "max_new_comments_per_run": 20,
            "max_new_comments_per_day": 100,
            "llm_budget_usd_per_day": 0.50,
        },
        "llm": {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "max_tokens": 250,
            "temperature": 0.7,
        },
        "rag": {
            "embedding_model": "BAAI/bge-small-zh-v1.5",
            "use_online_embedding": False,
            "top_k": 3,
            "similarity_threshold": 0.72,
            "history_turns": 6,
        },
        "vector_store": {"backend": "actions_cache", "max_size_mb": 500},
        "review": {"manual_mode": True},
        "filter": {
            "max_comment_tokens": 500,
            "spam_keywords": ["加微信"],
            "dedup_window_minutes": 60,
        },
        "alerting": {
            "github_issue": True,
            "consecutive_fail_limit": 3,
        },
    }
    with open(config_dir / "settings.yaml", "w") as f:
        yaml.dump(settings, f)

    articles = {
        "articles": [
            {
                "id": "99999",
                "title": "测试文章",
                "url": "https://example.com",
                "type": "article",
            }
        ]
    }
    with open(config_dir / "articles.yaml", "w") as f:
        yaml.dump(articles, f)

    # 创建必要目录
    (root / "data").mkdir()
    (root / "csm-wiki").mkdir()
    (root / "archive").mkdir()
    (root / "pending").mkdir()

    return root


@pytest.fixture
def runner(bot_root: Path) -> BotRunner:
    """创建 BotRunner 实例"""
    return BotRunner(project_root=str(bot_root))


def _make_comment(
    id: str, content: str, author: str = "user",
    is_author_reply: bool = False, parent_id: str | None = None,
) -> Comment:
    """构造 Comment 对象"""
    return Comment(
        id=id,
        parent_id=parent_id,
        content=content,
        author=author,
        created_time=1712000000,
        is_author_reply=is_author_reply,
    )


# ===== 配置加载测试 =====

class TestLoadConfig:
    """验证配置加载"""

    def test_load_config_success(self, runner):
        """应成功加载配置"""
        runner.load_config()
        assert len(runner.articles) == 1
        assert runner.articles[0]["id"] == "99999"
        assert runner.settings["bot"]["max_new_comments_per_day"] == 100


# ===== seen_ids 持久化测试 =====

class TestSeenIds:
    """验证已处理 ID 持久化"""

    def test_save_and_load_seen_ids(self, runner):
        """保存后加载应获得相同的 ID 集合"""
        runner.load_config()
        runner._seen_ids = {"1", "2", "3"}
        runner.save_seen_ids()

        # 新实例加载
        runner2 = BotRunner(project_root=runner.root)
        runner2.load_config()
        runner2.load_seen_ids()
        assert runner2._seen_ids == {"1", "2", "3"}

    def test_load_seen_ids_empty(self, runner):
        """无文件时应返回空集合"""
        runner.load_config()
        runner.load_seen_ids()
        assert runner._seen_ids == set()


# ===== 文章处理测试 =====

class TestProcessArticle:
    """验证文章处理流程"""

    def test_no_new_comments(self, runner):
        """无新评论时应安全跳过"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = []

        runner.process_article(runner.articles[0])
        assert runner._processed_count == 0

    def test_skip_seen_comments(self, runner):
        """已处理过的评论应被跳过"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("seen_1", "旧评论"),
        ]
        runner._seen_ids = {"seen_1"}

        # LLM 不应被调用
        runner.llm_client = MagicMock()

        runner.process_article(runner.articles[0])
        runner.llm_client.generate_reply.assert_not_called()

    def test_new_comment_generates_reply(self, runner, bot_root):
        """新评论应生成回复并写入 pending/"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("new_1", "CSM 是什么？"),
        ]

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = (
            "CSM 是客户成功管理的缩写。", 150
        )
        runner.llm_client.total_cost_usd = 0.001
        runner.llm_client.total_prompt_tokens = 100
        runner.llm_client.total_completion_tokens = 50
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = "摘要"

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # 验证回复被写入 pending/
        pending_files = list((bot_root / "pending").glob("*.md"))
        assert len(pending_files) >= 1

        # 验证 seen_ids 更新
        assert "new_1" in runner._seen_ids

    def test_spam_comment_skipped(self, runner):
        """广告评论应被跳过"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("spam_1", "加微信了解更多"),
        ]

        runner.llm_client = MagicMock()
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # LLM 不应被调用（广告被过滤）
        runner.llm_client.generate_reply.assert_not_called()
        # 但 seen_ids 应记录
        assert "spam_1" in runner._seen_ids

    def test_human_reply_indexed(self, runner):
        """真人回复应被索引"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment(
                "human_1", "这是作者的回复",
                author="文章作者",
                is_author_reply=True,
                parent_id="parent_1",
            ),
        ]

        runner.llm_client = MagicMock()
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # 验证 index_human_reply 被调用
        runner.rag_retriever.index_human_reply.assert_called_once()


# ===== 异常处理测试 =====

class TestExceptionHandling:
    """验证异常场景"""

    def test_auth_error_triggers_alert(self, runner):
        """ZhihuAuthError 应触发告警并终止"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.side_effect = ZhihuAuthError(
            "Cookie 过期"
        )

        runner.alert_manager = MagicMock()
        runner.rag_retriever = MagicMock()

        # 不应抛异常到 run()
        runner.load_seen_ids()
        runner.run_articles()
        runner.alert_manager.alert_cookie_expired.assert_called_once()

    def test_budget_exceeded_triggers_alert(self, runner, bot_root):
        """BudgetExceededError 应触发告警并终止"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("new_1", "测试"),
        ]

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.side_effect = BudgetExceededError(
            "超预算"
        )
        runner.llm_client.total_cost_usd = 0.55
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.alert_manager = MagicMock()

        runner.load_seen_ids()
        runner.run_articles()
        runner.alert_manager.alert_budget_exceeded.assert_called_once()


# ===== daily_limit 测试 =====

class TestDailyLimit:
    """验证每日上限"""

    def test_daily_limit_respected(self, runner):
        """达到每日上限后应停止处理"""
        runner.load_config()
        runner.settings["bot"]["max_new_comments_per_day"] = 2

        runner._processed_count = 2
        assert runner._check_daily_limit() is False

    def test_under_limit_continues(self, runner):
        """未达上限时应继续"""
        runner.load_config()
        runner._processed_count = 0
        assert runner._check_daily_limit() is True


# ===== pending 文件测试 =====

class TestWritePending:
    """验证 pending 文件写入"""

    def test_pending_file_created(self, runner, bot_root):
        """应创建 pending 文件"""
        runner.load_config()
        article = {"id": "111", "title": "测试文章"}

        runner._write_pending(
            article=article,
            comment_content="用户评论",
            reply_content="Bot 回复",
            comment_id="222",
        )

        pending_files = list((bot_root / "pending").glob("*.md"))
        assert len(pending_files) == 1

        content = pending_files[0].read_text()
        assert "用户评论" in content
        assert "Bot 回复" in content
        assert "111" in content
        assert "222" in content

    def test_pending_file_name_format(self, runner, bot_root):
        """pending 文件名格式: {article_id}_{comment_id}.md"""
        runner.load_config()
        article = {"id": "AAA", "title": ""}

        runner._write_pending(article, "评论", "回复", "BBB")

        assert (bot_root / "pending" / "AAA_BBB.md").exists()
