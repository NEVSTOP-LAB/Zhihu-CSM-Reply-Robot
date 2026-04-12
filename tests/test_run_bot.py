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

from scripts.run_bot import BotRunner, RecentLogsHandler
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
            "max_new_comments_per_run": 20,
            "max_new_comments_per_day": 100,
            "llm_budget_usd_per_day": 0.50,
            "reply_prefix": "[rob]",
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
        "review": {"auto_skip_patterns": []},
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
    (root / "data" / "pending").mkdir()
    (root / "csm-wiki").mkdir()
    (root / "archive").mkdir()

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


# ===== 冷启动归档测试（Issue #1）=====

class TestBootstrapMode:
    """验证冷启动归档逻辑（Issue #1）：首次运行只归档不回复"""

    def test_bootstrap_archives_existing_comments(self, runner, bot_root):
        """首次运行时，已有评论应被归档到 seen_ids，不生成回复"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("old_1", "旧评论1"),
            _make_comment("old_2", "旧评论2"),
        ]
        runner.llm_client = MagicMock()
        runner.rag_retriever = MagicMock()

        # 初始无 bootstrapped_articles，模拟首次运行
        runner.load_seen_ids()

        runner.run_articles()

        # 旧评论应被加入 seen_ids，不触发 LLM
        assert "old_1" in runner._seen_ids
        assert "old_2" in runner._seen_ids
        runner.llm_client.generate_reply.assert_not_called()

    def test_bootstrap_skips_already_seen(self, runner, bot_root):
        """冷启动时，已在 seen_ids 中的评论不重复计入"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("existing", "已有评论"),
        ]
        runner.llm_client = MagicMock()
        runner.rag_retriever = MagicMock()

        runner.load_seen_ids()
        runner._seen_ids.add("existing")  # 已存在

        runner.run_articles()

        # 已存在的 ID 不应重复处理
        runner.llm_client.generate_reply.assert_not_called()

    def test_bootstrap_marks_article_as_bootstrapped(self, runner, bot_root):
        """冷启动完成后，文章应被标记为已归档"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = []
        runner.llm_client = MagicMock()
        runner.rag_retriever = MagicMock()

        runner.load_seen_ids()

        runner.run_articles()

        assert "99999" in runner._bootstrapped_articles

    def test_bootstrapped_articles_persisted(self, runner, bot_root):
        """已归档文章集合应能保存并重新加载"""
        runner.load_config()
        runner._bootstrapped_articles = {"a1", "a2"}
        runner.save_bootstrapped_articles()

        runner2 = BotRunner(project_root=runner.root)
        runner2.load_config()
        runner2.load_bootstrapped_articles()
        assert runner2._bootstrapped_articles == {"a1", "a2"}

    def test_bootstrapped_articles_empty_when_no_file(self, runner, bot_root):
        """无文件时应返回空集合"""
        runner.load_config()
        runner.load_bootstrapped_articles()
        assert runner._bootstrapped_articles == set()

    def test_second_run_processes_new_comments(self, runner, bot_root):
        """冷启动后第二次运行，应正常处理新评论"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("new_1", "新评论"),
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复内容", 50)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        # 模拟已完成冷启动
        runner.load_seen_ids()
        runner._bootstrapped_articles.add("99999")

        runner.run_articles()

        # 新评论应被 LLM 处理
        runner.llm_client.generate_reply.assert_called_once()

    def test_bootstrap_auth_error_triggers_alert(self, runner, bot_root):
        """冷启动时认证失败应触发告警"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.side_effect = ZhihuAuthError(
            "Cookie 过期", status_code=401
        )
        runner.llm_client = MagicMock()
        runner.rag_retriever = MagicMock()
        runner.alert_manager = MagicMock()

        runner.load_seen_ids()
        runner.run_articles()

        runner.alert_manager.alert_cookie_expired.assert_called_once()

    def test_bootstrap_rate_limit_triggers_alert(self, runner, bot_root):
        """冷启动时限流应触发告警并终止（Issue review feedback）"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.side_effect = ZhihuRateLimitError("限流")
        runner.llm_client = MagicMock()
        runner.rag_retriever = MagicMock()
        runner.alert_manager = MagicMock()

        runner.load_seen_ids()
        runner.run_articles()

        runner.alert_manager.alert_rate_limited.assert_called_once()

    def test_bootstrap_generic_failure_does_not_mark_bootstrapped(self, runner, bot_root):
        """冷启动时发生通用异常后，文章不应被标记为已归档（避免跳过重试）"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.side_effect = Exception("临时网络错误")
        runner.llm_client = MagicMock()
        runner.rag_retriever = MagicMock()

        runner.load_seen_ids()
        runner.run_articles()

        # 文章不应被标记为已归档，以便下次运行时可以重试
        assert "99999" not in runner._bootstrapped_articles


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
        """新评论应生成回复并写入 pending/（高危时）"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("new_1", "CSM 是什么？"),
        ]

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = (
            "CSM 是 Communicable State Machine（通信状态机）框架的缩写。", 150
        )
        # AI 风险评估：高危 → 写入 pending/
        runner.llm_client.assess_risk.return_value = ("risky", "需人工确认")
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
        pending_files = list((bot_root / "data" / "pending").glob("*.md"))
        assert len(pending_files) >= 1

        # 验证 pending 文件包含 object_type 元数据
        content = pending_files[0].read_text()
        assert "object_type" in content

        # 验证 [rob]: 前缀被添加
        assert "[rob]:" in content

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

    def test_filtered_comment_recorded_to_thread(self, runner, bot_root):
        """被过滤的评论也应归档到对话线程，以供后期学习"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("spam_2", "加微信了解更多"),
        ]

        runner.llm_client = MagicMock()
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.thread_manager = MagicMock()
        runner.thread_manager.get_or_create_thread.return_value = "mock_thread_path"

        runner.process_article(runner.articles[0])

        # LLM 不应被调用（被过滤）
        runner.llm_client.generate_reply.assert_not_called()
        # 但线程记录应被调用（归档供学习）
        runner.thread_manager.append_turn.assert_called_once()
        call_kwargs = runner.thread_manager.append_turn.call_args[1]
        assert call_kwargs["author"] == "user"
        assert call_kwargs["content"] == "加微信了解更多"

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

    def test_expand_auth_error_triggers_alert_and_stops(self, runner):
        """展开监控目标时 ZhihuAuthError 应触发告警并终止"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_user_answers.side_effect = ZhihuAuthError(
            "认证失败 HTTP 403", status_code=403
        )

        runner.articles = [{"id": "some-user", "type": "user_answers"}]
        runner.alert_manager = MagicMock()

        runner.load_seen_ids()
        runner.run_articles()

        runner.alert_manager.alert_cookie_expired.assert_called_once_with(403)
        # 没有尝试处理任何文章
        runner.zhihu_client.get_comments.assert_not_called()

    def test_expand_zero_targets_triggers_alert(self, runner):
        """展开后目标为空时应触发 alert_expansion_failed 告警"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_column_articles.side_effect = Exception("网络错误")

        runner.articles = [{"id": "fail-col", "type": "column"}]
        runner.alert_manager = MagicMock()

        runner.load_seen_ids()
        runner.run_articles()

        runner.alert_manager.alert_expansion_failed.assert_called_once()
        call_kwargs = runner.alert_manager.alert_expansion_failed.call_args
        assert call_kwargs.kwargs.get("configured_count", call_kwargs.args[0] if call_kwargs.args else None) == 1

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
        runner._bootstrapped_articles.add("99999")  # 跳过冷启动，直接测试正式处理
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

        pending_files = list((bot_root / "data" / "pending").glob("*.md"))
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

        assert (bot_root / "data" / "pending" / "AAA_BBB.md").exists()


# ===== AI 风险评估与自动发布测试 =====

class TestRiskAssessment:
    """验证 AI 风险评估决定自动发布或写入 pending/"""

    def test_safe_reply_auto_published(self, runner, bot_root):
        """AI 判定安全的回复应自动发布"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("auto_1", "CSM 框架怎么用？"),
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = (
            "CSM 框架使用说明...", 100
        )
        # AI 风险评估：安全 → 自动发布
        runner.llm_client.assess_risk.return_value = ("safe", "CSM 技术问题")
        runner.llm_client.total_cost_usd = 0.001
        runner.llm_client.total_prompt_tokens = 80
        runner.llm_client.total_completion_tokens = 40
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = "摘要"

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # 验证自动发布被调用
        runner.zhihu_client.post_comment.assert_called_once()
        # 不应有 pending 文件
        pending_files = list((bot_root / "data" / "pending").glob("*.md"))
        assert len(pending_files) == 0

    def test_risky_reply_goes_to_pending(self, runner, bot_root):
        """AI 判定高危的回复应写入 pending/"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("risky_1", "公司财务怎么处理？"),
        ]

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = (
            "建议咨询专业人士...", 100
        )
        runner.llm_client.assess_risk.return_value = ("risky", "超出知识库范围")
        runner.llm_client.total_cost_usd = 0.001
        runner.llm_client.total_prompt_tokens = 80
        runner.llm_client.total_completion_tokens = 40
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = "摘要"

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # 不应自动发布
        runner.zhihu_client.post_comment.assert_not_called()
        # 应写入 pending/
        pending_files = list((bot_root / "data" / "pending").glob("*.md"))
        assert len(pending_files) == 1
        content = pending_files[0].read_text()
        assert "risk_reason" in content

    def test_publish_failure_falls_back_to_pending(self, runner, bot_root):
        """自动发布失败应回退到 pending/"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("fail_1", "CSM 问题"),
        ]
        runner.zhihu_client.post_comment.return_value = False  # 发布失败

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复内容", 100)
        runner.llm_client.assess_risk.return_value = ("safe", "CSM 话题")
        runner.llm_client.total_cost_usd = 0.001
        runner.llm_client.total_prompt_tokens = 80
        runner.llm_client.total_completion_tokens = 40
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = "摘要"

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # 发布被调用但失败，应回退到 pending
        runner.zhihu_client.post_comment.assert_called_once()
        pending_files = list((bot_root / "data" / "pending").glob("*.md"))
        assert len(pending_files) == 1


# ===== [rob]: 回复前缀测试 =====

class TestReplyPrefix:
    """验证回复前缀功能"""

    def test_reply_prefix_added(self, runner, bot_root):
        """回复应包含 [rob]: 前缀"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("prefix_1", "测试"),
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("原始回复", 50)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # 验证发布的内容包含 [rob]: 前缀
        call_args = runner.zhihu_client.post_comment.call_args
        posted_content = call_args.kwargs.get("content", call_args[0][2] if len(call_args[0]) > 2 else "")
        assert posted_content.startswith("[rob]:")


# ===== seen_ids 迁移测试 =====

class TestSeenIdsMigration:
    """验证 seen_ids 格式校验和迁移"""

    def test_load_dict_format_resets(self, runner, bot_root):
        """旧 dict 格式应被重置为空并迁移"""
        seen_path = bot_root / "data" / "seen_ids.json"
        seen_path.write_text('{"articles": {}, "last_run": null}')

        runner.load_config()
        runner.load_seen_ids()

        # 旧 dict 格式应被重置
        assert runner._seen_ids == set()
        # 文件应已被迁移为 list 格式
        import json
        with open(seen_path) as f:
            data = json.load(f)
        assert isinstance(data, list)

    def test_load_dict_with_seen_ids_key(self, runner, bot_root):
        """dict 包含 seen_ids key 时应正确迁移"""
        seen_path = bot_root / "data" / "seen_ids.json"
        seen_path.write_text('{"seen_ids": ["a", "b"]}')

        runner.load_config()
        runner.load_seen_ids()

        assert runner._seen_ids == {"a", "b"}


# ===== 文章类型展开测试 =====

class TestExpandArticles:
    """验证 column/user_answers 类型展开"""

    def test_article_and_answer_pass_through(self, runner):
        """article/answer 类型应直接传递（FIX-01）"""
        runner.load_config()
        runner.init_modules()
        runner.zhihu_client = MagicMock()

        runner.articles = [
            {"id": "1", "type": "article"},
            {"id": "20", "type": "answer"},
        ]
        result = runner._expand_articles()
        assert len(result) == 2
        assert result[0]["type"] == "article"
        assert result[1]["type"] == "answer"

    def test_question_expands_to_answers(self, runner):
        """question 类型应通过 API 展开为 answer 列表（FIX-01）"""
        runner.load_config()
        runner.init_modules()
        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_question_answers.return_value = [
            {"id": "30", "title": "回答1", "url": "url1", "type": "answer"},
            {"id": "31", "title": "回答2", "url": "url2", "type": "answer"},
        ]

        runner.articles = [
            {"id": "999", "type": "question", "title": "问题标题"},
        ]
        result = runner._expand_articles()
        assert len(result) == 2
        assert all(a["type"] == "answer" for a in result)
        runner.zhihu_client.get_question_answers.assert_called_once_with("999")

    def test_column_expands_to_articles(self, runner):
        """column 类型应展开为文章列表"""
        runner.load_config()
        runner.init_modules()
        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_column_articles.return_value = [
            {"id": "10", "title": "文章1", "url": "url1", "type": "article"},
            {"id": "11", "title": "文章2", "url": "url2", "type": "article"},
        ]

        runner.articles = [
            {"id": "my-column", "type": "column"},
        ]
        result = runner._expand_articles()
        assert len(result) == 2
        assert all(a["type"] == "article" for a in result)

    def test_user_answers_expands(self, runner):
        """user_answers 类型应展开为回答列表（type=answer，FIX-01）"""
        runner.load_config()
        runner.init_modules()
        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_user_answers.return_value = [
            {"id": "20", "title": "回答1", "url": "url1", "type": "answer"},
        ]

        runner.articles = [
            {"id": "some-user", "type": "user_answers"},
        ]
        result = runner._expand_articles()
        assert len(result) == 1
        assert result[0]["type"] == "answer"

    def test_column_expand_failure_skips(self, runner):
        """展开失败时应跳过该条目（非认证类错误）"""
        runner.load_config()
        runner.init_modules()
        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_column_articles.side_effect = Exception("网络错误")

        runner.articles = [
            {"id": "1", "type": "article"},
            {"id": "fail-column", "type": "column"},
        ]
        result = runner._expand_articles()
        # 仅保留 article 类型
        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_column_expand_auth_error_propagates(self, runner):
        """展开专栏时认证失败应向上传播（不被吞掉）"""
        runner.load_config()
        runner.init_modules()
        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_column_articles.side_effect = ZhihuAuthError(
            "认证失败 HTTP 403: Cookie 失效", status_code=403
        )

        runner.articles = [{"id": "col-1", "type": "column"}]
        with pytest.raises(ZhihuAuthError):
            runner._expand_articles()

    def test_user_answers_expand_auth_error_propagates(self, runner):
        """展开用户回答时认证失败应向上传播（不被吞掉）"""
        runner.load_config()
        runner.init_modules()
        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_user_answers.side_effect = ZhihuAuthError(
            "认证失败 HTTP 403", status_code=403
        )

        runner.articles = [{"id": "some-user", "type": "user_answers"}]
        with pytest.raises(ZhihuAuthError):
            runner._expand_articles()

    def test_question_expand_auth_error_propagates(self, runner):
        """展开问题时认证失败应向上传播"""
        runner.load_config()
        runner.init_modules()
        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_question_answers.side_effect = ZhihuAuthError(
            "认证失败 HTTP 403", status_code=403
        )

        runner.articles = [{"id": "123", "type": "question"}]
        with pytest.raises(ZhihuAuthError):
            runner._expand_articles()

# ===== post_comment 类型映射测试 =====

class TestTypeMapping:
    """验证 question→answer 类型映射"""

    def test_question_mapped_to_answer_on_publish(self, runner):
        """自动发布时 question 应映射为 answer"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("map_1", "LabVIEW 问题"),
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复", 50)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        # 使用 question 类型的文章（向后兼容）
        runner.articles = [{
            "id": "999",
            "title": "知乎问题",
            "url": "https://example.com",
            "type": "question",
        }]

        runner.process_article(runner.articles[0])

        # 验证 post_comment 使用 "answer" 而非 "question"
        call_args = runner.zhihu_client.post_comment.call_args
        assert call_args.kwargs.get("object_type") == "answer"

    def test_answer_type_published_as_answer(self, runner):
        """answer 类型（FIX-01 展开结果）应直接以 answer 发布"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("ans_1", "LabVIEW 问题"),
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复", 50)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        # 使用 answer 类型（FIX-01 展开后的条目）
        runner.articles = [{
            "id": "777",
            "title": "某回答",
            "url": "https://example.com",
            "type": "answer",
        }]

        runner.process_article(runner.articles[0])

        call_args = runner.zhihu_client.post_comment.call_args
        assert call_args.kwargs.get("object_type") == "answer"


# ===== 白名单用户测试 =====

class TestWhitelistUsers:
    """验证白名单用户仅记录不做 AI 处理"""

    def test_whitelist_user_skips_ai(self, runner, bot_root):
        """白名单用户的评论不应触发 LLM"""
        runner.load_config()
        runner.settings["bot"]["whitelist_users"] = ["maintainer"]
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("wl_1", "已修复", author="maintainer"),
        ]

        runner.llm_client = MagicMock()
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # LLM 不应被调用
        runner.llm_client.generate_reply.assert_not_called()
        # 但评论应被记录到 seen_ids
        assert "wl_1" in runner._seen_ids

    def test_whitelist_user_recorded_to_rag(self, runner, bot_root):
        """白名单用户的评论应索引到 RAG"""
        runner.load_config()
        runner.settings["bot"]["whitelist_users"] = ["maintainer"]
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("wl_2", "关于 CSM 架构说明", author="maintainer"),
        ]

        runner.llm_client = MagicMock()
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # RAG 索引应被调用
        runner.rag_retriever.index_human_reply.assert_called_once()

    def test_non_whitelist_user_processed_normally(self, runner, bot_root):
        """非白名单用户应正常走 AI 处理流程"""
        runner.load_config()
        runner.settings["bot"]["whitelist_users"] = ["maintainer"]
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("nwl_1", "CSM 是什么？", author="普通用户"),
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复内容", 100)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # LLM 应被调用
        runner.llm_client.generate_reply.assert_called_once()

    def test_empty_whitelist_no_skip(self, runner, bot_root):
        """空白名单不应跳过任何用户"""
        runner.load_config()
        runner.settings["bot"]["whitelist_users"] = []
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("ewl_1", "问题", author="anyone"),
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复", 50)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        runner.llm_client.generate_reply.assert_called_once()


# ===== 真人回复索引 question 取值测试（FIX-12）=====

class TestHumanReplyQuestion:
    """验证 _handle_human_reply 取最近 user 消息作 question（FIX-12）"""

    def test_human_reply_uses_last_user_message(self, runner, bot_root):
        """真人回复索引时 question 应为最近 user 消息而非 Bot 消息（FIX-12）"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("hr_1", "作者给出的人工回答", is_author_reply=True),
        ]

        runner.llm_client = MagicMock()
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        # 模拟 thread_manager 返回 Bot 回复在前、用户提问在后的历史
        runner.thread_manager = MagicMock()
        runner.thread_manager.get_or_create_thread.return_value = "mock_path"
        runner.thread_manager.build_context_messages.return_value = [
            {"role": "assistant", "content": "[rob]: 上一次 Bot 回复"},
            {"role": "user", "content": "用户的真实提问"},
        ]

        runner.process_article(runner.articles[0])

        call_args = runner.rag_retriever.index_human_reply.call_args
        question = call_args.kwargs.get("question", call_args[0][0] if call_args[0] else "")
        assert question == "用户的真实提问", (
            f"question 应为最近 user 消息，但实际为: {question!r}"
        )




class TestReplyIndexToRAG:
    """验证所有回复内容加入 RAG 学习"""

    def test_bot_reply_indexed_to_rag(self, runner, bot_root):
        """Bot 回复应被索引到 RAG"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("rag_1", "如何使用 CSM？"),
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("CSM 使用指南...", 100)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # RAG index_human_reply 应被调用（bot 回复也会被索引）
        runner.rag_retriever.index_human_reply.assert_called_once()
        call_args = runner.rag_retriever.index_human_reply.call_args
        assert "如何使用 CSM？" in call_args.kwargs.get("question", call_args[1].get("question", ""))
        # 验证 bot 回复内容包含 [rob]: 前缀
        reply_arg = call_args.kwargs.get("reply", call_args[1].get("reply", ""))
        assert "CSM 使用指南" in reply_arg

    def test_risky_reply_not_indexed_to_rag(self, runner, bot_root):
        """高危回复不应索引到 RAG（FIX-06）"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("risky_rag", "敏感问题"),
        ]

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复内容", 100)
        runner.llm_client.assess_risk.return_value = ("risky", "超出范围")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # 高危回复不应索引到 RAG
        runner.rag_retriever.index_human_reply.assert_not_called()

    def test_failed_post_not_indexed_to_rag(self, runner, bot_root):
        """发布失败的回复不应索引到 RAG（FIX-06）"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("fail_rag", "CSM 问题"),
        ]
        runner.zhihu_client.post_comment.return_value = False  # 发布失败

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复内容", 100)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # 发布失败不应索引到 RAG
        runner.rag_retriever.index_human_reply.assert_not_called()


# ===== RAG 检索使用评论内容测试（FIX-02 / TEST-01）=====

class TestRAGQueryByComment:
    """验证 RAG 检索使用评论内容而非文章标题（FIX-02 / TEST-01）"""

    def test_rag_retrieve_called_with_comment_content(self, runner, bot_root):
        """RAG retrieve 应以评论内容而非文章标题作为 query（FIX-02）"""
        runner.load_config()
        runner.init_modules()

        comment_text = "如何在 LabVIEW 中实现 CSM 状态机？"
        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("rag_q1", comment_text),
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复内容", 50)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # 验证 retrieve 被调用，且 query 参数为评论内容而非文章标题
        runner.rag_retriever.retrieve.assert_called_once()
        call_kwargs = runner.rag_retriever.retrieve.call_args.kwargs
        query_used = call_kwargs.get("query", runner.rag_retriever.retrieve.call_args[0][0] if runner.rag_retriever.retrieve.call_args[0] else "")
        assert query_used == comment_text, (
            f"RAG query 应为评论内容，但实际为: {query_used!r}"
        )

    def test_rag_retrieve_per_comment_not_shared(self, runner, bot_root):
        """多条评论各自独立检索（FIX-02）"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("rag_m1", "第一条评论", author="user_a"),
            _make_comment("rag_m2", "第二条评论", author="user_b"),
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复", 50)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # 两条评论各自触发一次 retrieve
        assert runner.rag_retriever.retrieve.call_count == 2
        queries = [
            call.kwargs.get("query", call[0][0] if call[0] else "")
            for call in runner.rag_retriever.retrieve.call_args_list
        ]
        assert "第一条评论" in queries
        assert "第二条评论" in queries


# ===== 文章摘要记录测试 =====

class TestArticleSummary:
    """验证文章使用 LLM 摘要而非全文"""

    def test_article_summary_stored_in_meta(self, runner, bot_root):
        """文章摘要应存储在 article_meta 中并传递给线程"""
        runner.load_config()
        runner.init_modules()

        # 文章有 content 时应调用 summarize_article（FIX-05）
        runner.articles = [{
            "id": "99999",
            "title": "测试文章",
            "url": "https://example.com",
            "type": "article",
            "content": "这是文章正文内容，介绍 CSM 方法论。",
        }]

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("sum_1", "问题"),
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复", 50)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = "LLM 生成的摘要"

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.thread_manager = MagicMock()
        runner.thread_manager.get_or_create_thread.return_value = "mock_path"
        runner.thread_manager.build_context_messages.return_value = []

        runner.process_article(runner.articles[0])

        # 验证 summarize_article 被调用
        runner.llm_client.summarize_article.assert_called_once()

        # 验证 get_or_create_thread 调用时 article_meta 包含 summary
        calls = runner.thread_manager.get_or_create_thread.call_args_list
        assert len(calls) > 0, "get_or_create_thread 应至少被调用一次"
        found_summary = False
        for call in calls:
            meta = call.kwargs.get("article_meta", call[1].get("article_meta", {}))
            if "summary" in meta:
                assert meta["summary"] == "LLM 生成的摘要"
                found_summary = True
        assert found_summary, "至少一次 get_or_create_thread 调用应包含 summary"

    def test_no_content_uses_title_as_summary(self, runner, bot_root):
        """无正文时应直接用 title 作为摘要，不调用 LLM（FIX-05）"""
        runner.load_config()
        runner.init_modules()

        # 文章无 content 字段（通常 articles.yaml 只有 title）
        runner.articles = [{
            "id": "99999",
            "title": "CSM 入门指南",
            "url": "https://example.com",
            "type": "article",
            # 无 content 字段
        }]

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("s1", "问题"),
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复", 50)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # 无正文时不应调用 summarize_article（直接用 title）
        runner.llm_client.summarize_article.assert_not_called()


# ===== max_new_comments_per_run 测试（FIX-16）=====

class TestMaxNewCommentsPerRun:
    """验证 max_new_comments_per_run 单次运行上限（FIX-16）"""

    def test_per_run_limit_respected(self, runner, bot_root):
        """max_new_comments_per_run 应限制每次 run 处理的评论数（FIX-16）"""
        runner.load_config()
        runner.settings["bot"]["max_new_comments_per_run"] = 2
        runner.init_modules()

        # 提供 5 条来自不同用户的评论
        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment(f"pr_{i}", f"问题{i}", author=f"user_{i}")
            for i in range(5)
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复", 50)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # 应只处理 2 条评论（max_new_comments_per_run=2）
        assert runner.llm_client.generate_reply.call_count == 2

    def test_no_per_run_limit_when_zero(self, runner, bot_root):
        """max_new_comments_per_run=0 表示无限制"""
        runner.load_config()
        runner.settings["bot"]["max_new_comments_per_run"] = 0
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment(f"pr_{i}", f"问题{i}", author=f"user_{i}")
            for i in range(4)
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复", 50)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # 无限制时 4 条全处理
        assert runner.llm_client.generate_reply.call_count == 4


# ===== 多级回复线程归档测试（Issue #3）=====

class TestCommentThreadMap:
    """验证评论 ID → 线程根 ID 映射（Issue #3：修复多级回复归档）"""

    def test_root_comment_maps_to_itself(self, runner, bot_root):
        """顶级评论（无 parent_id）的线程根应为其自身"""
        runner.load_config()
        root_comment = _make_comment("root_1", "顶级评论", parent_id=None)
        thread_root = runner._get_thread_root_id(root_comment)
        assert thread_root == "root_1"

    def test_reply_uses_parent_as_root_when_no_map(self, runner, bot_root):
        """无映射时，回复的线程根退回到直接父 ID"""
        runner.load_config()
        reply = _make_comment("reply_1", "回复", parent_id="root_1")
        thread_root = runner._get_thread_root_id(reply)
        assert thread_root == "root_1"

    def test_nested_reply_follows_map(self, runner, bot_root):
        """多级嵌套回复应通过映射找到真正的根"""
        runner.load_config()
        # 模拟：root_1 → root_1, reply_1 → root_1
        runner._comment_thread_map = {"reply_1": "root_1"}
        nested = _make_comment("nested_1", "嵌套回复", parent_id="reply_1")
        thread_root = runner._get_thread_root_id(nested)
        assert thread_root == "root_1"

    def test_map_updated_after_processing_root(self, runner, bot_root):
        """处理顶级评论后，映射应记录 comment_id → comment_id"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("root_1", "顶级评论", parent_id=None),
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复内容", 50)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        assert "root_1" in runner._comment_thread_map
        assert runner._comment_thread_map["root_1"] == "root_1"

    def test_map_updated_after_processing_reply(self, runner, bot_root):
        """处理回复评论后，映射应记录 reply_id → root_id"""
        runner.load_config()
        runner.init_modules()

        # root 已在映射中
        runner._comment_thread_map = {"root_1": "root_1"}

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("reply_1", "回复评论", parent_id="root_1"),
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复内容", 50)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        assert runner._comment_thread_map.get("reply_1") == "root_1"

    def test_multilevel_all_go_to_same_thread(self, runner, bot_root):
        """多级回复链应全部归入同一线程文件"""
        runner.load_config()
        runner.init_modules()

        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("root_1", "根评论", parent_id=None),
            _make_comment("reply_1", "一级回复", parent_id="root_1"),
            _make_comment("reply_2", "二级回复", parent_id="reply_1"),
        ]
        runner.zhihu_client.post_comment.return_value = True

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复内容", 50)
        runner.llm_client.assess_risk.return_value = ("safe", "安全")
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_article(runner.articles[0])

        # 三条评论应全部映射到 root_1
        assert runner._comment_thread_map.get("root_1") == "root_1"
        assert runner._comment_thread_map.get("reply_1") == "root_1"
        assert runner._comment_thread_map.get("reply_2") == "root_1"

        # 线程文件应只有一个（root_1.md），不应有 reply_1.md 或 reply_2.md
        thread_dir = bot_root / "archive" / "articles" / "99999" / "threads"
        if thread_dir.exists():
            thread_files = list(thread_dir.glob("*.md"))
            thread_ids = {f.stem for f in thread_files}
            assert "root_1" in thread_ids, "root_1.md 应存在"
            assert "reply_1" not in thread_ids, "reply_1.md 不应存在（应归入 root_1.md）"
            assert "reply_2" not in thread_ids, "reply_2.md 不应存在（应归入 root_1.md）"

    def test_comment_thread_map_persisted(self, runner, bot_root):
        """评论线程映射应能保存和重新加载"""
        runner.load_config()
        runner._comment_thread_map = {"c1": "r1", "c2": "r1", "c3": "r2"}
        runner.save_comment_thread_map()

        runner2 = BotRunner(project_root=runner.root)
        runner2.load_config()
        runner2.load_comment_thread_map()
        assert runner2._comment_thread_map == {"c1": "r1", "c2": "r1", "c3": "r2"}

    def test_comment_thread_map_empty_when_no_file(self, runner, bot_root):
        """无文件时应返回空映射"""
        runner.load_config()
        runner.load_comment_thread_map()
        assert runner._comment_thread_map == {}

    def test_deep_chain_resolved_to_root(self, runner, bot_root):
        """深层嵌套回复链应迭代追溯到最终根，而非仅做一次跳转"""
        runner.load_config()
        # chain: c4 → c3 → c2 → c1 → c1（c1 是根）
        runner._comment_thread_map = {
            "c1": "c1",
            "c2": "c1",
            "c3": "c2",   # 指向中间节点，仅做一次跳转时会返回 c2 而非 c1
        }
        deep = _make_comment("c4", "深层回复", parent_id="c3")
        thread_root = runner._get_thread_root_id(deep)
        assert thread_root == "c1", (
            f"深层回复链应追溯到真正的根 c1，但得到 {thread_root!r}"
        )

    def test_cycle_in_map_does_not_loop_forever(self, runner, bot_root):
        """映射中存在循环时，_get_thread_root_id 应安全返回而不死循环"""
        runner.load_config()
        # 人工构造一个损坏的循环映射
        runner._comment_thread_map = {
            "x": "y",
            "y": "x",
        }
        cycle = _make_comment("z", "某评论", parent_id="x")
        # 不应死循环，应返回某个非空字符串
        result = runner._get_thread_root_id(cycle)
        assert isinstance(result, str) and result != ""

    def test_path_compression_applied(self, runner, bot_root):
        """路径压缩：查询后中间节点应直接指向最终根"""
        runner.load_config()
        runner._comment_thread_map = {
            "c1": "c1",
            "c2": "c1",
            "c3": "c2",  # 未压缩时 c3 指向 c2，压缩后应指向 c1
        }
        c4 = _make_comment("c4", "深层", parent_id="c3")
        runner._get_thread_root_id(c4)
        # 路径压缩后，c3 应直接指向 c1
        assert runner._comment_thread_map.get("c3") == "c1"


# ===== RecentLogsHandler 测试 =====

class TestRecentLogsHandler:
    """验证内存日志缓冲处理器"""

    def test_captures_log_records(self):
        """发出的日志记录应被捕获到内部缓冲区"""
        import logging
        handler = RecentLogsHandler(maxlen=10)
        log = logging.getLogger("test.recent_logs_handler")
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
        try:
            log.error("测试错误信息")
        finally:
            log.removeHandler(handler)

        result = handler.format_markdown(10)
        assert "测试错误信息" in result

    def test_maxlen_respected(self):
        """超过 maxlen 时旧记录应被丢弃"""
        import logging
        handler = RecentLogsHandler(maxlen=3)
        log = logging.getLogger("test.maxlen")
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
        try:
            for i in range(5):
                log.info(f"消息{i}")
        finally:
            log.removeHandler(handler)

        result = handler.format_markdown(10)
        # 只应保留最后 3 条（消息2、消息3、消息4）
        assert "消息0" not in result
        assert "消息1" not in result
        assert "消息4" in result

    def test_format_markdown_empty(self):
        """无日志时 format_markdown 应返回空字符串"""
        handler = RecentLogsHandler()
        assert handler.format_markdown() == ""

    def test_format_markdown_contains_code_block(self):
        """有日志时 format_markdown 应返回 Markdown 代码块"""
        import logging
        handler = RecentLogsHandler()
        log = logging.getLogger("test.codeblock")
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
        try:
            log.warning("告警消息")
        finally:
            log.removeHandler(handler)

        result = handler.format_markdown()
        assert result.startswith("```")
        assert result.endswith("```")

    def test_consecutive_failures_alert_includes_logs(self, runner, bot_root):
        """连续失败告警时应将内存日志传入 alert_consecutive_failures"""
        runner.load_config()
        runner.init_modules()

        # 注入必要 mock
        runner.zhihu_client = MagicMock()
        # 使用不同作者避免 dedup 过滤（dedup_window_minutes 按 author 去重）
        runner.zhihu_client.get_comments.return_value = [
            _make_comment("fail_1", "评论A", author="user_a"),
            _make_comment("fail_2", "评论B", author="user_b"),
            _make_comment("fail_3", "评论C", author="user_c"),
        ]
        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.side_effect = Exception("模拟处理失败")
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""
        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []
        runner.alert_manager = MagicMock()

        runner.load_seen_ids()
        runner._bootstrapped_articles.add("99999")
        runner.run_articles()

        # alert_consecutive_failures 应被调用，且 recent_logs 参数非空
        runner.alert_manager.alert_consecutive_failures.assert_called_once()
        call_kwargs = runner.alert_manager.alert_consecutive_failures.call_args
        recent_logs = call_kwargs.kwargs.get("recent_logs", "")
        assert recent_logs, "recent_logs 应非空，以便 Issue 包含最近日志"
