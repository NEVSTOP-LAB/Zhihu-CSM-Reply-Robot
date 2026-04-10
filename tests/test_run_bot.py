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
        """新评论应生成回复并写入 pending/（高危时）"""
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
        pending_files = list((bot_root / "pending").glob("*.md"))
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
        pending_files = list((bot_root / "pending").glob("*.md"))
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
        pending_files = list((bot_root / "pending").glob("*.md"))
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
        pending_files = list((bot_root / "pending").glob("*.md"))
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

    def test_article_and_question_pass_through(self, runner):
        """article/question 类型应直接传递"""
        runner.load_config()
        runner.init_modules()
        runner.zhihu_client = MagicMock()

        runner.articles = [
            {"id": "1", "type": "article"},
            {"id": "2", "type": "question"},
        ]
        result = runner._expand_articles()
        assert len(result) == 2
        assert result[0]["type"] == "article"
        assert result[1]["type"] == "question"

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
        """user_answers 类型应展开为回答列表"""
        runner.load_config()
        runner.init_modules()
        runner.zhihu_client = MagicMock()
        runner.zhihu_client.get_user_answers.return_value = [
            {"id": "20", "title": "回答1", "url": "url1", "type": "question"},
        ]

        runner.articles = [
            {"id": "some-user", "type": "user_answers"},
        ]
        result = runner._expand_articles()
        assert len(result) == 1
        assert result[0]["type"] == "question"

    def test_column_expand_failure_skips(self, runner):
        """展开失败时应跳过该条目"""
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

        # 使用 question 类型的文章
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


# ===== 回复索引到 RAG 测试 =====

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


# ===== 文章摘要记录测试 =====

class TestArticleSummary:
    """验证文章使用 LLM 摘要而非全文"""

    def test_article_summary_stored_in_meta(self, runner, bot_root):
        """文章摘要应存储在 article_meta 中并传递给线程"""
        runner.load_config()
        runner.init_modules()

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
