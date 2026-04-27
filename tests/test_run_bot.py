# -*- coding: utf-8 -*-
"""
BotRunner 单元测试
==================

覆盖：
- 消息处理、过滤、LLM 回复、写入 pending/
- 已处理 ID 正确持久化
- BudgetExceededError → alert + 终止
- 专家回复 → index_human_reply
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from scripts.run_bot import BotRunner, RecentLogsHandler, Message
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


def _make_message(
    id: str, content: str, author: str = "user",
    is_author_reply: bool = False, parent_id: str | None = None,
) -> Message:
    """构造 Message 对象"""
    return Message(
        id=id,
        content=content,
        author=author,
        created_time=1712000000,
        parent_id=parent_id,
        is_author_reply=is_author_reply,
    )


# Keep backward-compatible alias for existing test code
_make_comment = _make_message


# ===== 配置加载测试 =====

class TestLoadConfig:
    """验证配置加载"""

    def test_load_config_success(self, runner):
        """应成功加载配置"""
        runner.load_config()
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


# ===== 消息处理测试 =====

class TestProcessMessages:
    """验证消息处理流程"""

    def test_no_new_messages(self, runner):
        """无新消息时应安全跳过"""
        runner.load_config()
        runner.init_modules()

        runner.process_messages([])
        assert runner._processed_count == 0

    def test_skip_seen_messages(self, runner):
        """已处理过的消息应被跳过"""
        runner.load_config()
        runner.init_modules()

        runner._seen_ids = {"seen_1"}
        runner.llm_client = MagicMock()

        runner.process_messages(
            [_make_message("seen_1", "旧消息")],
            topic={"id": "99999", "title": "测试话题"},
        )
        runner.llm_client.generate_reply.assert_not_called()

    def test_new_message_generates_reply(self, runner, bot_root):
        """新消息应生成回复并写入 pending/"""
        runner.load_config()
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = (
            "CSM 是 Communicable State Machine（通信状态机）框架的缩写。", 150
        )
        runner.llm_client.total_cost_usd = 0.001
        runner.llm_client.total_prompt_tokens = 100
        runner.llm_client.total_completion_tokens = 50
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = "摘要"

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message("new_1", "CSM 是什么？")],
            topic={"id": "99999", "title": "测试文章", "url": "https://example.com"},
        )

        # 验证回复被写入 pending/
        pending_files = list((bot_root / "data" / "pending").glob("*.md"))
        assert len(pending_files) >= 1

        # 验证 [rob]: 前缀被添加
        content = pending_files[0].read_text()
        assert "[rob]:" in content

        # 验证 seen_ids 更新
        assert "new_1" in runner._seen_ids

    def test_spam_message_skipped(self, runner):
        """广告消息应被跳过"""
        runner.load_config()
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message("spam_1", "加微信了解更多")],
            topic={"id": "99999", "title": "测试"},
        )

        # LLM 不应被调用（广告被过滤）
        runner.llm_client.generate_reply.assert_not_called()
        # 但 seen_ids 应记录
        assert "spam_1" in runner._seen_ids

    def test_filtered_message_recorded_to_thread(self, runner, bot_root):
        """被过滤的消息也应归档到对话线程，以供后期学习"""
        runner.load_config()
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.thread_manager = MagicMock()
        runner.thread_manager.get_or_create_thread.return_value = "mock_thread_path"

        runner.process_messages(
            [_make_message("spam_2", "加微信了解更多")],
            topic={"id": "99999", "title": "测试"},
        )

        # LLM 不应被调用（被过滤）
        runner.llm_client.generate_reply.assert_not_called()
        # 但线程记录应被调用（归档供学习）
        runner.thread_manager.append_turn.assert_called_once()
        call_kwargs = runner.thread_manager.append_turn.call_args[1]
        assert call_kwargs["author"] == "user"
        assert call_kwargs["content"] == "加微信了解更多"

    def test_expert_reply_indexed(self, runner):
        """专家回复应被索引"""
        runner.load_config()
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message(
                "expert_1", "这是专家的回复",
                author="专家",
                is_author_reply=True,
                parent_id="parent_1",
            )],
            topic={"id": "99999", "title": "测试"},
        )

        # 验证 index_human_reply 被调用
        runner.rag_retriever.index_human_reply.assert_called_once()


# ===== 异常处理测试 =====

class TestExceptionHandling:
    """验证异常场景"""

    def test_budget_exceeded_triggers_alert(self, runner, bot_root):
        """BudgetExceededError 应触发告警并终止（通过 run_inbox）"""
        runner.load_config()
        runner.init_modules()

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

        # 准备一条 inbox 消息
        inbox_dir = bot_root / "data" / "inbox"
        inbox_dir.mkdir(parents=True, exist_ok=True)
        msg_data = {
            "id": "budget_test_1",
            "content": "触发预算超限的消息",
            "author": "user",
            "created_time": 1712000000,
            "is_author_reply": False,
        }
        with open(inbox_dir / "msg1.json", "w") as f:
            json.dump(msg_data, f)

        # run_inbox 应捕获 BudgetExceededError 并触发告警，不向上传播
        runner.run_inbox()

        runner.alert_manager.alert_budget_exceeded.assert_called_once_with(
            cost=0.55,
            budget=runner.settings["bot"]["llm_budget_usd_per_day"],
        )


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
        topic = {"id": "111", "title": "测试文章"}

        runner._write_pending(
            topic=topic,
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
        """pending 文件名格式: {topic_id}_{comment_id}.md"""
        runner.load_config()
        topic = {"id": "AAA", "title": ""}

        runner._write_pending(topic, "评论", "回复", "BBB")

        assert (bot_root / "data" / "pending" / "AAA_BBB.md").exists()

    def test_pending_yaml_safe_with_quotes_in_risk_reason(self, runner, bot_root):
        """BUG-FIX-02: risk_reason 含双引号时 frontmatter 应仍可被正确解析"""
        import frontmatter as fm

        runner.load_config()
        topic = {"id": "QQ", "title": 'Title with "quotes"'}

        runner._write_pending(
            topic=topic,
            comment_content="评论内容",
            reply_content="回复内容",
            comment_id="RR",
            risk_reason='包含 "双引号" 的风险理由',
        )

        filepath = bot_root / "data" / "pending" / "QQ_RR.md"
        assert filepath.exists()

        # frontmatter.load 必须能正常解析，不抛出异常
        post = fm.load(str(filepath))
        assert post.metadata.get("article_id") == "QQ"
        assert post.metadata.get("comment_id") == "RR"
        assert '"双引号"' in post.metadata.get("risk_reason", "")


# ===== 回复总是写入 pending 测试 =====

class TestReplyAlwaysPending:
    """验证所有 AI 回复均写入 pending/ 等待人工审核"""

    def test_reply_goes_to_pending(self, runner, bot_root):
        """AI 生成的回复应写入 pending/"""
        runner.load_config()
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = (
            "CSM 框架使用说明...", 100
        )
        runner.llm_client.total_cost_usd = 0.001
        runner.llm_client.total_prompt_tokens = 80
        runner.llm_client.total_completion_tokens = 40
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = "摘要"

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message("reply_1", "CSM 框架怎么用？")],
            topic={"id": "99999", "title": "测试"},
        )

        pending_files = list((bot_root / "data" / "pending").glob("*.md"))
        assert len(pending_files) == 1

    def test_reply_goes_to_pending_with_content(self, runner, bot_root):
        """pending 文件应包含回复内容"""
        runner.load_config()
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("建议咨询专业人士...", 100)
        runner.llm_client.total_cost_usd = 0.001
        runner.llm_client.total_prompt_tokens = 80
        runner.llm_client.total_completion_tokens = 40
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = "摘要"

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message("risky_1", "公司财务怎么处理？")],
            topic={"id": "99999", "title": "测试"},
        )

        pending_files = list((bot_root / "data" / "pending").glob("*.md"))
        assert len(pending_files) == 1
        content = pending_files[0].read_text()
        assert "建议咨询专业人士" in content


# ===== [rob]: 回复前缀测试 =====

class TestReplyPrefix:
    """验证回复前缀功能"""

    def test_reply_prefix_added(self, runner, bot_root):
        """回复应包含 [rob]: 前缀"""
        runner.load_config()
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("原始回复", 50)
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message("prefix_1", "测试")],
            topic={"id": "99999", "title": "测试"},
        )

        # 验证写入 pending 的内容包含 [rob]: 前缀
        pending_files = list((bot_root / "data" / "pending").glob("*.md"))
        assert len(pending_files) == 1
        content = pending_files[0].read_text()
        assert "[rob]:" in content


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


# ===== 白名单用户测试 =====

class TestWhitelistUsers:
    """验证白名单用户仅记录不做 AI 处理"""

    def test_whitelist_user_skips_ai(self, runner, bot_root):
        """白名单用户的消息不应触发 LLM"""
        runner.load_config()
        runner.settings["bot"]["whitelist_users"] = ["maintainer"]
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message("wl_1", "已修复", author="maintainer")],
            topic={"id": "99999", "title": "测试"},
        )

        # LLM 不应被调用
        runner.llm_client.generate_reply.assert_not_called()
        # 但消息应被记录到 seen_ids
        assert "wl_1" in runner._seen_ids

    def test_whitelist_user_recorded_to_rag(self, runner, bot_root):
        """白名单用户的消息应索引到 RAG"""
        runner.load_config()
        runner.settings["bot"]["whitelist_users"] = ["maintainer"]
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message("wl_2", "关于 CSM 架构说明", author="maintainer")],
            topic={"id": "99999", "title": "测试"},
        )

        # RAG 索引应被调用
        runner.rag_retriever.index_human_reply.assert_called_once()

    def test_non_whitelist_user_processed_normally(self, runner, bot_root):
        """非白名单用户应正常走 AI 处理流程"""
        runner.load_config()
        runner.settings["bot"]["whitelist_users"] = ["maintainer"]
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复内容", 100)
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message("nwl_1", "CSM 是什么？", author="普通用户")],
            topic={"id": "99999", "title": "测试"},
        )

        # LLM 应被调用
        runner.llm_client.generate_reply.assert_called_once()

    def test_empty_whitelist_no_skip(self, runner, bot_root):
        """空白名单不应跳过任何用户"""
        runner.load_config()
        runner.settings["bot"]["whitelist_users"] = []
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复", 50)
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message("ewl_1", "问题", author="anyone")],
            topic={"id": "99999", "title": "测试"},
        )

        runner.llm_client.generate_reply.assert_called_once()


# ===== 专家回复索引 question 取值测试（FIX-12）=====

class TestHumanReplyQuestion:
    """验证 _handle_human_reply 取最近 user 消息作 question（FIX-12）"""

    def test_human_reply_uses_last_user_message(self, runner, bot_root):
        """专家回复索引时 question 应为最近 user 消息而非 Bot 消息（FIX-12）"""
        runner.load_config()
        runner.init_modules()


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

        runner.process_messages(
            [_make_message("hr_1", "作者给出的人工回答", is_author_reply=True)],
            topic={"id": "99999", "title": "测试"},
        )

        call_args = runner.rag_retriever.index_human_reply.call_args
        question = call_args.kwargs.get("question", call_args[0][0] if call_args[0] else "")
        assert question == "用户的真实提问", (
            f"question 应为最近 user 消息，但实际为: {question!r}"
        )

    def test_human_reply_question_not_self_indexed(self, runner, bot_root):
        """BUG-FIX-01（root fix）: _parse_turns 将"真人回复"识别为 assistant role，
        因此历史中的作者真人回复不会被 reversed 搜索误当成"用户提问"索引到 RAG。
        本测试用真实 ThreadManager 验证 question 抽取正确。"""
        from scripts.thread_manager import ThreadManager

        runner.load_config()
        runner.init_modules()
        # 使用真实的 ThreadManager
        runner.thread_manager = ThreadManager(archive_dir=str(bot_root / "archive"))

        article = {"id": "art1", "title": "Test Article", "url": "http://example.com"}

        # 在线程里预先写入一条用户提问（模拟之前的 bot 交互）
        thread_path = runner.thread_manager.get_or_create_thread(
            article_id="art1",
            root_comment={"id": "uc1", "author": "alice"},
            article_meta={"title": "Test Article", "url": "http://example.com"},
        )
        runner.thread_manager.append_turn(
            thread_path=thread_path,
            author="alice",
            content="CSM 是什么？",
            comment_id="uc1",
        )
        runner._comment_thread_map["uc1"] = "uc1"

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []
        runner.llm_client = MagicMock()
        runner.llm_client.summarize_article.return_value = ""

        # 作者人工回复该提问
        human_comment = _make_message(
            "hr2",
            "CSM 是 Communicable State Machine",
            is_author_reply=True,
            parent_id="uc1",
        )
        runner._comment_thread_map["hr2"] = "uc1"
        runner._seen_ids_path.parent.mkdir(parents=True, exist_ok=True)
        runner._seen_ids = set()

        runner.process_messages([human_comment], topic=article)

        call_args = runner.rag_retriever.index_human_reply.call_args
        assert call_args is not None, "index_human_reply 应被调用"
        # 兼容 keyword 参数和位置参数两种调用形式
        question = call_args.kwargs.get("question")
        if question is None and call_args.args:
            question = call_args.args[0]
        question = question or ""
        # 正确行为：question 应该是用户的提问，不应该是作者的回复内容
        assert "CSM 是什么" in question, (
            f"question 应为用户提问 'CSM 是什么？'，实际为: {question!r}. "
            "若此测试失败说明 _parse_turns 未将真人回复识别为 assistant role，"
            "导致真人回复内容被当成问题索引。"
        )
        assert "Communicable State Machine" not in question, (
            f"question 不应为作者的回复内容，实际为: {question!r}"
        )

    def test_prior_human_reply_not_mistaken_as_question(self, runner, bot_root):
        """BUG-FIX-01 扩展：即使线程历史中已有多条真人回复，
        reversed 搜索也应跳过所有 assistant 消息，返回最近的 user 提问。"""
        from scripts.thread_manager import ThreadManager

        runner.load_config()
        runner.init_modules()
        runner.thread_manager = ThreadManager(archive_dir=str(bot_root / "archive"))

        article = {"id": "art2", "title": "Test Article 2", "url": "http://example.com"}

        # 线程历史：用户提问 → 真人回复 → 用户追问（当前需要索引的真人回复对应的提问）
        thread_path = runner.thread_manager.get_or_create_thread(
            article_id="art2",
            root_comment={"id": "q1", "author": "bob"},
            article_meta={"title": "Test Article 2", "url": "http://example.com"},
        )
        runner.thread_manager.append_turn(thread_path, "bob", "第一个问题", comment_id="q1")
        runner.thread_manager.append_turn(thread_path, "author", "第一个答案", is_human=True)
        runner.thread_manager.append_turn(thread_path, "bob", "第二个问题", comment_id="q2")
        runner._comment_thread_map["q1"] = "q1"
        runner._comment_thread_map["q2"] = "q1"

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []
        runner.llm_client = MagicMock()
        runner.llm_client.summarize_article.return_value = ""

        # 作者回复第二个问题
        human_comment2 = _make_message(
            "hr3", "第二个答案", is_author_reply=True, parent_id="q2"
        )
        runner._comment_thread_map["hr3"] = "q1"
        runner._seen_ids_path.parent.mkdir(parents=True, exist_ok=True)
        runner._seen_ids = set()

        runner.process_messages([human_comment2], topic=article)

        call_args = runner.rag_retriever.index_human_reply.call_args
        assert call_args is not None
        question = call_args.kwargs.get("question")
        if question is None and call_args.args:
            question = call_args.args[0]
        question = question or ""
        # 应取到"第二个问题"，而不是更早的"第一个答案"（作者上一次真人回复）
        assert "第二个问题" in question, (
            f"question 应为最近用户提问'第二个问题'，实际为: {question!r}"
        )
        assert "第一个答案" not in question, (
            f"question 不应为历史真人回复内容，实际为: {question!r}"
        )




class TestReplyIndexToRAG:
    """验证所有回复内容加入 RAG 学习"""

    def test_bot_reply_indexed_to_rag(self, runner, bot_root):
        """Bot 回复应被索引到 RAG"""
        runner.load_config()
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("CSM 使用指南...", 100)
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message("rag_1", "如何使用 CSM？")],
            topic={"id": "99999", "title": "测试"},
        )

        # RAG index_human_reply 应被调用（bot 回复也会被索引）
        runner.rag_retriever.index_human_reply.assert_called_once()
        call_args = runner.rag_retriever.index_human_reply.call_args
        assert "如何使用 CSM？" in call_args.kwargs.get("question", call_args[1].get("question", ""))
        # 验证 bot 回复内容包含 [rob]: 前缀
        reply_arg = call_args.kwargs.get("reply", call_args[1].get("reply", ""))
        assert "CSM 使用指南" in reply_arg


# ===== RAG 检索使用评论内容测试（FIX-02 / TEST-01）=====

class TestRAGQueryByComment:
    """验证 RAG 检索使用消息内容而非话题标题（FIX-02 / TEST-01）"""

    def test_rag_retrieve_called_with_comment_content(self, runner, bot_root):
        """RAG retrieve 应以消息内容作为 query（FIX-02）"""
        runner.load_config()
        runner.init_modules()

        comment_text = "如何在 LabVIEW 中实现 CSM 状态机？"

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复内容", 50)
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message("rag_q1", comment_text)],
            topic={"id": "99999", "title": "测试文章"},
        )

        # 验证 retrieve 被调用，且 query 参数为消息内容而非话题标题
        runner.rag_retriever.retrieve.assert_called_once()
        call_kwargs = runner.rag_retriever.retrieve.call_args.kwargs
        query_used = call_kwargs.get("query", runner.rag_retriever.retrieve.call_args[0][0] if runner.rag_retriever.retrieve.call_args[0] else "")
        assert query_used == comment_text, (
            f"RAG query 应为消息内容，但实际为: {query_used!r}"
        )

    def test_rag_retrieve_per_comment_not_shared(self, runner, bot_root):
        """多条消息各自独立检索（FIX-02）"""
        runner.load_config()
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复", 50)
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [
                _make_message("rag_m1", "第一条消息", author="user_a"),
                _make_message("rag_m2", "第二条消息", author="user_b"),
            ],
            topic={"id": "99999", "title": "测试文章"},
        )

        # 两条消息各自触发一次 retrieve
        assert runner.rag_retriever.retrieve.call_count == 2
        queries = [
            call.kwargs.get("query", call[0][0] if call[0] else "")
            for call in runner.rag_retriever.retrieve.call_args_list
        ]
        assert "第一条消息" in queries
        assert "第二条消息" in queries


# ===== 话题摘要记录测试 =====

class TestArticleSummary:
    """验证话题使用 LLM 摘要而非全文"""

    def test_article_summary_stored_in_meta(self, runner, bot_root):
        """话题摘要应存储在 topic_meta 中并传递给线程"""
        runner.load_config()
        runner.init_modules()

        topic = {
            "id": "99999",
            "title": "测试文章",
            "url": "https://example.com",
            "content": "这是文章正文内容，介绍 CSM 方法论。",
        }

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复", 50)
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

        runner.process_messages(
            [_make_message("sum_1", "问题")],
            topic=topic,
        )

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

        topic = {
            "id": "99999",
            "title": "CSM 入门指南",
            "url": "https://example.com",
            # 无 content 字段
        }

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复", 50)
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message("s1", "问题")],
            topic=topic,
        )

        # 无正文时不应调用 summarize_article（直接用 title）
        runner.llm_client.summarize_article.assert_not_called()


# ===== max_new_comments_per_run 测试（FIX-16）=====

class TestMaxNewCommentsPerRun:
    """验证 max_new_comments_per_run 单次运行上限（FIX-16）"""

    def test_per_run_limit_respected(self, runner, bot_root):
        """max_new_comments_per_run 应限制每次 run 处理的消息数（FIX-16）"""
        runner.load_config()
        runner.settings["bot"]["max_new_comments_per_run"] = 2
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复", 50)
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message(f"pr_{i}", f"问题{i}", author=f"user_{i}") for i in range(5)],
            topic={"id": "99999", "title": "测试"},
        )

        # 应只处理 2 条消息（max_new_comments_per_run=2）
        assert runner.llm_client.generate_reply.call_count == 2

    def test_no_per_run_limit_when_zero(self, runner, bot_root):
        """max_new_comments_per_run=0 表示无限制"""
        runner.load_config()
        runner.settings["bot"]["max_new_comments_per_run"] = 0
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复", 50)
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message(f"pr_{i}", f"问题{i}", author=f"user_{i}") for i in range(4)],
            topic={"id": "99999", "title": "测试"},
        )

        # 无限制时 4 条全处理
        assert runner.llm_client.generate_reply.call_count == 4


# ===== 多级回复线程归档测试（Issue #3）=====

class TestCommentThreadMap:
    """验证消息 ID → 线程根 ID 映射（Issue #3：修复多级回复归档）"""

    def test_root_comment_maps_to_itself(self, runner, bot_root):
        """顶级消息（无 parent_id）的线程根应为其自身"""
        runner.load_config()
        root_msg = _make_message("root_1", "顶级消息", parent_id=None)
        thread_root = runner._get_thread_root_id(root_msg)
        assert thread_root == "root_1"

    def test_reply_uses_parent_as_root_when_no_map(self, runner, bot_root):
        """无映射时，回复的线程根退回到直接父 ID"""
        runner.load_config()
        reply = _make_message("reply_1", "回复", parent_id="root_1")
        thread_root = runner._get_thread_root_id(reply)
        assert thread_root == "root_1"

    def test_nested_reply_follows_map(self, runner, bot_root):
        """多级嵌套回复应通过映射找到真正的根"""
        runner.load_config()
        # 模拟：root_1 → root_1, reply_1 → root_1
        runner._comment_thread_map = {"reply_1": "root_1"}
        nested = _make_message("nested_1", "嵌套回复", parent_id="reply_1")
        thread_root = runner._get_thread_root_id(nested)
        assert thread_root == "root_1"

    def test_map_updated_after_processing_root(self, runner, bot_root):
        """处理顶级消息后，映射应记录 msg_id → msg_id"""
        runner.load_config()
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复内容", 50)
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message("root_1", "顶级消息", parent_id=None)],
            topic={"id": "99999", "title": "测试"},
        )

        assert "root_1" in runner._comment_thread_map
        assert runner._comment_thread_map["root_1"] == "root_1"

    def test_map_updated_after_processing_reply(self, runner, bot_root):
        """处理回复消息后，映射应记录 reply_id → root_id"""
        runner.load_config()
        runner.init_modules()

        # root 已在映射中
        runner._comment_thread_map = {"root_1": "root_1"}

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复内容", 50)
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [_make_message("reply_1", "回复消息", parent_id="root_1")],
            topic={"id": "99999", "title": "测试"},
        )

        assert runner._comment_thread_map.get("reply_1") == "root_1"

    def test_multilevel_all_go_to_same_thread(self, runner, bot_root):
        """多级回复链应全部归入同一线程文件"""
        runner.load_config()
        runner.init_modules()

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.return_value = ("回复内容", 50)
        runner.llm_client.total_cost_usd = 0.0
        runner.llm_client.total_prompt_tokens = 0
        runner.llm_client.total_completion_tokens = 0
        runner.llm_client.total_cache_hit_tokens = 0
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""

        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []

        runner.process_messages(
            [
                _make_message("root_1", "根消息", parent_id=None),
                _make_message("reply_1", "一级回复", parent_id="root_1"),
                _make_message("reply_2", "二级回复", parent_id="reply_1"),
            ],
            topic={"id": "99999", "title": "测试"},
        )

        # 三条消息应全部映射到 root_1
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
        """消息线程映射应能保存和重新加载"""
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
        deep = _make_message("c4", "深层回复", parent_id="c3")
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
        cycle = _make_message("z", "某消息", parent_id="x")
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
        c4 = _make_message("c4", "深层", parent_id="c3")
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

        runner.llm_client = MagicMock()
        runner.llm_client.generate_reply.side_effect = Exception("模拟处理失败")
        runner.llm_client.model = "deepseek-chat"
        runner.llm_client.summarize_article.return_value = ""
        runner.rag_retriever = MagicMock()
        runner.rag_retriever.retrieve.return_value = []
        runner.alert_manager = MagicMock()

        runner.load_seen_ids()
        runner.process_messages(
            [
                _make_message("fail_1", "消息A", author="user_a"),
                _make_message("fail_2", "消息B", author="user_b"),
                _make_message("fail_3", "消息C", author="user_c"),
            ],
            topic={"id": "99999", "title": "测试"},
        )

        # alert_consecutive_failures 应被调用，且 recent_logs 参数非空
        runner.alert_manager.alert_consecutive_failures.assert_called_once()
        call_kwargs = runner.alert_manager.alert_consecutive_failures.call_args
        recent_logs = call_kwargs.kwargs.get("recent_logs", "")
        assert recent_logs, "recent_logs 应非空，以便 Issue 包含最近日志"
