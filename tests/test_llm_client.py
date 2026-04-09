# -*- coding: utf-8 -*-
"""
LLMClient 单元测试
==================

实施计划关联：AI-006 验收标准
独立于实现的测试用例，覆盖：
- System Prompt 固定前缀（缓存友好）
- history_messages 正确拼接到 messages 列表
- 重试逻辑（前2次失败，第3次成功）
- 超预算时抛出 BudgetExceededError
- summarize_article 结果被缓存（第二次不触发 API）
"""
from unittest.mock import MagicMock, patch, PropertyMock
from types import SimpleNamespace

import pytest

from scripts.llm_client import LLMClient, BudgetExceededError, PRICING


# ===== Fixtures =====

def _make_mock_response(
    content: str = "这是一个测试回复",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    cached_tokens: int = 0,
):
    """构造模拟的 OpenAI API 响应"""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
    )
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def client():
    """创建测试用 LLMClient（使用假 API Key）"""
    return LLMClient(
        api_key="test-api-key",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        max_tokens=250,
        temperature=0.7,
        budget_usd_per_day=0.50,
        max_retries=3,
    )


# ===== System Prompt 测试 =====

class TestSystemPrompt:
    """验证 System Prompt 结构（缓存友好）"""

    @patch("scripts.llm_client.time.sleep")
    def test_system_prompt_has_fixed_prefix(self, mock_sleep, client):
        """System Prompt 应以固定前缀开头（最大化缓存命中）"""
        mock_response = _make_mock_response()

        with patch.object(
            client._client.chat.completions, "create",
            return_value=mock_response
        ) as mock_create:
            client.generate_reply(
                comment="测试评论",
                context_chunks=[],
                article_summary="",
            )

        call_args = mock_create.call_args
        messages = call_args[1]["messages"]

        # 第一条消息应为 system
        assert messages[0]["role"] == "system"
        # 应包含固定的角色定义前缀
        assert "客户成功" in messages[0]["content"]
        assert "回复规则" in messages[0]["content"]

    @patch("scripts.llm_client.time.sleep")
    def test_system_prompt_includes_context(self, mock_sleep, client):
        """System Prompt 应包含 RAG 检索结果"""
        mock_response = _make_mock_response()

        with patch.object(
            client._client.chat.completions, "create",
            return_value=mock_response
        ) as mock_create:
            client.generate_reply(
                comment="测试",
                context_chunks=["CSM 核心概念", "投诉处理流程"],
                article_summary="本文讨论 CSM",
            )

        messages = mock_create.call_args[1]["messages"]
        system = messages[0]["content"]
        assert "CSM 核心概念" in system
        assert "投诉处理流程" in system
        assert "参考资料" in system

    @patch("scripts.llm_client.time.sleep")
    def test_system_prompt_includes_article_summary(self, mock_sleep, client):
        """System Prompt 应包含文章摘要"""
        mock_response = _make_mock_response()

        with patch.object(
            client._client.chat.completions, "create",
            return_value=mock_response
        ) as mock_create:
            client.generate_reply(
                comment="测试",
                context_chunks=[],
                article_summary="本文讨论客户成功的最佳实践",
            )

        messages = mock_create.call_args[1]["messages"]
        system = messages[0]["content"]
        assert "客户成功的最佳实践" in system


# ===== 消息列表测试 =====

class TestMessageConstruction:
    """验证 messages 列表构建"""

    @patch("scripts.llm_client.time.sleep")
    def test_user_comment_is_last_message(self, mock_sleep, client):
        """用户评论应为最后一条 message"""
        mock_response = _make_mock_response()

        with patch.object(
            client._client.chat.completions, "create",
            return_value=mock_response
        ) as mock_create:
            client.generate_reply(
                comment="这是用户的评论",
                context_chunks=[],
                article_summary="",
            )

        messages = mock_create.call_args[1]["messages"]
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "这是用户的评论"

    @patch("scripts.llm_client.time.sleep")
    def test_history_messages_appended(self, mock_sleep, client):
        """历史消息应正确拼接到 messages 中"""
        mock_response = _make_mock_response()
        history = [
            {"role": "user", "content": "第一个问题"},
            {"role": "assistant", "content": "第一个回复"},
            {"role": "user", "content": "追问"},
            {"role": "assistant", "content": "追问回复"},
        ]

        with patch.object(
            client._client.chat.completions, "create",
            return_value=mock_response
        ) as mock_create:
            client.generate_reply(
                comment="最新评论",
                context_chunks=[],
                article_summary="",
                history_messages=history,
            )

        messages = mock_create.call_args[1]["messages"]
        # system + 4条历史 + 1条当前用户
        assert len(messages) == 6
        assert messages[0]["role"] == "system"
        assert messages[1]["content"] == "第一个问题"
        assert messages[2]["content"] == "第一个回复"
        assert messages[-1]["content"] == "最新评论"

    @patch("scripts.llm_client.time.sleep")
    def test_no_history_messages(self, mock_sleep, client):
        """无历史消息时只有 system + user"""
        mock_response = _make_mock_response()

        with patch.object(
            client._client.chat.completions, "create",
            return_value=mock_response
        ) as mock_create:
            client.generate_reply(
                comment="简单问题",
                context_chunks=[],
                article_summary="",
            )

        messages = mock_create.call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


# ===== 重试逻辑测试 =====

class TestRetryLogic:
    """验证指数退避重试"""

    @patch("scripts.llm_client.time.sleep")
    def test_retry_on_rate_limit(self, mock_sleep, client):
        """限流时应重试并最终成功"""
        from openai import RateLimitError
        mock_response = _make_mock_response(content="重试后成功")

        error_response = MagicMock()
        error_response.status_code = 429
        error_response.headers = {}

        with patch.object(
            client._client.chat.completions, "create",
            side_effect=[
                RateLimitError(
                    message="rate limit",
                    response=error_response,
                    body=None,
                ),
                RateLimitError(
                    message="rate limit",
                    response=error_response,
                    body=None,
                ),
                mock_response,
            ]
        ):
            reply, tokens = client.generate_reply(
                comment="测试", context_chunks=[], article_summary=""
            )

        assert reply == "重试后成功"
        assert mock_sleep.call_count == 2  # 前两次失败各 sleep 一次

    @patch("scripts.llm_client.time.sleep")
    def test_retry_exhausted_raises(self, mock_sleep, client):
        """重试次数耗尽应抛出异常"""
        from openai import RateLimitError

        error_response = MagicMock()
        error_response.status_code = 429
        error_response.headers = {}

        with patch.object(
            client._client.chat.completions, "create",
            side_effect=RateLimitError(
                message="rate limit",
                response=error_response,
                body=None,
            ),
        ):
            with pytest.raises(RateLimitError):
                client.generate_reply(
                    comment="测试", context_chunks=[], article_summary=""
                )


# ===== 预算控制测试 =====

class TestBudgetControl:
    """验证费用追踪与预算控制"""

    @patch("scripts.llm_client.time.sleep")
    def test_cost_accumulation(self, mock_sleep, client):
        """费用应累计"""
        mock_response = _make_mock_response(
            prompt_tokens=1000, completion_tokens=200
        )

        with patch.object(
            client._client.chat.completions, "create",
            return_value=mock_response
        ):
            client.generate_reply(
                comment="测试", context_chunks=[], article_summary=""
            )

        assert client.total_cost_usd > 0
        assert client.total_prompt_tokens == 1000
        assert client.total_completion_tokens == 200

    @patch("scripts.llm_client.time.sleep")
    def test_budget_exceeded_raises(self, mock_sleep):
        """超出预算时应抛出 BudgetExceededError"""
        # 设置极低预算
        client = LLMClient(
            api_key="test",
            budget_usd_per_day=0.0001,
        )

        mock_response = _make_mock_response(
            prompt_tokens=100000, completion_tokens=50000
        )

        with patch.object(
            client._client.chat.completions, "create",
            return_value=mock_response
        ):
            # 第一次调用应成功（但费用会累计）
            client.generate_reply(
                comment="测试1", context_chunks=[], article_summary=""
            )

            # 第二次调用应触发预算超限
            with pytest.raises(BudgetExceededError):
                client.generate_reply(
                    comment="测试2", context_chunks=[], article_summary=""
                )

    def test_initial_cost_is_zero(self, client):
        """初始费用应为零"""
        assert client.total_cost_usd == 0
        assert client.total_prompt_tokens == 0

    @patch("scripts.llm_client.time.sleep")
    def test_cache_hit_reduces_cost(self, mock_sleep, client):
        """缓存命中应降低费用"""
        # 无缓存
        resp_no_cache = _make_mock_response(
            prompt_tokens=1000, completion_tokens=200, cached_tokens=0
        )
        # 有缓存
        resp_with_cache = _make_mock_response(
            prompt_tokens=1000, completion_tokens=200, cached_tokens=800
        )

        client_no_cache = LLMClient(api_key="test", budget_usd_per_day=10)
        client_with_cache = LLMClient(api_key="test", budget_usd_per_day=10)

        with patch.object(
            client_no_cache._client.chat.completions, "create",
            return_value=resp_no_cache
        ):
            client_no_cache.generate_reply(
                comment="测试", context_chunks=[], article_summary=""
            )

        with patch.object(
            client_with_cache._client.chat.completions, "create",
            return_value=resp_with_cache
        ):
            client_with_cache.generate_reply(
                comment="测试", context_chunks=[], article_summary=""
            )

        assert client_with_cache.total_cost_usd < client_no_cache.total_cost_usd


# ===== 文章摘要测试 =====

class TestSummarizeArticle:
    """验证文章摘要生成与缓存"""

    @patch("scripts.llm_client.time.sleep")
    def test_summarize_returns_text(self, mock_sleep, client):
        """应返回摘要文本"""
        mock_response = _make_mock_response(
            content="本文讨论了 CSM 的核心方法论"
        )

        with patch.object(
            client._client.chat.completions, "create",
            return_value=mock_response
        ):
            summary = client.summarize_article("CSM 指南", "详细内容...")

        assert "CSM" in summary

    @patch("scripts.llm_client.time.sleep")
    def test_summarize_caches_result(self, mock_sleep, client):
        """第二次调用同一文章应使用缓存，不触发 API"""
        mock_response = _make_mock_response(content="摘要内容")

        with patch.object(
            client._client.chat.completions, "create",
            return_value=mock_response
        ) as mock_create:
            # 第一次调用
            summary1 = client.summarize_article("标题", "内容")
            # 第二次调用（应缓存命中）
            summary2 = client.summarize_article("标题", "内容")

        assert summary1 == summary2
        assert mock_create.call_count == 1  # 只调用了一次 API

    @patch("scripts.llm_client.time.sleep")
    def test_different_articles_not_cached(self, mock_sleep, client):
        """不同文章不应缓存混淆"""
        resp1 = _make_mock_response(content="摘要1")
        resp2 = _make_mock_response(content="摘要2")

        with patch.object(
            client._client.chat.completions, "create",
            side_effect=[resp1, resp2]
        ) as mock_create:
            s1 = client.summarize_article("标题1", "内容1")
            s2 = client.summarize_article("标题2", "内容2")

        assert s1 != s2
        assert mock_create.call_count == 2


# ===== 返回值测试 =====

class TestReturnValues:
    """验证返回值格式"""

    @patch("scripts.llm_client.time.sleep")
    def test_generate_reply_returns_tuple(self, mock_sleep, client):
        """generate_reply 应返回 (回复, token数) 元组"""
        mock_response = _make_mock_response(
            content="回复内容",
            prompt_tokens=100,
            completion_tokens=50,
        )

        with patch.object(
            client._client.chat.completions, "create",
            return_value=mock_response
        ):
            reply, tokens = client.generate_reply(
                comment="测试", context_chunks=[], article_summary=""
            )

        assert isinstance(reply, str)
        assert isinstance(tokens, int)
        assert tokens == 150  # 100 + 50
        assert reply == "回复内容"
