# -*- coding: utf-8 -*-
"""
LLM 客户端 — DeepSeek/OpenAI 调用封装
=======================================

实施计划关联：AI-006 LLMClient — 回复生成
参考文档：docs/调研/03-LLM接入与回复生成.md

功能：
- OpenAI 兼容接口调用（支持 DeepSeek）
- Prompt Caching 优化（System Prompt 固定前缀）
- 费用追踪与预算控制
- 文章摘要生成与缓存
- 指数退避重试

使用方式：
    client = LLMClient(
        api_key="sk-xxx",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        budget_usd_per_day=0.50,
    )
    reply, tokens = client.generate_reply(
        comment="如何做好客户成功？",
        context_chunks=["..."],
        article_summary="本文讨论...",
    )
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

from openai import OpenAI, APIError, RateLimitError, APIConnectionError

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """每日 LLM 费用预算超限异常

    当累计费用超过 llm_budget_usd_per_day 时抛出，
    主流程捕获后应将剩余评论写入 pending/ 并触发告警。

    实施计划关联：AI-006 任务 5
    """
    pass


# DeepSeek-V3 价格（参考 docs/调研/07-费用评估.md）
# 单位：USD per 1M tokens
PRICING = {
    "deepseek-chat": {
        "input": 0.27,          # 标准输入
        "input_cached": 0.07,   # 缓存命中输入
        "output": 1.10,         # 输出
    },
    "deepseek-reasoner": {
        "input": 0.55,
        "input_cached": 0.14,
        "output": 2.19,
    },
    # 通用默认价格（未知模型）
    "default": {
        "input": 0.50,
        "input_cached": 0.25,
        "output": 1.50,
    },
}


class LLMClient:
    """LLM 客户端

    实施计划关联：AI-006 LLMClient — 回复生成
    参考：docs/调研/03-LLM接入与回复生成.md

    封装 OpenAI 兼容接口（支持 DeepSeek），提供：
    - 回复生成（固定 System Prompt 前缀优化缓存）
    - 文章摘要生成与缓存
    - 费用追踪与预算控制
    - 指数退避重试

    Args:
        api_key: LLM API Key
        base_url: API 端点（默认 DeepSeek）
        model: 模型名称
        max_tokens: 回复最大 token 数
        temperature: 生成温度
        budget_usd_per_day: 每日费用预算上限
        max_retries: 最大重试次数
    """

    # System Prompt 固定前缀（最大化 Prompt Caching 命中）
    # 参考：docs/调研/06-Token优化策略.md
    SYSTEM_PROMPT_PREFIX = (
        "你是一个专业的客户成功（CSM）领域助手，在知乎上回复与客户成功相关的评论。\n\n"
        "## 回复规则\n"
        "1. 语气友好专业，避免过于正式\n"
        "2. 回复简洁有用，控制在 200 字以内\n"
        "3. 如果引用了参考资料，自然融入回复，不要列出引用来源\n"
        "4. 不要使用 Markdown 格式（知乎评论不支持）\n"
        "5. 如果不确定答案，诚实说明并建议进一步了解的方向\n"
        "6. 回复应该自然流畅，像一个资深 CSM 从业者的口吻\n"
    )

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        max_tokens: int = 250,
        temperature: float = 0.7,
        budget_usd_per_day: float = 0.50,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "LLM_BASE_URL", "https://api.deepseek.com"
        )
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.budget_usd_per_day = budget_usd_per_day
        self.max_retries = max_retries

        # OpenAI 兼容客户端
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # 费用追踪
        self._total_cost_usd = 0.0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cache_hit_tokens = 0

        # 文章摘要缓存
        # 参考 AI-006 任务 3：结果缓存，避免重复调用
        self._summary_cache: dict[str, str] = {}

    @property
    def total_cost_usd(self) -> float:
        """当日累计费用（USD）"""
        return self._total_cost_usd

    @property
    def total_prompt_tokens(self) -> int:
        """累计输入 token 数"""
        return self._total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        """累计输出 token 数"""
        return self._total_completion_tokens

    @property
    def total_cache_hit_tokens(self) -> int:
        """累计缓存命中 token 数"""
        return self._total_cache_hit_tokens

    def _get_pricing(self) -> dict:
        """获取当前模型的价格配置"""
        return PRICING.get(self.model, PRICING["default"])

    def _calculate_cost(
        self, prompt_tokens: int, completion_tokens: int, cache_hit_tokens: int
    ) -> float:
        """计算单次调用费用

        参考：docs/调研/07-费用评估.md

        Args:
            prompt_tokens: 输入 token 数（不含缓存命中）
            completion_tokens: 输出 token 数
            cache_hit_tokens: 缓存命中的输入 token 数

        Returns:
            费用（USD）
        """
        pricing = self._get_pricing()
        non_cached_input = prompt_tokens - cache_hit_tokens

        cost = (
            non_cached_input * pricing["input"] / 1_000_000
            + cache_hit_tokens * pricing["input_cached"] / 1_000_000
            + completion_tokens * pricing["output"] / 1_000_000
        )
        return cost

    def _check_budget(self):
        """检查是否超出每日预算

        Raises:
            BudgetExceededError: 超出预算时抛出
        """
        if self._total_cost_usd >= self.budget_usd_per_day:
            raise BudgetExceededError(
                f"每日 LLM 费用已达 ${self._total_cost_usd:.4f}，"
                f"超出预算 ${self.budget_usd_per_day:.2f}"
            )

    def _call_api(self, messages: list[dict], max_tokens: int | None = None) -> dict:
        """调用 LLM API（含重试逻辑）

        实施计划关联：AI-006 任务 4（指数退避重试最多 3 次）

        Args:
            messages: OpenAI messages 格式的消息列表
            max_tokens: 最大输出 token 数

        Returns:
            包含 reply, prompt_tokens, completion_tokens, cache_hit_tokens 的字典

        Raises:
            各种 API 异常
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=self.temperature,
                )

                # 解析响应
                reply = response.choices[0].message.content or ""
                usage = response.usage

                prompt_tokens = usage.prompt_tokens if usage else 0
                completion_tokens = usage.completion_tokens if usage else 0

                # DeepSeek 返回的缓存命中 token 数
                # 参考 docs/调研/06-Token优化策略.md
                cache_hit_tokens = 0
                if usage and hasattr(usage, "prompt_tokens_details"):
                    details = usage.prompt_tokens_details
                    if details and hasattr(details, "cached_tokens"):
                        cache_hit_tokens = details.cached_tokens or 0

                # 累计统计
                cost = self._calculate_cost(
                    prompt_tokens, completion_tokens, cache_hit_tokens
                )
                self._total_cost_usd += cost
                self._total_prompt_tokens += prompt_tokens
                self._total_completion_tokens += completion_tokens
                self._total_cache_hit_tokens += cache_hit_tokens

                logger.info(
                    f"LLM 调用: prompt={prompt_tokens}, "
                    f"completion={completion_tokens}, "
                    f"cache_hit={cache_hit_tokens}, "
                    f"cost=${cost:.6f}"
                )

                return {
                    "reply": reply,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cache_hit_tokens": cache_hit_tokens,
                    "cost": cost,
                }

            except RateLimitError as e:
                last_exception = e
                wait_time = 2 ** attempt
                logger.warning(
                    f"LLM 限流，第 {attempt + 1} 次重试，"
                    f"等待 {wait_time} 秒"
                )
                time.sleep(wait_time)

            except (APIError, APIConnectionError) as e:
                last_exception = e
                wait_time = 2 ** attempt
                logger.warning(
                    f"LLM API 错误: {e}，第 {attempt + 1} 次重试"
                )
                time.sleep(wait_time)

        raise last_exception  # type: ignore

    def generate_reply(
        self,
        comment: str,
        context_chunks: list[str],
        article_summary: str,
        history_messages: list[dict] | None = None,
    ) -> tuple[str, int]:
        """生成回复

        实施计划关联：AI-006 任务 2

        使用固定 System Prompt 前缀（角色+规则+wiki_context），
        最大化 Prompt Caching 命中。

        Args:
            comment: 用户评论内容
            context_chunks: RAG 检索到的相关文档片段
            article_summary: 文章摘要
            history_messages: 历史对话（追问场景，OpenAI messages 格式）

        Returns:
            (回复内容, 消耗的总 token 数)

        Raises:
            BudgetExceededError: 超出每日预算
        """
        # 预算检查
        self._check_budget()

        # 构建 System Prompt（固定前缀 + 动态 context）
        # 参考 docs/调研/06-Token优化策略.md: 固定前缀最大化缓存命中
        system_content = self.SYSTEM_PROMPT_PREFIX

        if context_chunks:
            wiki_context = "\n\n".join(context_chunks)
            system_content += (
                f"\n## 参考资料\n"
                f"以下是与当前话题相关的知识库内容，请参考但不要直接复制：\n\n"
                f"{wiki_context}\n"
            )

        if article_summary:
            system_content += (
                f"\n## 文章背景\n{article_summary}\n"
            )

        messages: list[dict] = [
            {"role": "system", "content": system_content},
        ]

        # 追加历史消息（追问上下文）
        # 参考 AI-006 任务 2：history_messages 正确拼接
        if history_messages:
            messages.extend(history_messages)

        # 当前用户评论
        messages.append({"role": "user", "content": comment})

        result = self._call_api(messages)

        total_tokens = result["prompt_tokens"] + result["completion_tokens"]
        return result["reply"], total_tokens

    def summarize_article(self, title: str, content: str) -> str:
        """生成文章摘要

        实施计划关联：AI-006 任务 3

        结果缓存：同一文章只调用一次 API，后续直接返回缓存。

        Args:
            title: 文章标题
            content: 文章内容

        Returns:
            文章摘要（≤200 tokens）
        """
        cache_key = f"{title}:{hash(content)}"
        if cache_key in self._summary_cache:
            logger.debug(f"文章摘要缓存命中: {title}")
            return self._summary_cache[cache_key]

        # 预算检查
        self._check_budget()

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个文本摘要助手。请用简洁的中文概括以下文章的核心观点，"
                    "不超过 100 字。"
                ),
            },
            {
                "role": "user",
                "content": f"标题：{title}\n\n{content[:2000]}",
            },
        ]

        result = self._call_api(messages, max_tokens=200)
        summary = result["reply"]

        # 写入缓存
        self._summary_cache[cache_key] = summary
        logger.info(f"已生成并缓存文章摘要: {title}")

        return summary
