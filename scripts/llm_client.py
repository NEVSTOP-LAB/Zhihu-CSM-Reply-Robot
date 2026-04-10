"""
AI-006: LLMClient — DeepSeek/OpenAI 回复生成
参考: docs/plan/README.md § AI-006, docs/调研/03-LLM接入与回复生成.md

功能：
1. 从环境变量读取 LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
2. generate_reply(): 生成回复，System Prompt 固定前缀（缓存友好）
3. summarize_article(): 文章摘要生成（结果缓存）
4. 指数退避重试最多 3 次
5. 累计 token 费用，超 llm_budget_usd_per_day 时抛出 BudgetExceededError

设计说明：
- 使用 OpenAI 兼容接口，通过 base_url 切换 provider
- System Prompt 固定部分放最前，最大化 Prompt Caching 命中率
- 费用追踪基于 DeepSeek 定价模型
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from openai import OpenAI, RateLimitError, APIError, APIConnectionError

logger = logging.getLogger(__name__)


# ─── 异常定义 ──────────────────────────────────────────────────

class BudgetExceededError(Exception):
    """每日 LLM 费用预算超限"""
    pass


# ─── System Prompt 模板（固定前缀，触发 Prompt Caching）─────────

# 参考: docs/调研/03-LLM接入与回复生成.md § Prompt 结构（三段式）
# 参考: docs/调研/06-Token优化策略.md § 2. Prompt Caching
SYSTEM_PROMPT_PREFIX = (
    "你是 CSM（客户成功管理）助理，代表专栏作者回复知乎评论。\n"
    "回复规则：\n"
    "1. 专业、友善、简洁（200字以内）\n"
    "2. 基于知识库内容回复，不编造信息\n"
    "3. 如果知识库无相关内容，坦诚告知并建议查阅相关资料\n"
    "4. 对追问引用历史上下文，不重复解释\n"
    "5. 不讨论政治、宗教等敏感话题\n"
    "6. 回复风格要像真人，不要有明显的 AI 痕迹\n"
)


# ─── 费用定价（DeepSeek 基准）───────────────────────────────────

# 参考: docs/调研/06-Token优化策略.md § 5. 选用低成本模型
PRICING = {
    "deepseek-chat": {
        "input": 0.27 / 1_000_000,        # $/token（标准）
        "input_cached": 0.07 / 1_000_000,  # $/token（缓存命中）
        "output": 1.10 / 1_000_000,        # $/token
    },
    "deepseek-reasoner": {
        "input": 0.55 / 1_000_000,
        "input_cached": 0.14 / 1_000_000,
        "output": 2.19 / 1_000_000,
    },
    "default": {
        "input": 0.50 / 1_000_000,
        "input_cached": 0.15 / 1_000_000,
        "output": 1.50 / 1_000_000,
    },
}


class LLMClient:
    """
    LLM 调用封装（OpenAI 兼容接口）
    参考: docs/plan/README.md § AI-006, docs/调研/03-LLM接入与回复生成.md

    使用方式：
        client = LLMClient()
        reply, tokens = client.generate_reply(
            comment="如何处理客户投诉？",
            context_chunks=["Wiki 片段1", "Wiki 片段2"],
            article_summary="本文介绍 CSM 方法论..."
        )
    """

    MAX_RETRIES = 3

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 250,
        temperature: float = 0.7,
        budget_usd_per_day: float = 0.50,
        max_retries: int = 3,
    ) -> None:
        """
        初始化 LLM 客户端

        Args:
            api_key: API Key（默认从 LLM_API_KEY 环境变量读取）
            base_url: API Base URL（默认从 LLM_BASE_URL 读取）
            model: 模型名称（默认从 LLM_MODEL 读取）
            max_tokens: 最大生成 token 数
            temperature: 生成温度
            budget_usd_per_day: 每日费用预算上限（USD）
            max_retries: 最大重试次数（默认 3）
        """
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-chat")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.budget_usd_per_day = budget_usd_per_day
        self.MAX_RETRIES = max_retries

        # 初始化 OpenAI 客户端
        # 参考: docs/调研/03-LLM接入与回复生成.md § 1. API 兼容性
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # 费用追踪
        self._daily_cost_usd = 0.0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cache_hit_tokens = 0

        # 文章摘要缓存
        # 参考: docs/plan/README.md § AI-006 第 3 点 — 结果缓存
        self._article_summary_cache: dict[str, str] = {}

        logger.info(
            "LLMClient 初始化: model=%s, base_url=%s, budget=%.2f USD/day",
            self.model, self.base_url, self.budget_usd_per_day,
        )

    def generate_reply(
        self,
        comment: str,
        context_chunks: list[str],
        article_summary: str,
        history_messages: Optional[list[dict]] = None,
    ) -> tuple[str, int]:
        """
        生成评论回复
        参考: docs/plan/README.md § AI-006 第 2 点

        Prompt 结构：
        - System: 固定前缀（角色+规则）+ Wiki 片段（同文章批量时缓存命中）
        - History: 历史对话轮次（追问场景）
        - User: 当前评论

        Args:
            comment: 用户评论内容
            context_chunks: RAG 检索到的 Wiki 片段
            article_summary: 文章摘要
            history_messages: 历史对话消息（OpenAI messages 格式）

        Returns:
            (回复文本, 使用的 token 总数)

        Raises:
            BudgetExceededError: 超过每日预算
        """
        # 检查预算
        if self._daily_cost_usd >= self.budget_usd_per_day:
            raise BudgetExceededError(
                f"每日 LLM 费用已达 ${self._daily_cost_usd:.4f}，"
                f"超过预算 ${self.budget_usd_per_day:.2f}"
            )

        # 组装 System Prompt
        # 参考: docs/调研/06-Token优化策略.md § 2. Prompt Caching
        wiki_context = "\n\n".join(context_chunks) if context_chunks else "（无相关知识库内容）"
        system_content = (
            f"{SYSTEM_PROMPT_PREFIX}\n"
            f"文章背景：{article_summary}\n\n"
            f"参考资料：\n{wiki_context}"
        )

        # 构建 messages
        messages: list[dict] = [
            {"role": "system", "content": system_content},
        ]

        # 添加历史对话（追问场景）
        # 参考: docs/plan/README.md § AI-006 第 2 点 — history_messages 正确拼接
        if history_messages:
            messages.extend(history_messages)

        # 添加当前评论
        messages.append({"role": "user", "content": comment})

        # 调用 LLM（含重试）
        response = self._call_with_retry(messages)

        # 提取回复和 token 使用
        reply = response.choices[0].message.content or ""
        usage = response.usage
        total_tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0

        # 记录费用
        # 参考: docs/plan/README.md § AI-006 第 5 点 — 累计 token 费用
        self._track_cost(usage)

        logger.info(
            "生成回复: tokens=%d, cost=+$%.4f, daily_total=$%.4f",
            total_tokens,
            self._calculate_cost(usage),
            self._daily_cost_usd,
        )

        return reply, total_tokens

    def summarize_article(self, title: str, content: str) -> str:
        """
        生成文章摘要（≤200 tokens）
        参考: docs/plan/README.md § AI-006 第 3 点

        结果被缓存，同一文章第二次调用不触发 API。

        Args:
            title: 文章标题
            content: 文章正文

        Returns:
            文章摘要文本
        """
        # 检查缓存
        cache_key = f"{title}_{hash(content)}"
        if cache_key in self._article_summary_cache:
            logger.debug("文章摘要命中缓存: %s", title)
            return self._article_summary_cache[cache_key]

        messages = [
            {
                "role": "system",
                "content": "你是文档摘要助手。请用 200 字以内概括文章核心内容。",
            },
            {
                "role": "user",
                "content": f"标题：{title}\n\n正文：{content[:2000]}",
            },
        ]

        response = self._call_with_retry(messages, max_tokens=200)
        summary = response.choices[0].message.content or ""

        # 缓存结果
        self._article_summary_cache[cache_key] = summary

        # 记录费用
        self._track_cost(response.usage)

        logger.info("生成文章摘要: %s（%d字）", title, len(summary))
        return summary

    def assess_risk(self, comment: str, reply: str) -> tuple[str, str]:
        """
        AI 风险评估：判断回复是否可以自动发布

        根据评论内容和生成的回复，判断是否需要人工审核。
        CSM/LabVIEW 相关的明确回复可直接发布；
        敏感话题、超出知识库范围的回复需人工介入。

        Args:
            comment: 用户评论内容
            reply: LLM 生成的回复内容

        Returns:
            (risk_level, reason):
            - risk_level: "safe"（可自动发布）或 "risky"（需人工审核）
            - reason: 判断理由

        Raises:
            BudgetExceededError: 超过每日预算
        """
        # FIX-20：预算检查，与 generate_reply 保持一致
        if self._daily_cost_usd >= self.budget_usd_per_day:
            raise BudgetExceededError(
                f"每日 LLM 费用已达 ${self._daily_cost_usd:.4f}，"
                f"超过预算 ${self.budget_usd_per_day:.2f}"
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是风险评估助手。判断一条自动生成的知乎评论回复是否可以直接发布。\n"
                    "判断规则：\n"
                    "1. 如果回复是关于 CSM（客户成功管理）、LabVIEW、NI、JKISM/CSM框架 "
                    "等技术话题的正常回答，回复 SAFE\n"
                    "2. 如果回复涉及以下情况，回复 RISKY：\n"
                    "   - 政治、宗教等敏感话题\n"
                    "   - 回复内容超出知识库范围，可能不准确\n"
                    "   - 用户的问题需要专业人工判断\n"
                    "   - 回复包含具体承诺或商业建议\n"
                    "   - 回复语气不当或可能引起争议\n"
                    "只需回复一个词：SAFE 或 RISKY，然后换行给出简短理由。"
                ),
            },
            {
                "role": "user",
                "content": f"用户评论：{comment}\n\n生成的回复：{reply}",
            },
        ]

        try:
            response = self._call_with_retry(messages, max_tokens=50)
            result = (response.choices[0].message.content or "").strip()
            self._track_cost(response.usage)

            # 解析结果
            lines = result.split("\n", 1)
            level = lines[0].strip().upper()
            reason = lines[1].strip() if len(lines) > 1 else ""

            if "SAFE" in level:
                logger.info("风险评估: SAFE - %s", reason)
                return "safe", reason
            else:
                logger.info("风险评估: RISKY - %s", reason)
                return "risky", reason

        except Exception as e:
            # 评估失败时保守处理，标记为 risky
            logger.warning("风险评估异常，默认标记为 risky: %s", e)
            return "risky", f"评估异常: {e}"

    def _call_with_retry(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
    ):
        """
        带指数退避重试的 LLM 调用
        参考: docs/调研/03-LLM接入与回复生成.md § 6. 错误处理

        Args:
            messages: OpenAI messages 列表
            max_tokens: 最大生成 token 数

        Returns:
            OpenAI ChatCompletion 响应

        Raises:
            RuntimeError: 重试耗尽后仍失败
        """
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens or self.max_tokens,
                    temperature=self.temperature,
                )
            except RateLimitError as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning("LLM 限流，第 %d/%d 次重试，等待 %ds", attempt + 1, self.MAX_RETRIES, wait)
                time.sleep(wait)
            except (APIError, APIConnectionError) as e:
                last_error = e
                if isinstance(e, APIConnectionError) or (hasattr(e, 'status_code') and e.status_code and e.status_code >= 500):
                    wait = 2 ** attempt
                    logger.warning("LLM 服务端错误 (%s)，第 %d/%d 次重试", e, attempt + 1, self.MAX_RETRIES)
                    time.sleep(wait)
                else:
                    raise

        raise last_error  # type: ignore[misc]

    def _track_cost(self, usage) -> None:
        """
        记录 token 使用和费用
        参考: docs/plan/README.md § AI-006 第 5 点
        """
        if not usage:
            return

        self._total_prompt_tokens += usage.prompt_tokens
        self._total_completion_tokens += usage.completion_tokens

        # 检查 DeepSeek 缓存命中（通过 prompt_tokens_details.cached_tokens）
        details = getattr(usage, 'prompt_tokens_details', None)
        cache_hit = getattr(details, 'cached_tokens', 0) or 0
        # 降级到旧字段名（兼容不同版本 API）
        if not cache_hit:
            cache_hit = getattr(usage, 'prompt_cache_hit_tokens', 0) or 0
        self._total_cache_hit_tokens += cache_hit

        cost = self._calculate_cost(usage)
        self._daily_cost_usd += cost

    def _calculate_cost(self, usage) -> float:
        """计算单次调用费用"""
        if not usage:
            return 0.0

        pricing = PRICING.get(self.model, PRICING["default"])

        # 优先从 prompt_tokens_details.cached_tokens 读取缓存命中数
        details = getattr(usage, 'prompt_tokens_details', None)
        cache_hit = getattr(details, 'cached_tokens', 0) or 0
        if not cache_hit:
            cache_hit = getattr(usage, 'prompt_cache_hit_tokens', 0) or 0
        regular_input = usage.prompt_tokens - cache_hit

        cost = (
            regular_input * pricing["input"]
            + cache_hit * pricing["input_cached"]
            + usage.completion_tokens * pricing["output"]
        )
        return cost

    @property
    def total_cost_usd(self) -> float:
        """当日累计费用（USD）"""
        return self._daily_cost_usd

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

    def reset_daily_cost(self) -> None:
        """重置每日费用计数器（新一天开始时调用）"""
        self._daily_cost_usd = 0.0
