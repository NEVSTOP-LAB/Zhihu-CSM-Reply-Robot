# -*- coding: utf-8 -*-
"""
GitHub Issue 告警模块
=====================

实施计划关联：AI-010 告警模块 — GitHub Issue 自动创建
参考文档：docs/Review-方案/ 评审建议

功能：
- 失败告警（401/403 Cookie 失效、429 限流、预算超限）
- 幂等创建（防止重复 Issue）
- Cookie 存活状态记录

告警场景：
- Cookie 失效（401/403）
- 持续限流（429 多次）
- 连续失败 ≥ N 次
- 每日预算超限
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class AlertManager:
    """GitHub Issue 告警管理器

    实施计划关联：AI-010

    通过 GitHub API 创建 Issue 实现告警，
    支持幂等创建（同 title 的 open issue 不重复创建）。

    Args:
        github_token: GitHub API Token
        repo: 仓库全名（owner/repo）
        health_file: Cookie 健康状态记录文件路径
    """

    GITHUB_API_BASE = "https://api.github.com"

    def __init__(
        self,
        github_token: str = "",
        repo: str = "",
        health_file: str = "data/health.json",
    ):
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN", "")
        self.repo = repo or os.environ.get("GITHUB_REPOSITORY", "")
        self.health_file = Path(health_file)

        self._session = requests.Session()
        if self.github_token:
            self._session.headers.update({
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
            })

    def _has_open_issue(self, title: str, labels: list[str] | None = None) -> bool:
        """检查是否已有同 title 的 open issue

        实施计划关联：AI-010 任务 3（防重复）

        通过分页遍历所有 open issues，避免 > 50 条时漏查（FIX-18/REV-3）。

        Args:
            title: Issue 标题
            labels: 仅搜索含这些标签的 Issue（FIX-18：缩小搜索范围）

        Returns:
            True 如果已有同名 open issue
        """
        if not self.github_token or not self.repo:
            return False

        url = f"{self.GITHUB_API_BASE}/repos/{self.repo}/issues"
        params: dict = {"state": "open", "per_page": 100, "page": 1}
        # FIX-18：按标签过滤，缩小搜索范围
        if labels:
            params["labels"] = ",".join(labels)

        try:
            while url:
                resp = self._session.get(url, params=params, timeout=10)
                if resp.status_code != 200:
                    break
                issues = resp.json()
                if any(issue.get("title") == title for issue in issues):
                    return True
                # 分页：读取 Link header 中的 next 链接（REV-3）
                link_header = resp.headers.get("Link", "")
                next_url = None
                for part in link_header.split(","):
                    part = part.strip()
                    if 'rel="next"' in part:
                        next_url = part.split(";")[0].strip().strip("<>")
                        break
                # next_url 已包含所有分页参数，置空 params 避免与 URL 内参数冲突
                url = next_url
                params = {}
        except Exception as e:
            logger.warning(f"检查已有 Issue 失败: {e}")

        return False

    def create_issue(
        self,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> bool:
        """创建 GitHub Issue

        实施计划关联：AI-010 任务 1

        幂等操作：如果已有同 title 的 open issue，不重复创建。

        Args:
            title: Issue 标题
            body: Issue 内容（Markdown）
            labels: 标签列表

        Returns:
            True 创建成功或已存在，False 创建失败
        """
        if not self.github_token or not self.repo:
            logger.warning("未配置 GITHUB_TOKEN 或 GITHUB_REPOSITORY，跳过告警")
            return False

        # 幂等检查（FIX-18：传入 labels 缩小搜索范围）
        if self._has_open_issue(title, labels=labels or ["bot-alert"]):
            logger.info(f"已有同名 open issue，跳过: {title}")
            return True

        url = f"{self.GITHUB_API_BASE}/repos/{self.repo}/issues"
        payload = {
            "title": title,
            "body": body,
            "labels": labels or ["bot-alert"],
        }

        try:
            resp = self._session.post(url, json=payload, timeout=10)
            if resp.status_code in (201, 200):
                logger.info(f"告警 Issue 已创建: {title}")
                return True
            else:
                logger.error(
                    f"创建 Issue 失败: {resp.status_code} {resp.text}"
                )
                return False
        except Exception as e:
            logger.error(f"创建 Issue 异常: {e}")
            return False

    def alert_cookie_expired(self, error_code: int):
        """Cookie 失效告警

        Args:
            error_code: HTTP 状态码（401 或 403）
        """
        title = "🔴 Bot Alert: 知乎 Cookie 已失效"
        body = (
            f"## Cookie 认证失败\n\n"
            f"- **HTTP 状态码**: {error_code}\n"
            f"- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"### 处理步骤\n"
            f"1. 登录知乎，获取新的 Cookie\n"
            f"2. 更新 GitHub Secrets 中的 `ZHIHU_COOKIE`\n"
            f"3. 手动触发 workflow 验证\n"
        )
        self.create_issue(title, body, labels=["bot-alert", "cookie-expired"])
        self.record_health("cookie_expired", {"error_code": error_code})

    def alert_rate_limited(self):
        """持续限流告警"""
        title = "🟡 Bot Alert: 知乎 API 持续限流 (429)"
        body = (
            f"## 持续限流\n\n"
            f"- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"知乎 API 返回 429，已超过最大重试次数。\n"
            f"请检查请求频率是否正常。\n"
        )
        self.create_issue(title, body, labels=["bot-alert", "rate-limit"])

    def alert_budget_exceeded(self, cost: float, budget: float):
        """预算超限告警

        Args:
            cost: 当前累计费用
            budget: 预算上限
        """
        title = "🟡 Bot Alert: LLM 每日预算超限"
        body = (
            f"## 预算超限\n\n"
            f"- **当前费用**: ${cost:.4f}\n"
            f"- **预算上限**: ${budget:.2f}\n"
            f"- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"剩余评论已写入 `pending/` 等待人工处理。\n"
        )
        self.create_issue(title, body, labels=["bot-alert", "budget"])

    def alert_consecutive_failures(self, count: int, recent_logs: str = ""):
        """连续失败告警

        Args:
            count: 连续失败次数
            recent_logs: 最近日志内容（Markdown 格式），用于快速定位问题
        """
        title = f"🔴 Bot Alert: 连续失败 {count} 次"
        body = (
            f"## 连续失败\n\n"
            f"- **连续失败次数**: {count}\n"
            f"- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Bot 已暂停处理，请检查日志。\n"
        )
        if recent_logs:
            body += f"\n### 最近日志\n\n{recent_logs}\n"
        self.create_issue(title, body, labels=["bot-alert", "failure"])

    def alert_expansion_failed(self, configured_count: int, recent_logs: str = ""):
        """监控目标展开失败告警

        所有配置的 column/user_answers/question 类型均展开失败，
        导致本轮没有任何监控目标可处理。

        Args:
            configured_count: 配置的监控目标数量
            recent_logs: 最近日志内容（Markdown 格式），用于快速定位问题
        """
        title = "🔴 Bot Alert: 监控目标全部展开失败"
        body = (
            f"## 监控目标展开失败\n\n"
            f"- **配置目标数**: {configured_count}\n"
            f"- **展开后目标数**: 0\n"
            f"- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"所有监控目标均展开失败，本轮未处理任何评论。\n"
            f"请检查 Cookie 是否有效，以及专栏/用户配置是否正确。\n"
        )
        if recent_logs:
            body += f"\n### 最近日志\n\n{recent_logs}\n"
        self.create_issue(title, body, labels=["bot-alert", "expansion-failed"])
        self.record_health("expansion_failed", {"configured_count": configured_count})

    def record_health(self, status: str, details: dict | None = None):
        """记录 Cookie 健康状态

        实施计划关联：AI-010 任务 4

        将每次运行的 Cookie 状态写入 data/health.json

        Args:
            status: 状态（"ok", "cookie_expired", "rate_limited", "error"）
            details: 额外详情
        """
        self.health_file.parent.mkdir(parents=True, exist_ok=True)

        health_data = {
            "last_check": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "status": status,
            "details": details or {},
        }

        try:
            with open(self.health_file, "w", encoding="utf-8") as f:
                json.dump(health_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"健康状态已记录: {status}")
        except Exception as e:
            logger.warning(f"写入健康状态失败: {e}")
