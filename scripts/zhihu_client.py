# -*- coding: utf-8 -*-
"""
知乎 API v4 客户端封装
=======================

实施计划关联：AI-003 ZhihuClient — 读取与写入接口
参考文档：docs/调研/01-知乎数据获取.md

功能：
- 知乎评论读取（文章/问题）
- 评论发布（Cookie + CSRF token）
- 认证管理与错误处理
- 限流退避与反爬策略
"""
from __future__ import annotations

import hashlib
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class ZhihuAuthError(Exception):
    """知乎认证失败异常（Cookie 失效 / 401 / 403）

    当检测到 Cookie 过期或认证被拒绝时抛出，
    主流程捕获后应触发 GitHub Issue 告警（参考 AI-010）
    """
    pass


class ZhihuRateLimitError(Exception):
    """知乎限流异常（429 Too Many Requests）

    指数退避重试逻辑会在内部处理此异常，
    超过最大重试次数后向上抛出
    """
    pass


@dataclass
class Comment:
    """知乎评论数据结构

    字段映射参考：docs/调研/01-知乎数据获取.md API 响应格式

    Attributes:
        id: 评论唯一 ID
        parent_id: 父评论 ID（顶级评论为 None）
        content: 评论正文
        author: 评论者昵称
        created_time: 创建时间戳（Unix timestamp）
        is_author_reply: 是否为文章作者的回复
    """
    id: str
    parent_id: Optional[str]
    content: str
    author: str
    created_time: int
    is_author_reply: bool = False


class ZhihuClient:
    """知乎 API v4 客户端

    实施计划关联：AI-003
    参考：docs/调研/01-知乎数据获取.md

    使用 Cookie 认证调用知乎 API v4，支持：
    - 文章评论读取（GET /api/v4/articles/{id}/comments）
    - 问题回答评论读取（GET /api/v4/answers/{id}/comments）
    - 评论发布（POST /api/v4/comments, Cookie + CSRF）

    Args:
        cookie: 完整的知乎 Cookie 字符串（包含 z_c0 和 _xsrf）
        max_retries: 429 限流最大重试次数（默认 3）
    """

    # 浏览器指纹（参考 zhihu-cli 的 User-Agent 设置）
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.zhihu.com/",
        "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        "sec-ch-ua-platform": '"Windows"',
    }

    # API 端点（参考 AI-001 验证结果）
    READ_API_BASE = "https://www.zhihu.com/api/v4"
    WRITE_API_BASE = "https://api.zhihu.com/v4"

    def __init__(self, cookie: str, max_retries: int = 3):
        self.cookie = cookie
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)
        self.session.headers["Cookie"] = cookie

        # 从 Cookie 提取 _xsrf（CSRF token）
        # 参考：zhihu-cli 的 CSRF 处理方式
        self._xsrf = self._extract_xsrf(cookie)

    @staticmethod
    def _extract_xsrf(cookie: str) -> Optional[str]:
        """从 Cookie 字符串中提取 _xsrf 值

        知乎使用 _xsrf cookie 作为 CSRF token，
        写操作需要在请求头中设置 x-xsrftoken

        Args:
            cookie: 完整的 Cookie 字符串

        Returns:
            _xsrf 值，未找到时返回 None
        """
        match = re.search(r'_xsrf=([^;]+)', cookie)
        return match.group(1) if match else None

    def _build_read_url(self, object_id: str, object_type: str) -> str:
        """构建评论读取 API URL

        参考：docs/调研/01-知乎数据获取.md API 端点说明

        Args:
            object_id: 文章 ID 或回答 ID
            object_type: "article" 或 "question"

        Returns:
            完整的 API URL
        """
        if object_type == "article":
            return f"{self.READ_API_BASE}/articles/{object_id}/comments"
        elif object_type == "question":
            # 问题评论通过回答 ID 获取
            return f"{self.READ_API_BASE}/answers/{object_id}/comments"
        else:
            raise ValueError(f"不支持的 object_type: {object_type}")

    def _request_with_retry(
        self, method: str, url: str, **kwargs
    ) -> requests.Response:
        """带限流退避重试的 HTTP 请求

        实施计划关联：AI-003 任务 2（429 指数退避最多 3 次）

        Args:
            method: HTTP 方法（GET/POST）
            url: 请求 URL
            **kwargs: 传递给 requests 的其他参数

        Returns:
            HTTP 响应对象

        Raises:
            ZhihuAuthError: 401/403 认证失败
            ZhihuRateLimitError: 429 超过最大重试次数
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.request(method, url, **kwargs)

                # 认证失败：Cookie 过期或被封禁
                if resp.status_code in (401, 403):
                    raise ZhihuAuthError(
                        f"知乎认证失败 (HTTP {resp.status_code})，"
                        f"请检查 Cookie 是否过期"
                    )

                # 限流：指数退避重试
                if resp.status_code == 429:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"知乎限流 (429)，第 {attempt + 1} 次重试，"
                        f"等待 {wait_time:.1f} 秒"
                    )
                    time.sleep(wait_time)
                    continue

                resp.raise_for_status()
                return resp

            except ZhihuAuthError:
                raise
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    last_exception = e
                    continue
                raise

        raise ZhihuRateLimitError(
            f"知乎限流，已重试 {self.max_retries} 次仍然失败"
        )

    def get_comments(
        self,
        object_id: str,
        object_type: str,
        since_id: Optional[str] = None
    ) -> list[Comment]:
        """获取知乎评论列表

        实施计划关联：AI-003 任务 2

        自动分页直到 is_end=True，每次请求间随机延迟 1~2 秒。

        Args:
            object_id: 文章 ID 或回答 ID
            object_type: "article" 或 "question"
            since_id: 只返回此 ID 之后的评论（增量获取）

        Returns:
            评论列表（Comment 对象）

        Raises:
            ZhihuAuthError: Cookie 失效
            ZhihuRateLimitError: 限流超过重试次数
        """
        url = self._build_read_url(object_id, object_type)
        all_comments: list[Comment] = []
        params: dict = {"limit": 20, "order": "normal"}

        if since_id:
            params["after_id"] = since_id

        while True:
            resp = self._request_with_retry("GET", url, params=params)
            data = resp.json()

            for item in data.get("data", []):
                comment = self._parse_comment(item)
                all_comments.append(comment)

            # 检查分页是否结束
            paging = data.get("paging", {})
            if paging.get("is_end", True):
                break

            # 使用下一页 URL 继续分页
            next_url = paging.get("next")
            if next_url:
                url = next_url
                params = {}  # 下一页 URL 已包含参数
            else:
                break

            # 随机延迟 1~2 秒（反爬策略）
            time.sleep(random.uniform(1.0, 2.0))

        return all_comments

    def post_comment(
        self,
        object_id: str,
        object_type: str,
        content: str,
        parent_id: Optional[str] = None
    ) -> bool:
        """发布知乎评论

        实施计划关联：AI-003 任务 3, AI-014

        使用 Cookie + CSRF token（_xsrf）方式发布评论。
        参考 zhihu-cli 的 CSRF 处理方式。

        Args:
            object_id: 目标文章/回答 ID
            object_type: "article" 或 "question"
            content: 评论内容
            parent_id: 父评论 ID（回复他人评论时使用）

        Returns:
            True 发布成功，False 发布失败（主流程应写入 pending/）
        """
        if not self._xsrf:
            logger.error("Cookie 中未找到 _xsrf，无法发布评论")
            return False

        url = f"{self.WRITE_API_BASE}/comments"
        headers = {
            "x-xsrftoken": self._xsrf,
            "Content-Type": "application/json",
        }

        payload: dict = {
            "content": content,
            "object_id": object_id,
            "object_type": object_type,
        }
        if parent_id:
            payload["parent_id"] = parent_id

        try:
            resp = self._request_with_retry(
                "POST", url, json=payload, headers=headers
            )
            if resp.status_code in (200, 201):
                logger.info(
                    f"评论发布成功: object_id={object_id}, "
                    f"object_type={object_type}"
                )
                return True
            else:
                logger.warning(
                    f"评论发布返回非预期状态码: {resp.status_code}"
                )
                return False
        except ZhihuAuthError:
            logger.error("评论发布失败：认证错误（Cookie 可能已失效）")
            return False
        except Exception as e:
            logger.error(f"评论发布失败: {e}")
            return False

    @staticmethod
    def _parse_comment(item: dict) -> Comment:
        """解析 API 返回的评论 JSON 为 Comment 对象

        Args:
            item: API 返回的单条评论 JSON

        Returns:
            Comment 数据对象
        """
        author_info = item.get("author", {})
        reply_to = item.get("reply_to_author", {})

        return Comment(
            id=str(item.get("id", "")),
            parent_id=str(item["reply_comment"]["id"])
            if item.get("reply_comment")
            else None,
            content=item.get("content", ""),
            author=author_info.get("name", "unknown"),
            created_time=item.get("created_time", 0),
            is_author_reply=item.get("is_author", False),
        )
