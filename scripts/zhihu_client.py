"""
AI-003: 知乎 API v4 封装 — 评论读取与写入
参考: docs/plan/README.md § AI-003, docs/调研/01-知乎数据获取.md

功能：
- ZhihuClient(cookie): 从 Cookie 初始化，自动提取 _xsrf CSRF token
- get_comments(object_id, object_type, since_id): 获取评论列表
  - 支持 article / question 两种类型
  - 自动分页（is_end=True 停止）
  - 请求间随机延迟 1~2 秒
  - 429 指数退避重试最多 3 次
- post_comment(object_id, object_type, content, parent_id): 发布评论
  - Cookie+CSRF 方式，x-xsrftoken 请求头
  - 发布失败时返回 False（主流程处理写入 pending/）
- Comment dataclass: id, parent_id, content, author, created_time, is_author_reply
- Cookie 失效（401/403）时抛出 ZhihuAuthError

设计说明：
- 请求头参考 zhihu-cli 的浏览器指纹设计
- _xsrf 从 Cookie 字符串中自动解析
- 所有 HTTP 错误统一处理：401/403 → ZhihuAuthError, 429 → 重试
"""

from __future__ import annotations

import logging
import re
import time
import random
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)


# ─── 异常定义 ──────────────────────────────────────────────────

class ZhihuAuthError(Exception):
    """Cookie 失效或认证失败（HTTP 401/403）"""
    pass


class ZhihuRateLimitError(Exception):
    """请求频率超限（HTTP 429），重试后仍失败"""
    pass


# ─── 数据模型 ──────────────────────────────────────────────────

@dataclass
class Comment:
    """
    知乎评论数据模型
    参考: docs/plan/README.md § AI-003 第 4 点

    Attributes:
        id: 评论 ID
        parent_id: 父评论 ID（追问时非空）
        content: 评论内容
        author: 评论者用户名
        created_time: 创建时间戳（Unix 秒）
        is_author_reply: 是否为文章作者的回复
    """
    id: str
    parent_id: Optional[str]
    content: str
    author: str
    created_time: int
    is_author_reply: bool = False


# ─── 客户端主类 ──────────────────────────────────────────────────

class ZhihuClient:
    """
    知乎 API v4 客户端封装
    参考: docs/plan/README.md § AI-003, docs/调研/01-知乎数据获取.md

    使用方式：
        client = ZhihuClient(cookie="z_c0=xxx; _xsrf=yyy")
        comments = client.get_comments("98765432", "article")
    """

    # 知乎 API 基础 URL
    API_READ_BASE = "https://www.zhihu.com/api/v4"
    API_WRITE_BASE = "https://api.zhihu.com/v4"

    # 浏览器指纹（参考 zhihu-cli）
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.zhihu.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        "sec-ch-ua-platform": '"macOS"',
    }

    # 重试配置
    MAX_RETRIES = 3
    REQUEST_DELAY_MIN = 1.0  # 秒
    REQUEST_DELAY_MAX = 2.0  # 秒
    PAGE_LIMIT = 20  # 每页最大数量

    def __init__(self, cookie: str, max_retries: int = 3) -> None:
        """
        初始化知乎客户端

        Args:
            cookie: 完整的知乎 Cookie 字符串（如 "z_c0=xxx; _xsrf=yyy"）
            max_retries: 429 限流最大重试次数（默认 3）

        Raises:
            ValueError: Cookie 为空
        """
        if not cookie:
            raise ValueError("Cookie 不能为空")

        self.cookie = cookie
        self.max_retries = max_retries
        self._xsrf = self._extract_xsrf(cookie)
        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)
        self.session.headers["Cookie"] = cookie

        logger.info("ZhihuClient 初始化完成，_xsrf=%s...", self._xsrf[:8] if self._xsrf else "N/A")

    @staticmethod
    def _extract_xsrf(cookie: str) -> Optional[str]:
        """
        从 Cookie 字符串中提取 _xsrf 值
        参考: docs/plan/README.md § AI-003 第 3 点

        Args:
            cookie: Cookie 字符串

        Returns:
            _xsrf token 值，未找到时返回 None
        """
        match = re.search(r'_xsrf=([^;]+)', cookie)
        return match.group(1) if match else None

    def _build_read_url(self, object_id: str, object_type: str) -> str:
        """
        构建评论读取 API URL
        参考: docs/调研/01-知乎数据获取.md API 端点说明

        Args:
            object_id: 文章 ID 或回答 ID
            object_type: "article" 或 "question"

        Returns:
            完整的 API URL

        Raises:
            ValueError: 不支持的 object_type
        """
        if object_type == "article":
            return f"{self.API_READ_BASE}/articles/{object_id}/comments"
        elif object_type == "question":
            return f"{self.API_READ_BASE}/answers/{object_id}/comments"
        else:
            raise ValueError(f"不支持的 object_type: {object_type}，应为 'article' 或 'question'")

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        带指数退避重试的 HTTP 请求
        参考: docs/plan/README.md § AI-003 第 2 点 — 429 指数退避重试最多 3 次

        Args:
            method: HTTP 方法（GET/POST）
            url: 请求 URL
            **kwargs: requests 参数

        Returns:
            HTTP 响应对象

        Raises:
            ZhihuAuthError: 401/403
            ZhihuRateLimitError: 429 重试后仍失败
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, url, **kwargs)

                # 认证失败 → 立即抛出，不重试
                if response.status_code in (401, 403):
                    logger.error("认证失败 (HTTP %d)，Cookie 可能已失效", response.status_code)
                    raise ZhihuAuthError(
                        f"认证失败 HTTP {response.status_code}: Cookie 失效或权限不足"
                    )

                # 限流 → 指数退避重试
                if response.status_code == 429:
                    wait = (2 ** attempt) + random.random()
                    logger.warning(
                        "HTTP 429 限流，第 %d/%d 次重试，等待 %.1f 秒",
                        attempt + 1, self.max_retries, wait
                    )
                    last_exception = ZhihuRateLimitError(
                        f"HTTP 429 限流（第 {attempt + 1} 次）"
                    )
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                return response

            except ZhihuAuthError:
                raise
            except requests.RequestException as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait = (2 ** attempt) + random.random()
                    logger.warning(
                        "请求异常 (%s)，第 %d/%d 次重试，等待 %.1f 秒",
                        e, attempt + 1, self.max_retries, wait
                    )
                    time.sleep(wait)

        raise ZhihuRateLimitError(
            f"请求失败，已重试 {self.max_retries} 次: {last_exception}"
        )

    def get_comments(
        self,
        object_id: str,
        object_type: str,
        since_id: Optional[str] = None,
    ) -> list[Comment]:
        """
        获取指定文章/问题的评论列表
        参考: docs/plan/README.md § AI-003 第 2 点

        根据 object_type 选择不同的 API 端点：
        - article: GET /api/v4/articles/{id}/comments
        - question: GET /api/v4/answers/{answer_id}/comments

        Args:
            object_id: 文章 ID 或回答 ID
            object_type: "article" 或 "question"
            since_id: 从此 ID 之后开始获取（用于增量检测）

        Returns:
            Comment 列表（按时间升序）

        Raises:
            ZhihuAuthError: Cookie 失效
            ValueError: 不支持的 object_type
        """
        base_url = self._build_read_url(object_id, object_type)

        all_comments: list[Comment] = []
        offset = 0

        while True:
            params = {"limit": self.PAGE_LIMIT, "offset": offset}

            # 随机延迟 1~2 秒，降低反爬风险
            # 参考: docs/调研/01-知乎数据获取.md § 5. 反爬与长期稳定访问
            delay = random.uniform(self.REQUEST_DELAY_MIN, self.REQUEST_DELAY_MAX)
            time.sleep(delay)

            logger.debug("请求评论: %s offset=%d", base_url, offset)
            response = self._request_with_retry("GET", base_url, params=params)
            data = response.json()

            comments_data = data.get("data", [])
            for item in comments_data:
                comment = self._parse_comment(item)
                all_comments.append(comment)

            # 检查分页结束标志
            # 参考: docs/调研/01-知乎数据获取.md § 2. 核心读取接口
            paging = data.get("paging", {})
            if paging.get("is_end", True):
                break

            offset += self.PAGE_LIMIT

        # 按 since_id 过滤（若提供），使用数值比较而非字典序
        if since_id is not None:
            try:
                since_id_int = int(since_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("since_id must be a numeric comment ID") from exc

            filtered_comments: list[Comment] = []
            for c in all_comments:
                try:
                    comment_id_int = int(c.id)
                except (TypeError, ValueError):
                    continue
                if comment_id_int > since_id_int:
                    filtered_comments.append(c)
            all_comments = filtered_comments

        logger.info(
            "获取 %s/%s 评论 %d 条",
            object_type, object_id, len(all_comments)
        )
        return all_comments

    @staticmethod
    def _parse_comment(item: dict) -> Comment:
        """
        将 API 返回的 JSON 解析为 Comment 对象
        参考: docs/plan/README.md § AI-003 第 4 点

        Args:
            item: API 返回的单条评论数据

        Returns:
            Comment 对象
        """
        author_info = item.get("author", {})

        return Comment(
            id=str(item.get("id", "")),
            parent_id=str(item["reply_comment"]["id"]) if item.get("reply_comment") else None,
            content=item.get("content", ""),
            author=author_info.get("name", "unknown"),
            created_time=item.get("created_time", 0),
            is_author_reply=item.get("is_author", False),
        )

    def post_comment(
        self,
        object_id: str,
        object_type: str,
        content: str,
        parent_id: Optional[str] = None,
    ) -> bool:
        """
        发布评论（Cookie+CSRF 方式）
        参考: docs/plan/README.md § AI-003 第 3 点, AI-014

        使用 POST https://api.zhihu.com/v4/comments 接口
        请求头包含 x-xsrftoken（从 Cookie 中提取的 _xsrf 值）

        Args:
            object_id: 目标文章/回答 ID
            object_type: "article" 或 "answer"
            content: 评论内容
            parent_id: 父评论 ID（回复追问时使用）

        Returns:
            True 表示发布成功，False 表示发布失败（由主流程写入 pending/）
        """
        url = f"{self.API_WRITE_BASE}/comments"

        # 构建请求体
        # 参考: docs/调研/01-知乎数据获取.md § 3. 写操作接口
        payload: dict = {
            "content": content,
            "object_id": object_id,
            "object_type": object_type,
        }
        # _xsrf 缺失时无法发布，直接返回 False
        if not self._xsrf:
            logger.warning("Cookie 中无 _xsrf token，无法发布评论")
            return False

        if parent_id:
            payload["parent_id"] = parent_id

        # 设置 CSRF header
        # 参考: docs/plan/README.md § AI-003 第 3 点 — 从 Cookie 提取 _xsrf
        headers = {
            "x-xsrftoken": self._xsrf,
            "Content-Type": "application/json",
        }

        try:
            response = self._request_with_retry("POST", url, json=payload, headers=headers)
            if response.status_code in (200, 201):
                logger.info("评论发布成功: object_id=%s", object_id)
                return True
            else:
                logger.warning(
                    "评论发布返回非预期状态码 %d: %s",
                    response.status_code, response.text[:200]
                )
                return False
        except ZhihuAuthError:
            # 认证失败时不抛出，返回 False，让主流程写入 pending/
            # 参考: docs/plan/README.md § AI-003 第 3 点
            logger.error("评论发布失败: Cookie 失效或权限不足")
            return False
        except Exception as e:
            logger.error("评论发布异常: %s", e)
            return False

    def get_column_articles(self, column_id: str) -> list[dict]:
        """
        获取专栏下所有文章列表

        用于展开 type="column" 的监控目标，自动获取专栏内全部文章。

        Args:
            column_id: 专栏 ID（如 "csm-practice"）

        Returns:
            文章字典列表，每项含 id/title/url/type
        """
        url = f"{self.API_READ_BASE}/columns/{column_id}/articles"
        all_articles: list[dict] = []
        offset = 0

        while True:
            params = {"limit": self.PAGE_LIMIT, "offset": offset}
            delay = random.uniform(self.REQUEST_DELAY_MIN, self.REQUEST_DELAY_MAX)
            time.sleep(delay)

            logger.debug("请求专栏文章: %s offset=%d", url, offset)
            response = self._request_with_retry("GET", url, params=params)
            data = response.json()

            items = data.get("data", [])
            for item in items:
                all_articles.append({
                    "id": str(item.get("id", "")),
                    "title": item.get("title", ""),
                    "url": item.get("url", f"https://zhuanlan.zhihu.com/p/{item.get('id', '')}"),
                    "type": "article",
                })

            paging = data.get("paging", {})
            if paging.get("is_end", True):
                break
            offset += self.PAGE_LIMIT

        logger.info("专栏 %s 获取到 %d 篇文章", column_id, len(all_articles))
        return all_articles

    def get_user_answers(self, user_id: str) -> list[dict]:
        """
        获取某用户的全部回答列表

        用于展开 type="user_answers" 的监控目标。

        Args:
            user_id: 用户 URL ID（如 "nevstop"）

        Returns:
            回答字典列表，每项含 id/title/url/type
        """
        url = f"{self.API_READ_BASE}/members/{user_id}/answers"
        all_answers: list[dict] = []
        offset = 0

        while True:
            params = {"limit": self.PAGE_LIMIT, "offset": offset}
            delay = random.uniform(self.REQUEST_DELAY_MIN, self.REQUEST_DELAY_MAX)
            time.sleep(delay)

            logger.debug("请求用户回答: %s offset=%d", url, offset)
            response = self._request_with_retry("GET", url, params=params)
            data = response.json()

            items = data.get("data", [])
            for item in items:
                answer_id = str(item.get("id", ""))
                question = item.get("question", {})
                all_answers.append({
                    "id": answer_id,
                    "title": question.get("title", item.get("excerpt", "")),
                    "url": f"https://www.zhihu.com/question/{question.get('id', '')}/answer/{answer_id}",
                    "type": "question",
                })

            paging = data.get("paging", {})
            if paging.get("is_end", True):
                break
            offset += self.PAGE_LIMIT

        logger.info("用户 %s 获取到 %d 个回答", user_id, len(all_answers))
        return all_answers
