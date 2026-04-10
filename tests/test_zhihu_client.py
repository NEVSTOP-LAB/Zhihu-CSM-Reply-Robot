# -*- coding: utf-8 -*-
"""
ZhihuClient 单元测试
=====================

实施计划关联：AI-003 ZhihuClient — 读取与写入接口
独立于实现的测试用例，覆盖：
- 正常分页返回（article 和 question 类型）
- is_end=True 停止分页
- 429 指数退避重试
- 401 抛出 ZhihuAuthError
- Comment dataclass 字段映射
- post_comment 目标 URL 和 CSRF token
- POST 成功/失败场景
"""
import json
from unittest.mock import MagicMock, patch, call

import pytest
import requests

from scripts.zhihu_client import (
    ZhihuClient,
    ZhihuAuthError,
    ZhihuRateLimitError,
    Comment,
)


# ===== Fixtures =====

SAMPLE_COOKIE = 'z_c0=test_z_c0_value; _xsrf=test_xsrf_token_123; d_c0=test_dc0'


@pytest.fixture
def client():
    """创建一个使用测试 Cookie 的 ZhihuClient 实例"""
    return ZhihuClient(cookie=SAMPLE_COOKIE, max_retries=3)


def _make_comment_json(
    comment_id: str = "100",
    content: str = "测试评论",
    author_name: str = "test_user",
    created_time: int = 1712000000,
    is_author: bool = False,
    reply_comment: dict | None = None,
) -> dict:
    """构造模拟的知乎评论 JSON"""
    result = {
        "id": int(comment_id),
        "content": content,
        "author": {"name": author_name},
        "created_time": created_time,
        "is_author": is_author,
    }
    if reply_comment:
        result["reply_comment"] = reply_comment
    return result


def _make_response(
    status_code: int = 200,
    json_data: dict | None = None,
) -> MagicMock:
    """构造模拟的 HTTP 响应"""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=resp
        )
    return resp


# ===== CSRF Token 提取测试 =====

class TestXsrfExtraction:
    """验证从 Cookie 中提取 _xsrf 的逻辑"""

    def test_extract_xsrf_from_cookie(self, client):
        """应从 Cookie 字符串中正确提取 _xsrf 值"""
        assert client._xsrf == "test_xsrf_token_123"

    def test_extract_xsrf_missing(self):
        """Cookie 中无 _xsrf 时应返回 None"""
        client = ZhihuClient(cookie="z_c0=value; d_c0=value")
        assert client._xsrf is None

    def test_extract_xsrf_at_end(self):
        """_xsrf 在 Cookie 末尾时应正确提取"""
        client = ZhihuClient(cookie="z_c0=value; _xsrf=end_token")
        assert client._xsrf == "end_token"


# ===== URL 构建测试 =====

class TestBuildReadUrl:
    """验证读取 API URL 构建逻辑"""

    def test_article_url(self, client):
        """文章评论应调用 articles 端点"""
        url = client._build_read_url("12345", "article")
        assert url == "https://www.zhihu.com/api/v4/articles/12345/comments"

    def test_question_url(self, client):
        """问题评论应调用 answers 端点"""
        url = client._build_read_url("67890", "question")
        assert url == "https://www.zhihu.com/api/v4/answers/67890/comments"

    def test_invalid_type_raises(self, client):
        """不支持的类型应抛出 ValueError"""
        with pytest.raises(ValueError, match="不支持的 object_type"):
            client._build_read_url("12345", "invalid_type")


# ===== 评论读取测试 =====

class TestGetComments:
    """验证 get_comments 方法的各种场景"""

    @patch("scripts.zhihu_client.time.sleep")  # 跳过延迟
    def test_single_page_article_comments(self, mock_sleep, client):
        """测试文章评论单页返回（is_end=True 停止分页）"""
        comments_data = [
            _make_comment_json("1", "第一条", "user_a", 1712000001),
            _make_comment_json("2", "第二条", "user_b", 1712000002),
        ]
        resp = _make_response(200, {
            "data": comments_data,
            "paging": {"is_end": True, "next": ""},
        })

        with patch.object(client.session, "request", return_value=resp):
            result = client.get_comments("99999", "article")

        assert len(result) == 2
        assert result[0].id == "1"
        assert result[0].content == "第一条"
        assert result[0].author == "user_a"
        assert result[1].id == "2"

    @patch("scripts.zhihu_client.time.sleep")
    def test_multi_page_article_comments(self, mock_sleep, client):
        """测试文章评论多页分页"""
        page1 = _make_response(200, {
            "data": [_make_comment_json("1", "page1")],
            "paging": {
                "is_end": False,
                "next": "https://www.zhihu.com/api/v4/articles/99/comments?offset=20",
            },
        })
        page2 = _make_response(200, {
            "data": [_make_comment_json("2", "page2")],
            "paging": {"is_end": True},
        })

        with patch.object(
            client.session, "request", side_effect=[page1, page2]
        ):
            result = client.get_comments("99", "article")

        assert len(result) == 2
        assert result[0].content == "page1"
        assert result[1].content == "page2"
        # 验证分页间有延迟调用
        assert mock_sleep.called

    @patch("scripts.zhihu_client.time.sleep")
    def test_question_type_comments(self, mock_sleep, client):
        """测试问题（回答）评论获取"""
        resp = _make_response(200, {
            "data": [_make_comment_json("10", "question_comment")],
            "paging": {"is_end": True},
        })

        with patch.object(client.session, "request", return_value=resp) as mock_req:
            result = client.get_comments("55555", "question")

        assert len(result) == 1
        assert result[0].content == "question_comment"
        # 验证调用了 answers 端点
        call_args = mock_req.call_args
        assert "answers/55555/comments" in str(call_args)

    @patch("scripts.zhihu_client.time.sleep")
    def test_empty_comments(self, mock_sleep, client):
        """无评论时返回空列表"""
        resp = _make_response(200, {
            "data": [],
            "paging": {"is_end": True},
        })

        with patch.object(client.session, "request", return_value=resp):
            result = client.get_comments("99", "article")

        assert result == []


# ===== 认证错误测试 =====

class TestAuthErrors:
    """验证认证失败场景"""

    @patch("scripts.zhihu_client.time.sleep")
    def test_401_raises_auth_error(self, mock_sleep, client):
        """401 响应应抛出 ZhihuAuthError"""
        resp = _make_response(401)

        with patch.object(client.session, "request", return_value=resp):
            with pytest.raises(ZhihuAuthError, match="认证失败"):
                client.get_comments("99", "article")

    @patch("scripts.zhihu_client.time.sleep")
    def test_403_raises_auth_error(self, mock_sleep, client):
        """403 响应应抛出 ZhihuAuthError"""
        resp = _make_response(403)

        with patch.object(client.session, "request", return_value=resp):
            with pytest.raises(ZhihuAuthError, match="认证失败"):
                client.get_comments("99", "article")


# ===== 限流重试测试 =====

class TestRateLimit:
    """验证 429 限流退避重试逻辑"""

    @patch("scripts.zhihu_client.time.sleep")
    def test_429_retry_then_success(self, mock_sleep, client):
        """429 后重试成功"""
        resp_429 = _make_response(429)
        resp_ok = _make_response(200, {
            "data": [_make_comment_json("1", "success after retry")],
            "paging": {"is_end": True},
        })

        with patch.object(
            client.session, "request", side_effect=[resp_429, resp_429, resp_ok]
        ):
            result = client.get_comments("99", "article")

        assert len(result) == 1
        assert result[0].content == "success after retry"
        # 验证 sleep 被调用（指数退避）
        assert mock_sleep.call_count >= 2

    @patch("scripts.zhihu_client.time.sleep")
    def test_429_max_retries_exceeded(self, mock_sleep, client):
        """429 超过最大重试次数应抛出 ZhihuRateLimitError"""
        resp_429 = _make_response(429)

        with patch.object(
            client.session, "request", return_value=resp_429
        ):
            with pytest.raises(ZhihuRateLimitError, match="重试"):
                client.get_comments("99", "article")


# ===== Comment 解析测试 =====

class TestCommentParsing:
    """验证 Comment dataclass 字段映射"""

    def test_basic_comment_fields(self):
        """基本评论字段映射正确"""
        raw = _make_comment_json(
            comment_id="777",
            content="测试内容",
            author_name="张三",
            created_time=1712345678,
            is_author=True,
        )
        comment = ZhihuClient._parse_comment(raw)

        assert comment.id == "777"
        assert comment.content == "测试内容"
        assert comment.author == "张三"
        assert comment.created_time == 1712345678
        assert comment.is_author_reply is True
        assert comment.parent_id is None

    def test_reply_comment_parent_id(self):
        """带父评论的回复应正确设置 parent_id"""
        raw = _make_comment_json("888", "回复")
        raw["reply_comment"] = {"id": 666}

        comment = ZhihuClient._parse_comment(raw)
        assert comment.parent_id == "666"

    def test_missing_author_defaults(self):
        """缺少 author 字段时应使用默认值"""
        raw = {"id": 999, "content": "no author", "created_time": 0}
        comment = ZhihuClient._parse_comment(raw)
        assert comment.author == "unknown"


# ===== 评论发布测试 =====

class TestPostComment:
    """验证 post_comment 方法"""

    @patch("scripts.zhihu_client.time.sleep")
    def test_post_success(self, mock_sleep, client):
        """发布成功应返回 True"""
        resp = _make_response(201, {"id": "new_comment_id"})

        with patch.object(client.session, "request", return_value=resp) as mock_req:
            result = client.post_comment("99", "article", "好文章！")

        assert result is True
        # 验证调用了正确的 URL
        call_args = mock_req.call_args
        assert call_args[0][0] == "POST"
        assert "api.zhihu.com/v4/comments" in call_args[0][1]

    @patch("scripts.zhihu_client.time.sleep")
    def test_post_includes_xsrf_header(self, mock_sleep, client):
        """发布请求应包含 x-xsrftoken 头"""
        resp = _make_response(200)

        with patch.object(client.session, "request", return_value=resp) as mock_req:
            client.post_comment("99", "article", "内容")

        call_args = mock_req.call_args
        headers = call_args[1].get("headers", {})
        assert headers.get("x-xsrftoken") == "test_xsrf_token_123"

    @patch("scripts.zhihu_client.time.sleep")
    def test_post_includes_object_params(self, mock_sleep, client):
        """发布请求应包含正确的 object_id 和 object_type"""
        resp = _make_response(200)

        with patch.object(client.session, "request", return_value=resp) as mock_req:
            client.post_comment("12345", "article", "内容")

        call_args = mock_req.call_args
        json_data = call_args[1].get("json", {})
        assert json_data["object_id"] == "12345"
        assert json_data["object_type"] == "article"

    @patch("scripts.zhihu_client.time.sleep")
    def test_post_with_parent_id(self, mock_sleep, client):
        """回复评论应包含 parent_id"""
        resp = _make_response(200)

        with patch.object(client.session, "request", return_value=resp) as mock_req:
            client.post_comment("99", "article", "回复", parent_id="777")

        call_args = mock_req.call_args
        json_data = call_args[1].get("json", {})
        assert json_data["parent_id"] == "777"

    @patch("scripts.zhihu_client.time.sleep")
    def test_post_auth_failure_returns_false(self, mock_sleep, client):
        """发布时遇到 401 应返回 False（不抛异常）"""
        resp = _make_response(401)

        with patch.object(client.session, "request", return_value=resp):
            result = client.post_comment("99", "article", "内容")

        assert result is False

    def test_post_without_xsrf_returns_false(self):
        """没有 _xsrf 时应返回 False"""
        client = ZhihuClient(cookie="z_c0=value_only")
        result = client.post_comment("99", "article", "内容")
        assert result is False

    @patch("scripts.zhihu_client.time.sleep")
    def test_post_network_error_returns_false(self, mock_sleep, client):
        """网络异常应返回 False"""
        with patch.object(
            client.session, "request",
            side_effect=requests.exceptions.ConnectionError("network error")
        ):
            result = client.post_comment("99", "article", "内容")

        assert result is False
