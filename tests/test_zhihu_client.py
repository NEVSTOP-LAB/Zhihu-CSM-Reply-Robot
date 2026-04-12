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
    ZhihuRequestError,
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

    def test_cookie_stripped_of_whitespace(self):
        """Cookie 值中的首尾空白/换行应被清除（防止 HTTP 头注入）"""
        client = ZhihuClient(cookie="  z_c0=value; _xsrf=tok\n")
        assert "\n" not in client.cookie
        assert client.cookie == "z_c0=value; _xsrf=tok"
        assert client.session.headers["Cookie"] == "z_c0=value; _xsrf=tok"
        assert client._xsrf == "tok"


# ===== URL 构建测试 =====

class TestBuildReadUrl:
    """验证读取 API URL 构建逻辑"""

    def test_article_url(self, client):
        """文章评论应调用 articles 端点"""
        url = client._build_read_url("12345", "article")
        assert url == "https://www.zhihu.com/api/v4/articles/12345/comments"

    def test_question_url(self, client):
        """问题评论（向后兼容）应调用 answers 端点"""
        url = client._build_read_url("67890", "question")
        assert url == "https://www.zhihu.com/api/v4/answers/67890/comments"

    def test_answer_url(self, client):
        """answer 类型应调用 answers 端点（FIX-01）"""
        url = client._build_read_url("67890", "answer")
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

        with patch.object(client.read_session, "request", return_value=resp):
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
            client.read_session, "request", side_effect=[page1, page2]
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

        with patch.object(client.read_session, "request", return_value=resp) as mock_req:
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

        with patch.object(client.read_session, "request", return_value=resp):
            result = client.get_comments("99", "article")

        assert result == []


# ===== 认证错误测试 =====

class TestAuthErrors:
    """验证认证失败场景"""

    @patch("scripts.zhihu_client.time.sleep")
    def test_401_raises_auth_error(self, mock_sleep, client):
        """401 响应应抛出 ZhihuAuthError"""
        resp = _make_response(401)

        with patch.object(client.read_session, "request", return_value=resp):
            with pytest.raises(ZhihuAuthError, match="认证失败"):
                client.get_comments("99", "article")

    @patch("scripts.zhihu_client.time.sleep")
    def test_403_raises_auth_error(self, mock_sleep, client):
        """403 响应应抛出 ZhihuAuthError 并携带 status_code=403（FIX-11）"""
        resp = _make_response(403)

        with patch.object(client.read_session, "request", return_value=resp):
            with pytest.raises(ZhihuAuthError) as exc_info:
                client.get_comments("99", "article")

        assert exc_info.value.status_code == 403

    @patch("scripts.zhihu_client.time.sleep")
    def test_401_auth_error_carries_status_code(self, mock_sleep, client):
        """401 ZhihuAuthError 应携带 status_code=401（FIX-11）"""
        resp = _make_response(401)

        with patch.object(client.read_session, "request", return_value=resp):
            with pytest.raises(ZhihuAuthError) as exc_info:
                client.get_comments("99", "article")

        assert exc_info.value.status_code == 401


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
            client.read_session, "request", side_effect=[resp_429, resp_429, resp_ok]
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
            client.read_session, "request", return_value=resp_429
        ):
            with pytest.raises(ZhihuRateLimitError, match="重试"):
                client.get_comments("99", "article")

    @patch("scripts.zhihu_client.time.sleep")
    def test_network_error_raises_request_error(self, mock_sleep, client):
        """网络类异常耗尽重试后应抛出 ZhihuRequestError 而非 ZhihuRateLimitError（FIX-10）"""
        with patch.object(
            client.read_session, "request",
            side_effect=requests.exceptions.ConnectionError("DNS 失败")
        ):
            with pytest.raises(ZhihuRequestError):
                client.get_comments("99", "article")

    @patch("scripts.zhihu_client.time.sleep")
    def test_timeout_error_raises_request_error(self, mock_sleep, client):
        """超时异常耗尽重试后应抛出 ZhihuRequestError（FIX-10）"""
        with patch.object(
            client.read_session, "request",
            side_effect=requests.exceptions.Timeout("请求超时")
        ):
            with pytest.raises(ZhihuRequestError):
                client.get_comments("99", "article")

    @patch("scripts.zhihu_client.time.sleep")
    def test_http_5xx_raises_request_error(self, mock_sleep, client):
        """HTTP 5xx 耗尽重试后应抛出 ZhihuRequestError（REV-4：区分 HTTPError vs 纯网络异常）"""
        resp_500 = _make_response(500)
        resp_500.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Server Error"
        )

        with patch.object(client.read_session, "request", return_value=resp_500):
            with pytest.raises(ZhihuRequestError, match="HTTP"):
                client.get_comments("99", "article")

    @patch("scripts.zhihu_client.time.sleep")
    def test_network_error_message_says_network(self, mock_sleep, client):
        """纯网络异常的错误信息应包含'网络'（REV-4：错误文案区分）"""
        with patch.object(
            client.read_session, "request",
            side_effect=requests.exceptions.ConnectionError("网络中断")
        ):
            with pytest.raises(ZhihuRequestError, match="网络"):
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
    def test_post_auth_failure_raises(self, mock_sleep, client):
        """发布时遇到 401 应重新抛出 ZhihuAuthError（FIX-09）"""
        resp = _make_response(401)

        with patch.object(client.session, "request", return_value=resp):
            with pytest.raises(ZhihuAuthError):
                client.post_comment("99", "article", "内容")

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


# ===== 问题回答列表测试（FIX-01）=====

class TestGetQuestionAnswers:
    """验证 get_question_answers 通过问题 ID 获取回答列表"""

    @patch("scripts.zhihu_client.time.sleep")
    def test_returns_answer_type(self, mock_sleep, client):
        """返回的条目 type 应为 'answer'（FIX-01）"""
        resp = _make_response(200, {
            "data": [
                {
                    "id": 111111,
                    "author": {"name": "作者A"},
                    "question": {"id": 99999, "title": "问题标题"},
                }
            ],
            "paging": {"is_end": True},
        })

        with patch.object(client.read_session, "request", return_value=resp):
            result = client.get_question_answers("99999")

        assert len(result) == 1
        assert result[0]["id"] == "111111"
        assert result[0]["type"] == "answer"
        assert "99999" in result[0]["url"]
        assert "111111" in result[0]["url"]

    @patch("scripts.zhihu_client.time.sleep")
    def test_calls_questions_endpoint(self, mock_sleep, client):
        """应调用 /questions/{id}/answers 端点（FIX-01）"""
        resp = _make_response(200, {
            "data": [],
            "paging": {"is_end": True},
        })

        with patch.object(client.read_session, "request", return_value=resp) as mock_req:
            client.get_question_answers("12345")

        call_args = mock_req.call_args
        assert "questions/12345/answers" in str(call_args)

    @patch("scripts.zhihu_client.time.sleep")
    def test_get_user_answers_returns_answer_type(self, mock_sleep, client):
        """get_user_answers 返回的条目 type 应为 'answer'（FIX-01）"""
        resp = _make_response(200, {
            "data": [
                {
                    "id": 222222,
                    "question": {"id": 88888, "title": "回答的问题"},
                }
            ],
            "paging": {"is_end": True},
        })

        with patch.object(client.write_session, "request", return_value=resp):
            result = client.get_user_answers("some_user")

        assert result[0]["type"] == "answer"


# ===== Cookie 隔离测试（Issue #2）=====

class TestCookieIsolation:
    """验证读操作不携带 Cookie、写操作携带 Cookie（Issue #2）

    知乎不登录即可查看文章和评论，只有回复时才需要 Cookie。
    通过分离 read_session（无 Cookie）和 write_session（含 Cookie），
    可以降低被反爬追踪的风险。
    """

    @patch("scripts.zhihu_client.time.sleep")
    def test_get_comments_does_not_send_cookie(self, mock_sleep, client):
        """get_comments 请求不应包含 Cookie 头（read_session）"""
        resp = _make_response(200, {
            "data": [],
            "paging": {"is_end": True},
        })

        with patch.object(client.read_session, "request", return_value=resp) as mock_req:
            client.get_comments("99999", "article")

        # 验证确实通过 read_session 发起了请求
        mock_req.assert_called_once()

        # read_session 的永久请求头中不应存在 Cookie
        assert "Cookie" not in client.read_session.headers, (
            "read_session 不应包含 Cookie 头"
        )

        # 验证运行时传入的额外 headers 参数（若有）中也不含 Cookie
        call_kwargs = mock_req.call_args[1] if mock_req.call_args else {}
        extra_headers = call_kwargs.get("headers", {}) or {}
        assert "Cookie" not in extra_headers, (
            "get_comments 调用时不应在 headers 参数中传递 Cookie"
        )

    @patch("scripts.zhihu_client.time.sleep")
    def test_post_comment_sends_cookie(self, mock_sleep, client):
        """post_comment 请求应使用含 Cookie 的 write_session"""
        resp = _make_response(201, {"id": "new_comment"})

        with patch.object(client.write_session, "request", return_value=resp) as mock_req:
            client.post_comment("99", "article", "回复内容")

        mock_req.assert_called_once()
        # write_session 应含 Cookie
        assert "Cookie" in client.write_session.headers, (
            "write_session 应包含 Cookie 头"
        )

    @patch("scripts.zhihu_client.time.sleep")
    def test_get_column_articles_uses_read_session(self, mock_sleep, client):
        """get_column_articles 应使用 read_session（无 Cookie）— 专栏文章为公开内容"""
        resp = _make_response(200, {
            "data": [],
            "paging": {"is_end": True},
        })

        with patch.object(client.read_session, "request", return_value=resp) as mock_req:
            client.get_column_articles("c_1234567890")

        mock_req.assert_called_once()

    @patch("scripts.zhihu_client.time.sleep")
    def test_get_column_articles_uses_items_endpoint(self, mock_sleep, client):
        """get_column_articles 应调用 /columns/{id}/items 端点（/articles 已返回 404）"""
        resp = _make_response(200, {
            "data": [],
            "paging": {"is_end": True},
        })

        with patch.object(client.read_session, "request", return_value=resp) as mock_req:
            client.get_column_articles("c_1234567890")

        call_args = mock_req.call_args
        assert "columns/c_1234567890/items" in str(call_args)

    @patch("scripts.zhihu_client.time.sleep")
    def test_get_column_articles_parses_items_format(self, mock_sleep, client):
        """get_column_articles 应正确解析 /items 返回的 {type, content} 格式"""
        resp = _make_response(200, {
            "data": [
                {
                    "type": "article",
                    "content": {
                        "id": 123456,
                        "title": "测试文章",
                        "url": "https://zhuanlan.zhihu.com/p/123456",
                    },
                },
                {
                    "type": "pin",  # 非文章类型，应被忽略
                    "content": {"id": 999},
                },
            ],
            "paging": {"is_end": True},
        })

        with patch.object(client.read_session, "request", return_value=resp):
            result = client.get_column_articles("c_1234567890")

        assert len(result) == 1
        assert result[0]["id"] == "123456"
        assert result[0]["title"] == "测试文章"
        assert result[0]["type"] == "article"

    @patch("scripts.zhihu_client.time.sleep")
    def test_get_column_articles_handles_non_dict_response(self, mock_sleep, client):
        """get_column_articles 遇到非 dict 响应体时应安全返回空列表"""
        # API 偶尔会在 200 状态下返回一个 JSON 字符串而非对象
        resp = _make_response(200, "认证失败")  # response.json() 返回字符串

        with patch.object(client.read_session, "request", return_value=resp):
            result = client.get_column_articles("c_1234567890")

        assert result == []

    @patch("scripts.zhihu_client.time.sleep")
    def test_get_column_articles_skips_non_dict_items(self, mock_sleep, client):
        """get_column_articles data[] 中出现非 dict 条目时应跳过，不引发 AttributeError"""
        resp = _make_response(200, {
            "data": [
                "unexpected_string_item",  # 非 dict，应被跳过
                {
                    "type": "article",
                    "content": {"id": 42, "title": "正常文章", "url": "https://zhuanlan.zhihu.com/p/42"},
                },
            ],
            "paging": {"is_end": True},
        })

        with patch.object(client.read_session, "request", return_value=resp):
            result = client.get_column_articles("c_1234567890")

        assert len(result) == 1
        assert result[0]["id"] == "42"

    @patch("scripts.zhihu_client.time.sleep")
    def test_get_user_answers_uses_write_session(self, mock_sleep, client):
        """get_user_answers 应使用 write_session（含 Cookie）— 该接口需要认证"""
        resp = _make_response(200, {
            "data": [],
            "paging": {"is_end": True},
        })

        with patch.object(client.write_session, "request", return_value=resp) as mock_req:
            client.get_user_answers("some_user")

        mock_req.assert_called_once()

    @patch("scripts.zhihu_client.time.sleep")
    def test_get_question_answers_uses_read_session(self, mock_sleep, client):
        """get_question_answers 应使用 read_session（无 Cookie）"""
        resp = _make_response(200, {
            "data": [],
            "paging": {"is_end": True},
        })

        with patch.object(client.read_session, "request", return_value=resp) as mock_req:
            client.get_question_answers("12345")

        mock_req.assert_called_once()

    def test_read_session_has_no_cookie_header(self, client):
        """read_session 不应设置 Cookie 头"""
        assert "Cookie" not in client.read_session.headers

    def test_write_session_has_cookie_header(self, client):
        """write_session 应包含 Cookie 头"""
        assert "Cookie" in client.write_session.headers
        assert client.write_session.headers["Cookie"] == SAMPLE_COOKIE.strip()

    def test_read_write_sessions_are_different_objects(self, client):
        """read_session 和 write_session 应是不同的 Session 对象"""
        assert client.read_session is not client.write_session
