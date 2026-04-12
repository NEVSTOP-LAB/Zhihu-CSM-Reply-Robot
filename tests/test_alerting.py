# -*- coding: utf-8 -*-
"""
AlertManager 单元测试
=====================

实施计划关联：AI-010 验收标准
独立于实现的测试用例，覆盖：
- Issue 标题和标签正确
- 重复告警时不创建第二个 issue（幂等）
- health.json 记录了 Cookie 状态
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.alerting import AlertManager


# ===== Fixtures =====

@pytest.fixture
def health_file(tmp_path: Path) -> Path:
    """临时健康状态文件路径"""
    return tmp_path / "data" / "health.json"


@pytest.fixture
def manager(health_file: Path) -> AlertManager:
    """创建测试用 AlertManager"""
    return AlertManager(
        github_token="test-token",
        repo="test-owner/test-repo",
        health_file=str(health_file),
    )


def _mock_get_no_issues(url, **kwargs):
    """模拟 GitHub API 返回空 issue 列表（无分页）"""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = []
    resp.headers = {}
    return resp


def _mock_get_with_issue(title):
    """模拟 GitHub API 返回包含指定 title 的 issue（无分页）"""
    def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = [{"title": title}]
        resp.headers = {}
        return resp
    return mock_get


def _mock_post_success(url, **kwargs):
    """模拟 GitHub API 创建 issue 成功"""
    resp = MagicMock()
    resp.status_code = 201
    resp.json.return_value = {"id": 1, "number": 42}
    return resp


# ===== Issue 创建测试 =====

class TestCreateIssue:
    """验证 create_issue 逻辑"""

    def test_create_issue_success(self, manager):
        """成功创建 Issue"""
        with patch.object(
            manager._session, "get", side_effect=_mock_get_no_issues
        ), patch.object(
            manager._session, "post", side_effect=_mock_post_success
        ) as mock_post:
            result = manager.create_issue(
                title="测试告警",
                body="告警内容",
                labels=["test"],
            )

        assert result is True
        # 验证 POST 请求的 payload
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["title"] == "测试告警"
        assert payload["body"] == "告警内容"
        assert "test" in payload["labels"]

    def test_create_issue_idempotent(self, manager):
        """已有同名 open issue 时不重复创建（幂等）"""
        with patch.object(
            manager._session, "get",
            side_effect=_mock_get_with_issue("测试告警")
        ), patch.object(
            manager._session, "post", side_effect=_mock_post_success
        ) as mock_post:
            result = manager.create_issue("测试告警", "内容")

        assert result is True
        # 不应调用 POST
        mock_post.assert_not_called()

    def test_create_issue_without_token(self):
        """无 token 时跳过"""
        m = AlertManager(github_token="", repo="")
        result = m.create_issue("标题", "内容")
        assert result is False

    def test_create_issue_api_failure(self, manager):
        """API 调用失败时返回 False"""
        fail_resp = MagicMock()
        fail_resp.status_code = 500
        fail_resp.text = "Internal Server Error"

        with patch.object(
            manager._session, "get", side_effect=_mock_get_no_issues
        ), patch.object(
            manager._session, "post", return_value=fail_resp
        ):
            result = manager.create_issue("标题", "内容")

        assert result is False


# ===== 告警场景测试 =====

class TestAlertScenarios:
    """验证各告警场景"""

    def test_alert_cookie_expired(self, manager):
        """Cookie 失效告警应创建正确的 Issue"""
        with patch.object(
            manager._session, "get", side_effect=_mock_get_no_issues
        ), patch.object(
            manager._session, "post", side_effect=_mock_post_success
        ) as mock_post:
            manager.alert_cookie_expired(401)

        payload = mock_post.call_args[1]["json"]
        assert "Cookie" in payload["title"]
        assert "401" in payload["body"]
        assert "cookie-expired" in payload["labels"]

    def test_alert_rate_limited(self, manager):
        """限流告警应创建正确的 Issue"""
        with patch.object(
            manager._session, "get", side_effect=_mock_get_no_issues
        ), patch.object(
            manager._session, "post", side_effect=_mock_post_success
        ) as mock_post:
            manager.alert_rate_limited()

        payload = mock_post.call_args[1]["json"]
        assert "429" in payload["title"]
        assert "rate-limit" in payload["labels"]

    def test_alert_budget_exceeded(self, manager):
        """预算超限告警应包含费用信息"""
        with patch.object(
            manager._session, "get", side_effect=_mock_get_no_issues
        ), patch.object(
            manager._session, "post", side_effect=_mock_post_success
        ) as mock_post:
            manager.alert_budget_exceeded(cost=0.55, budget=0.50)

        payload = mock_post.call_args[1]["json"]
        assert "预算" in payload["title"]
        assert "$0.55" in payload["body"]
        assert "budget" in payload["labels"]

    def test_alert_consecutive_failures(self, manager):
        """连续失败告警应包含次数"""
        with patch.object(
            manager._session, "get", side_effect=_mock_get_no_issues
        ), patch.object(
            manager._session, "post", side_effect=_mock_post_success
        ) as mock_post:
            manager.alert_consecutive_failures(count=5)

        payload = mock_post.call_args[1]["json"]
        assert "5" in payload["title"]
        assert "failure" in payload["labels"]

    def test_alert_consecutive_failures_with_logs(self, manager):
        """连续失败告警应在 Issue 正文中包含最近日志"""
        sample_logs = "```\n2026-01-01 [ERROR] 处理评论失败: timeout\n```"
        with patch.object(
            manager._session, "get", side_effect=_mock_get_no_issues
        ), patch.object(
            manager._session, "post", side_effect=_mock_post_success
        ) as mock_post:
            manager.alert_consecutive_failures(count=3, recent_logs=sample_logs)

        payload = mock_post.call_args[1]["json"]
        assert "3" in payload["title"]
        assert "最近日志" in payload["body"]
        assert "timeout" in payload["body"]

    def test_alert_consecutive_failures_no_logs(self, manager):
        """不传日志时 Issue 正文不应包含日志节（保持向后兼容）"""
        with patch.object(
            manager._session, "get", side_effect=_mock_get_no_issues
        ), patch.object(
            manager._session, "post", side_effect=_mock_post_success
        ) as mock_post:
            manager.alert_consecutive_failures(count=3)

        payload = mock_post.call_args[1]["json"]
        assert "最近日志" not in payload["body"]

    def test_alert_expansion_failed(self, manager, health_file):
        """展开失败告警应包含配置数量和最近日志"""
        sample_logs = "```\n[WARNING] 展开专栏失败\n```"
        with patch.object(
            manager._session, "get", side_effect=_mock_get_no_issues
        ), patch.object(
            manager._session, "post", side_effect=_mock_post_success
        ) as mock_post:
            manager.alert_expansion_failed(configured_count=3, recent_logs=sample_logs)

        payload = mock_post.call_args[1]["json"]
        assert "展开失败" in payload["title"]
        assert "3" in payload["body"]
        assert "最近日志" in payload["body"]
        assert "展开专栏失败" in payload["body"]
        assert "expansion-failed" in payload["labels"]

        # 验证 health.json 被写入
        assert health_file.exists()
        data = json.loads(health_file.read_text())
        assert data["status"] == "expansion_failed"
        assert data["details"]["configured_count"] == 3

    def test_alert_expansion_failed_no_logs(self, manager):
        """不传日志时展开失败告警仍应正常创建"""
        with patch.object(
            manager._session, "get", side_effect=_mock_get_no_issues
        ), patch.object(
            manager._session, "post", side_effect=_mock_post_success
        ) as mock_post:
            manager.alert_expansion_failed(configured_count=2)

        payload = mock_post.call_args[1]["json"]
        assert "expansion-failed" in payload["labels"]
        assert "最近日志" not in payload["body"]


# ===== 健康状态记录测试 =====

class TestHealthRecord:
    """验证 health.json 记录"""

    def test_record_health_ok(self, manager, health_file):
        """记录正常状态"""
        manager.record_health("ok")

        assert health_file.exists()
        data = json.loads(health_file.read_text())
        assert data["status"] == "ok"
        assert "last_check" in data

    def test_record_health_with_details(self, manager, health_file):
        """记录包含详情的状态"""
        manager.record_health(
            "cookie_expired", {"error_code": 401}
        )

        data = json.loads(health_file.read_text())
        assert data["status"] == "cookie_expired"
        assert data["details"]["error_code"] == 401

    def test_record_health_creates_directory(self, tmp_path):
        """目录不存在时应自动创建"""
        deep_path = tmp_path / "deep" / "nested" / "health.json"
        m = AlertManager(
            github_token="test",
            repo="test/test",
            health_file=str(deep_path),
        )
        m.record_health("ok")
        assert deep_path.exists()

    def test_record_health_overwrites(self, manager, health_file):
        """多次记录应覆盖旧数据"""
        manager.record_health("ok")
        manager.record_health("error", {"msg": "test"})

        data = json.loads(health_file.read_text())
        assert data["status"] == "error"


# ===== 分页查重测试 =====

class TestPaginatedIssueCheck:
    """验证 _has_open_issue 跨分页查重（REV-3）"""

    def test_finds_issue_on_second_page(self, manager):
        """目标 issue 在第二页时应正确检测到"""
        target_title = "第二页的告警"
        call_count = [0]

        def paginated_get(url, params=None, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            call_count[0] += 1
            if call_count[0] == 1:
                # 第一页：100 条不同 issue，并附 Link: next
                resp.json.return_value = [{"title": f"issue-{i}"} for i in range(100)]
                resp.headers = {
                    "Link": f'<{manager.GITHUB_API_BASE}/repos/test-owner/test-repo/issues?page=2>; rel="next"'
                }
            else:
                # 第二页：包含目标 issue，无下一页
                resp.json.return_value = [{"title": target_title}]
                resp.headers = {}
            return resp

        with patch.object(manager._session, "get", side_effect=paginated_get):
            result = manager._has_open_issue(target_title)

        assert result is True
        assert call_count[0] == 2

    def test_returns_false_when_not_on_any_page(self, manager):
        """所有页均无目标 issue 时应返回 False"""
        call_count = [0]

        def paginated_get(url, params=None, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            call_count[0] += 1
            if call_count[0] == 1:
                resp.json.return_value = [{"title": "其他告警"}]
                resp.headers = {
                    "Link": f'<{manager.GITHUB_API_BASE}/repos/test-owner/test-repo/issues?page=2>; rel="next"'
                }
            else:
                resp.json.return_value = []
                resp.headers = {}
            return resp

        with patch.object(manager._session, "get", side_effect=paginated_get):
            result = manager._has_open_issue("不存在的告警")

        assert result is False
        assert call_count[0] == 2
