"""wiki_updater 模块测试（无网络、无真实 git）。"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from csm_qa.wiki_updater import (
    WikiSource,
    _repo_api_url,
    check_and_update_wiki,
    fetch_latest_commit_id,
    pull_wiki,
)


# ─── _repo_api_url ────────────────────────────────────────────────────────────

class TestRepoApiUrl:
    def test_plain_url(self):
        assert (
            _repo_api_url("https://github.com/NEVSTOP-LAB/CSM-Wiki")
            == "https://api.github.com/repos/NEVSTOP-LAB/CSM-Wiki"
        )

    def test_git_suffix(self):
        assert (
            _repo_api_url("https://github.com/NEVSTOP-LAB/CSM-Wiki.git")
            == "https://api.github.com/repos/NEVSTOP-LAB/CSM-Wiki"
        )

    def test_trailing_slash(self):
        assert (
            _repo_api_url("https://github.com/NEVSTOP-LAB/CSM-Wiki/")
            == "https://api.github.com/repos/NEVSTOP-LAB/CSM-Wiki"
        )

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="不支持的 GitHub 仓库 URL 格式"):
            _repo_api_url("https://gitlab.com/owner/repo")


# ─── WikiSource ───────────────────────────────────────────────────────────────

class TestWikiSource:
    def test_load(self, tmp_dir: Path):
        f = tmp_dir / "wiki_source.json"
        f.write_text(
            json.dumps({"url": "https://github.com/A/B", "commit_id": "abc123"}),
            encoding="utf-8",
        )
        src = WikiSource.load(f)
        assert src.url == "https://github.com/A/B"
        assert src.commit_id == "abc123"

    def test_load_missing_commit_id(self, tmp_dir: Path):
        f = tmp_dir / "wiki_source.json"
        f.write_text(json.dumps({"url": "https://github.com/A/B"}), encoding="utf-8")
        src = WikiSource.load(f)
        assert src.commit_id == ""

    def test_save_roundtrip(self, tmp_dir: Path):
        f = tmp_dir / "wiki_source.json"
        src = WikiSource(url="https://github.com/A/B", commit_id="deadbeef")
        src.save(f)
        loaded = WikiSource.load(f)
        assert loaded.url == src.url
        assert loaded.commit_id == src.commit_id

    def test_save_ends_with_newline(self, tmp_dir: Path):
        f = tmp_dir / "wiki_source.json"
        WikiSource(url="https://github.com/A/B", commit_id="x").save(f)
        assert f.read_text(encoding="utf-8").endswith("\n")


# ─── fetch_latest_commit_id ───────────────────────────────────────────────────

SHA = "a" * 40


def _make_response(sha: str, status: int = 200):
    """构造 urllib.request.urlopen 的 mock 返回值。"""
    import io

    body = json.dumps({"sha": sha}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestFetchLatestCommitId:
    def test_returns_sha(self):
        with patch("urllib.request.urlopen", return_value=_make_response(SHA)):
            result = fetch_latest_commit_id("https://github.com/A/B")
        assert result == SHA

    def test_falls_back_to_master(self):
        import urllib.error

        http_error_404 = urllib.error.HTTPError(
            url="", code=404, msg="Not Found", hdrs=None, fp=None
        )
        calls = [http_error_404, _make_response(SHA)]

        with patch("urllib.request.urlopen", side_effect=calls):
            result = fetch_latest_commit_id("https://github.com/A/B", branch="main")
        assert result == SHA

    def test_raises_on_both_404(self):
        import urllib.error

        http_error_404 = urllib.error.HTTPError(
            url="", code=404, msg="Not Found", hdrs=None, fp=None
        )
        with patch("urllib.request.urlopen", side_effect=[http_error_404, http_error_404]):
            with pytest.raises(ValueError, match="无法获取"):
                fetch_latest_commit_id("https://github.com/A/B")

    def test_raises_on_non_404_http_error(self):
        import urllib.error

        http_error_500 = urllib.error.HTTPError(
            url="", code=500, msg="Server Error", hdrs=None, fp=None
        )
        with patch("urllib.request.urlopen", side_effect=http_error_500):
            with pytest.raises(urllib.error.HTTPError):
                fetch_latest_commit_id("https://github.com/A/B")

    def test_missing_sha_raises(self):
        import io

        body = json.dumps({}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(ValueError, match="sha"):
                fetch_latest_commit_id("https://github.com/A/B")


# ─── pull_wiki ────────────────────────────────────────────────────────────────

class TestPullWiki:
    def test_clone_when_missing(self, tmp_dir: Path):
        local = tmp_dir / "wiki"
        with patch("subprocess.run") as mock_run:
            pull_wiki("https://github.com/A/B", local)
        mock_run.assert_called_once_with(
            ["git", "clone", "https://github.com/A/B", str(local)],
            check=True,
            capture_output=True,
        )

    def test_pull_when_exists(self, tmp_dir: Path):
        local = tmp_dir / "wiki"
        (local / ".git").mkdir(parents=True)
        with patch("subprocess.run") as mock_run:
            pull_wiki("https://github.com/A/B", local)
        mock_run.assert_called_once_with(
            ["git", "-C", str(local), "pull", "--ff-only"],
            check=True,
            capture_output=True,
        )

    def test_creates_parent_dirs(self, tmp_dir: Path):
        local = tmp_dir / "deep" / "nested" / "wiki"
        with patch("subprocess.run"):
            pull_wiki("https://github.com/A/B", local)
        assert local.parent.exists()


# ─── check_and_update_wiki ────────────────────────────────────────────────────

class TestCheckAndUpdateWiki:
    def _source_file(self, tmp_dir: Path, commit_id: str = "") -> Path:
        f = tmp_dir / "wiki_source.json"
        WikiSource(url="https://github.com/A/B", commit_id=commit_id).save(f)
        return f

    def test_no_update_when_same_commit(self, tmp_dir: Path):
        source = self._source_file(tmp_dir, commit_id=SHA)
        retriever = MagicMock()
        with patch(
            "csm_qa.wiki_updater.fetch_latest_commit_id", return_value=SHA
        ):
            result = check_and_update_wiki(
                source_file=source,
                local_dir=tmp_dir / "wiki",
                retriever=retriever,
            )
        assert result is False
        retriever.sync_wiki.assert_not_called()

    def test_update_when_different_commit(self, tmp_dir: Path):
        source = self._source_file(tmp_dir, commit_id="old" + "0" * 37)
        retriever = MagicMock()
        retriever.sync_wiki.return_value = {"updated": 5, "skipped": 0, "removed": 0}

        with (
            patch("csm_qa.wiki_updater.fetch_latest_commit_id", return_value=SHA),
            patch("csm_qa.wiki_updater.pull_wiki") as mock_pull,
        ):
            result = check_and_update_wiki(
                source_file=source,
                local_dir=tmp_dir / "wiki",
                retriever=retriever,
            )

        assert result is True
        mock_pull.assert_called_once()
        retriever.sync_wiki.assert_called_once_with(force=False)

        # commit_id 应已更新到文件
        updated_src = WikiSource.load(source)
        assert updated_src.commit_id == SHA

    def test_force_sync_even_when_same_commit(self, tmp_dir: Path):
        source = self._source_file(tmp_dir, commit_id=SHA)
        retriever = MagicMock()
        retriever.sync_wiki.return_value = {"updated": 3, "skipped": 0, "removed": 0}

        with (
            patch("csm_qa.wiki_updater.fetch_latest_commit_id", return_value=SHA),
            patch("csm_qa.wiki_updater.pull_wiki"),
        ):
            result = check_and_update_wiki(
                source_file=source,
                local_dir=tmp_dir / "wiki",
                retriever=retriever,
                force_sync=True,
            )

        assert result is True
        retriever.sync_wiki.assert_called_once_with(force=True)

    def test_update_when_empty_stored_commit(self, tmp_dir: Path):
        source = self._source_file(tmp_dir, commit_id="")
        retriever = MagicMock()
        retriever.sync_wiki.return_value = {"updated": 10, "skipped": 0, "removed": 0}

        with (
            patch("csm_qa.wiki_updater.fetch_latest_commit_id", return_value=SHA),
            patch("csm_qa.wiki_updater.pull_wiki"),
        ):
            result = check_and_update_wiki(
                source_file=source,
                local_dir=tmp_dir / "wiki",
                retriever=retriever,
            )

        assert result is True
        updated_src = WikiSource.load(source)
        assert updated_src.commit_id == SHA
