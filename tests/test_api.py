"""``CSM_QA`` 主类的端到端测试（mock LLM + fake embedding）。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from csm_qa import AnswerResult, CSM_QA, Message
from csm_qa.types import Usage
from tests.test_rag import FakeEmbedding


@pytest.fixture
def qa(tmp_dir: Path):
    """构造一个 CSM_QA 实例，LLM 与 embedding 全部被替换为 mock。"""
    wiki = tmp_dir / "wiki"
    store = tmp_dir / "store"
    wiki.mkdir()
    (wiki / "csm.md").write_text(
        "# CSM 框架\nCSM 是 Communicable State Machine，状态切换 通过 消息 通信",
        encoding="utf-8",
    )

    with patch("csm_qa.api.LLMClient") as mock_llm_cls, \
            patch("csm_qa.api.EmbeddingFunction", return_value=FakeEmbedding()):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = ("回答内容", Usage(10, 5, 15))
        mock_llm_cls.return_value = mock_llm

        instance = CSM_QA(
            api_key="sk-test",
            wiki_dir=wiki,
            vector_store_dir=store,
            similarity_threshold=0.0,  # 让 fake embedding 也能命中
        )
        instance._mock_llm = mock_llm  # 测试用引用
    return instance


def test_init_requires_api_key():
    with pytest.raises(ValueError):
        CSM_QA(api_key="")


def test_init_resolves_deepseek_defaults(tmp_dir):
    with patch("csm_qa.api.LLMClient") as mock_llm_cls, \
            patch("csm_qa.api.EmbeddingFunction", return_value=FakeEmbedding()):
        mock_llm_cls.return_value = MagicMock()
        qa = CSM_QA(
            api_key="sk",
            wiki_dir=tmp_dir / "wiki",
            vector_store_dir=tmp_dir / "store",
            auto_sync_wiki=False,
        )
        assert qa.provider == "deepseek"
        assert qa.model == "deepseek-chat"
        assert qa.base_url == "https://api.deepseek.com"


def test_openai_compatible_requires_base_and_model(tmp_dir):
    with pytest.raises(ValueError):
        CSM_QA(
            api_key="sk",
            provider="openai_compatible",
            wiki_dir=tmp_dir / "wiki",
            vector_store_dir=tmp_dir / "store",
            auto_sync_wiki=False,
        )


def test_openai_compatible_works_with_overrides(tmp_dir):
    with patch("csm_qa.api.LLMClient") as mock_llm_cls, \
            patch("csm_qa.api.EmbeddingFunction", return_value=FakeEmbedding()):
        mock_llm_cls.return_value = MagicMock()
        qa = CSM_QA(
            api_key="sk",
            provider="openai_compatible",
            base_url="https://api.moonshot.cn/v1",
            model="moonshot-v1-8k",
            wiki_dir=tmp_dir / "wiki",
            vector_store_dir=tmp_dir / "store",
            auto_sync_wiki=False,
        )
        assert qa.base_url == "https://api.moonshot.cn/v1"
        assert qa.model == "moonshot-v1-8k"


def test_ask_returns_string(qa):
    answer = qa.ask("CSM 是什么？")
    assert answer == "回答内容"


def test_ask_detailed_returns_full_result(qa):
    result = qa.ask_detailed("CSM 状态机 切换")
    assert isinstance(result, AnswerResult)
    assert result.answer == "回答内容"
    assert result.usage.total_tokens == 15
    assert result.model == "deepseek-chat"
    # system + user 至少 2 条
    assert len(result.prompt_messages) >= 2
    assert result.prompt_messages[0]["role"] == "system"
    assert result.prompt_messages[-1]["role"] == "user"
    assert result.prompt_messages[-1]["content"] == "CSM 状态机 切换"


def test_ask_includes_history_in_messages(qa):
    history = [
        Message(role="user", content="先前问题"),
        Message(role="assistant", content="先前回答"),
    ]
    qa.ask("追问问题", history=history)
    sent = qa._mock_llm.chat.call_args.args[0]
    roles = [m["role"] for m in sent]
    contents = [m["content"] for m in sent]
    # system, user(history), assistant(history), user(current)
    assert roles == ["system", "user", "assistant", "user"]
    assert "先前问题" in contents
    assert "先前回答" in contents
    assert "追问问题" in contents


def test_ask_accepts_dict_history(qa):
    history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    qa.ask("z", history=history)
    sent = qa._mock_llm.chat.call_args.args[0]
    assert [m["content"] for m in sent[1:]] == ["x", "y", "z"]


def test_ask_rejects_invalid_history(qa):
    with pytest.raises(ValueError):
        qa.ask("z", history=[{"role": "robot", "content": "x"}])
    with pytest.raises(TypeError):
        qa.ask("z", history=[123])


def test_ask_empty_question_raises(qa):
    with pytest.raises(ValueError):
        qa.ask("")
    with pytest.raises(ValueError):
        qa.ask("   ")


def test_ask_uses_default_system_prompt(qa):
    qa.ask("hi")
    sent = qa._mock_llm.chat.call_args.args[0]
    assert sent[0]["role"] == "system"
    assert "CSM" in sent[0]["content"]
    assert "LabVIEW" in sent[0]["content"]


def test_custom_system_prompt_overrides_default(tmp_dir):
    with patch("csm_qa.api.LLMClient") as mock_llm_cls, \
            patch("csm_qa.api.EmbeddingFunction", return_value=FakeEmbedding()):
        mock_llm = MagicMock()
        mock_llm.chat.return_value = ("ok", Usage())
        mock_llm_cls.return_value = mock_llm

        qa = CSM_QA(
            api_key="sk",
            wiki_dir=tmp_dir / "wiki",
            vector_store_dir=tmp_dir / "store",
            system_prompt="You are a pirate.",
            auto_sync_wiki=False,
        )
        qa.ask("ahoy")
        sys_msg = mock_llm.chat.call_args.args[0][0]["content"]
        assert sys_msg.startswith("You are a pirate.")
        # 默认 prompt 中独有的中文规则不应出现
        assert "回答原则" not in sys_msg
        assert "LabVIEW" not in sys_msg


def test_ask_passes_rag_contexts_into_system(qa):
    result = qa.ask_detailed("CSM 状态机 切换")
    sys_msg = result.prompt_messages[0]["content"]
    # contexts 应被塞到 system message 中
    if result.contexts:
        assert any(c[:10] in sys_msg for c in result.contexts)


def test_from_env_reads_legacy_env_vars(monkeypatch, tmp_dir):
    monkeypatch.setenv("LLM_API_KEY", "sk-from-env")
    monkeypatch.delenv("CSM_QA_API_KEY", raising=False)
    monkeypatch.delenv("CSM_QA_PROVIDER", raising=False)

    with patch("csm_qa.api.LLMClient") as mock_llm_cls, \
            patch("csm_qa.api.EmbeddingFunction", return_value=FakeEmbedding()):
        mock_llm_cls.return_value = MagicMock()
        qa = CSM_QA.from_env(
            wiki_dir=tmp_dir / "wiki",
            vector_store_dir=tmp_dir / "store",
            auto_sync_wiki=False,
        )
        assert qa.provider == "deepseek"
        # LLMClient 收到了正确的 api_key
        kwargs = mock_llm_cls.call_args.kwargs
        assert kwargs["api_key"] == "sk-from-env"


def test_sync_wiki_proxies_to_retriever(qa):
    res = qa.sync_wiki(force=True)
    # 至少包含三个键
    assert set(res.keys()) >= {"updated", "skipped", "removed"}


# ─── auto_sync_wiki: remote 模式 ──────────────────────────────────────────────

def _write_wiki_source(parent: Path, url: str = "https://github.com/A/B") -> Path:
    """在 parent 目录写 wiki_source.json，返回其路径。"""
    f = parent / "wiki_source.json"
    f.write_text(
        json.dumps({"url": url, "commit_id": ""}),
        encoding="utf-8",
    )
    return f


def test_auto_sync_wiki_triggers_remote_when_dir_missing(tmp_dir):
    """wiki 目录不存在 + wiki_source.json 存在 → 调用 check_and_update_wiki。"""
    wiki_parent = tmp_dir / "csm-wiki"
    wiki_parent.mkdir()
    _write_wiki_source(wiki_parent)
    wiki_dir = wiki_parent / "remote"  # 故意不创建

    with (
        patch("csm_qa.api.LLMClient") as mock_llm_cls,
        patch("csm_qa.api.EmbeddingFunction", return_value=FakeEmbedding()),
        patch("csm_qa.api.check_and_update_wiki") as mock_remote,
    ):
        mock_llm_cls.return_value = MagicMock()
        mock_remote.return_value = True

        CSM_QA(
            api_key="sk-test",
            wiki_dir=wiki_dir,
            vector_store_dir=tmp_dir / "store",
            auto_sync_wiki=True,
        )

    mock_remote.assert_called_once()
    _, call_kwargs = mock_remote.call_args
    assert call_kwargs["local_dir"] == wiki_dir


def test_auto_sync_wiki_uses_regular_sync_when_dir_exists(tmp_dir):
    """wiki 目录已存在 → 使用普通 sync_wiki()，不调用 check_and_update_wiki。"""
    wiki_dir = tmp_dir / "wiki"
    wiki_dir.mkdir()
    (wiki_dir / "a.md").write_text("# test", encoding="utf-8")

    with (
        patch("csm_qa.api.LLMClient") as mock_llm_cls,
        patch("csm_qa.api.EmbeddingFunction", return_value=FakeEmbedding()),
        patch("csm_qa.api.check_and_update_wiki") as mock_remote,
    ):
        mock_llm_cls.return_value = MagicMock()

        CSM_QA(
            api_key="sk-test",
            wiki_dir=wiki_dir,
            vector_store_dir=tmp_dir / "store",
            auto_sync_wiki=True,
        )

    mock_remote.assert_not_called()


def test_auto_sync_wiki_uses_regular_sync_when_no_source_json(tmp_dir):
    """wiki 目录不存在且无 wiki_source.json → 使用普通 sync_wiki()（静默无操作）。"""
    wiki_dir = tmp_dir / "csm-wiki" / "remote"  # 不创建，父目录也无 source json

    with (
        patch("csm_qa.api.LLMClient") as mock_llm_cls,
        patch("csm_qa.api.EmbeddingFunction", return_value=FakeEmbedding()),
        patch("csm_qa.api.check_and_update_wiki") as mock_remote,
    ):
        mock_llm_cls.return_value = MagicMock()

        CSM_QA(
            api_key="sk-test",
            wiki_dir=wiki_dir,
            vector_store_dir=tmp_dir / "store",
            auto_sync_wiki=True,
        )

    mock_remote.assert_not_called()


def test_auto_sync_wiki_remote_exception_is_swallowed(tmp_dir):
    """远程同步失败时不抛出，仅记录 warning，初始化正常完成。"""
    wiki_parent = tmp_dir / "csm-wiki"
    wiki_parent.mkdir()
    _write_wiki_source(wiki_parent)
    wiki_dir = wiki_parent / "remote"

    with (
        patch("csm_qa.api.LLMClient") as mock_llm_cls,
        patch("csm_qa.api.EmbeddingFunction", return_value=FakeEmbedding()),
        patch(
            "csm_qa.api.check_and_update_wiki",
            side_effect=RuntimeError("network error"),
        ),
    ):
        mock_llm_cls.return_value = MagicMock()

        # 不应抛出异常
        qa = CSM_QA(
            api_key="sk-test",
            wiki_dir=wiki_dir,
            vector_store_dir=tmp_dir / "store",
            auto_sync_wiki=True,
        )
    assert qa is not None
