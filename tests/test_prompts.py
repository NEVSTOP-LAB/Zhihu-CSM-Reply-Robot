"""Prompt 模板测试。"""

from csm_qa.prompts import (
    CONTEXT_BLOCK_TEMPLATE,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_WIKI_BASE_URL,
    build_system_message,
)


def test_default_system_prompt_mentions_csm():
    assert "CSM" in DEFAULT_SYSTEM_PROMPT
    assert "LabVIEW" in DEFAULT_SYSTEM_PROMPT


def test_default_system_prompt_requires_wiki_links():
    """默认提示词应要求把关键信息写成指向 csm-wiki 的 Markdown 超链接。"""
    assert "Markdown" in DEFAULT_SYSTEM_PROMPT
    assert "链接" in DEFAULT_SYSTEM_PROMPT
    assert "csm-wiki" in DEFAULT_SYSTEM_PROMPT


def test_default_wiki_base_url_points_to_csm_wiki_repo():
    assert DEFAULT_WIKI_BASE_URL.startswith("https://")
    assert "CSM-Wiki" in DEFAULT_WIKI_BASE_URL


def test_build_system_message_with_contexts():
    out = build_system_message(
        DEFAULT_SYSTEM_PROMPT, ["片段A", "片段B"]
    )
    assert out.startswith(DEFAULT_SYSTEM_PROMPT)
    assert "[片段 1]" in out
    assert "片段A" in out
    assert "[片段 2]" in out
    assert "片段B" in out


def test_build_system_message_empty_contexts():
    out = build_system_message(DEFAULT_SYSTEM_PROMPT, [])
    # 没有片段时应显示"（无）"占位，避免模型困惑
    assert "（无）" in out


def test_build_system_message_with_metadata_includes_wiki_link():
    """传入带 source 的 dict 时，应把 source 拼成指向 csm-wiki 的链接。"""
    contexts = [
        {"text": "正文A", "source": "guide/intro.md", "heading": "概述"},
        {"text": "正文B", "source": "api/state.md", "heading": "Untitled"},
    ]
    out = build_system_message(DEFAULT_SYSTEM_PROMPT, contexts)
    assert "正文A" in out and "正文B" in out
    assert "来源: guide/intro.md" in out
    assert "小节: 概述" in out
    # Untitled 不显示
    assert "小节: Untitled" not in out
    # 链接以默认 wiki base url 拼接
    assert f"{DEFAULT_WIKI_BASE_URL}/guide/intro.md" in out
    assert f"{DEFAULT_WIKI_BASE_URL}/api/state.md" in out


def test_build_system_message_with_custom_wiki_base_url():
    contexts = [{"text": "x", "source": "foo.md", "heading": "H"}]
    out = build_system_message(
        DEFAULT_SYSTEM_PROMPT, contexts, wiki_base_url="https://wiki.example.com/docs"
    )
    assert "https://wiki.example.com/docs/foo.md" in out


def test_context_block_template_structure():
    # 模板必须包含 {contexts} 占位符，以便上层注入
    assert "{contexts}" in CONTEXT_BLOCK_TEMPLATE
