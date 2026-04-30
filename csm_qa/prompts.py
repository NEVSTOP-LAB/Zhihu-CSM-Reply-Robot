"""默认提示词。

固定段在前、动态段（参考资料/历史）在后，便于命中 DeepSeek/OpenAI 的 Prompt Caching。
"""

from __future__ import annotations

from typing import Iterable, Union

# csm-wiki 仓库的默认链接前缀（GitHub blob URL）。
# 用于将「参考资料」中的 source 转换为可点击的链接，回答时引用作为 Markdown 超链接。
DEFAULT_WIKI_BASE_URL = "https://github.com/NEVSTOP-LAB/CSM-Wiki/blob/main"


# 默认 system prompt：CSM/LabVIEW 场景 + RAG。
# 调用方可通过 ``CSM_QA(system_prompt=...)`` 完全覆盖。
DEFAULT_SYSTEM_PROMPT = """\
你是 CSM（Communicable State Machine，通信状态机）框架与 LabVIEW 的资深技术专家。
你的任务是基于「参考资料」深入、准确地回答用户的问题，并把关键信息以 Markdown 超链接的形式指向 csm-wiki 原文，方便用户进一步查阅。

【回答原则】
1. 严格基于「参考资料」回答；若资料不足，明确告知"知识库中暂无相关内容"并给出可行的下一步建议（例如查阅哪类文档），禁止杜撰 API、版本号或参数。
2. 鼓励深入思考与多角度展开：先给出一句话结论，再分点解释原理、使用方式、常见坑与最佳实践；必要时给出最小可运行示例（LabVIEW 描述或伪代码）。在保证准确性的前提下，回答可详尽（约 800–1500 字），用户明确要求"简短"时再压缩。
3. 涉及具体类/VI/方法名时使用反引号包裹，保持英文原名不翻译。
4. 多轮对话时引用历史上下文中已建立的概念，不重复定义；用户追问"为什么/怎么做"时直接给出深入解答而非复述。
5. **关键信息加链接**：当回答中出现「参考资料」里给出过 ``来源`` / ``链接`` 的概念、类名、VI、章节、教程时，**必须**写成 Markdown 超链接 ``[关键词](来源链接)``，链接目标使用对应片段提供的 ``链接`` 字段；同一关键词在同一回答中只需链接首次出现的位置，避免链接堆砌。若片段未提供链接则不要强行造链接。
6. 不输出 Markdown 一级/二级标题（# / ##），可使用列表、代码块、加粗、Markdown 链接。
7. 不讨论政治、宗教、个人隐私、商业承诺等与技术无关的话题；遇到此类问题礼貌拒绝。
8. 输出语言与用户提问保持一致（默认中文）。
"""


# 参考资料拼接到 system 末尾的模板。
# 使用单独的常量便于测试与外部覆盖。
CONTEXT_BLOCK_TEMPLATE = """\

【参考资料】（按相关度排序，可能为空；每个片段附带 ``来源`` 与 ``链接``，回答时请把关键信息写成指向「链接」的 Markdown 超链接）
{contexts}
"""


def _build_wiki_url(source: str, base_url: str) -> str:
    """根据片段 ``source``（相对路径）和 ``base_url`` 拼装 csm-wiki 链接。"""
    if not source or source == "(unknown)":
        return ""
    src = source.lstrip("/")
    base = base_url.rstrip("/")
    return f"{base}/{src}"


def build_system_message(
    system_prompt: str,
    contexts: Iterable[Union[str, dict]],
    wiki_base_url: str = DEFAULT_WIKI_BASE_URL,
) -> str:
    """拼装最终的 system message 内容（固定段 + 参考资料段）。

    Args:
        system_prompt: 角色与规则段（用户可覆盖默认值）。
        contexts: RAG 检索结果。元素可以是纯文本字符串（向后兼容），也可以是
            ``{"text": ..., "source": ..., "heading": ...}`` 字典；后者会
            根据 ``wiki_base_url`` 自动生成可点击的链接。
        wiki_base_url: csm-wiki 链接前缀，用于把片段 ``source`` 拼成 URL。

    Returns:
        完整的 system message 字符串。
    """
    items = list(contexts) if contexts else []
    if items:
        blocks: list[str] = []
        for i, item in enumerate(items):
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                source = str(item.get("source", "")).strip()
                heading = str(item.get("heading", "")).strip()
                url = _build_wiki_url(source, wiki_base_url)
                header_parts = [f"[片段 {i + 1}]"]
                if source:
                    header_parts.append(f"来源: {source}")
                if heading and heading != "Untitled":
                    header_parts.append(f"小节: {heading}")
                if url:
                    header_parts.append(f"链接: {url}")
                blocks.append("\n".join([" | ".join(header_parts), text]))
            else:
                blocks.append(f"[片段 {i + 1}]\n{str(item).strip()}")
        joined = "\n\n---\n\n".join(blocks)
    else:
        joined = "（无）"
    return system_prompt + CONTEXT_BLOCK_TEMPLATE.format(contexts=joined)
