"""最简单的单轮问答示例。

运行前：
    pip install -e .
"""

from __future__ import annotations

import logging

from csm_qa import CSM_QA


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    qa = CSM_QA.from_ini("config.ini")  # 自动读取 LLM_API_KEY
    answer = qa.ask("csm 的 log 怎么使用？")
    print(answer)


if __name__ == "__main__":
    main()
