# GitHub Actions 自动化调研

## 1. 定时触发（Cron）

```yaml
on:
  schedule:
    - cron: '0 */6 * * *'   # 每 6 小时检查一次（UTC）
  workflow_dispatch:          # 支持手动触发调试
```

- 最高频率：每 5 分钟一次（本项目无需如此高频）
- 时区：固定 UTC，需自行换算
- Free 账号 Public repo：无限制；Private repo：2000 分钟/月
- **注意**：低活跃仓库的 schedule 可能延迟触发，加 `workflow_dispatch` 保底

参考：[Scheduled Cron Jobs in GitHub Actions](https://dylanbritz.dev/writing/scheduled-cron-jobs-github/)

## 2. 典型 Workflow 结构

```yaml
name: Zhihu Reply Bot

on:
  schedule:
    - cron: '0 2,8,14,20 * * *'
  workflow_dispatch:

jobs:
  check-and-reply:
    runs-on: ubuntu-latest
    permissions:
      contents: write   # 需要写权限以推送存档

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Cache pip
        uses: actions/cache@v4
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}

      - name: Install deps
        run: pip install -r requirements.txt

      - name: Run bot
        env:
          ZHIHU_COOKIE: ${{ secrets.ZHIHU_COOKIE }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          OPENAI_BASE_URL: ${{ secrets.OPENAI_BASE_URL }}
        run: python scripts/run_bot.py

      - name: Commit archive
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data/ archive/
          git diff --cached --quiet || git commit -m "bot: update reply archive [skip ci]"
          git push
```

## 3. Secrets 管理

在 **Settings → Secrets and variables → Actions** 中添加：

| Secret 名 | 内容 |
|-----------|------|
| `ZHIHU_COOKIE` | 完整的知乎登录 Cookie 字符串（z_c0=...; _xsrf=...） |
| `OPENAI_API_KEY` | GPT-like API 的 Key |
| `OPENAI_BASE_URL` | 自定义 endpoint（如非 OpenAI 官方则必填） |

> Secrets 在日志中自动被 mask，不会泄露。

## 4. 状态持久化策略

GitHub Actions 每次 job 是全新容器，状态需持久化到仓库文件或外部存储：

| 数据 | 存储方式 |
|------|----------|
| 已处理评论 ID | `data/seen_ids.json` → git commit |
| 向量索引（FAISS/Chroma） | `data/vector_store/` → git commit（小型）或 Actions Cache |
| 回复存档 | `archive/YYYY/MM/` Markdown → git commit |
| CSM Wiki 快照哈希 | `data/wiki_hash.json` → git commit |

**[skip ci]** 标签防止存档 commit 再次触发 workflow 死循环。

## 5. 调试技巧

- 使用 `workflow_dispatch` 手动触发测试
- 用 `act`（本地 Actions 模拟器）在本地调试：[https://github.com/nektos/act](https://github.com/nektos/act)
- 在脚本中保存详细日志，通过 Actions 界面查看输出

## 6. 参考资源

- [swyxio/gh-action-data-scraping](https://github.com/swyxio/gh-action-data-scraping) — 数据抓取存入仓库的范例
- [GitHub Actions 官方文档 - Events that trigger workflows](https://docs.github.com/en/actions/writing-workflows/choosing-when-your-workflow-runs/events-that-trigger-workflows)
- [Scheduled Scraping with GitHub Actions](https://tds.s-anand.net/2025-05/scheduled-scraping-with-github-actions/)
