# Webhook Relay（实时触发组织级 Discussion 回答）

> 让组织级（`/orgs/{org}/discussions/`）Discussion 创建后能**实时**触发 CSM-QA-Bot，
> 而不必等待 30 分钟一次的定时扫描。

## 整体架构

```
组织 Discussion 创建
    │
    ▼
GitHub App Webhook  (event: discussion.created)
    │  (HTTPS POST, X-Hub-Signature-256)
    ▼
Cloudflare Worker  (cloudflare-worker.js)
    │  1. 验签
    │  2. 用 App 私钥签 JWT
    │  3. 拿 installation access token
    │  4. POST /repos/<owner>/<repo>/dispatches
    │       event_type: "org_discussion_created"
    │       client_payload: { discussion_number: <N> }
    ▼
GitHub Actions  (csm-discussion-bot.yml on: repository_dispatch)
    │
    ▼
scripts/discussion_bot.py --org-discussion-number <N>
```

定时扫描 (`schedule: */30 * * * *`) 仍保留作兜底，确保 webhook 偶发漏发时不会丢消息。

---

## 一次性预备工作（你需要做）

### 1. 创建 GitHub App

1. 打开 `https://github.com/organizations/<你的组织>/settings/apps/new`
2. 配置：
   - **App name**：`CSM-QA-Bot-Webhook-Relay`（任意）
   - **Homepage URL**：本仓库 URL
   - **Webhook URL**：先留空或随便填，部署完 Worker 后再回来更新
   - **Webhook secret**：用 `openssl rand -hex 32` 生成强随机字符串，**记下来**
3. **Repository permissions**：
   - **Contents**：`Read-only`（用于触发 dispatches API）
   - 或 **Metadata**: `Read-only`（必选，所有 App 默认包含）
4. **Organization permissions**：
   - **Discussions**：`Read & write`
5. **Subscribe to events** 勾选：`Discussion`
6. 创建后：
   - 记下 **App ID**（在 App 设置页顶部的数字）
   - 滚动到 "Private keys" 区，点 **Generate a private key**，下载 `.pem` 文件

### 2. 安装 App 到组织

在 App 设置页 → 左侧 **Install App** → 选择目标组织 → 选择 "All repositories"
或仅勾选 `CSM-QA-Robot`。

### 3. 把私钥转成 PKCS#8

GitHub 下发的私钥是 PKCS#1（`-----BEGIN RSA PRIVATE KEY-----`），
而 Cloudflare Worker 的 Web Crypto API 只接受 PKCS#8：

```bash
openssl pkcs8 -topk8 -nocrypt \
  -in csm-qa-bot-webhook-relay.<date>.private-key.pem \
  -out app.pkcs8.pem
```

转换后开头应是 `-----BEGIN PRIVATE KEY-----`。

### 4. 部署 Cloudflare Worker

```bash
cd webhook
npm install -g wrangler          # 已装可跳过
wrangler login                   # 浏览器登录 Cloudflare 账号

# 注入三个 secret（命令会进入交互式输入）
wrangler secret put WEBHOOK_SECRET           # 粘贴步骤 1 生成的 webhook secret
wrangler secret put GITHUB_APP_ID            # 粘贴步骤 1 的 App ID
wrangler secret put GITHUB_APP_PRIVATE_KEY   # 粘贴 app.pkcs8.pem 全文（含 BEGIN/END 行）

wrangler deploy
```

部署完成后 wrangler 会输出 Worker URL，类似
`https://csm-qa-bot-webhook-relay.<你的子域>.workers.dev`。

### 5. 把 Worker URL 填回 GitHub App

回到 App 设置页 → **Webhook URL** 改为上一步拿到的 Worker URL → 保存。
然后在 App 设置页 → **Advanced** → **Recent Deliveries** 里点
"Redeliver" 触发一次 ping，看 Worker 是否返回 `200 pong`，并通过响应体确认验签通过。

### 6. （已完成）确认仓库 Secrets

`csm-discussion-bot.yml` 仍使用现有的两项 secrets，无需为 webhook 新增：

| Secret | 用途 |
| --- | --- |
| `CSM_QA_GH_TOKEN` | GitHub Fine-grained PAT，用于 GraphQL 读写 Discussions 以及 Actions checkout 拉取 submodule。**必须同时授权两个仓库**：`<org>/CSM-QA-Robot`（自身）和 `<org>/.github`（org-level Q&A discussions 的实际归属仓库），权限均需 `Discussions: Read & Write`（自身仓库还需 `Contents: Read`）。如组织级 discussions 实际归属其他仓库，请同步设置 workflow 的 `DISCUSSION_SOURCE_REPO` 环境变量。 |
| `LLM_API_KEY` | DeepSeek / 其他 LLM key |

> Worker 的三个 secret（`WEBHOOK_SECRET` / `GITHUB_APP_ID` / `GITHUB_APP_PRIVATE_KEY`）
> 是 **Cloudflare Worker** 的 secret，不是仓库 Actions secret。

---

## 端到端验证

1. 在组织级 Q&A 分类下随便发一条新 Discussion。
2. **Cloudflare Dashboard → Workers → csm-qa-bot-webhook-relay → Logs**：
   应能看到 `202 {"ok":true,"discussion_number":N}`。
3. **GitHub App → Advanced → Recent Deliveries**：应显示 ✅ 200。
4. **本仓库 → Actions → CSM Q&A Discussion Bot**：应在 ~10 秒内出现一次
   `repository_dispatch` 触发的运行。
5. 等待几分钟后回到 Discussion 页面，应看到 Bot 评论。

---

## 故障排查

| 现象 | 原因 / 处理 |
| --- | --- |
| Worker 返回 401 `Invalid signature` | `WEBHOOK_SECRET` 与 GitHub App 中配置不一致 |
| Worker 返回 500 `App auth failed` | 私钥未转 PKCS#8，或 `GITHUB_APP_ID` 错误，或 App 未安装到该 repo |
| Worker 返回 502 `Dispatch failed: HTTP 404` | App 安装时未授权访问该 repo / Contents 权限不足 |
| Worker 返回 200 `Ignored event` | 收到了非 `discussion` 事件（正常，过滤掉了） |
| Actions 中触发了但 step 报 `client_payload.discussion_number 为空` | Worker 端 payload 解析失败，检查 Worker 日志 |
| Discussion 已创建但 30 分钟内无回复 | Webhook 链路异常，定时扫描会兜底处理 |
