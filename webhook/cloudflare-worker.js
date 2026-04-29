/**
 * CSM-QA-Robot Webhook Relay (Cloudflare Worker)
 *
 * 工作流程
 * ────────
 * 1. 接收 GitHub App 推送的 webhook（事件订阅 `discussion`）。
 * 2. 校验请求头 `X-Hub-Signature-256`（HMAC-SHA256，密钥 = WEBHOOK_SECRET）。
 * 3. 仅处理 `action == "created"` 的 discussion 事件，提取 `discussion.number`。
 * 4. 用 GitHub App 的 App ID + 私钥（PKCS#8 PEM）签发 JWT
 *    → 调 `/repos/{owner}/{repo}/installation` 拿到 installation_id
 *    → 换取 installation access token。
 * 5. 用 installation token 调
 *    POST /repos/{REPO_OWNER}/{REPO_NAME}/dispatches
 *    {
 *       event_type: "org_discussion_created",
 *       client_payload: { discussion_number: <N>, source: "webhook" }
 *    }
 *
 * 必需的 Worker 环境变量 / Secrets
 * ────────────────────────────────
 *   WEBHOOK_SECRET           – GitHub App 配置的 Webhook secret（强随机字符串）
 *   GITHUB_APP_ID            – GitHub App 的数字 App ID
 *   GITHUB_APP_PRIVATE_KEY   – GitHub App 私钥（PKCS#8 PEM 全文，含 BEGIN/END 行）
 *   REPO_OWNER               – 目标仓库 owner（默认 "NEVSTOP-LAB"）
 *   REPO_NAME                – 目标仓库名（默认 "CSM-QA-Robot"）
 *
 * 注意：GitHub App 默认下发的私钥是 PKCS#1（BEGIN RSA PRIVATE KEY）。
 *       Web Crypto API 仅接受 PKCS#8。请用以下命令转换后再粘贴：
 *           openssl pkcs8 -topk8 -nocrypt -in app.pem -out app.pkcs8.pem
 */

const GITHUB_API = "https://api.github.com";
const USER_AGENT = "csm-qa-bot-webhook-relay/1.0";

export default {
  async fetch(request, env, _ctx) {
    if (request.method !== "POST") {
      return new Response("Method Not Allowed", { status: 405 });
    }

    // 1. 读取 raw body 字节（验签必须对原始字节做 HMAC，避免任何
    //    text decode → encode 来回过程引入编码/规范化差异导致误判）
    const rawBodyBuf = await request.arrayBuffer();

    // 2. 验签（直接传原始字节）
    const signature = request.headers.get("x-hub-signature-256") || "";
    const ok = await verifySignature(env.WEBHOOK_SECRET, rawBodyBuf, signature);
    if (!ok) {
      return new Response("Invalid signature", { status: 401 });
    }

    // 3. 仅处理 discussion 事件
    const eventType = request.headers.get("x-github-event") || "";
    if (eventType === "ping") {
      return new Response("pong", { status: 200 });
    }
    if (eventType !== "discussion") {
      return new Response(`Ignored event: ${eventType}`, { status: 200 });
    }

    // 验签通过后再把字节解码为字符串供 JSON.parse 使用
    let payload;
    try {
      const rawBodyText = new TextDecoder("utf-8").decode(rawBodyBuf);
      payload = JSON.parse(rawBodyText);
    } catch (_e) {
      return new Response("Invalid JSON", { status: 400 });
    }

    if (payload.action !== "created") {
      return new Response(`Ignored action: ${payload.action}`, { status: 200 });
    }

    const discussionNumber = payload?.discussion?.number;
    if (!discussionNumber) {
      return new Response("Missing discussion.number", { status: 400 });
    }

    // 4. 取 installation token
    const owner = env.REPO_OWNER || "NEVSTOP-LAB";
    const repo = env.REPO_NAME || "CSM-QA-Robot";

    let installationToken;
    try {
      installationToken = await getInstallationToken(env, owner, repo);
    } catch (e) {
      return new Response(`App auth failed: ${e.message}`, { status: 500 });
    }

    // 5. 触发 repository_dispatch
    const dispatchResp = await fetch(
      `${GITHUB_API}/repos/${owner}/${repo}/dispatches`,
      {
        method: "POST",
        headers: {
          Authorization: `token ${installationToken}`,
          Accept: "application/vnd.github+json",
          "User-Agent": USER_AGENT,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          event_type: "org_discussion_created",
          client_payload: {
            discussion_number: discussionNumber,
            source: "webhook",
          },
        }),
      },
    );

    if (!dispatchResp.ok) {
      const text = await dispatchResp.text();
      return new Response(
        `Dispatch failed: HTTP ${dispatchResp.status} ${text.slice(0, 400)}`,
        { status: 502 },
      );
    }

    return new Response(
      JSON.stringify({ ok: true, discussion_number: discussionNumber }),
      { status: 202, headers: { "Content-Type": "application/json" } },
    );
  },
};

// ── 辅助函数 ──────────────────────────────────────────────────────────────────

/** 用 HMAC-SHA256 校验 GitHub webhook 签名，常量时间比较。
 *  rawBody 必须是 ArrayBuffer（直接对原始字节计算，避免编码差异）。 */
async function verifySignature(secret, rawBody, signatureHeader) {
  if (!secret || !signatureHeader.startsWith("sha256=")) return false;
  const expected = signatureHeader.slice("sha256=".length);

  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sigBuf = await crypto.subtle.sign("HMAC", key, rawBody);
  const actual = bufToHex(sigBuf);

  return timingSafeEqual(expected, actual);
}

function timingSafeEqual(a, b) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return diff === 0;
}

function bufToHex(buf) {
  const bytes = new Uint8Array(buf);
  let s = "";
  for (let i = 0; i < bytes.length; i++) {
    s += bytes[i].toString(16).padStart(2, "0");
  }
  return s;
}

/** 用 App 私钥签 JWT（RS256，9 分钟有效期）。 */
async function signAppJwt(appId, privateKeyPem) {
  const now = Math.floor(Date.now() / 1000);
  const header = { alg: "RS256", typ: "JWT" };
  // iat 回退 60s 以容忍时钟漂移（GitHub 推荐做法）
  const payload = { iat: now - 60, exp: now + 9 * 60, iss: String(appId) };

  const encode = (obj) =>
    base64UrlEncode(new TextEncoder().encode(JSON.stringify(obj)));
  const signingInput = `${encode(header)}.${encode(payload)}`;

  const key = await importPkcs8PrivateKey(privateKeyPem);
  const sigBuf = await crypto.subtle.sign(
    { name: "RSASSA-PKCS1-v1_5" },
    key,
    new TextEncoder().encode(signingInput),
  );
  return `${signingInput}.${base64UrlEncode(new Uint8Array(sigBuf))}`;
}

function base64UrlEncode(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function importPkcs8PrivateKey(pem) {
  const cleaned = pem
    .replace(/-----BEGIN [^-]+-----/g, "")
    .replace(/-----END [^-]+-----/g, "")
    .replace(/\s+/g, "");
  const der = Uint8Array.from(atob(cleaned), (c) => c.charCodeAt(0));
  return crypto.subtle.importKey(
    "pkcs8",
    der.buffer,
    { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" },
    false,
    ["sign"],
  );
}

/** 用 App JWT 拿到目标 repo 的 installation token。 */
async function getInstallationToken(env, owner, repo) {
  const jwt = await signAppJwt(env.GITHUB_APP_ID, env.GITHUB_APP_PRIVATE_KEY);

  // 查询此 repo 对应的 installation
  const instResp = await fetch(
    `${GITHUB_API}/repos/${owner}/${repo}/installation`,
    {
      headers: {
        Authorization: `Bearer ${jwt}`,
        Accept: "application/vnd.github+json",
        "User-Agent": USER_AGENT,
      },
    },
  );
  if (!instResp.ok) {
    const text = await instResp.text();
    throw new Error(
      `get installation HTTP ${instResp.status}: ${text.slice(0, 200)}`,
    );
  }
  const inst = await instResp.json();

  // 用 JWT 换 installation token
  const tokResp = await fetch(
    `${GITHUB_API}/app/installations/${inst.id}/access_tokens`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${jwt}`,
        Accept: "application/vnd.github+json",
        "User-Agent": USER_AGENT,
      },
    },
  );
  if (!tokResp.ok) {
    const text = await tokResp.text();
    throw new Error(
      `access_tokens HTTP ${tokResp.status}: ${text.slice(0, 200)}`,
    );
  }
  const tok = await tokResp.json();
  return tok.token;
}
