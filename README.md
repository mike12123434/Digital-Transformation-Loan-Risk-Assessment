# 貸款風險評估系統 — 部署步驟

## 整體架構（全部免費）

```
使用者瀏覽器
    ↓ 上傳 PDF
Streamlit Community Cloud  ←  你的公開連結在這裡
    ↓ API call (~2–4秒)
Groq Cloud（免費，14,400次/天）
    ↓
llama-3.1-8b-instant 模型
```

---

## Step 1 — 取得免費 Groq API Key（2分鐘）

1. 前往 https://console.groq.com
2. Sign up（Google 帳號即可）
3. 左側 **API Keys** → **Create API Key**
4. 複製 `gsk_xxxxx...` 備用

---

## Step 2 — 上傳到 GitHub（3分鐘）

上傳這四個檔案到一個新的 GitHub repo（public 或 private 都可）：

```
你的-repo/
├── app.py
├── requirements.txt
├── .gitignore
└── .streamlit/
    └── config.toml
```

---

## Step 3 — Streamlit Community Cloud 部署（3分鐘）

1. 前往 https://share.streamlit.io → **Sign in with GitHub**
2. **New app** → 選你的 repo → Main file: `app.py`
3. **Deploy**（等約 90 秒自動安裝套件）
4. 取得公開連結：`https://xxx.streamlit.app`

---

## 使用方式

1. 開啟連結
2. 在左側輸入 Groq API Key（`gsk_...`）
3. 上傳 PDF
4. 點「開始評估」→ 2–4 秒出結果

API Key 只存在 session 中，不會被記錄或儲存。

---

## 免費限制

| 項目 | 限制 |
|------|------|
| Groq API | 30次/分鐘，14,400次/天，完全免費 |
| Streamlit Cloud | 無用量限制，免費公開 app |
| PDF 大小 | 最大 50MB |

每次評估 = 1次 API 呼叫，14,400次/天對一般審核量綽綽有餘。
