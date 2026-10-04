# Kaggle Account & API Setup Guide

Kaggleアカウントを作成し、Kaggle CLI、kagglehub、MCP、headless agentで使うcredentialを安全に設定するための正本。この内容を他のmoduleへ複製せず、client固有の差分だけを各moduleに記載する。

## 1. Create a Kaggle Account

1. Go to [https://www.kaggle.com/account/login](https://www.kaggle.com/account/login)
2. Click **Register** (or sign in with Google/GitHub if you prefer)
3. Fill in:
   - **Email**: your email address
   - **Password**: choose a strong password
   - **Username**: choose a username (this becomes your Kaggle handle, e.g., `yourname`)
4. Click **Create Account**
5. Verify your email by clicking the link Kaggle sends you

### Account Verification

verificationの要否、方式、対象機能、UIラベルは変更され得る。対象competitionの
rulesと[account settings](https://www.kaggle.com/settings)の現在の案内を確認する。
本人確認情報をエージェントへ渡さず、ユーザー自身がKaggle UIで完了する。

## 2. Generate Your API Credentials

| Client | 利用できる方式 |
|--------|----------------|
| Kaggle CLI / Kaggle Python API (`kaggle==2.2.4`) | OAuth、API token、legacy username/key |
| kagglehub (`kagglehub==1.0.2`) | API token、legacy username/key（CLIのOAuthファイルは読まない） |
| Kaggle MCP Server | API token |

### OAuth Login for Kaggle CLI and Python API

For local CLI or Kaggle Python API use, first log in with a browser:

```bash
uv run kaggle auth login
```

Use `uv run kaggle auth login --no-launch-browser` if the CLI cannot open a browser automatically. This creates `~/.kaggle/credentials.json`. Kaggle CLIとKaggle Python APIの`KaggleApi.authenticate()`は同じOAuthファイルを読み、必要に応じてtokenを更新する。kagglehubはこのファイルを読み込まない。

Do not log or share output from `kaggle auth print-access-token`.

### API Token

| Credential | Variable | How to Get |
|-----------|----------|------------|
| API Token | `KAGGLE_API_TOKEN` | "Generate New Token" button under "API Tokens (Recommended)" |

1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
2. Scroll to the **API** section
3. Under **API Tokens (Recommended)**, click **Generate New Token**
4. Name the token (e.g., "claude-code") and copy the generated value
5. This single token works with kaggle CLI (>= 1.8.0), kagglehub (>= 0.4.1), and MCP Server

**Note:** Creating a new token does not expire existing tokens or legacy keys. You can create multiple named tokens for different tools/projects.

### Legacy API Key

| Credential | Variables | How to Get |
|-----------|-----------|------------|
| Legacy Key | `KAGGLE_USERNAME` + `KAGGLE_KEY` | "Create Legacy API Key" under "Legacy API Credentials" |

Kaggle CLI、Kaggle Python API、kagglehubでは、API tokenの代わりにlegacy username/keyも使用できる。MCPのBearer認証には使用できない。

1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
2. Under **Legacy API Credentials**, click **Create Legacy API Key**
3. A `kaggle.json` file downloads containing `{"username":"...","key":"..."}`

**Warning:** Creating a legacy key expires any existing legacy keys.

## 3. Install Your Credentials

### Method 1: OAuth Login (Local CLI / Kaggle Python API)

```bash
uv run kaggle auth login
chmod 600 ~/.kaggle/credentials.json
```

### Method 2: Access Token File (Recommended for Agents/CI)

Run the repository helper from a local interactive terminal. Token input is hidden and is never passed as a command argument:

```bash
uv run python .agents/skills/kaggle-platform/modules/registration/scripts/configure_token.py
```

### Method 3: Environment Variable

Use the secret-variable facility provided by CI, Colab, or the managed execution environment. Do not put token values in command arguments or shell profiles copied into logs.

### Method 4: kaggle.json File (Legacy)

If you created a legacy API key, place the downloaded `kaggle.json`:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Note: `kaggle.json` only stores username + legacy key. For the API token, use Methods 2-3.

このリポジトリはproject-levelまたはhome-levelの`.env`を自動読込しない。環境変数を使う場合は実行環境のsecret storeやprocess environmentへ明示的に設定する。

## 4. Verify Your Setup

### Using the Registration Checker

```bash
uv run python .agents/skills/kaggle-platform/shared/check_all_credentials.py --require python-api
# kagglehubを使う場合は --require kagglehub、MCPは --require api-token
```

checkerはローカルの設定を調べるだけで、認証・期限・失効・利用権限・通信の確認はしない。OAuthは`credentials.json`内の非空の`refresh_token`等の形式を確認し、ファイルが存在するだけでは成功扱いにしない。Kaggle/Colab実行環境の専用secretやclientへ直接渡した設定は、このローカルcheckerの対象外。

Expected output when credentials are configured:
```
[OK] API Token: *****abcd (from env)
[OK] OAuth credentials: /home/user/.kaggle/credentials.json (from kaggle auth login)
[OK] KAGGLE_USERNAME: your_username (from kaggle.json)
[OK] KAGGLE_KEY: ****wxyz (Legacy API key, from kaggle.json)

API token found — you're ready to go!
```

### Manual Verification

```bash
# Test with kaggle CLI
uv run kaggle datasets list --search "titanic" --page-size 1

# Test with kagglehub
uv run python -c "import kagglehub; print(kagglehub.whoami())"
```

## 5. Credential Priority Order

`uv.lock`で固定したKaggle 2.2.4の`KaggleApi.authenticate()`（CLIとPython API共通）は、API token、legacy username/key、OAuthの順に試す。kagglehub 1.0.2の`config.get_kaggle_credentials()`は別実装であり、OAuthファイルを読み込まない。

| Priority | Source | Used By |
|----------|--------|----------|
| 1 | `KAGGLE_API_TOKEN` env var（token文字列または既存tokenファイルのパス） | CLI, Kaggle Python API, kagglehub, MCP |
| 2 | `~/.kaggle/access_token` file、空または不在なら`access_token.txt` | CLI, Kaggle Python API, kagglehub, MCP |
| 3 | `KAGGLE_USERNAME` + `KAGGLE_KEY` env vars | CLI, Kaggle Python API, kagglehub (legacy) |
| 4 | `~/.kaggle/kaggle.json` file | CLI, Kaggle Python API, kagglehub (legacy) |
| 5 | `~/.kaggle/credentials.json` from `uv run kaggle auth login` | CLI, Kaggle Python API |

tokenの探索はcheckerと同梱MCP clientも固定版`kagglesdk.kaggle_env.get_access_token_from_env()`を使う。`KAGGLE_API_TOKEN`が既存ファイルを指す場合は内容を読み、そのファイルが空なら別のtokenへfallbackしない。ファイルのパス文字列そのものをtokenとして送らない。

これは標準ローカル保存先の優先順位。CLI/APIはlegacyの環境変数と設定ファイルを組み合わせて読む場合がある。kagglehubではprocess内に明示設定したcredentialが先に使われ、環境変数のlegacy pairは両方が必要。clientの`KAGGLE_CONFIG_DIR`やLinuxのXDG fallback、Kaggle/Colabの専用認証を使う場合はclientの実際の認証結果で確認し、標準保存先だけを調べるcheckerの失敗を未設定と断定しない。

## 6. Common Misconfigurations

| Issue | Fix |
|-------|-----|
| `KAGGLE_TOKEN` set instead of `KAGGLE_API_TOKEN` | Rename to `KAGGLE_API_TOKEN` |
| OAuth credentialしかなくkagglehubを使う | API tokenまたはlegacy username/keyを設定する（Kaggle Python APIならOAuthのまま利用可能） |
| Legacy credentialしかなくMCPを使う | Kaggle SettingsでAPI tokenを生成する |
| API token is not stored locally | Run `uv run python .agents/skills/kaggle-platform/modules/registration/scripts/configure_token.py` in a local interactive terminal |
| Old kaggle CLI (< 1.8.0) doesn't recognize new tokens | Run `uv sync --locked` to restore the repository-pinned CLI, or use a legacy key |
| `kaggle forums` or `competitions topics show` missing | Run `uv sync --locked` to restore the repository-pinned Kaggle CLI v2.2.0+ |
| Old kagglehub (< 0.4.1) doesn't recognize new tokens | Run `uv sync --locked --extra kaggle-platform` to restore the repository-pinned kagglehub, or use a legacy key |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `kaggle: command not found` | Run `uv sync --locked`, then invoke it as `uv run kaggle` |
| `401 Unauthenticated` | Check that credentials exist and are correct |
| `403 Forbidden` on competition | Accept competition rules at kaggle.com |
| `403 Forbidden` on model | Accept model license at kaggle.com |
| `kaggle.json permissions warning` | Run `chmod 600 ~/.kaggle/kaggle.json` |
| `credentials.json permissions warning` | Run `chmod 600 ~/.kaggle/credentials.json` |
| MCP "Unauthenticated" | Use API token (from "Generate New Token") as Bearer token |
| `HTTP 429 Too Many Requests` | Dynamic rate limiting — wait a few minutes and retry |
