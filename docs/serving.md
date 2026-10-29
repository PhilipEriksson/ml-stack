# Serving

Two backends, one choice to make:

| | vLLM (Docker) | llama.cpp (`ml serve-model`) |
|---|---|---|
| Input | HF checkpoint directory or repo id | a single `.gguf` file |
| Best for | throughput, batching, long context, HF-native features | single-file models, quick local chat, low setup |
| Config | env file in `configs/vllm/` | CLI flags |
| GPU sharing | reserves VRAM up front | loads layers on demand |

Both expose an OpenAI-compatible API, so the proxy, Open WebUI, `claude-local`, and `ml eval` work against either.

---

## vLLM service

Based on `vllm/vllm-openai:v0.20.1`. All model settings come from environment variables — swapping models does not require a rebuild.

**How it works**

1. `services/vllm/start.sh` reads the env file and builds the `vllm serve` command.
2. `HF_HOME=/hf-cache` is set and the host HF cache is mounted there.
3. `MODEL_NAME` is resolved by vLLM's normal HF resolution, or read from `/opt/models/...` for local directories.

**Selecting a config**

```bash
cp configs/vllm/example.env configs/vllm/my-model.env   # edit MODEL_NAME
ml set-vllm-env my-model.env                            # → configs/vllm/active.env
docker compose -f services/docker/docker-compose.yml up -d vllm
```

`configs/vllm/active.env` is gitignored local state; `docker-compose.yml` always reads it.

**Variables** (`services/vllm/start.sh` maps each to a flag; unset variables are omitted)

| Variable | Flag | Notes |
|---|---|---|
| `MODEL_NAME` | positional | HF repo id or container path — required |
| `MAX_MODEL_LEN` | `--max-model-len` | context window |
| `GPU_MEMORY_UTILIZATION` | `--gpu-memory-utilization` | 0.0–1.0 |
| `MAX_NUM_SEQS` | `--max-num-seqs` | concurrent sequences |
| `ATTENTION_BACKEND` | `--attention-backend` | e.g. `flashinfer` |
| `KV_CACHE_DTYPE` | `--kv-cache-dtype` | e.g. `fp8_e4m3` |
| `QUANTIZATION` | `--quantization` | usually auto-detected from the checkpoint |
| `REASONING_PARSER` | `--reasoning-parser` | parser name is model-family specific |
| `TOOL_CALL_PARSER` | `--tool-call-parser` | same |
| `ENABLE_AUTO_TOOL_CHOICE` | `--enable-auto-tool-choice` | `true` / omitted |
| `ENABLE_PREFIX_CACHING` | `--enable-prefix-caching` | `true` / omitted |
| `ENABLE_CHUNKED_PREFILL` | `--enable-chunked-prefill` | `true` / omitted |
| `PERFORMANCE_MODE` | `--performance-mode` | e.g. `interactivity`, `throughput` |
| `SPECULATIVE_CONFIG` | `--speculative-config` | JSON; method depends on the checkpoint |
| `LANGUAGE_MODEL_ONLY` | `--language-model-only` | `true` / omitted |
| `SKIP_MM_PROFILING` | `--skip-mm-profiling` | `true` / omitted |
| `DTYPE` / `BLOCK_SIZE` / `DOWNLOAD_DIR` | matching flags | rarely needed |
| `HOST` / `PORT` | `--host` / `--port` | container bind address |

**Local model directories**

The host `models/` directory is mounted read-only at `/opt/models`, so a registered model can be served without touching the HF cache:

```
MODEL_NAME=/opt/models/quantized/<family>/<format>/<bits>/<variant>
```

**Ports**

| Host | Container | Service |
|---|---|---|
| 8080 | 8000 | vLLM (direct API) |
| 8000 | 8000 | FastAPI proxy |
| 3000 | 8080 | Open WebUI |
| 8888 | 8080 | SearXNG |

---

## API proxy + Open WebUI

`services/api-webui` runs two things in one container:

- a **FastAPI proxy** on port 8000 that forwards to the LLM backend, normalises request defaults, and exposes `/search` backed by SearXNG;
- **Open WebUI** on port 8080 (host 3000).

`LLM_SERVERS` is a comma-separated list of backend base URLs; the proxy uses the first one that answers `/v1/models`. Default is `http://vllm-server:8000` — set it to a host address if the model runs outside the compose network:

```bash
LLM_SERVERS=http://172.17.0.1:8000 docker compose -f services/docker/docker-compose.yml up -d api-webui
```

`services/api-webui/proxy.env` sets request defaults (`DEFAULT_MAX_TOKENS`, `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_P`). Speculative decoding requires `temperature=0` and `top_p=1`; any other value disables draft-verify alignment.

**Web search in Open WebUI**

1. Settings → Integrations → Tool Servers → Add New.
2. URL: `http://<host>:8000` — the URL is resolved by your *browser*, so `localhost` only works when the browser runs on the same machine. Use the host name or VPN address otherwise.
3. Save; Open WebUI discovers the `search_web` tool from `/openapi.json`.

```bash
curl "http://localhost:8000/search?q=<query>"
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" \
  -d '{"query": "<query>", "max_results": 5}'
```

---

## llama.cpp

```bash
ml serve-model <family> <variant>
ml serve-model <family> <variant> --ctx 32768 --port 8000 --fa
```

The script resolves the registry entry, finds the `.gguf` file (or uses the recorded `file`), and launches `runtimes/llama.cpp/build/bin/llama-server`. If the entry has an `mmproj`, the vision projection is loaded too.

Before launching it checks whether the vLLM container is running and offers to stop it — both backends on one GPU will contend for VRAM.

The active model is written to `configs/llama/.active-llama`, which `claude-local` and `ml eval` read.

---

## Docker Compose

```bash
# build images
docker compose -f services/docker/docker-compose.yml build

# everything
docker compose -f services/docker/docker-compose.yml up -d

# one service
docker compose -f services/docker/docker-compose.yml up -d vllm
docker compose -f services/docker/docker-compose.yml up -d api-webui

# logs / stop
docker compose -f services/docker/docker-compose.yml logs -f vllm
docker compose -f services/docker/docker-compose.yml down
```

`bash scripts/build.sh [service...]` is a thin wrapper around `build`.

**Auto-start on WSL2** — containers do not survive a reboot. Two scheduled tasks on the Windows host:

```powershell
schtasks /create /tn "ML-Stack-vLLM" /tr "wsl bash -c 'cd /home/<user>/ml-stack && docker compose -f services/docker/docker-compose.yml up -d vllm'" /sc onlogon /ru <user>
schtasks /create /tn "ML-Stack-API-WebUI" /tr "wsl bash -c 'cd /home/<user>/ml-stack && docker compose -f services/docker/docker-compose.yml up -d api-webui'" /sc onlogon /ru <user>
```

---

## `claude-local`

`init.sh --install` adds a `claude-local` shell function that sources `.claude_env`, which in turn runs `scripts/utils/get-claude-env`. That script detects the active backend in order:

1. a running `vllm-server` container (reads `MODEL_NAME` from its environment);
2. `configs/vllm/active.env`;
3. a recorded llama.cpp model in `configs/llama/.active-llama` (only if something is listening on port 8000).

It then exports `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, and the three model aliases, and launches `claude`. Override the base URL or token by exporting them before calling `claude-local`.

```bash
ml set-vllm-env my-model.env
docker compose -f services/docker/docker-compose.yml up -d vllm
claude-local
```
