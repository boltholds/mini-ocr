# Docker deployment

There are two Compose files:

- `docker-compose.cpu.yml` — CPU OCR runtime.
- `docker-compose.gpu.yml` — NVIDIA GPU OCR runtime.

The default `docker-compose.yml` points to the CPU setup for backward compatibility.

## CPU run

```bash
cp .env.cpu.example .env
docker compose -f docker-compose.cpu.yml --env-file .env up --build
```

API:

```bash
curl http://localhost:8000/health
```

## GPU run

Host requirements:

- NVIDIA driver installed.
- NVIDIA Container Toolkit installed.
- `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` works.

Run:

```bash
cp .env.gpu.example .env
docker compose -f docker-compose.gpu.yml --env-file .env up --build
```


### Paddle GPU wheel index

The GPU image uses `python:3.12-slim` because Paddle GPU wheels may be unavailable for Python 3.13. The image installs app dependencies from `pyproject.toml`, removes the CPU `paddlepaddle` package if it appears transitively, and then installs the configured GPU wheel from the Paddle CUDA index.

Defaults:

```env
PADDLE_GPU_PACKAGE=paddlepaddle-gpu==3.2.2
PADDLE_GPU_INDEX_URL=https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

If your host/driver stack requires another CUDA build, override `PADDLE_GPU_INDEX_URL` in `.env`, for example `https://www.paddlepaddle.org.cn/packages/stable/cu118/`.

## PostgreSQL memory limit

Both Compose files limit PostgreSQL with:

```env
POSTGRES_MEMORY_LIMIT=2g
POSTGRES_SHM_SIZE=512m
POSTGRES_SHARED_BUFFERS=512MB
POSTGRES_WORK_MEM=16MB
POSTGRES_MAINTENANCE_WORK_MEM=256MB
POSTGRES_EFFECTIVE_CACHE_SIZE=1536MB
POSTGRES_MAX_CONNECTIONS=50
```

For this OCR app PostgreSQL is not the bottleneck, so `2g` is intentionally conservative. Raise it only if you start storing many documents and running heavy validation queries.

## GPU / 32GB VRAM note

Docker Compose can request/select a GPU, but it cannot enforce a hard per-container VRAM limit. The environment contains:

```env
GPU_MEMORY_LIMIT_GB=32
PADDLE_GPU_MEMORY_FRACTION=0.95
```

These values are configuration hints. They are not a Docker-level VRAM cgroup.

If you need a real 32GB VRAM boundary, use NVIDIA MIG on the host and expose only the selected MIG device to the container:

```env
NVIDIA_VISIBLE_DEVICES=MIG-<uuid>
```

Without MIG, the practical controls are:

- expose only one GPU with `NVIDIA_VISIBLE_DEVICES=0`;
- reduce `PADDLE_GPU_MEMORY_FRACTION`;
- keep LLM inference outside this app container, for example in a separate Ollama service/host process;
- monitor with `nvidia-smi`.

## LLM runtime options

The app can work with an external Ollama server, with Ollama inside Compose, or in a degraded deterministic fallback mode.

### Option A — Ollama on the host

Use this when Ollama is already installed on the developer machine:

```env
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_MODEL=qwen2.5:7b-instruct
ENABLE_REGEX_FALLBACK=true
FALLBACK_ON_LLM_ERROR=true
LLM_MAX_RETRIES=0
```

On Linux `host.docker.internal` is enabled by `extra_hosts: host.docker.internal:host-gateway` in Compose.

Check availability:

```bash
curl http://localhost:8000/health/llm
```

### Option B — self-contained Ollama in Compose

This avoids depending on Ollama installed on the reviewer machine. It starts an `ollama` service and an `ollama-pull` init service that downloads the configured model into the `ollama_data` volume.

CPU:

```bash
cp .env.cpu.example .env
docker compose -f docker-compose.cpu.yml -f docker-compose.ollama.yml --env-file .env up --build
```

GPU OCR app with the same Ollama override:

```bash
cp .env.gpu.example .env
docker compose -f docker-compose.gpu.yml -f docker-compose.ollama.yml --env-file .env up --build
```

The model can be changed before startup:

```env
OLLAMA_MODEL=qwen2.5:7b-instruct
LLM_MODEL=qwen2.5:7b-instruct
```

For weaker reviewer machines you may choose a smaller Ollama model, but keep `LLM_MODEL` and `OLLAMA_MODEL` identical.

### Option C — no LLM / deterministic fallback

This mode is useful when no local LLM is available. It will not be as accurate as LLM extraction, but the pipeline will still produce conservative candidates instead of returning an empty result after timeouts.

```env
ENABLE_LLM=false
ENABLE_REGEX_FALLBACK=true
FALLBACK_ON_LLM_ERROR=true
```

When LLM is enabled but unavailable or too slow, `ENABLE_REGEX_FALLBACK=true` and `FALLBACK_ON_LLM_ERROR=true` make extraction degrade to the regex/table fallback.

## Useful commands

```bash
# CPU logs
docker compose -f docker-compose.cpu.yml logs -f app

# GPU logs
docker compose -f docker-compose.gpu.yml logs -f app

# DB shell
docker compose -f docker-compose.cpu.yml exec db psql -U postgres -d ocr_db

# Stop and keep volumes
docker compose -f docker-compose.cpu.yml down

# Stop and delete volumes
docker compose -f docker-compose.cpu.yml down -v
```

### Ollama port conflicts

`docker-compose.ollama.yml` does **not** publish Ollama's `11434` port to the host by default.
The app container talks to Ollama through the internal Compose network:

```env
LLM_BASE_URL=http://ollama:11434/v1
```

This avoids the common error:

```text
ports are not available: exposing port TCP 0.0.0.0:11434 ... Only one usage of each socket address
```

That error means you already have Ollama running on the host and occupying port `11434`.

If you want to expose the Compose Ollama service to the host anyway, use the optional port override. It publishes to host port `11435` by default:

```bash
docker compose -f docker-compose.cpu.yml -f docker-compose.ollama.yml -f docker-compose.ollama-port.yml --env-file .env up --build
```

Then from the host you can use:

```bash
curl http://localhost:11435/v1/models
```

If you want to use your already-running host Ollama instead of containerized Ollama, do not include `docker-compose.ollama.yml`; keep:

```env
LLM_BASE_URL=http://host.docker.internal:11434/v1
```
