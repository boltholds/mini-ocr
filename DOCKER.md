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

## Ollama on host

By default the app expects Ollama on the host:

```env
LLM_BASE_URL=http://host.docker.internal:11434/v1
```

On Linux this is enabled by `extra_hosts: host.docker.internal:host-gateway` in Compose.

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
