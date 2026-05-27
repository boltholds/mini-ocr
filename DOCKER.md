# Docker-развёртывание

В проекте есть два Compose-файла:

- `docker-compose.cpu.yml` — runtime для OCR на CPU.
- `docker-compose.gpu.yml` — runtime для OCR на NVIDIA GPU.

Файл `docker-compose.yml` по умолчанию указывает на CPU-настройку для обратной совместимости.

## Запуск на CPU

```bash
cp .env.cpu.example .env
docker compose -f docker-compose.cpu.yml --env-file .env up --build
```

API:

```bash
curl http://localhost:8000/health
```

## Запуск на GPU

Требования к хосту:

- установлен NVIDIA driver;
- установлен NVIDIA Container Toolkit;
- команда `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` работает.

Запуск:

```bash
cp .env.gpu.example .env
docker compose -f docker-compose.gpu.yml --env-file .env up --build
```

### Индекс Paddle GPU wheel

GPU-образ использует `python:3.12-slim`, потому что Paddle GPU wheels могут быть недоступны для Python 3.13. Образ устанавливает зависимости приложения из `pyproject.toml`, удаляет CPU-пакет `paddlepaddle`, если он появился транзитивно, а затем устанавливает настроенный GPU wheel из Paddle CUDA index.

Значения по умолчанию:

```env
PADDLE_GPU_PACKAGE=paddlepaddle-gpu==3.2.2
PADDLE_GPU_INDEX_URL=https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

Если стек хоста/драйвера требует другую CUDA-сборку, переопределите `PADDLE_GPU_INDEX_URL` в `.env`, например `https://www.paddlepaddle.org.cn/packages/stable/cu118/`.

## Ограничение памяти PostgreSQL

Оба Compose-файла ограничивают PostgreSQL следующими настройками:

```env
POSTGRES_MEMORY_LIMIT=2g
POSTGRES_SHM_SIZE=512m
POSTGRES_SHARED_BUFFERS=512MB
POSTGRES_WORK_MEM=16MB
POSTGRES_MAINTENANCE_WORK_MEM=256MB
POSTGRES_EFFECTIVE_CACHE_SIZE=1536MB
POSTGRES_MAX_CONNECTIONS=50
```

Для этого OCR-приложения PostgreSQL не является узким местом, поэтому `2g` выбран как консервативное значение. Увеличивайте его только если начнёте хранить много документов и запускать тяжёлые validation queries.

## Примечание про GPU / 32 GB VRAM

Docker Compose может запросить или выбрать GPU, но не может задать жёсткий лимит VRAM для конкретного контейнера. В окружении есть:

```env
GPU_MEMORY_LIMIT_GB=32
PADDLE_GPU_MEMORY_FRACTION=0.95
```

Эти значения являются конфигурационными подсказками. Это не Docker-level VRAM cgroup.

Если нужна реальная граница в 32 GB VRAM, используйте NVIDIA MIG на хосте и пробросьте в контейнер только выбранное MIG-устройство:

```env
NVIDIA_VISIBLE_DEVICES=MIG-<uuid>
```

Без MIG практические варианты контроля такие:

- пробросить только одну GPU через `NVIDIA_VISIBLE_DEVICES=0`;
- уменьшить `PADDLE_GPU_MEMORY_FRACTION`;
- держать LLM inference вне контейнера приложения, например в отдельном Ollama service или host process;
- мониторить потребление через `nvidia-smi`.

## Варианты запуска LLM

Приложение может работать с внешним Ollama server, с Ollama внутри Compose или в degraded deterministic fallback mode.

### Вариант A — Ollama на хосте

Используйте этот вариант, если Ollama уже установлена на машине разработчика:

```env
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_MODEL=qwen2.5:7b-instruct
ENABLE_REGEX_FALLBACK=true
FALLBACK_ON_LLM_ERROR=true
LLM_MAX_RETRIES=0
```

На Linux `host.docker.internal` включается через `extra_hosts: host.docker.internal:host-gateway` в Compose.

Проверка доступности:

```bash
curl http://localhost:8000/health/llm
```

### Вариант B — self-contained Ollama в Compose

Этот вариант не зависит от Ollama, установленной на машине проверяющего. Он запускает service `ollama` и init service `ollama-pull`, который скачивает настроенную модель в volume `ollama_data`.

CPU:

```bash
cp .env.cpu.example .env
docker compose -f docker-compose.cpu.yml -f docker-compose.ollama.yml --env-file .env up --build
```

GPU OCR app с тем же Ollama override:

```bash
cp .env.gpu.example .env
docker compose -f docker-compose.gpu.yml -f docker-compose.ollama.yml --env-file .env up --build
```

Модель можно изменить до запуска:

```env
OLLAMA_MODEL=qwen2.5:7b-instruct
LLM_MODEL=qwen2.5:7b-instruct
```

Для более слабых машин проверяющих можно выбрать меньшую Ollama-модель, но `LLM_MODEL` и `OLLAMA_MODEL` должны совпадать.

### Вариант C — без LLM / deterministic fallback

Этот режим полезен, когда локальная LLM недоступна. Он не будет таким точным, как LLM extraction, но pipeline всё равно будет выдавать консервативных кандидатов вместо пустого результата после timeout.

```env
ENABLE_LLM=false
ENABLE_REGEX_FALLBACK=true
FALLBACK_ON_LLM_ERROR=true
```

Когда LLM включена, но недоступна или отвечает слишком медленно, `ENABLE_REGEX_FALLBACK=true` и `FALLBACK_ON_LLM_ERROR=true` переводят extraction в regex/table fallback.

## Полезные команды

```bash
# CPU logs
docker compose -f docker-compose.cpu.yml logs -f app

# GPU logs
docker compose -f docker-compose.gpu.yml logs -f app

# DB shell
docker compose -f docker-compose.cpu.yml exec db psql -U postgres -d ocr_db

# Остановить и сохранить volumes
docker compose -f docker-compose.cpu.yml down

# Остановить и удалить volumes
docker compose -f docker-compose.cpu.yml down -v
```

### Конфликты портов Ollama

`docker-compose.ollama.yml` по умолчанию **не публикует** порт Ollama `11434` на хост. Контейнер приложения обращается к Ollama через внутреннюю Compose-сеть:

```env
LLM_BASE_URL=http://ollama:11434/v1
```

Это позволяет избежать распространённой ошибки:

```text
ports are not available: exposing port TCP 0.0.0.0:11434 ... Only one usage of each socket address
```

Эта ошибка означает, что Ollama уже запущена на хосте и занимает порт `11434`.

Если всё же нужно открыть Compose Ollama service на хосте, используйте optional port override. По умолчанию он публикует сервис на host port `11435`:

```bash
docker compose -f docker-compose.cpu.yml -f docker-compose.ollama.yml -f docker-compose.ollama-port.yml --env-file .env up --build
```

После этого с хоста можно использовать:

```bash
curl http://localhost:11435/v1/models
```

Если нужно использовать уже запущенную host Ollama вместо containerized Ollama, не подключайте `docker-compose.ollama.yml`; оставьте:

```env
LLM_BASE_URL=http://host.docker.internal:11434/v1
```
