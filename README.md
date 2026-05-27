# OCR + LLM извлечение данных из документов

MVP-сервис для автоматического извлечения перечней сокращений, терминов и определений из PDF-сканов без текстового слоя.

Решение рассчитано на документы с разным качеством сканов, произвольным расположением разделов и возможным отсутствием нужных разделов. Все промежуточные результаты сохраняются в PostgreSQL, поэтому сервис можно безопасно перезапускать: уже распознанные страницы и уже выполненные задания извлечения повторно не обрабатываются.

## Что внутри

- FastAPI API
- PostgreSQL
- PaddleOCR для OCR PDF-сканов
- PyMuPDF для рендера PDF-страниц в изображения
- OpenCV для предварительной обработки изображений
- нечёткий поиск разделов через RapidFuzz
- структурированное LLM-извлечение через OpenAI-compatible API
- Pydantic-валидация результата
- привязка к источнику: сохраняется исходный OCR-фрагмент
- хэш файла и хэш входного текста для дедупликации
- статусы документов, страниц и заданий извлечения
- Docker Compose

## Pipeline

```text
PDF-скан
  ↓
регистрация документа + sha256(file)
  ↓
рендер страниц в изображения
  ↓
предварительная обработка изображений страниц
  ↓
PaddleOCR по каждой странице
  ↓
сохранение OCR-текста + OCR-блоков с bbox/confidence
  ↓
нечёткий поиск нужных разделов
  ↓
LLM извлекает JSON из фрагментов-кандидатов
  ↓
Pydantic-валидация + привязка к исходному тексту
  ↓
сохранение extracted_items в PostgreSQL
```

## Внутренняя архитектура pipeline

`ProcessingPipeline` теперь является тонким фасадом. Он не содержит OCR/render/cleanup-логику напрямую, а только собирает зависимости:

```text
DocumentRegistry                 — регистрация PDF, file hash, storage path
DocumentProcessingOrchestrator   — порядок стадий и статусы документа
RenderPagesStage                 — PDF -> images + render cache
OCRPagesStage                    — OCR per page + сохранение блоков/аналитики
ExtractItemsStage                — section detection + LangGraph extraction
PageStore                        — DB-операции по страницам, OCR cache, cleanup extraction output
DocumentStatusService            — переходы статусов registered/rendering/ocr/extracting/processed/failed
```

Благодаря этому сохранение в БД, OCR, рендеринг, регистрация документа и оркестрация больше не смешаны в одном классе. Оркестратор тестируется через fake stages без реальной БД и OCR.

## Быстрый запуск

```bash
docker compose up --build
```

Проверка health endpoint:

```bash
curl http://localhost:8000/health
```

Загрузка PDF:

```bash
curl -F "file=@samples/test.pdf" http://localhost:8000/documents/upload
```

Ответ содержит `id` документа. Запуск обработки:

```bash
curl -X POST http://localhost:8000/documents/<DOCUMENT_ID>/process
```

Получение результатов:

```bash
curl http://localhost:8000/documents/<DOCUMENT_ID>/items
```

Получение результатов по названию документа:

```bash
curl http://localhost:8000/documents/by-title/test.pdf/items
```

## LLM-настройки

Сервис поддерживает OpenAI-compatible API:

```bash
cp .env.example .env
```

```env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=...
LLM_MODEL=gpt-4o-mini
ENABLE_LLM=true
```

Если `LLM_API_KEY` не задан, сервис использует простой regex fallback. Это удобно для локальной проверки пайплайна без внешнего API, но для качественного извлечения в тестовом сценарии предполагается LLM.

## GPU

По умолчанию OCR работает без GPU:

```env
OCR_USE_GPU=false
```

При наличии GPU можно включить:

```env
OCR_USE_GPU=true
```

Ограничение 32 GB VRAM учтено архитектурно: OCR выполняется постранично, без загрузки всего документа в память. Для production-режима batch size OCR/LLM должен задаваться конфигурацией.

## Отказоустойчивость и рестарт

Решение идемпотентно по этапам:

- если документ с таким `file_hash` уже есть, он не загружается повторно;
- если страница уже имеет `ocr_status=done`, OCR повторно не запускается;
- если extraction job с тем же `input_text_hash + prompt_version + model_name` уже выполнен, LLM повторно не вызывается;
- если нужный раздел отсутствует, документ может получить статус `processed`, а список items будет пустым;
- если страница OCR упала, она получает `ocr_status=failed`, остальные страницы могут быть обработаны.

## Основные таблицы

- `documents` — документы, hash, статус обработки;
- `document_pages` — страницы, image path, OCR text, OCR status;
- `ocr_blocks` — OCR-блоки с координатами и confidence;
- `extraction_jobs` — кэш LLM-извлечения по input hash;
- `extracted_items` — итоговые сокращения, термины и определения.

## CLI

Можно использовать CLI внутри контейнера:

```bash
docker compose run --rm app python -m mini_ocr.cli init-db
```

```bash
docker compose run --rm app python -m mini_ocr.cli process /data/input
```

```bash
docker compose run --rm app python -m mini_ocr.cli list-documents
```

```bash
docker compose run --rm app python -m mini_ocr.cli get-items "document.pdf"
```

## Пример результата

```json
[
  {
    "id": "...",
    "item_type": "abbreviation",
    "key": "БД",
    "value": "база данных",
    "source_text": "БД — база данных",
    "page_from": 3,
    "page_to": 3,
    "confidence": 0.91,
    "status": "auto",
    "extractor": "llm"
  },
  {
    "id": "...",
    "item_type": "term",
    "key": "Документ",
    "value": "зафиксированная на носителе информация...",
    "source_text": "Документ — зафиксированная на носителе информация...",
    "page_from": 5,
    "page_to": 6,
    "confidence": 0.86,
    "status": "auto",
    "extractor": "llm"
  }
]
```

## Что можно улучшить дальше

- вынести обработку в background worker;
- добавить UI для `needs_review`;
- добавить отдельный LLM-проход для валидации;
- добавить table-aware extraction по OCR bbox;
- добавить Alembic migrations вместо `create_all`;
- добавить метрики: OCR time, pages/sec, LLM tokens, items extracted, average confidence.

## Агент коррекции OCR

LangGraph workflow включает отдельный узел `OCRCorrectionAgent` между извлечением и валидацией:

```text
extract -> save -> normalize -> validate
```

Агент коррекции запускается для каждого кандидата с низкой уверенностью / `needs_review`, а также для ключей, которые выглядят как OCR-noisy. Он не перезаписывает исходные поля `key`, `value` и `source_text`. Вместо этого он записывает поля с предложениями:

- `normalized_key`
- `normalized_value`
- `correction_confidence`
- `correction_reason`

Для неоднозначных OCR-случаев агент всё равно возвращает гипотезу в `normalized_key`, но оставляет item в статусе `needs_review`. Это полезно для старых сканов, где OCR часто смешивает кириллические и латинские символы.

Если база данных была создана до этой версии, один раз выполните:

```sql
ALTER TABLE extracted_items ADD COLUMN IF NOT EXISTS normalized_key TEXT;
ALTER TABLE extracted_items ADD COLUMN IF NOT EXISTS normalized_value TEXT;
ALTER TABLE extracted_items ADD COLUMN IF NOT EXISTS correction_confidence DOUBLE PRECISION;
ALTER TABLE extracted_items ADD COLUMN IF NOT EXISTS correction_reason TEXT;
```

Рекомендуемые флаги:

```env
ENABLE_LANGGRAPH_WORKFLOW=true
ENABLE_OCR_CORRECTION_AGENT=true
ENABLE_AGENT_VALIDATION=true
ENABLE_RAG_VALIDATION=true
PROMPT_VERSION=terms_abbrev_extractor_v6_aggressive_correction
```

## Трассировка и тайминги агентов

LangGraph workflow теперь логирует каждую важную стадию и каждый вызов агента. Трассировка управляется переменными окружения:

```env
ENABLE_AGENT_TRACING=true
AGENT_LOG_LEVEL=INFO
AGENT_LOG_FILE=logs/agents.log
```

Формат лога — построчный и JSON-like. Каждая измеряемая операция пишет событие `start` и событие `end` с `duration_ms`; ошибки пишутся как событие `error` с типом и сообщением исключения.

Примеры стадий:

```text
pipeline.render_pages
pipeline.ocr_pages
ocr.page
section_detector.detect
langgraph.workflow
agent.extractor
workflow.save_node
workflow.normalize_node
agent.ocr_correction
rag.retrieve_for_correction
workflow.validate_node
agent.rag_validation
rag.retrieve_for_validation
```

Так проще увидеть, где pipeline тратит время и прошёл ли каждый кандидат через извлечение, коррекцию и валидацию.

## Рефакторинг чистой архитектуры

Код обработки разделён на небольшие тестируемые модули:

```text
src/mini_ocr/services/langgraph_workflow.py      # только оркестрация на уровне документа
src/mini_ocr/services/agents/extraction.py       # LLM-агент извлечения
src/mini_ocr/services/agents/correction.py       # LangGraph router/subgraph коррекции
src/mini_ocr/services/agents/validation.py       # RAG-assisted агент валидации
src/mini_ocr/utils/json_utils.py                 # устойчивый парсинг JSON из LLM-ответов
src/mini_ocr/utils/text.py                       # OCR/text эвристики
src/mini_ocr/core/schema.py                      # небольшая runtime migration для совместимости схемы
```

Подграф коррекции теперь явный:

```text
route
  -> keep | capitalizer | corrector | restorer | skip
  -> post_filter
```

Исходные OCR-поля никогда не перезаписываются. Система записывает коррекции только в `normalized_key`, `normalized_value`, `correction_confidence`, `correction_reason`, `correction_strategy`, `correction_status` и `correction_orchestrator_reason`.

## Тесты

Запуск:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Текущие тесты покрывают:

- relaxed JSON parsing для LLM-ответов;
- OCR/text эвристики для русских, латинских, mixed-script и caps-ключей;
- guardrails deterministic extraction validator.

## Docker

Docker-файлы для CPU/GPU-развёртывания описаны в [DOCKER.md](DOCKER.md).
