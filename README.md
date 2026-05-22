# OCR + LLM Document Extractor

MVP-сервис для автоматического извлечения перечней сокращений, терминов и определений из PDF-сканов без текстового слоя.

Решение рассчитано на документы с разным качеством сканов, произвольным расположением разделов и возможным отсутствием нужных разделов. Все промежуточные результаты сохраняются в PostgreSQL, поэтому сервис можно безопасно перезапускать: уже распознанные страницы и уже выполненные extraction jobs не обрабатываются повторно.

## Что внутри

- FastAPI API
- PostgreSQL
- PaddleOCR для OCR PDF-сканов
- PyMuPDF для рендера PDF-страниц в изображения
- OpenCV preprocessing
- fuzzy section detection через RapidFuzz
- LLM structured extraction через OpenAI-compatible API
- Pydantic validation результата
- source grounding: сохраняется исходный OCR-фрагмент
- file hash и input text hash для дедупликации
- статусы документов, страниц и extraction jobs
- Docker Compose

## Pipeline

```text
PDF scan
  ↓
register document + sha256(file)
  ↓
render pages to images
  ↓
preprocess page images
  ↓
PaddleOCR per page
  ↓
save OCR text + OCR blocks with bbox/confidence
  ↓
fuzzy search for sections
  ↓
LLM extracts JSON from candidate fragments
  ↓
Pydantic validation + source grounding
  ↓
save extracted_items in PostgreSQL
```

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

Если `LLM_API_KEY` не задан, сервис использует простой regex fallback. Это удобно для локальной проверки пайплайна без внешнего API, но для качества извлечения в тестовом сценарии предполагается LLM.

## GPU

По умолчанию OCR работает без GPU:

```env
OCR_USE_GPU=false
```

При наличии GPU можно включить:

```env
OCR_USE_GPU=true
```

Ограничение 32 GB VRAM учитывается архитектурно: OCR выполняется постранично, без загрузки всего документа в память. Для production-режима batch size OCR/LLM должен задаваться конфигурацией.

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
- добавить отдельный LLM validation pass;
- добавить table-aware extraction по OCR bbox;
- добавить Alembic migrations вместо `create_all`;
- добавить метрики: OCR time, pages/sec, LLM tokens, items extracted, average confidence.
