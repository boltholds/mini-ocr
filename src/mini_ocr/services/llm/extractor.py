import json
import re
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from mini_ocr.core.config import settings
from mini_ocr.schemas.extraction import ExtractedEntity, ExtractionResult
from mini_ocr.services.llm.prompt import SYSTEM_PROMPT


class StructuredExtractor:
    extractor_name = "base"

    def extract(self, text: str) -> ExtractionResult:
        raise NotImplementedError


class LLMStructuredExtractor(StructuredExtractor):
    extractor_name = "llm"

    def __init__(self) -> None:
        provider = settings.llm_provider.lower().strip()
        base_url = settings.llm_base_url
        api_key = settings.llm_api_key

        if provider == "ollama":
            base_url = base_url or "http://localhost:11434/v1"
            api_key = api_key or "ollama"

        if not api_key:
            raise RuntimeError("LLM_API_KEY is not set. For Ollama use LLM_API_KEY=ollama")

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.provider = provider

    def extract(self, text: str) -> ExtractionResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text[:30000]},
        ]

        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": 0,
        }

        # Ollama's OpenAI-compatible endpoint is more reliable without response_format
        # for small local models. For OpenAI-compatible cloud providers, keep JSON mode.
        if self.provider != "ollama":
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or "{}"
        data = _loads_json_relaxed(content)
        result = ExtractionResult.model_validate(data)
        return self._ground_to_source(result, text)

    def _ground_to_source(self, result: ExtractionResult, source: str) -> ExtractionResult:
        normalized_source = _compact(source).lower()
        for group in (result.abbreviations, result.terms):
            for item in group:
                evidence = item.source_text or f"{item.key} {item.value}"
                grounded = _compact(evidence).lower() in normalized_source
                if not grounded:
                    item.confidence = min(item.confidence or 0.5, 0.49)
        return result


class RegexFallbackExtractor(StructuredExtractor):
    """Conservative fallback. It is intentionally disabled by default."""

    extractor_name = "regex"
    PAIR_RE = re.compile(r"^\s*([A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9 ./_-]{1,80})\s+[—–-]\s+(.{3,})$")

    def extract(self, text: str) -> ExtractionResult:
        abbreviations: list[ExtractedEntity] = []
        terms: list[ExtractedEntity] = []
        lower = text.lower()
        target = abbreviations if any(w in lower for w in ("сокращ", "обознач")) else terms

        for line in text.splitlines():
            line = line.strip()
            match = self.PAIR_RE.match(line)
            if not match:
                continue
            key, value = match.group(1).strip(), match.group(2).strip()
            if _looks_like_garbage(key, value):
                continue
            entity = ExtractedEntity(key=key, value=value, source_text=line, confidence=0.65)
            target.append(entity)

        return ExtractionResult(abbreviations=abbreviations, terms=terms)


class HybridExtractor(StructuredExtractor):
    def __init__(self) -> None:
        self.fallback = RegexFallbackExtractor()
        self.llm: LLMStructuredExtractor | None = None
        self.extractor_name = "none"
        if settings.enable_llm:
            self.llm = LLMStructuredExtractor()
            self.extractor_name = self.llm.extractor_name
        elif settings.enable_regex_fallback:
            self.extractor_name = self.fallback.extractor_name

    def extract(self, text: str) -> ExtractionResult:
        if self.llm is not None:
            try:
                self.extractor_name = self.llm.extractor_name
                return self.llm.extract(text)
            except Exception:
                if settings.enable_regex_fallback:
                    self.extractor_name = self.fallback.extractor_name
                    return self.fallback.extract(text)
                raise

        if settings.enable_regex_fallback:
            self.extractor_name = self.fallback.extractor_name
            return self.fallback.extract(text)

        return ExtractionResult(abbreviations=[], terms=[])


def _loads_json_relaxed(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content.strip(), flags=re.IGNORECASE).strip()
        content = re.sub(r"```$", "", content.strip()).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _compact(text: str) -> str:
    return " ".join(text.split())


def _looks_like_garbage(key: str, value: str) -> bool:
    key_norm = _compact(key).lower()
    value_norm = _compact(value).lower()
    joined = f"{key_norm} {value_norm}"

    if len(key_norm) < 2 or len(value_norm) < 5:
        return True
    if key_norm.count(".") >= 2:
        return True
    if any(fragment in joined for fragment in (
        "применение терминов",
        "терминов синонимов",
        "синонимов не допускается",
        "недопустим",
    )):
        return True
    return False
