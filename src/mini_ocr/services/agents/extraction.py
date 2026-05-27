from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from mini_ocr.schemas.extraction import ExtractionResult
from mini_ocr.services.llm.client import build_chat_model
from mini_ocr.services.llm.prompt import SYSTEM_PROMPT
from mini_ocr.services.section_detector import SectionCandidate
from mini_ocr.utils.text import compact


class ExtractionAgent:
    """LLM extractor for one section candidate.

    The agent uses LangChain structured output, so the LLM boundary returns an
    ExtractionResult directly instead of a JSON string that has to be parsed and
    repaired manually.
    """

    extractor_name = "langchain_llm"

    def __init__(self) -> None:
        self.llm = build_chat_model().with_structured_output(ExtractionResult)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "OCR fragment metadata:\n"
                "section_type={section_type}\n"
                "page_from={page_from}\n"
                "page_to={page_to}\n\n"
                "OCR text:\n{text}\n\n"
                "Return a structured ExtractionResult object.",
            ),
        ])
        self.chain = self.prompt | self.llm

    def extract(self, candidate: SectionCandidate) -> ExtractionResult:
        result = self.chain.invoke({
            "section_type": candidate.section_type,
            "page_from": candidate.page_from,
            "page_to": candidate.page_to,
            "text": candidate.text[:30000],
        })
        return self._ground_to_source(ExtractionResult.model_validate(result), candidate.text)

    def _ground_to_source(self, result: ExtractionResult, source: str) -> ExtractionResult:
        normalized_source = compact(source).lower()
        for group in (result.abbreviations, result.terms):
            for item in group:
                evidence = item.source_text or f"{item.key} {item.value}"
                grounded = compact(evidence).lower() in normalized_source
                if not grounded:
                    item.confidence = min(item.confidence or 0.5, 0.49)
        return result
