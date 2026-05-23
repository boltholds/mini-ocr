from __future__ import annotations

import json
from typing import Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from mini_ocr.services.llm.client import build_chat_model
from mini_ocr.services.validation_models import (
    RagMatchLike,
    ValidationCandidate,
    ValidationDecision,
    decision_from_agent_payload,
    rag_evidence_from_matches,
)
from mini_ocr.utils.json_utils import loads_json_relaxed


class CandidateValidationAgent:
    """Pure LLM validation agent.

    It knows how to validate a candidate using RAG evidence, but it does not
    know about SQLAlchemy sessions, persistence, item status transitions or
    concrete RAG storage.
    """

    agent_name = "langchain_rag_validation_agent"

    def __init__(self) -> None:
        self.llm = build_chat_model()
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a strict validation agent for OCR extraction. "
                "Validate one already extracted candidate. Do not extract new terms. "
                "Return only JSON.",
            ),
            (
                "human",
                "Task: validate candidate term/abbreviation from OCR.\n\n"
                "Rules:\n"
                "- decision must be exactly one of: auto, needs_review, rejected.\n"
                "- auto only when the candidate is clearly grounded in source_text and the definition is clean.\n"
                "- needs_review when candidate may be real but OCR noise is high.\n"
                "- rejected for service phrases, empty/unrelated text, or hallucinations.\n"
                "- RAG matches are hints, not proof.\n\n"
                "Candidate JSON:\n{candidate_json}\n\n"
                "RAG matches JSON:\n{rag_json}\n\n"
                "Output schema:\n"
                "{{\"decision\": \"auto|needs_review|rejected\", "
                "\"confidence\": 0.0, "
                "\"reason\": \"short reason\", "
                "\"normalized_key\": null, "
                "\"normalized_value\": null}}",
            ),
        ])
        self.chain = self.prompt | self.llm | StrOutputParser()

    def validate(self, candidate: ValidationCandidate, matches: Sequence[RagMatchLike]) -> ValidationDecision:
        evidence = rag_evidence_from_matches(matches)
        content = self.chain.invoke({
            "candidate_json": candidate.model_dump_json(exclude_none=False),
            "rag_json": json.dumps([item.model_dump() for item in evidence], ensure_ascii=False),
        })
        data = loads_json_relaxed(content)
        return decision_from_agent_payload(data)
