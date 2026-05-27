from __future__ import annotations

from typing import Sequence

from langchain_core.prompts import ChatPromptTemplate

from mini_ocr.services.llm.client import build_chat_model
from mini_ocr.services.validation_models import (
    RagMatchLike,
    ValidationCandidate,
    ValidationDecision,
    rag_evidence_from_matches,
)


class CandidateValidationAgent:
    """Pure LLM validation agent.

    The agent uses ``llm.with_structured_output(ValidationDecision)``. It does
    not parse or repair JSON strings; LangChain/Pydantic owns the output schema
    boundary.
    """

    agent_name = "langchain_rag_validation_agent"

    def __init__(self) -> None:
        self.llm = build_chat_model().with_structured_output(ValidationDecision)
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a strict validation agent for OCR extraction. "
                "Validate one already extracted candidate. Do not extract new terms. "
                "Return a structured ValidationDecision object.",
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
                "Candidate:\n{candidate}\n\n"
                "RAG matches:\n{rag_evidence}",
            ),
        ])
        self.chain = self.prompt | self.llm

    def validate(self, candidate: ValidationCandidate, matches: Sequence[RagMatchLike]) -> ValidationDecision:
        evidence = rag_evidence_from_matches(matches)
        decision = self.chain.invoke({
            "candidate": candidate.model_dump(exclude_none=False),
            "rag_evidence": [item.model_dump() for item in evidence],
        })
        return ValidationDecision.model_validate(decision)
