from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    title: str
    status: str
    error_message: str | None = None

    class Config:
        from_attributes = True


class ItemOut(BaseModel):
    id: str
    item_type: str
    key: str
    value: str
    source_text: str | None
    page_from: int | None
    page_to: int | None
    confidence: float | None
    normalized_key: str | None = None
    normalized_value: str | None = None
    correction_confidence: float | None = None
    correction_reason: str | None = None
    correction_strategy: str | None = None
    correction_status: str | None = None
    correction_orchestrator_reason: str | None = None
    status: str
    extractor: str

    class Config:
        from_attributes = True


class ValidationOut(BaseModel):
    id: str
    agent_name: str
    decision: str
    confidence: float | None
    reason: str | None
    normalized_key: str | None
    normalized_value: str | None
    rag_evidence: dict | None

    class Config:
        from_attributes = True


class KnowledgeEntryOut(BaseModel):
    id: str
    term: str
    definition: str
    status: str
    source_document_id: str | None
    source_item_id: str | None

    class Config:
        from_attributes = True
