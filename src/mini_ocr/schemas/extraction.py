from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    key: str = Field(min_length=1)
    value: str = Field(min_length=1)
    source_text: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class ExtractionResult(BaseModel):
    abbreviations: list[ExtractedEntity] = Field(default_factory=list)
    terms: list[ExtractedEntity] = Field(default_factory=list)
