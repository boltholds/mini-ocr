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
    status: str
    extractor: str

    class Config:
        from_attributes = True
