from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine


def ensure_runtime_schema(engine: Engine) -> None:
    """Small compatibility migration for the prototype.

    The assignment does not require a full Alembic setup, but existing local
    databases should not fail after adding correction routing columns.
    """
    statements = [
        "ALTER TABLE extracted_items ADD COLUMN IF NOT EXISTS correction_strategy TEXT",
        "ALTER TABLE extracted_items ADD COLUMN IF NOT EXISTS correction_status TEXT",
        "ALTER TABLE extracted_items ADD COLUMN IF NOT EXISTS correction_orchestrator_reason TEXT",
    ]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))
