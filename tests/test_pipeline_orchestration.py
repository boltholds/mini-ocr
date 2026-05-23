from __future__ import annotations

import unittest
from types import SimpleNamespace

from mini_ocr.services.processing_orchestrator import DocumentProcessingOrchestrator


class FakeDb:
    def __init__(self, document):
        self.document = document
        self.commits = 0
        self.refreshes = 0

    def get(self, model, document_id):
        if self.document and self.document.id == document_id:
            return self.document
        return None

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        self.refreshes += 1


class FakeStatus:
    def __init__(self):
        self.transitions = []
        self.failed = None
        self.processed_failed_pages = None

    def set_status(self, db, document, status, error_message=None):
        document.status = status
        document.error_message = error_message
        self.transitions.append(status)
        db.commit()

    def mark_processed(self, db, document, failed_pages=0):
        self.processed_failed_pages = failed_pages
        document.status = "processed_with_warnings" if failed_pages else "processed"
        document.error_message = None if not failed_pages else f"Processed with warnings: OCR failed for {failed_pages} page(s)"
        db.commit()
        db.refresh(document)

    def mark_failed(self, db, document, exc):
        self.failed = str(exc)
        document.status = "failed"
        document.error_message = str(exc)
        db.commit()


class RecordingStage:
    def __init__(self, name, calls, result=None):
        self.name = name
        self.calls = calls
        self.result = result

    def run(self, db, document):
        self.calls.append(self.name)
        return self.result


class FakePageStore:
    def __init__(self, done_pages=1, failed_pages=0):
        self.done_pages = done_pages
        self.failed_pages = failed_pages

    def count_done_pages(self, db, document):
        return self.done_pages

    def count_failed_pages(self, db, document):
        return self.failed_pages


class ProcessingPipelineOrchestrationTest(unittest.TestCase):
    def test_runs_render_ocr_extract_in_order(self):
        calls = []
        document = SimpleNamespace(id="doc-1", title="test.pdf", status="registered", error_message=None)
        db = FakeDb(document)
        status = FakeStatus()

        pipeline = DocumentProcessingOrchestrator(
            status=status,
            renderer=RecordingStage("render", calls),
            ocr=RecordingStage("ocr", calls),
            extractor=RecordingStage("extract", calls, result=[]),
            page_store=FakePageStore(done_pages=3, failed_pages=0),
        )

        result = pipeline.process(db, document)

        self.assertIs(result, document)
        self.assertEqual(calls, ["render", "ocr", "extract"])
        self.assertEqual(status.transitions, ["rendering", "ocr_running", "extracting"])
        self.assertEqual(document.status, "processed")

    def test_stops_before_extraction_when_all_ocr_failed(self):
        calls = []
        document = SimpleNamespace(id="doc-1", title="test.pdf", status="registered", error_message=None)
        db = FakeDb(document)
        status = FakeStatus()

        pipeline = DocumentProcessingOrchestrator(
            status=status,
            renderer=RecordingStage("render", calls),
            ocr=RecordingStage("ocr", calls),
            extractor=RecordingStage("extract", calls, result=[]),
            page_store=FakePageStore(done_pages=0, failed_pages=5),
        )

        result = pipeline.process(db, document)

        self.assertIs(result, document)
        self.assertEqual(calls, ["render", "ocr"])
        self.assertEqual(document.status, "failed")
        self.assertEqual(document.error_message, "OCR failed for all pages")

    def test_marks_failed_on_stage_exception(self):
        class FailingStage:
            def run(self, db, document):
                raise RuntimeError("boom")

        document = SimpleNamespace(id="doc-1", title="test.pdf", status="registered", error_message=None)
        db = FakeDb(document)
        status = FakeStatus()
        pipeline = DocumentProcessingOrchestrator(
            status=status,
            renderer=FailingStage(),
            ocr=RecordingStage("ocr", []),
            extractor=RecordingStage("extract", []),
            page_store=FakePageStore(),
        )

        with self.assertRaises(RuntimeError):
            pipeline.process(db, document)

        self.assertEqual(document.status, "failed")
        self.assertEqual(status.failed, "boom")


if __name__ == "__main__":
    unittest.main()
