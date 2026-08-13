# ADR 0007: New manuals use immutable staged releases

Status: accepted

Manual ingestion never mutates the production index. A document is copied into an
isolated staging package, inspected, parsed, chunked, evaluated, and approved.
Publishing creates a new immutable version directory and atomically replaces the
scope's active pointer. Rollback switches that pointer to an existing version.

Document text is always untrusted evidence and never an instruction to the Agent.
MIME/content mismatch, suspicious prompt injection, missing OCR for scanned PDFs,
and failed quality gates block publication.

The frozen competition knowledge remains a separate version and is not rewritten
by the enterprise lifecycle.
