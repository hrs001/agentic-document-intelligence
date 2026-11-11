# Component 2A: Document Text Extraction

## Scope

This sub-component extracts page-by-page text from the fixed PDF corpus source and writes
an inspectable JSON artifact.

Implemented here:

- PDF text extraction
- page marker preservation
- extracted text artifact generation

Not implemented here:

- cleaning and normalization
- chunking
- embedding generation
- graph input selection

## Output artifact

- `artifacts/experiments/component2_document_text_extraction/document_text/microsoft_fy2025_10k_summary.json`

## Required artifact fields

- `document_id`
- `text`
- `pages`
- `extraction_method`
- `generated_at`

## Review goal

This step is approved only if the raw extracted text is real, non-empty, page-aware,
and good enough to feed the next sub-component: cleaning and normalization.
