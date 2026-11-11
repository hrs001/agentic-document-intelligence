# Component 2B: Document Cleaning and Normalization

## Scope

This sub-component takes the raw extracted document text artifact and cleans it for later chunking.

Implemented here:

- common PDF text artifact cleanup
- page label removal
- whitespace normalization
- bullet normalization
- cleaned text artifact generation

Not implemented here:

- chunking
- embeddings
- graph extraction

## Input artifact

- `artifacts/experiments/component2_document_text_extraction/document_text/microsoft_fy2025_10k_summary.json`

## Output artifact

- `artifacts/experiments/component2_document_text_cleaning/document_text/microsoft_fy2025_10k_summary_cleaned.json`
