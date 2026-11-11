# Component 2C: Chunk Generation

## Scope

This sub-component converts cleaned document text into parent/child chunk records.

Implemented here:

- parent chunk splitting
- child chunk splitting
- chunk metadata generation
- chunk artifact output

Not implemented here:

- embeddings
- graph extraction
- retrieval

## Input artifact

- `artifacts/experiments/component2_document_text_cleaning/document_text/microsoft_fy2025_10k_summary_cleaned.json`

## Output artifact

- `artifacts/experiments/component2_chunk_generation/chunks/microsoft_fy2025_10k_summary_chunks.json`
