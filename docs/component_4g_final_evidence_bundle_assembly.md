# Component 4G: Final Evidence Bundle Assembly

## What this sub-component does

- takes the diversified final child chunks
- keeps the parent chunk linkage for grounded answering and citations
- builds a stable answer-ready evidence payload

## Output contents

For each original sub-query:

- selected child evidence items
- `parent_id`
- `child_text`
- `parent_text`
- citation fields:
  - section
  - page range
  - content type
- scores:
  - retrieval
  - rerank
  - MMR
- provenance history

## Why this exists

Later answer generation should consume one clean evidence bundle instead of reading raw retrieval artifacts directly.
