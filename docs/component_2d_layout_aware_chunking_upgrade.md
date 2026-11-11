# Component 2D: Layout-Aware and Table-Aware Chunking Upgrade

## Scope

This upgrade replaces plain-text-only chunking with:

- layout-aware PDF parsing
- native table extraction before chunking
- section-first chunk generation
- explicit table chunks

## New upstream artifact

- `artifacts/experiments/component2_layout_aware_extraction/document_layout/microsoft_fy2025_10k_summary_layout.json`

## Upgraded chunk artifact

- `artifacts/experiments/component2_chunk_generation/chunks/microsoft_fy2025_10k_summary_chunks.json`
