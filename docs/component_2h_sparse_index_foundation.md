# Component 2H: Sparse/BM25 Retrieval Foundation

## Scope

This sub-component builds a reusable BM25-style sparse index artifact from child chunks.

Implemented here:

- child chunk tokenization
- document frequency and IDF computation
- BM25 parameterized sparse index artifact
- parent and metadata preservation

Not implemented here:

- live sparse retrieval queries
- retrieval correctness evaluation
- hybrid fusion
