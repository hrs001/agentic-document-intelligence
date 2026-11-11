# Component 2I: Pinecone Sparse Vector Enrichment

## Scope

This sub-component upgrades the Pinecone records from dense-only to hybrid-capable by adding
Pinecone sparse vectors to the existing child-chunk records.

Implemented here:

- Pinecone sparse embedding generation
- existing dense vector fetch
- merged dense+sparse upsert

Not implemented here:

- sparse retrieval verification
- hybrid score fusion evaluation
