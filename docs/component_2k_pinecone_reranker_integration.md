# Component 2K: Pinecone Reranker Integration

## Scope

This sub-component reranks hybrid retrieval candidates using Pinecone's reranker.

Implemented here:

- rerank document shaping
- Pinecone rerank call
- reranked result mapping back to hydrated chunks

Testing includes a live gold-query evaluation over the reranked output.
