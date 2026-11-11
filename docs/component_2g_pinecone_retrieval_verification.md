# Component 2G: Pinecone Retrieval Verification

## Scope

This sub-component verifies that Pinecone retrieval returns child-chunk matches that can be hydrated
back into the original child and parent text needed by downstream RAG components.

Implemented here:

- query embedding
- Pinecone vector query
- child-match hydration
- parent-text recovery
- retrieval report output

Not implemented here:

- reranking
- answer generation
- multi-query retrieval
