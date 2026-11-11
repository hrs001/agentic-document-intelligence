# Component 3J: Transformed Query Bundle Orchestrator

## What this sub-component does

- applies the active fixed transformation policy
- uses LLM decomposition capped at 3 sub-queries
- runs multi-query generation and one step-back query for each sub-query
- does not run HyDE in the initial bundle
- marks sub-queries that may deserve HyDE later if retrieval is weak

## Current active policy

- decomposition: on
- multi-query: on for each sub-query
- step-back: on for each sub-query
- HyDE: off by default, candidate-only
