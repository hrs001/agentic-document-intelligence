# Component 3H: Transformation Gating and Selection

## What this sub-component does

- decides which query-transformation tools should run
- caps decomposition fanout at 3 sub-queries
- identifies ambiguous cases for possible later LLM routing

## Current design

- deterministic first-pass policy
- cost-aware routing
- generic ambiguity signals

## Current limits

- no retrieval-feedback loop yet
- no LLM fallback selector yet
