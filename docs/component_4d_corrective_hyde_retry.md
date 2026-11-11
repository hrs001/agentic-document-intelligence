# Component 4D: Corrective HyDE Trigger and Retrieval Retry

## What this sub-component does

- looks at first-pass retrieval coverage per original sub-query
- triggers HyDE only for clearly weak cases
- runs one HyDE retrieval retry for those sub-queries only
- merges HyDE hits back into the existing sub-query candidate pool

## Trigger policy

This version uses conservative retrieval-feedback signals, not raw query text alone.

It can trigger when:

- coverage is explicitly weak
- overlap with the original sub-query is very low and the top score is not convincing
- overlap and reinforcement are both weak
- result depth and score strength are both shallow

## Why this exists

HyDE is useful, but expensive enough that it should not run by default. This step makes it a corrective retry rather than a standard always-on transform.

## Important limitation

The trigger is still heuristic. It is better than raw query-only gating, but it is not yet the final long-term judge. Later rerank-aware feedback should improve this decision.
