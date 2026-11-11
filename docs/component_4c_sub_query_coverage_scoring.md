# Component 4C: Sub-Query Coverage Scoring

## What this sub-component does

- scores whether each original sub-query looks well-covered by merged retrieval evidence
- uses cheap deterministic signals only
- flags sub-queries that may need corrective HyDE later

## Signals used

- top retrieved score strength
- lexical overlap between the original sub-query and top evidence
- reinforcement from repeated matches across transformed variants
- result depth

## Output

For each original sub-query:

- `coverage_score`
- `coverage_label`
  - `strong`
  - `moderate`
  - `weak`
- `should_consider_hyde`
- `signal_breakdown`

## Why this exists

The system should not run HyDE blindly. This step gives a cheap first pass for deciding whether retrieval looks too weak to trust.

## Important limitation

This is a deterministic heuristic, not a full semantic judge. It is good enough to guide later corrective retrieval experiments, but it is not a final faithfulness or relevance guarantee.
