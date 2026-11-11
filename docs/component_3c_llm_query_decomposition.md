# Component 3C: LLM-Assisted Query Decomposition

## What this sub-component does

- uses `gpt-5-mini` as the primary query decomposition model
- forces structured JSON output
- evaluates the model against a gold query set
- compares model performance against the deterministic baseline

## Inputs

- one user query
- optional baseline decomposition result

## Outputs

- `needs_decomposition`
- `sub_queries`
- `reasoning_type`
- `decomposition_strategy`

## Evaluation

- gold-case pass/fail scoring
- missing-sub-query detection
- extra-sub-query detection
- baseline comparison

## Current limits

- no confidence score yet
- no judge model yet
- no retrieval-grounded quality check yet
