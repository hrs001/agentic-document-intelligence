# Component 3I: LLM-Assisted Gating for Ambiguous Cases

## What this sub-component does

- keeps deterministic gating as the first pass
- calls `gpt-5-mini` only when the deterministic gate marks a query ambiguous
- returns an adjusted transformation plan when ambiguity is high

## Current goal

- improve gating quality only where deterministic routing is weak
- avoid paying LLM cost for clear cases
