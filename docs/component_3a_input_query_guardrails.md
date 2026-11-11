# Component 3A: Input Query Guardrails

## Scope

This sub-component blocks obviously malicious or unsafe queries before query transformation or retrieval.

Implemented here:

- query sanitization
- deterministic prompt-injection detection
- deterministic secret-exfiltration detection
- structured guard result output

Not implemented here:

- semantic guard model
- output guardrails
- final deterministic answer checks
