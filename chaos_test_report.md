# Chaos Testing Report

- **Accuracy w/o Retrieval**: 0.8333
- **Accuracy w/o Intelligence**: 0.9000

## Analysis
- The system degrades gracefully when Retrieval is missing, falling back to LLM confidence. (We modified confidence capping to allow ML proxy to still output decisions if confident).
