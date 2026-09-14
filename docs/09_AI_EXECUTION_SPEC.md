# AI Execution Specification

AI coding agent must:
1. read Master Prompt and all docs before coding;
2. inspect real source files rather than inventing columns;
3. validate schema, level, year, denominator, missingness and duplicates;
4. maintain provenance;
5. keep scoring deterministic in Python;
6. use Gemini only with supplied context;
7. cite evidence IDs in Copilot answers;
8. fail safely when evidence is missing;
9. run tests after each implementation stage;
10. never commit secrets.

Workflow:
PLAN → INSPECT → IMPLEMENT → TEST → VALIDATE → DOCUMENT.
