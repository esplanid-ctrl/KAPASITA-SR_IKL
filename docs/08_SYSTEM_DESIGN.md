# System Design

Google Drive
→ Drive Adapter
→ Validation & Harmonization
→ Feature Builder
→ Deterministic Score Engine
→ Policy Rule Engine
→ Evidence Registry
→ Streamlit Dashboard
→ Copilot Context Builder
→ Gemini

### Future-ready separation
The MVP keeps clear boundaries between data, analytics, policy, Copilot and presentation. A future React/FastAPI architecture can replace the presentation/API layer without changing the scoring contract.

### Google Drive
Use Google Drive API + service account. Share the data and policy folders with the service account. Store folder IDs and service-account JSON only in deployment secrets.
