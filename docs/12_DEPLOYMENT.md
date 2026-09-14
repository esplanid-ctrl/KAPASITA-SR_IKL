# Deployment

Recommended MVP: Streamlit Community Cloud.

Set secrets in the deployment UI, not Git:
GEMINI_API_KEY
GEMINI_MODEL
GOOGLE_DRIVE_DATA_FOLDER_ID
GOOGLE_DRIVE_POLICY_FOLDER_ID
GOOGLE_SERVICE_ACCOUNT_JSON

Smoke tests:
- app starts;
- Drive authentication works;
- data ingestion works;
- scoring works;
- evidence panel works;
- Copilot works or degrades gracefully;
- no secret is present in Git.
