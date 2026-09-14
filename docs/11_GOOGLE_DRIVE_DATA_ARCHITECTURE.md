# Google Drive Data Architecture

Recommended folders:
- `/KAPASITA-MVP/DATA/`
- `/KAPASITA-MVP/POLICY/`

The repository contains only logical source keys:
`DATA_DRIVE` and `POLICY_DRIVE`.

Secrets:
- GEMINI_API_KEY
- GEMINI_MODEL
- GOOGLE_DRIVE_DATA_FOLDER_ID
- GOOGLE_DRIVE_POLICY_FOLDER_ID
- GOOGLE_DRIVE_REGION_FOLDER_ID
- GOOGLE_SERVICE_ACCOUNT_JSON

The region master (cahyadsn/wilayah) uses its own folder
(`GOOGLE_DRIVE_REGION_FOLDER_ID`), separate from DATA and POLICY, because
its lifecycle is different: it is fetched and preprocessed ONCE into a
canonical CSV artifact (app.data.region_master.build_canonical_region_master
+ save_canonical_artifact), not re-downloaded or re-parsed on every
Streamlit page load. See docs/generated/REGION_MASTER_MAPPING_SPEC.json
for the verified schema of the source ZIP.

The application downloads approved files into temporary runtime storage, validates them, and never exposes raw private files through the dashboard.
