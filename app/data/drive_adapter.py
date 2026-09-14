import os, json
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def service():
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON is missing")
    info = json.loads(raw)
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)

def list_files(folder_id):
    svc = service()
    q = f"'{folder_id}' in parents and trashed=false"
    return svc.files().list(q=q, fields="files(id,name,mimeType,modifiedTime,size)").execute().get("files", [])

def download_file(file_id, target_dir="/tmp/kapasita"):
    target = Path(target_dir); target.mkdir(parents=True, exist_ok=True)
    svc = service()
    meta = svc.files().get(fileId=file_id, fields="name").execute()
    path = target / meta["name"]
    req = svc.files().get_media(fileId=file_id)
    with path.open("wb") as fh:
        dl = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
    return path


# ---------------------------------------------------------------------------
# Region master source (GOOGLE_DRIVE_REGION_FOLDER_ID) -- P0 wiring.
#
# Kept as its own function, separate from the generic list_files/download_file
# above and from any DATA_DRIVE/POLICY_DRIVE logic, because the region
# master has a different lifecycle: it must be fetched and built into a
# canonical artifact ONCE, then read from a local cache on every subsequent
# call -- never re-downloaded or re-parsed on every Streamlit page load
# (project-owner decision). See app.data.region_master for the actual
# parsing/build logic; this module only handles the Drive I/O.
# ---------------------------------------------------------------------------
from .region_master import (
    build_canonical_region_master,
    load_canonical_artifact,
    save_canonical_artifact,
    RegionMasterError,
)


def _find_file(files, predicate):
    return next((f for f in files if predicate(f["name"].lower())), None)


def load_region_master_from_drive(
    folder_id=None,
    cache_path="/tmp/kapasita/region_master_canonical.csv",
    force_rebuild=False,
    target_dir="/tmp/kapasita",
):
    """Loads the canonical region master, sourced from the region Drive
    folder (GOOGLE_DRIVE_REGION_FOLDER_ID), never inventing another source.

    Order of preference, each one strictly cheaper than the next:
      1. A local cached canonical artifact at `cache_path` -- if present
         and `force_rebuild` is False, this is used directly with NO Drive
         call at all. This is what makes "do not re-download/reparse on
         every page load" true across repeated calls in the same
         deployment.
      2. A pre-built canonical artifact file already sitting in the Drive
         folder (name containing "region_master_canonical", a .csv) --
         downloaded once and cached to `cache_path`.
      3. The raw `wilayah.sql` + `wilayah_level_1_2.sql` pair in the Drive
         folder -- downloaded once, built into the canonical artifact via
         app.data.region_master.build_canonical_region_master, and cached
         to `cache_path` for all future calls.

    Raises RegionMasterError if none of the above are found in the given
    folder -- this function does not fall back to any other geographic
    source (project-owner decision: no new geographic source unless this
    one is proven insufficient).
    """
    if not force_rebuild and Path(cache_path).exists():
        return load_canonical_artifact(cache_path)

    if folder_id is None:
        from config.settings import load_region_source_config
        folder_id = load_region_source_config(required=True).region_folder_id

    files = list_files(folder_id)

    artifact_file = _find_file(
        files, lambda n: "region_master_canonical" in n and n.endswith(".csv")
    )
    if artifact_file:
        local_path = download_file(artifact_file["id"], target_dir=target_dir)
        region_master = load_canonical_artifact(str(local_path))
        save_canonical_artifact(region_master, cache_path)
        return region_master

    wilayah_file = _find_file(files, lambda n: n == "wilayah.sql")
    level12_file = _find_file(files, lambda n: n == "wilayah_level_1_2.sql")
    if wilayah_file and level12_file:
        wilayah_path = download_file(wilayah_file["id"], target_dir=target_dir)
        level12_path = download_file(level12_file["id"], target_dir=target_dir)
        region_master = build_canonical_region_master(str(wilayah_path), str(level12_path))
        save_canonical_artifact(region_master, cache_path)
        return region_master

    raise RegionMasterError(
        f"Region Drive folder '{folder_id}' contains neither a pre-built "
        "region_master_canonical*.csv artifact nor the raw wilayah.sql + "
        "wilayah_level_1_2.sql pair. Refusing to guess or substitute "
        "another geographic source."
    )
