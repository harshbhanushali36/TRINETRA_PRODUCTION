"""
🌌 Unified Orbital Data Fetcher (SpaceTrack + NASA NEO via JPL SBDB - full refresh)
-------------------------------------------------------------------------------
Enhancements:
• Multi-threaded NeoWs (browse) fetching
• Multi-threaded JPL SBDB orbital data fetching
• Full daily refresh (no incremental merging)
• Keeps identical SpaceTrack, compression, and structure
"""

import os
import json
import gzip
import time
import math
import requests
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# =============================================
# CONFIGURATION
# =============================================
NASA_API_KEY = os.getenv("NASA_API_KEY")
SPACETRACK_USER = os.getenv("SPACETRACK_USERNAME")
SPACETRACK_PASS = os.getenv("SPACETRACK_PASSWORD")

NEOWS_BROWSE_URL = "https://api.nasa.gov/neo/rest/v1/neo/browse"
JPL_SBDB_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"

SPACETRACK_LOGIN = "https://www.space-track.org/ajaxauth/login"
SPACETRACK_TLE_URL = "https://www.space-track.org/basicspacedata/query/class/tle_latest/ORDINAL/1/format/json"

OBJECT_TYPES = ["PAYLOAD", "ROCKET BODY", "DEBRIS", "UNKNOWN"]
TLE_AGE_DAYS = 5

OUTPUT_LATEST = "data/latest"
OUTPUT_RAW = "data/raw"
os.makedirs(OUTPUT_LATEST, exist_ok=True)
os.makedirs(OUTPUT_RAW, exist_ok=True)

timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

# =============================================
# FETCH FROM SPACETRACK
# =============================================
print("🌍 [SpaceTrack] Logging in...")
session = requests.Session()
resp = session.post(SPACETRACK_LOGIN, data={"identity": SPACETRACK_USER, "password": SPACETRACK_PASS})
resp.raise_for_status()
print("✅ Logged into Space-Track")

type_counts = defaultdict(int)
all_tle_data = []

for obj_type in OBJECT_TYPES:
    print(f"📡 Fetching {obj_type}...")
    url = f"{SPACETRACK_TLE_URL}/OBJECT_TYPE/{obj_type.replace(' ', '%20')}"
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        subset = r.json()
        type_counts[obj_type] = len(subset)
        all_tle_data.extend(subset)
        print(f"   ↳ {len(subset)} records")
    except Exception as e:
        print(f"⚠️ Failed for {obj_type}: {e}")

df_tle = pd.DataFrame(all_tle_data)
df_tle = df_tle.dropna(subset=["TLE_LINE1", "TLE_LINE2", "OBJECT_NAME"])
df_tle["EPOCH"] = pd.to_datetime(df_tle["EPOCH"], errors="coerce")
cutoff = datetime.utcnow() - timedelta(days=TLE_AGE_DAYS)
df_tle = df_tle[df_tle["EPOCH"] >= cutoff].drop_duplicates(subset=["NORAD_CAT_ID"])
print(f"🛰️ [SpaceTrack] {len(df_tle)} valid objects (TLE < {TLE_AGE_DAYS} days)")

tles_json = [
    {
        "id": int(row["NORAD_CAT_ID"]),
        "name": row["OBJECT_NAME"].strip(),
        "tle1": row["TLE_LINE1"].strip(),
        "tle2": row["TLE_LINE2"].strip(),
    }
    for _, row in df_tle.iterrows()
]

sat_info_json = {
    str(int(row["NORAD_CAT_ID"])): {
        "name": row["OBJECT_NAME"].strip(),
        "object_type": row.get("OBJECT_TYPE", "UNKNOWN"),
        "country": row.get("COUNTRY_CODE", "N/A"),
        "launch_date": row.get("LAUNCH_DATE", "N/A"),
        "launch_site": row.get("SITE", "N/A"),
        "inclination": float(row["INCLINATION"]) if pd.notna(row.get("INCLINATION")) else None,
        "apogee": float(row["APOGEE"]) if pd.notna(row.get("APOGEE")) else None,
        "perigee": float(row["PERIGEE"]) if pd.notna(row.get("PERIGEE")) else None,
        "period": float(row["PERIOD"]) if pd.notna(row.get("PERIOD")) else None,
        "decay_date": "ACTIVE" if pd.isna(row.get("DECAY_DATE")) else row.get("DECAY_DATE"),
    }
    for _, row in df_tle.iterrows()
}

# =============================================
# FETCH NEO LIST FROM NeoWs (multi-threaded)
# =============================================
print("\n☄️ [NASA NeoWs] Enumerating NEOs (browse, threaded)...")

def fetch_neows_page(page, api_key):
    params = {"api_key": api_key, "page": page, "size": 20}
    try:
        r = requests.get(NEOWS_BROWSE_URL, params=params, timeout=25)
        if r.status_code == 200:
            return r.json().get("near_earth_objects", [])
    except Exception:
        return []
    return []

def get_neows_first_page(api_key, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(NEOWS_BROWSE_URL, params={"api_key": api_key, "page": 0, "size": 20}, timeout=25)
            if r.status_code == 200:
                return r.json()
        except Exception:
            time.sleep(2 ** attempt)
    return {}

# --- Primary attempt with your API key ---
first_page = get_neows_first_page(NASA_API_KEY or "DEMO_KEY")
if not first_page or first_page.get("page", {}).get("total_elements", 0) == 0:
    print("⚠️ NeoWs returned 0 elements — retrying with DEMO_KEY...")
    first_page = get_neows_first_page("DEMO_KEY")

total_neos = first_page.get("page", {}).get("total_elements", 0)
total_pages = first_page.get("page", {}).get("total_pages", 1)
print(f"   Total estimated NEOs: {total_neos} across {total_pages} pages")

all_neows = first_page.get("near_earth_objects", [])
if total_neos > 0:
    MAX_PAGES = min(total_pages, 2000)  # safety cap
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(fetch_neows_page, i, NASA_API_KEY or "DEMO_KEY"): i for i in range(1, MAX_PAGES)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Enumerating NeoWs", ncols=90):
            data = fut.result()
            if data:
                all_neows.extend(data)

print(f"✅ [NASA NeoWs] Total NEO entries enumerated: {len(all_neows)}")

# Extract unique IDs and mapping for SBDB lookup
neows_by_id = {str(neo["id"]): neo for neo in all_neows if "id" in neo}
unique_ids = list(neows_by_id.keys())

# =============================================
# FULL SBDB PARALLEL FETCH (no incremental)
# =============================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Trinetra-NEO-Fetcher/1.0"})
MAX_THREADS = 20

def fetch_sbdb_single(nid, retries=4):
    params = {"des": nid, "cov": "0", "phys-par": "1"}
    attempt = 0
    while attempt < retries:
        try:
            r = SESSION.get(JPL_SBDB_URL, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                orbit = data.get("orbit") or {}
                phys = data.get("phys_par") or {}
                moid = orbit.get("moid") or orbit.get("earth_moid") or orbit.get("e_moid")
                return {"raw": data, "orbit": orbit, "phys": phys, "moid": moid}
            elif r.status_code == 429:
                time.sleep((2 ** attempt) * 2)
                attempt += 1
                continue
            else:
                return None
        except Exception:
            time.sleep((2 ** attempt) * 2)
            attempt += 1
    return None

sbdb_results = {}
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = {executor.submit(fetch_sbdb_single, nid): nid for nid in unique_ids}
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching SBDB", ncols=90):
        nid = futures[fut]
        res = fut.result()
        if res:
            sbdb_results[nid] = res

print(f"✅ [SBDB] Orbital/phys data fetched for {len(sbdb_results)} objects")

# =============================================
# PROCESS AND SAVE
# =============================================
processed_neos = []
for nid, sb in sbdb_results.items():
    neo_meta = neows_by_id.get(nid, {})
    orbit = sb.get("orbit", {})
    record = {
        "id": nid,
        "name": neo_meta.get("name") or neo_meta.get("designation"),
        "is_hazardous": neo_meta.get("is_potentially_hazardous_asteroid", False),
        "semi_major_axis_au": orbit.get("a"),
        "eccentricity": orbit.get("e"),
        "inclination_deg": orbit.get("i"),
        "ascending_node_longitude_deg": orbit.get("om"),
        "argument_of_perihelion_deg": orbit.get("w"),
        "mean_anomaly_deg": orbit.get("ma"),
        "epoch": orbit.get("epoch"),
        "moid_au": sb.get("moid"),
    }
    processed_neos.append(record)

# =============================================
# SAVE COMPRESSED FILES
# =============================================
def save_gzip(data, path):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)

save_gzip(tles_json, os.path.join(OUTPUT_LATEST, "tles.json.gz"))
save_gzip(sat_info_json, os.path.join(OUTPUT_LATEST, "sat_info.json.gz"))
save_gzip(processed_neos, os.path.join(OUTPUT_LATEST, "neos.json.gz"))

print("\n📊 FINAL SUMMARY")
print("──────────────────────────────")
print(f"🛰️ SpaceTrack Objects: {len(df_tle)}")
print(f"☄️ NASA NEOs total: {len(processed_neos)}")
print(f"✅ Full Parallel Fetch Complete at {timestamp}")
