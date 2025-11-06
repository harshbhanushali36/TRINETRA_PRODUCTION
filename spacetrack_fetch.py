"""
🌌 SpaceTrack Data Fetcher
--------------------------
Fetches TLE and satellite data from Space-Track.org
"""

import os
import json
import gzip
import pandas as pd
import requests
from datetime import datetime, timedelta
from collections import defaultdict

# =============================================
# CONFIGURATION
# =============================================
SPACETRACK_USER = os.getenv("SPACETRACK_USERNAME")
SPACETRACK_PASS = os.getenv("SPACETRACK_PASSWORD")

SPACETRACK_LOGIN = "https://www.space-track.org/ajaxauth/login"
SPACETRACK_TLE_URL = "https://www.space-track.org/basicspacedata/query/class/tle_latest/ORDINAL/1/format/json"

OBJECT_TYPES = ["PAYLOAD", "ROCKET BODY", "DEBRIS", "UNKNOWN"]
TLE_AGE_DAYS = 5

OUTPUT_LATEST = "data/latest"
OUTPUT_RAW = "data/raw"
os.makedirs(OUTPUT_LATEST, exist_ok=True)
os.makedirs(OUTPUT_RAW, exist_ok=True)

timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
RAW_DIR = os.path.join(OUTPUT_RAW, timestamp)
os.makedirs(RAW_DIR, exist_ok=True)

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
# SAVE COMPRESSED FILES
# =============================================
def save_gzip(data, path):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)

def save_gzip_final(data, name):
    save_gzip(data, os.path.join(OUTPUT_LATEST, name))
    save_gzip(data, os.path.join(RAW_DIR, name))

save_gzip_final(tles_json, "tles.json.gz")
save_gzip_final(sat_info_json, "sat_info.json.gz")

# =============================================
# FINAL SUMMARY
# =============================================
print("\n📊 SPACETRACK SUMMARY")
print("──────────────────────────────")
print(f"🛰️ SpaceTrack Objects: {len(df_tle)}")
for obj_type, count in type_counts.items():
    print(f"   {obj_type}: {count}")
print(f"✅ SpaceTrack Fetch Complete at {timestamp}")