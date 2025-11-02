"""
🚀 SpaceTrack Data Fetcher (All Orbital Objects - Optimized)
------------------------------------------------------------
This script:
  • Fetches all active orbital objects (payloads, debris, rockets, unknown)
    with TLE age < 5 days
  • Generates:
      - data/latest/tles.json.br  → For satellite.js live propagation
      - data/latest/sat_info.json.br → For metadata popups
      - data/raw/<timestamped>.json.br → For historical records
  • Brotli compression for ultra-fast frontend loading
"""

import os
import requests
import pandas as pd
import json
import brotli
from datetime import datetime, timedelta
from collections import defaultdict

# =============================================
# CONFIG
# =============================================
USERNAME = os.getenv("SPACETRACK_USERNAME")
PASSWORD = os.getenv("SPACETRACK_PASSWORD")

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
OBJECT_TYPES = ["PAYLOAD", "ROCKET BODY", "DEBRIS", "UNKNOWN"]

BASE_TLE_URL = (
    "https://www.space-track.org/basicspacedata/query/class/tle_latest/"
    "ORDINAL/1/format/json"
)

OUTPUT_LATEST = "data/latest"
OUTPUT_RAW = "data/raw"
os.makedirs(OUTPUT_LATEST, exist_ok=True)
os.makedirs(OUTPUT_RAW, exist_ok=True)

TLE_AGE_DAYS = 5

# =============================================
# LOGIN
# =============================================
print("🌍 Logging into Space-Track...")
session = requests.Session()
resp = session.post(LOGIN_URL, data={"identity": USERNAME, "password": PASSWORD})
resp.raise_for_status()
print("✅ Logged in successfully")

# =============================================
# FETCH DATA
# =============================================
type_counts = defaultdict(int)
all_data = []

for obj_type in OBJECT_TYPES:
    print(f"📡 Fetching TLE data for {obj_type}...")
    url = f"{BASE_TLE_URL}/OBJECT_TYPE/{obj_type.replace(' ', '%20')}"
    try:
        resp = session.get(url)
        resp.raise_for_status()
        subset = resp.json()
        count = len(subset)
        type_counts[obj_type] = count
        print(f"   ↳ Retrieved {count} records")
        all_data.extend(subset)
    except Exception as e:
        print(f"⚠️ Failed for {obj_type}: {e}")

print(f"\n✅ Total {len(all_data)} objects retrieved across all categories")

# =============================================
# CLEAN + FILTER
# =============================================
df = pd.DataFrame(all_data)
df = df.dropna(subset=["TLE_LINE1", "TLE_LINE2", "OBJECT_NAME"])

df["EPOCH"] = pd.to_datetime(df["EPOCH"], errors="coerce")
cutoff = datetime.utcnow() - timedelta(days=TLE_AGE_DAYS)
df = df[df["EPOCH"] >= cutoff]

# Remove duplicates by NORAD_CAT_ID (some objects appear multiple times)
df = df.drop_duplicates(subset=["NORAD_CAT_ID"])
print(f"🛰️ Final dataset: {len(df)} valid objects (TLE age < {TLE_AGE_DAYS} days)")

# =============================================
# GENERATE JSON DATA
# =============================================
print("🧠 Generating JSON structures...")

# TLE JSON
tles_data = [
    {
        "id": int(row["NORAD_CAT_ID"]),
        "name": row["OBJECT_NAME"].strip(),
        "tle1": row["TLE_LINE1"].strip(),
        "tle2": row["TLE_LINE2"].strip(),
    }
    for _, row in df.iterrows()
]

# Info JSON
sat_info = {
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
    for _, row in df.iterrows()
}

# =============================================
# SAVE COMPRESSED FILES (.br)
# =============================================
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

# Save timestamped versions in /raw/
raw_tles_path = os.path.join(OUTPUT_RAW, f"tles_{timestamp}.json.br")
raw_info_path = os.path.join(OUTPUT_RAW, f"sat_info_{timestamp}.json.br")

with open(raw_tles_path, "wb") as f:
    f.write(brotli.compress(json.dumps(tles_data).encode("utf-8")))

with open(raw_info_path, "wb") as f:
    f.write(brotli.compress(json.dumps(sat_info).encode("utf-8")))

# Save/overwrite latest versions in /latest/
latest_tles = os.path.join(OUTPUT_LATEST, "tles.json.br")
latest_info = os.path.join(OUTPUT_LATEST, "sat_info.json.br")

for src, dst in [(raw_tles_path, latest_tles), (raw_info_path, latest_info)]:
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        fdst.write(fsrc.read())

# =============================================
# SUMMARY
# =============================================
tles_size = os.path.getsize(latest_tles) / 1024
info_size = os.path.getsize(latest_info) / 1024

print("\n📊 Summary by Object Type:")
for obj_type, count in type_counts.items():
    print(f"   • {obj_type:<12}: {count} objects")

print(f"\n✅ Files Saved:")
print(f"   • Latest TLEs → {latest_tles} ({tles_size:.2f} KB)")
print(f"   • Latest Info → {latest_info} ({info_size:.2f} KB)")
print(f"   • Archived raw data → data/raw/tles_{timestamp}.json.br & sat_info_{timestamp}.json.br")

print(f"\n🛰️ Total orbital objects processed: {len(df)}")
print("🚀 Done.")
