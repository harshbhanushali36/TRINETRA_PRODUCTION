#!/usr/bin/env python3
"""
spacetrack_to_czml_tle_age_filter.py

Improvements over previous:
 - Parses each TLE's epoch (from Satrec) and computes age (days).
 - Optionally skips TLEs older than TLE_MAX_AGE_DAYS (default 30).
 - Tests TLE by attempting a propagation at its own epoch (basic sanity check).
 - Then propagates valid TLEs to the visualization time grid.
 - Produces stats and writes clean CZML (no NaNs).
"""
import requests
import pandas as pd
import json
from sgp4.api import Satrec, jday
import numpy as np
from datetime import datetime, timedelta, timezone
import math
from tqdm import tqdm
from collections import Counter
import statistics

# -----------------------
# User / runtime params
# -----------------------
USERNAME = "harshbhanushali36@gmail.com"
PASSWORD = "Trinetra_12345678"
LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
CATALOG_URL = "https://www.space-track.org/basicspacedata/query/class/gp/format/json"

OUTPUT_EXCEL = "spacetrack_data.xlsx"
OUTPUT_CZML = "orbital_objects.czml"

DURATION_HOURS = 1           # how many hours from start to propagate
STEP_SECONDS = 60            # sample step
WANTED_TYPES = {"PAYLOAD", "ROCKET BODY", "DEBRIS"}

# NEW: TLE age filter (days). Set None to disable filtering.
TLE_MAX_AGE_DAYS = 30    # DEFAULT: 30 days. Increase to include older TLEs (less accurate, more errors)

# -----------------------
# Utility functions
# -----------------------
def gmst_from_jd(jd_ut1):
    T = (jd_ut1 - 2451545.0) / 36525.0
    gmst_sec = 67310.54841 + (876600.0 * 3600 + 8640184.812866) * T \
               + 0.093104 * T**2 - 6.2e-6 * T**3
    gmst_sec = gmst_sec % 86400.0
    return (gmst_sec / 86400.0) * 2.0 * math.pi

def rotate_z(vec, angle_rad):
    c = math.cos(angle_rad); s = math.sin(angle_rad)
    x, y, z = vec
    return (c*x - s*y, s*x + c*y, z)

def teme_to_ecef_km(r_teme_km, dt_utc):
    jd, jd_frac = jday(dt_utc.year, dt_utc.month, dt_utc.day,
                       dt_utc.hour, dt_utc.minute, dt_utc.second + dt_utc.microsecond*1e-6)
    jd_ut1 = jd + jd_frac
    gmst = gmst_from_jd(jd_ut1)
    return rotate_z(r_teme_km, gmst)

def build_czml_cartesian_array(positions_m, offsets_s):
    arr = []
    for off, (x, y, z) in zip(offsets_s, positions_m):
        arr.extend([off, float(x), float(y), float(z)])
    return arr

def satrec_epoch_to_datetime(sat):
    """
    Satrec.twoline2rv returns an object with attributes jdsatepoch and jdsatepochF
    Representing TLE epoch as Julian date = jdsatepoch + jdsatepochF
    Convert to timezone-aware UTC datetime.
    """
    try:
        jd = float(sat.jdsatepoch) + float(sat.jdsatepochF)
    except Exception:
        # fallback: some Satrec versions use sat.jdsatepoch and sat.jdsatepochF names; if missing, return None
        return None
    # Convert JD to datetime (UTC)
    # Algorithm: use astronomical conversion
    # Note: this conversion uses standard formula, result in UTC (no leap-second handling)
    jd += 0.5
    F, I = math.modf(jd)
    I = int(I)
    A = int((I - 1867216.25) / 36524.25)
    if I > 2299160:
        B = I + 1 + A - int(A / 4)
    else:
        B = I
    C = B + 1524
    D = int((C - 122.1) / 365.25)
    E = int(365.25 * D)
    G = int((C - E) / 30.6001)
    day = C - E + F - int(30.6001 * G)
    if G < 13.5:
        month = G - 1
    else:
        month = G - 13
    if month > 2.5:
        year = D - 4716
    else:
        year = D - 4715
    day_frac, day_int = math.modf(day)
    day_int = int(day_int)
    secs = day_frac * 86400.0
    hour = int(secs // 3600)
    minute = int((secs % 3600) // 60)
    second = secs - hour*3600 - minute*60
    microsecond = int((second - int(second)) * 1e6)
    second = int(second)
    try:
        dt = datetime(year, month, day_int, hour, minute, second, microsecond, tzinfo=timezone.utc)
    except Exception:
        return None
    return dt

# -----------------------
# 1) Fetch & Save (your working fetch)
# -----------------------
session = requests.Session()
login_payload = {"identity": USERNAME, "password": PASSWORD}
resp = session.post(LOGIN_URL, data=login_payload)
resp.raise_for_status()
print("✅ Logged into Space-Track successfully!")

response = session.get(CATALOG_URL)
response.raise_for_status()
data = response.json()
df = pd.DataFrame(data)

rename_map = {
    "OBJECT_NAME": "Name",
    "OBJECT_ID": "ID",
    "OBJECT_TYPE": "Type",
    "LAUNCH_DATE": "Launch Date",
    "PERIOD": "Period (min)",
    "INCLINATION": "Inclination (deg)",
    "APOGEE": "Apogee (km)",
    "PERIGEE": "Perigee (km)",
    "ECCENTRICITY": "Eccentricity",
    "MEAN_MOTION": "Mean Motion",
    "RA_OF_ASC_NODE": "RAAN (deg)",
    "ARG_OF_PERICENTER": "Arg Perigee (deg)",
    "TLE_LINE1": "TLE Line 1",
    "TLE_LINE2": "TLE Line 2"
}
df = df.rename(columns=rename_map)
columns_to_keep = list(rename_map.values())
df_filtered = df[[col for col in columns_to_keep if col in df.columns]]

df_satellites = df_filtered[df_filtered.get("Type") == "PAYLOAD"]
df_rockets = df_filtered[df_filtered.get("Type") == "ROCKET BODY"]
df_debris = df_filtered[df_filtered.get("Type") == "DEBRIS"]

with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Raw Data", index=False)
    df_filtered.to_excel(writer, sheet_name="Filtered Data", index=False)
    df_satellites.to_excel(writer, sheet_name="Satellites", index=False)
    df_rockets.to_excel(writer, sheet_name="Rockets", index=False)
    df_debris.to_excel(writer, sheet_name="Debris", index=False)

print(f"✅ Data saved to {OUTPUT_EXCEL} with TLEs and filtered sheets.")

# -----------------------
# 2) Prepare propagation grid & counters
# -----------------------
start_utc = datetime.now(timezone.utc)
end_utc = start_utc + timedelta(hours=DURATION_HOURS)
num_steps = int((end_utc - start_utc).total_seconds() // STEP_SECONDS) + 1
time_offsets = [i * STEP_SECONDS for i in range(num_steps)]
time_datetimes = [start_utc + timedelta(seconds=off) for off in time_offsets]
epoch_for_czml = start_utc.isoformat()

print(f"CZML epoch (UTC): {epoch_for_czml}")
print(f"Propagating from {start_utc.isoformat()} to {end_utc.isoformat()} "
      f"({DURATION_HOURS} hr, {num_steps} steps @ {STEP_SECONDS}s)")

# -----------------------
# 3) Build candidate list and compute TLE ages
# -----------------------
candidates = []
tle_ages_days = []
for idx, row in df_filtered.iterrows():
    obj_type = row.get("Type", "")
    tle1 = row.get("TLE Line 1") or row.get("TLE_LINE1") or row.get("TLE Line1")
    tle2 = row.get("TLE Line 2") or row.get("TLE_LINE2") or row.get("TLE Line2")
    if pd.isna(tle1) or pd.isna(tle2):
        continue
    if WANTED_TYPES and obj_type not in WANTED_TYPES:
        continue
    name = str(row.get("Name") or row.get("ID") or f"object_{idx}")
    candidates.append((name, tle1.strip(), tle2.strip(), obj_type))

print(f"Found {len(candidates)} TLE candidates matching types {WANTED_TYPES}.")

# -----------------------
# 4) Propagate with TLE-age filtering and sanity check
# -----------------------
czml = []
czml.append({
    "id": "document",
    "name": "Space-Track Objects (propagated, age-filtered)",
    "version": "1.0",
    "clock": {
        "interval": f"{start_utc.isoformat()}/{end_utc.isoformat()}",
        "currentTime": start_utc.isoformat(),
        "multiplier": 1,
        "range": "LOOP_STOP"
    }
})

objects_written = 0
objects_skipped_tle_age = 0
objects_skipped_no_valid = 0
sgp4_error_counts = Counter()
bad_examples = []

# For stats: collect ages
ages_list = []

for name, tle1, tle2, obj_type in tqdm(candidates, desc="Testing & propagating", unit="obj"):
    # Parse satrec
    try:
        sat = Satrec.twoline2rv(tle1, tle2)
    except Exception as e:
        # completely unparsable TLE
        objects_skipped_no_valid += 1
        if len(bad_examples) < 20:
            bad_examples.append({"name": name, "reason": f"parse_error: {e}"})
        continue

    # Determine TLE epoch datetime from sat.jdsatepoch + sat.jdsatepochF
    tle_epoch_dt = satrec_epoch_to_datetime(sat)
    if tle_epoch_dt is None:
        # can't find epoch - treat as bad
        objects_skipped_no_valid += 1
        if len(bad_examples) < 20:
            bad_examples.append({"name": name, "reason": "no_epoch_in_satrec"})
        continue

    # Age in days
    age_days = (start_utc - tle_epoch_dt).total_seconds() / 86400.0
    ages_list.append(age_days)

    # If max age set and TLE too old -> skip
    if (TLE_MAX_AGE_DAYS is not None) and (age_days > TLE_MAX_AGE_DAYS):
        objects_skipped_tle_age += 1
        if len(bad_examples) < 20:
            bad_examples.append({"name": name, "reason": f"old_tle_age={age_days:.1f}d"})
        continue

    # Sanity-check propagation at its own epoch (should give e==0 normally)
    jd_epoch, fr_epoch = jday(tle_epoch_dt.year, tle_epoch_dt.month, tle_epoch_dt.day,
                              tle_epoch_dt.hour, tle_epoch_dt.minute,
                              tle_epoch_dt.second + tle_epoch_dt.microsecond*1e-6)
    e0, r0, v0 = sat.sgp4(jd_epoch, fr_epoch)
    if e0 != 0:
        # TLE fails even at its own epoch -> skip
        sgp4_error_counts[e0] += 1
        objects_skipped_no_valid += 1
        if len(bad_examples) < 20:
            bad_examples.append({"name": name, "reason": f"sgp4_epoch_fail_code_{e0}"})
        continue

    # Propagate across time_datetimes, collecting valid samples only
    valid_samples = []
    for dt in time_datetimes:
        jd, fr = jday(dt.year, dt.month, dt.day,
                      dt.hour, dt.minute, dt.second + dt.microsecond*1e-6)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            sgp4_error_counts[e] += 1
            continue
        try:
            r_ecef_km = teme_to_ecef_km(r, dt)
            r_ecef_m = (r_ecef_km[0] * 1000.0, r_ecef_km[1] * 1000.0, r_ecef_km[2] * 1000.0)
            valid_samples.append((dt, r_ecef_m))
        except Exception as e:
            if len(bad_examples) < 20:
                bad_examples.append({"name": name, "reason": f"convert_error: {e}"})
            continue

    if not valid_samples:
        objects_skipped_no_valid += 1
        continue

    # Build CZML packet
    offsets_s = [(t - start_utc).total_seconds() for t, _ in valid_samples]
    positions_m = [p for _, p in valid_samples]
    packet = {
        "id": name.replace(" ", "_")[:64],
        "name": name,
        "availability": f"{start_utc.isoformat()}/{end_utc.isoformat()}",
        "properties": {"object_type": obj_type, "tle_age_days": float(f"{age_days:.2f}")},
        "position": {
            "epoch": epoch_for_czml,
            "cartesian": build_czml_cartesian_array(positions_m, offsets_s)
        },
        "point": {"pixelSize": 2}
    }
    czml.append(packet)
    objects_written += 1

# -----------------------
# 5) Output results & save CZML
# -----------------------
with open(OUTPUT_CZML, "w") as f:
    json.dump(czml, f, indent=2)

print("\n✅ CZML saved to:", OUTPUT_CZML)
print("Overview:")
print(" - Total TLE candidates:", len(candidates))
print(" - Objects written to CZML:", objects_written)
print(" - Skipped due to TLE age >", TLE_MAX_AGE_DAYS, "days:", objects_skipped_tle_age)
print(" - Skipped due to parse/propagation issues:", objects_skipped_no_valid)
print(" - sgp4 error counts (during propagation attempts):")
for code, cnt in sgp4_error_counts.most_common():
    print(f"    code {code}: {cnt}")

if ages_list:
    print("\nTLE age summary (days):")
    print(" - min:", min(ages_list))
    print(" - median:", statistics.median(ages_list))
    print(" - mean:", statistics.mean(ages_list))
    print(" - max:", max(ages_list))

if bad_examples:
    print("\nExamples of problematic entries (up to 20):")
    for ex in bad_examples[:20]:
        print(" ", ex)

print("\nNotes & next steps:")
print(" - If you still see many skips, try increasing TLE_MAX_AGE_DAYS (but huge ages produce inaccurate positions).")
print(" - If you want 'everything' visualized regardless of TLE validity, I can switch to placeholder orbits for skipped items.")
print(" - For higher accuracy TEME->ITRF conversion use astropy (slower).")
