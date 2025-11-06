#!/usr/bin/env python3
"""
esa_fetch.py — ESA + JPL (SBDB + CAD fallback) threaded orbital fetcher (no cache)

- Fetches ESA CSVs (risk_list, upcoming, recent)
- Tries to fetch full orbital elements from JPL SBDB (sbdb.api) with smart name variants
- Falls back to JPL CAD API (cad.api) if SBDB doesn't provide full elements
- Tags each record with "incomplete_orbit": true/false
- Saves only one final GZIP JSON:
    data/latest/esa_neos_orbital_complete.json.gz
"""
import os
import csv
import json
import gzip
import time
import random
import requests
from io import StringIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ----------------- CONFIG -----------------
BASE_URL = "https://neo.ssa.esa.int/PSDB-portlet/download"
SBDB_API = "https://ssd-api.jpl.nasa.gov/sbdb.api"
CAD_API = "https://ssd-api.jpl.nasa.gov/cad.api"

LATEST_DIR = "data/latest"
os.makedirs(LATEST_DIR, exist_ok=True)

ENDPOINTS = {
    "risk_list": ["file=esa_risk_list"],
    "upcoming": ["file=esa_upcoming_close_app", "file=esa_closeapproaches"],
    "recent": ["file=esa_recent_close_app"]
}

MAX_WORKERS = 20
REQUEST_TIMEOUT = 15
RETRIES = 3
BACKOFF_BASE = 0.5  # seconds
ESSENTIAL_KEYS = ("a", "e", "i", "om", "w", "ma", "epoch")  # required for full orbit

# ----------------- ESA CSV -----------------
def fetch_esa_csv(param):
    url = f"{BASE_URL}?{param}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    text = r.text
    if not text.strip():
        return []
    first_line = text.splitlines()[0]
    delimiter = "|" if "|" in first_line else ","
    reader = csv.DictReader(StringIO(text), delimiter=delimiter)
    rows = []
    for row in reader:
        # strip whitespace from headers and values
        clean = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
        rows.append(clean)
    return rows

# ----------------- Name variants -----------------
def normalize_name(s):
    if not s:
        return None
    s = str(s)
    for token in ["(PHA)", "(NEA)", "(NEO)", "Asteroid", "asteroid", "—", "–", '"']:
        s = s.replace(token, "")
    s = s.replace("[", "").replace("]", "")
    s = " ".join(s.split())
    return s.strip()

def candidate_names_from_record(rec):
    """Return prioritized list of name variants to try for SBDB/CAD."""
    candidates = []
    possible_fields = ["Object", "Name", "Designation", "Asteroid designation", "Asteroid", "Target"]
    for f in possible_fields:
        if f in rec and rec[f]:
            n = normalize_name(rec[f])
            if n and n not in candidates:
                candidates.append(n)

    # Also attempt extracting tokens (numeric IDs, provisional designations)
    for val in rec.values():
        try:
            s = normalize_name(val)
            if not s:
                continue
            for tok in s.replace(",", " ").split():
                if tok.isdigit() and tok not in candidates:
                    candidates.append(tok)
            # provisional like '2004 MN4' or '2024 PT5'
            parts = s.split()
            for p in parts:
                if len(p) >= 4 and any(ch.isdigit() for ch in p) and any(ch.isalpha() for ch in p):
                    if p not in candidates:
                        candidates.append(p)
        except Exception:
            continue

    # Add underscore and nospace variants
    expanded = []
    for c in candidates:
        if c and c not in expanded:
            expanded.append(c)
        ns = c.replace(" ", "")
        us = c.replace(" ", "_")
        if ns and ns not in expanded:
            expanded.append(ns)
        if us and us not in expanded:
            expanded.append(us)
    return expanded

# ----------------- SBDB helpers -----------------
def sbdb_query(params):
    try:
        r = requests.get(SBDB_API, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def parse_sbdb_orbit(sbdb_json):
    """Return dict of orbit fields (keys from ESSENTIAL_KEYS plus extras) or None."""
    if not sbdb_json:
        return None

    # Possible places:
    # 1) top-level 'orbit' as dict, sometimes with 'elements' array inside
    # 2) 'object' -> 'orbit'
    orbit_src = None
    if isinstance(sbdb_json, dict):
        if sbdb_json.get("orbit"):
            orbit_src = sbdb_json["orbit"]
        elif sbdb_json.get("object", {}).get("orbit"):
            orbit_src = sbdb_json["object"]["orbit"]

    if not orbit_src:
        return None

    orbit_data = {}

    # If orbit_src has 'elements' as list, convert it
    if isinstance(orbit_src, dict) and "elements" in orbit_src and isinstance(orbit_src["elements"], list):
        for el in orbit_src["elements"]:
            # expect el to be dict with 'name' and 'value'
            name = el.get("name") or el.get("label")
            val = el.get("value")
            if name and val is not None:
                # normalize common names to short keys if possible
                key = name.strip()
                # many APIs return single-letter keys (a,e,i,om,w,ma) already
                # but sometimes return full names like 'eccentricity'. We'll attempt mapping.
                keymap = {
                    "eccentricity": "e",
                    "semimajor axis (au)": "a",
                    "semimajor axis": "a",
                    "semimajor axis (AU)": "a",
                    "inclination": "i",
                    "arg of perihelion": "w",
                    "long. of ascending node": "om",
                    "mean anomaly": "ma",
                    "epoch": "epoch",
                    "q": "q",
                    "ad": "ad",
                    "n": "n",
                }
                lk = keymap.get(key.lower(), key.lower())
                orbit_data[lk] = val
    # If orbit_src is dict with direct keys
    if isinstance(orbit_src, dict):
        for k, v in orbit_src.items():
            if k in ("a","e","i","om","w","ma","epoch","q","ad","n","tp"):
                orbit_data[k] = v
            # sometimes keys are full words
            if isinstance(k, str):
                kl = k.lower()
                if kl in ("semimajoraxis","semimajor_axis","semi_major_axis","semimajor axis"):
                    orbit_data["a"] = v
                if kl in ("eccentricity",):
                    orbit_data["e"] = v
                if kl in ("inclination",):
                    orbit_data["i"] = v
                if kl in ("longnode","longnode","longitude_of_ascending_node","ascending_node_longitude"):
                    orbit_data["om"] = v
                if kl in ("argperi","argument_of_perihelion","argument of perihelion"):
                    orbit_data["w"] = v
                if kl in ("mean_anomaly","mean anomaly"):
                    orbit_data["ma"] = v

    # Phys params sometimes under 'phys_par'
    phys = sbdb_json.get("phys_par") if isinstance(sbdb_json, dict) else None
    if phys and isinstance(phys, dict):
        if "H" in phys and phys["H"] is not None:
            orbit_data["H"] = phys["H"]

    if not orbit_data:
        return None
    return orbit_data

# ----------------- CAD fallback -----------------
def cad_query(params):
    try:
        r = requests.get(CAD_API, params=params, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def parse_cad_orbit(cad_json):
    # cad API primarily returns close approach records; some responses may include 'orbit' info
    # We'll attempt to extract orbit-like fields if present
    if not cad_json:
        return None
    # cad returns list under 'data' sometimes with fields; check first element for orbit keys
    # This is a best-effort attempt.
    if isinstance(cad_json, dict):
        # Some cad endpoints include 'orbit' at top-level for object-specific queries
        if cad_json.get("orbit"):
            return parse_sbdb_orbit({"orbit": cad_json.get("orbit")})
        # If 'data' exists, try keys inside the first row
        data = cad_json.get("data")
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            row = data[0]
            orbit = {}
            for k in ("a","e","i","om","w","ma","epoch","q","ad"):
                if k in row and row[k] not in (None,""):
                    orbit[k] = row[k]
            if orbit:
                return orbit
    return None

# ----------------- worker -----------------
def try_variants_for_record(variants):
    # Try des param first (precise)
    for v in variants:
        if not v:
            continue
        for attempt in range(RETRIES):
            sbdb = sbdb_query({"des": v})
            if sbdb:
                parsed = parse_sbdb_orbit(sbdb)
                if parsed:
                    # include fullname if present
                    fullname = None
                    if isinstance(sbdb.get("object"), dict):
                        fullname = sbdb["object"].get("fullname")
                    if fullname and "fullname" not in parsed:
                        parsed["fullname"] = fullname
                    return parsed, "SBDB(des)"
            time.sleep(BACKOFF_BASE * (2 ** attempt) * (0.3 + random.random()))
    # try sstr loose search
    for v in variants:
        if not v:
            continue
        for attempt in range(RETRIES):
            sbdb = sbdb_query({"sstr": v})
            if sbdb:
                parsed = parse_sbdb_orbit(sbdb)
                if parsed:
                    fullname = None
                    if isinstance(sbdb.get("object"), dict):
                        fullname = sbdb["object"].get("fullname")
                    if fullname and "fullname" not in parsed:
                        parsed["fullname"] = fullname
                    return parsed, "SBDB(sstr)"
            time.sleep(BACKOFF_BASE * (2 ** attempt) * (0.3 + random.random()))
    # CAD fallback (try des and sstr)
    for v in variants:
        try:
            cad = cad_query({"des": v})
            parsed = parse_cad_orbit(cad)
            if parsed:
                return parsed, "CAD(des)"
        except Exception:
            pass
        try:
            cad = cad_query({"sstr": v})
            parsed = parse_cad_orbit(cad)
            if parsed:
                return parsed, "CAD(sstr)"
        except Exception:
            pass
    return None, None

def worker_enrich(record):
    variants = candidate_names_from_record(record)
    if not variants:
        # nothing to query, mark incomplete
        out = dict(record)
        out["incomplete_orbit"] = True
        out["_orbit_source"] = None
        return out

    orbit_dict, source = try_variants_for_record(variants)
    out = dict(record)
    out["_matched_variants"] = variants[:4]
    if orbit_dict:
        out["orbit"] = orbit_dict
        out["_orbit_source"] = source
        # decide completeness
        incomplete = any(k not in orbit_dict or orbit_dict.get(k) in (None,"","null") for k in ESSENTIAL_KEYS)
        out["incomplete_orbit"] = bool(incomplete)
    else:
        out["orbit"] = {}
        out["_orbit_source"] = None
        out["incomplete_orbit"] = True
    return out

# ----------------- main -----------------
def main():
    print("🌍 [ESA + JPL] Full-orbit enrichment (threaded, no cache)")
    t0 = time.time()

    # fetch ESA datasets
    all_records = []
    for label, params in ENDPOINTS.items():
        print(f"🔹 Fetching ESA dataset: {label} ...")
        for p in params:
            try:
                rows = fetch_esa_csv(p)
                if rows:
                    for r in rows:
                        r["type"] = label
                    all_records.extend(rows)
                    print(f"   ↳ {len(rows)} rows fetched from '{p}'")
                    break
            except Exception as e:
                print(f"   ↳ attempt failed ({p}): {e}")
                continue

    total = len(all_records)
    print(f"\n📥 Total ESA rows collected: {total}")
    if total == 0:
        print("❌ No ESA data. Exiting.")
        return

    print("\n🛰️  Enriching with SBDB/CAD (threaded)...")
    enriched = []
    failures = 0
    matched_examples = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {exe.submit(worker_enrich, rec): rec for rec in all_records}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Enriching", ncols=90):
            try:
                res = fut.result()
                if res:
                    enriched.append(res)
                    if not res.get("incomplete_orbit") and len(matched_examples) < 6:
                        matched_examples.append(res)
                else:
                    failures += 1
            except Exception:
                failures += 1

    complete_count = sum(1 for r in enriched if not r.get("incomplete_orbit"))
    incomplete_count = sum(1 for r in enriched if r.get("incomplete_orbit"))
    print(f"\n🔭 Enrichment finished: total={len(enriched)}, complete={complete_count}, incomplete={incomplete_count}, failures={failures}")

    if matched_examples:
        print("\n📌 Sample COMPLETE matches:")
        for ex in matched_examples:
            name_guess = ex.get("Object") or ex.get("Name") or ex.get("Designation") or "<unknown>"
            source = ex.get("_orbit_source")
            keys = list(ex.get("orbit", {}).keys())
            print(f" - {name_guess}  (source={source}) -> orbit keys: {keys}")

    # save only entries (both complete and incomplete) but file is 'complete' name per user request
    out_name = "esa_neos_orbital_complete.json.gz"
    out_path = os.path.join(LATEST_DIR, out_name)
    with gzip.open(out_path, "wt", encoding="utf-8") as fh:
        json.dump(enriched, fh)

    print(f"\n💾 Saved full result to: {out_path}")
    print(f"⏱ Completed in {time.time() - t0:.2f} seconds")

if __name__ == "__main__":
    main()
