"""Downloads from CollegeFootballData, The Odds API, and Open-Meteo.

CFBD's free tier allows 1,000 calls a month, so every request pulls a whole
season at once and past seasons are cached forever in data/raw/.
"""
import json
import time
from datetime import datetime, timezone

import requests

import cfb_config as C


def _cache_path(name):
    return C.RAW / f"{name}.json"


def _load(name):
    p = _cache_path(name)
    return json.loads(p.read_text()) if p.exists() else None


def _save(name, obj):
    C.RAW.mkdir(parents=True, exist_ok=True)
    _cache_path(name).write_text(json.dumps(obj))


def cfbd(endpoint, params, name, refresh=False):
    """Fetch a CFBD endpoint, using the cache unless refresh=True."""
    cached = _load(name)
    if cached is not None and (not refresh or C.OFFLINE):
        return cached
    if C.OFFLINE:
        return []
    if not C.CFBD_KEY:
        raise SystemExit("CFBD_API_KEY is not set.")
    r = requests.get(
        f"{C.CFBD_BASE}{endpoint}",
        params=params,
        headers={"Authorization": f"Bearer {C.CFBD_KEY}", "Accept": "application/json"},
        timeout=120,
    )
    if r.status_code == 400 and params.get("seasonType") == "both":
        # Fallback if an endpoint doesn't accept seasonType=both.
        out = []
        for st in ("regular", "postseason"):
            out += cfbd(endpoint, {**params, "seasonType": st}, f"{name}_{st}", refresh)
        _save(name, out)
        return out
    r.raise_for_status()
    data = r.json()
    _save(name, data)
    time.sleep(0.3)
    return data


def season_data(year, refresh=False):
    """Everything the model needs for one season (5 calls, cached)."""
    return {
        "games": cfbd("/games", {"year": year, "seasonType": "both"}, f"games_{year}", refresh),
        "adv": cfbd(
            "/stats/game/advanced",
            {"year": year, "seasonType": "both", "excludeGarbageTime": "true"},
            f"adv_{year}",
            refresh,
        ),
        "lines": cfbd("/lines", {"year": year, "seasonType": "both"}, f"lines_{year}", refresh),
        "talent": cfbd("/talent", {"year": year}, f"talent_{year}"),
        "returning": cfbd("/player/returning", {"year": year}, f"returning_{year}"),
    }


def venues():
    return cfbd("/venues", {}, "venues")


def odds_snapshot():
    """Current spreads and totals for every listed NCAAF game (2 Odds API credits)."""
    if C.OFFLINE:
        return _load("odds_latest") or []
    if not C.ODDS_KEY:
        print("ODDS_API_KEY not set; skipping line snapshot.")
        return []
    r = requests.get(
        C.ODDS_URL,
        params={
            "apiKey": C.ODDS_KEY,
            "regions": "us",
            "markets": "spreads,totals",
            "oddsFormat": "american",
        },
        timeout=60,
    )
    if r.status_code != 200:
        print(f"Odds API returned {r.status_code}: {r.text[:200]}")
        return []
    print("Odds API credits remaining:", r.headers.get("x-requests-remaining"))
    data = r.json()
    _save("odds_latest", data)
    return data


def weather(points):
    """Hourly wind/precip forecasts for a list of (lat, lon). One request per 50 venues."""
    if C.OFFLINE or not points:
        return [None] * len(points)
    out = []
    for i in range(0, len(points), 50):
        chunk = points[i : i + 50]
        try:
            r = requests.get(
                C.WEATHER_URL,
                params={
                    "latitude": ",".join(f"{p[0]:.4f}" for p in chunk),
                    "longitude": ",".join(f"{p[1]:.4f}" for p in chunk),
                    "hourly": "wind_speed_10m,precipitation,temperature_2m",
                    "wind_speed_unit": "mph",
                    "temperature_unit": "fahrenheit",
                    "precipitation_unit": "inch",
                    "timezone": "UTC",
                    "forecast_days": 8,
                },
                timeout=60,
            )
            r.raise_for_status()
            js = r.json()
            out += js if isinstance(js, list) else [js]
        except Exception as e:  # weather is optional; never fail the run over it
            print("Weather fetch failed:", e)
            out += [None] * len(chunk)
    return out


def now():
    return datetime.now(timezone.utc)
