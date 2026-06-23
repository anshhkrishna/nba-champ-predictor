"""
build_dataset.py
================
Reproducible data pipeline for the NBA championship predictor.

Scrapes regular-season team / opponent / advanced stats from
Basketball-Reference, converts every stat to a league rank (1 = best for the
team), attaches each playoff team's actual playoff wins, and writes:

    past_league_rankings.csv   -> training data (all playoff teams, 2003..LAST_DONE)
    <YEAR>_prediction.csv      -> the current season, ready to predict (all 30 teams)

Re-run it once a year (bump CURRENT_SEASON) to keep the model up to date.

Usage:
    python build_dataset.py
"""
from __future__ import annotations
import html as _html
import re
import sys
import time
import urllib.request
from collections import defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd

# --- configuration -----------------------------------------------------------
FIRST_SEASON = 2003          # earliest season in the original dataset
LAST_COMPLETE_SEASON = 2025  # most recent season whose playoffs have finished
CURRENT_SEASON = 2026        # season to generate a prediction file for
CACHE = Path(__file__).parent / ".cache"
REQUEST_DELAY = 4.0          # seconds between requests (be polite to BBRef)
HEADERS = {"User-Agent": "Mozilla/5.0 (nba-champ-predictor data refresh)"}

# --- ranking direction -------------------------------------------------------
# A rank of 1 always means "best for this team".
# higher-is-better stats are ranked descending; lower-is-better ascending.
LOWER_IS_BETTER = {
    "TOV", "PF",                    # team per-game: fewer is better
    "O_MP", "O_FG", "O_FGA", "O_FG%", "O_3P", "O_3PA", "O_3P%",
    "O_2P", "O_2PA", "O_2P%", "O_FT", "O_FTA", "O_FT%", "O_ORB",
    "O_DRB", "O_TRB", "O_AST", "O_STL", "O_BLK", "O_PTS",  # opponent: fewer is better
    "L", "PL", "DRtg",              # advanced
    "TOV%", "O_eFG%", "O_FT/FGA",  # four factors (offense TOV%, defense eFG%/FT rate)
}
# Opponent turnovers / fouls you draw are GOOD, so they are higher-is-better
# (O_TOV, O_PF) and intentionally NOT in LOWER_IS_BETTER.

TEAM_PG_COLS = ["MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA",
                "2P%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL",
                "BLK", "TOV", "PF", "PTS"]
ADV_SIMPLE = ["W", "L", "PW", "PL", "MOV", "SOS", "SRS", "ORtg", "DRtg",
              "Pace", "FTr", "3PAr"]


def fetch(url: str) -> str:
    """Fetch a URL with on-disk caching and basic 429 back-off."""
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / (re.sub(r"[^A-Za-z0-9]", "_", url) + ".html")
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
            # BBRef ships many tables inside HTML comments
            raw = raw.replace("<!--", "").replace("-->", "")
            cache_file.write_text(raw, encoding="utf-8")
            time.sleep(REQUEST_DELAY)
            return raw
        except urllib.error.HTTPError as exc:  # type: ignore[attr-defined]
            if exc.code == 429:
                wait = REQUEST_DELAY * (attempt + 2)
                print(f"  rate limited, waiting {wait:.0f}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"giving up on {url}")


def read_table(raw_html: str, table_id: str) -> pd.DataFrame:
    m = re.search(rf'<table[^>]*id="{table_id}".*?</table>', raw_html, re.S)
    if not m:
        raise ValueError(f"table id {table_id!r} not found")
    df = pd.read_html(StringIO(m.group(0)))[0]
    df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    df = df[df["Team"].astype(str) != "League Average"].copy()
    df["Team"] = df["Team"].astype(str).str.replace("*", "", regex=False).str.strip()
    df = df[df["Team"].ne("nan") & df["Team"].ne("")]
    return df


def clean_advanced(adv: pd.DataFrame) -> pd.DataFrame:
    """Advanced table has duplicate eFG%/TOV%/FT/FGA columns (offense + defense)."""
    cols = list(adv.columns)
    out = pd.DataFrame({"Team": adv["Team"].values})
    for c in ADV_SIMPLE:
        out[c] = pd.to_numeric(adv[c], errors="coerce").values
    # four factors: offense block then defense block, by position
    ff = [i for i, c in enumerate(cols) if c in ("eFG%", "TOV%", "ORB%", "DRB%", "FT/FGA")]
    eFG = [i for i in ff if cols[i] == "eFG%"]
    tov = [i for i in ff if cols[i] == "TOV%"]
    ftr = [i for i in ff if cols[i] == "FT/FGA"]
    out["eFG%"] = pd.to_numeric(adv.iloc[:, eFG[0]], errors="coerce").values
    out["TOV%"] = pd.to_numeric(adv.iloc[:, tov[0]], errors="coerce").values
    out["ORB%"] = pd.to_numeric(adv[[c for c in cols if c == "ORB%"][0]], errors="coerce").values
    out["FT/FGA"] = pd.to_numeric(adv.iloc[:, ftr[0]], errors="coerce").values
    out["O_eFG%"] = pd.to_numeric(adv.iloc[:, eFG[1]], errors="coerce").values
    out["O_TOV%"] = pd.to_numeric(adv.iloc[:, tov[1]], errors="coerce").values
    out["DRB%"] = pd.to_numeric(adv[[c for c in cols if c == "DRB%"][0]], errors="coerce").values
    out["O_FT/FGA"] = pd.to_numeric(adv.iloc[:, ftr[1]], errors="coerce").values
    arena = [c for c in cols if c == "Arena"]
    out["Arena"] = adv[arena[0]].values if arena else ""
    attend = [c for c in cols if str(c).startswith("Attend")]
    out["Attendance_raw"] = (
        pd.to_numeric(adv[attend[0]], errors="coerce").values if attend else 0
    )
    return out


def to_rank(series: pd.Series, col: str) -> pd.Series:
    ascending = col in LOWER_IS_BETTER
    return series.rank(method="min", ascending=ascending).astype("Int64")


def playoff_wins(year: int, names: list[str]) -> dict[str, int]:
    raw = fetch(f"https://www.basketball-reference.com/playoffs/NBA_{year}.html")
    txt = re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", raw)))
    wins: dict[str, int] = defaultdict(int)
    for m in re.finditer(r" over .{0,40}?\((\d)-(\d)\)", txt):
        pre = txt[max(0, m.start() - 45):m.start()]
        post = txt[m.start() + len(" over "):m.end()]
        w, l = int(m.group(1)), int(m.group(2))
        winner = next((n for n in names if pre.rstrip().endswith(n)), None)
        loser = next((n for n in names if post.lstrip().startswith(n)), None)
        if winner and loser:
            wins[winner] += w
            wins[loser] += l
    return dict(wins)


def build_season(year: int) -> pd.DataFrame:
    """Return one row per team with every stat converted to a league rank."""
    league = fetch(f"https://www.basketball-reference.com/leagues/NBA_{year}.html")
    pg = read_table(league, "per_game-team").reset_index(drop=True)
    op = read_table(league, "per_game-opponent").reset_index(drop=True)
    adv = clean_advanced(read_table(league, "advanced-team"))

    # rank team and opponent stats within their own table, then join on Team
    df = pd.DataFrame({"Team": pg["Team"], "Year": year})
    for c in TEAM_PG_COLS:
        df[c] = to_rank(pd.to_numeric(pg[c], errors="coerce"), c).values
    op_ranks = pd.DataFrame({"Team": op["Team"]})
    for c in TEAM_PG_COLS:
        op_ranks["O_" + c] = to_rank(pd.to_numeric(op[c], errors="coerce"), "O_" + c).values
    df = df.merge(op_ranks, on="Team", how="left")
    df = df.merge(adv, on="Team", how="left")
    rank_adv = ADV_SIMPLE + ["eFG%", "TOV%", "ORB%", "FT/FGA",
                             "O_eFG%", "O_TOV%", "DRB%", "O_FT/FGA"]
    for c in rank_adv:
        df[c] = to_rank(df[c], c)
    df["Attendance"] = to_rank(df["Attendance_raw"], "Attendance")
    df = df.drop(columns=["Attendance_raw"])
    return df


def main() -> None:
    here = Path(__file__).parent

    # ---- training data: every completed season -----------------------------
    rows = []
    for year in range(FIRST_SEASON, LAST_COMPLETE_SEASON + 1):
        print(f"[train] {year} ...", file=sys.stderr)
        season = build_season(year)
        wins = playoff_wins(year, list(season["Team"]))
        season = season[season["Team"].isin(wins)].copy()  # playoff teams only
        season["Playoff Wins"] = season["Team"].map(wins)
        season["Rk"] = season["Playoff Wins"].rank(method="first", ascending=False).astype(int)
        rows.append(season)
    train = pd.concat(rows, ignore_index=True)
    front = ["Rk", "Team", "Year", "Playoff Wins"]
    train = train[front + [c for c in train.columns if c not in front]]
    train = train.sort_values(["Year", "Rk"]).reset_index(drop=True)
    train.to_csv(here / "past_league_rankings.csv", index=False)
    print(f"wrote past_league_rankings.csv ({len(train)} rows, "
          f"{FIRST_SEASON}-{LAST_COMPLETE_SEASON})")

    # ---- prediction data: the current season (all 30 teams) ----------------
    print(f"[predict] {CURRENT_SEASON} ...", file=sys.stderr)
    cur = build_season(CURRENT_SEASON)
    cur.insert(0, "Playoff Wins", pd.NA)
    cur.insert(0, "Rk", pd.NA)
    out = here / f"{CURRENT_SEASON}_prediction.csv"
    cur.to_csv(out, index=False)
    print(f"wrote {out.name} ({len(cur)} teams)")


if __name__ == "__main__":
    main()
