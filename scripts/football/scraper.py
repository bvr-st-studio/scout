"""
scout football definitions scraper
-----------------------------------
Scrapes football terminology from Wikipedia and structures it
into Q&A training examples for scout-1-football-instruct.

Sources:
- Glossary of American football (Wikipedia HTML)
- Individual football terms via Wikipedia REST API

Usage:
    python3 scripts/football/scraper.py --output data/football/instruct/scout-1-football.jsonl
    python3 scripts/football/scraper.py --output data/football/instruct/scout-1-football.jsonl --dry-run
"""

import argparse
import json
import re
import time

import requests
from bs4 import BeautifulSoup

GLOSSARY_URL = "https://en.wikipedia.org/wiki/Glossary_of_American_football"
API_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary"
HEADERS = {"User-Agent": "ScoutScraper/1.0 (scout@bvr-st.com)"}

MIN_DEF_LEN = 80
MAX_DEF_LEN = 1200

SKIP_TERMS = {
    "football",
    "game",
    "player",
    "team",
    "season",
    "league",
    "nfl",
    "ncaa",
    "coach",
    "stadium",
    "field",
    "score",
    "point",
    "yard",
    "down",
    "half",
    "quarter",
    "overtime",
    "penalty",
}

# Football terms to fetch via Wikipedia API
FOOTBALL_TERMS = [
    # Offensive positions
    "Quarterback",
    "Running_back",
    "Fullback_(American_football)",
    "Wide_receiver",
    "Tight_end",
    "Center_(gridiron_football)",
    "Offensive_guard",
    "Offensive_tackle",
    "Slot_receiver",
    # Defensive positions
    "Defensive_end",
    "Defensive_tackle",
    "Nose_tackle",
    "Middle_linebacker",
    "Outside_linebacker",
    "Cornerback",
    "Free_safety",
    "Strong_safety",
    "Nickelback_(gridiron_football)",
    "Linebacker",
    # Special teams
    "Punter_(gridiron_football)",
    "Placekicker",
    "Long_snapper",
    "Kick_returner",
    # Offensive formations and schemes
    "Shotgun_formation",
    "I_formation",
    "West_Coast_offense",
    "Air_raid_offense",
    "Spread_offense",
    "Option_offense",
    "Wildcat_formation",
    "Pistol_formation",
    "Single_wing_formation",
    "T_formation",
    "Trips_formation",
    "Bunch_formation",
    # Offensive concepts
    "Play_action",
    "Screen_pass",
    "Draw_play",
    "Jet_sweep",
    "Flea_flicker",
    "Double_reverse",
    "Two-point_conversion",
    "Onside_kick",
    # Run plays
    "Inside_zone",
    "Outside_zone_(gridiron_football)",
    "Trap_play",
    # Defensive schemes
    "3–4_defense",
    "4–3_defense",
    "46_defense",
    "Nickel_defense",
    "Dime_defense",
    "Prevent_defense",
    "Tampa_2",
    "Cover_2",
    "Cover_3",
    "Cover_4",
    "Blitz_(gridiron_football)",
    "Zone_defense_(gridiron_football)",
    "Man-to-man_defense",
    "Press_coverage_(gridiron_football)",
    # Situational football
    "Red_zone_(gridiron_football)",
    "Two-minute_drill",
    "Hail_Mary_pass",
    "Prevent_defense",
    # General concepts
    "Formation_(American_football)",
    "Personnel_grouping_(gridiron_football)",
    "Down_(gridiron_football)",
    "Line_of_scrimmage",
    "Snap_(gridiron_football)",
    "Pass_rush",
    "Sack_(gridiron_football)",
    "Interception_(gridiron_football)",
    "Fumble",
    "Punt_(gridiron_football)",
    "Field_goal_(gridiron_football)",
    "Kickoff_(gridiron_football)",
    "Special_teams_(gridiron_football)",
    "Penalty_(gridiron_football)",
    "Turnover_(gridiron_football)",
]


def clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[edit\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_useful_definition(term: str, definition: str) -> bool:
    if len(definition) < MIN_DEF_LEN:
        return False
    if len(definition) > MAX_DEF_LEN:
        return False
    if term.lower() in SKIP_TERMS:
        return False
    if not any(c.isalpha() for c in definition):
        return False
    if definition.lower().startswith("see ") and len(definition) < 100:
        return False
    return True


def format_example(instruction: str, response: str) -> dict:
    text = f"### Instruction: {instruction}\n### Response: {response}"
    return {"text": text}


def scrape_glossary() -> list[dict]:
    print(f"\nScraping glossary: {GLOSSARY_URL}")
    examples = []

    try:
        resp = requests.get(GLOSSARY_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        content = soup.find("div", {"class": "mw-parser-output"})

        if not content:
            print("  Warning: could not find main content div")
            return examples

        for dt in content.find_all("dt"):
            term = clean_text(dt.get_text())
            if not term:
                continue

            definition_parts = []
            sibling = dt.find_next_sibling()
            while sibling and sibling.name == "dd":
                definition_parts.append(clean_text(sibling.get_text()))
                sibling = sibling.find_next_sibling()

            definition = " ".join(definition_parts)

            if is_useful_definition(term, definition):
                examples.append(
                    format_example(f"What is {term} in football?", definition)
                )

    except Exception as e:
        print(f"  Error: {e}")

    print(f"  Found {len(examples)} examples from glossary")
    return examples


def scrape_wikipedia_api() -> list[dict]:
    print(f"\nScraping {len(FOOTBALL_TERMS)} terms via Wikipedia API")
    examples = []
    skipped = 0

    for term in FOOTBALL_TERMS:
        try:
            url = f"{API_BASE}/{term}"
            resp = requests.get(url, headers=HEADERS, timeout=15)

            if resp.status_code != 200:
                skipped += 1
                continue

            data = resp.json()
            extract = data.get("extract", "").strip()
            title = data.get("title", term.replace("_", " "))

            # Clean up disambiguation suffixes from titles
            title = re.sub(r"\s*\(.*?\)", "", title).strip()

            if not extract or len(extract) < MIN_DEF_LEN:
                skipped += 1
                continue

            # Trim to max length at sentence boundary
            if len(extract) > MAX_DEF_LEN:
                extract = extract[:MAX_DEF_LEN].rsplit(".", 1)[0] + "."

            examples.append(format_example(f"What is {title} in football?", extract))

            time.sleep(0.5)

        except Exception as e:
            print(f"  Error on {term}: {e}")
            skipped += 1
            continue

    print(f"  Found {len(examples)} examples ({skipped} skipped)")
    return examples


def deduplicate(examples: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for ex in examples:
        key = ex["text"].split("\n### Response:")[0].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Scrape football definitions for Scout dataset"
    )
    parser.add_argument(
        "--output", required=True, help="Output JSONL file path (appends if exists)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print examples without writing"
    )
    args = parser.parse_args()

    all_examples = []

    glossary_examples = scrape_glossary()
    all_examples.extend(glossary_examples)

    api_examples = scrape_wikipedia_api()
    all_examples.extend(api_examples)

    all_examples = deduplicate(all_examples)
    print(f"\nTotal unique examples after deduplication: {len(all_examples)}")

    if args.dry_run:
        print("\n--- DRY RUN: First 5 examples ---")
        for ex in all_examples[:5]:
            print(json.dumps(ex, indent=2))
            print()
        print("--- Last 5 examples (API) ---")
        for ex in all_examples[-5:]:
            print(json.dumps(ex, indent=2))
            print()
        return

    existing = 0
    try:
        with open(args.output, "r") as f:
            existing = sum(1 for line in f if line.strip())
    except FileNotFoundError:
        pass

    with open(args.output, "a") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nDone.")
    print(f"  Existing examples: {existing}")
    print(f"  New examples added: {len(all_examples)}")
    print(f"  Total in dataset: {existing + len(all_examples)}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
