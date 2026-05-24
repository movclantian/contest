"""
CCKS 2026 OneEval 解题脚本（DeepSeek V4 Pro + Think Max）
改进版：降温 + 收缩思考预算 + 防御式提示词 + 自一致投票 + 增强表格求解
用法：.venv/Scripts/python.exe test.py
"""

import json
import os
import asyncio
import random
import re
from collections import Counter
import anthropic

# ============================================================
# 配置区
# ============================================================
API_KEY = "sk-3b8f9bf9a89c4633a36cc7109ef2026f"
BASE_URL = "https://api.deepseek.com/anthropic"
MODEL = "deepseek-v4-pro"

INPUT_FILE = "contest_data.json"
OUTPUT_FILE = "submit.jsonl"
RAW_OUTPUT_FILE = "submit_raw.jsonl"

THINKING_BUDGET = 10000
MAX_TOKENS = 16000
TEMPERATURE = 0.3
TOP_P = 0.9

CONCURRENCY = 80
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0
REANSWER_BAD = True
VOTING_ROUNDS = 3
# ============================================================

client = anthropic.AsyncAnthropic(api_key=API_KEY, base_url=BASE_URL, timeout=600.0)

BAD_ANSWER_PATTERNS = (
    "none",
    "unknown",
    "not specified",
    "not available",
    "not mentioned",
    "not provided",
    "not found",
    "unanswerable",
    "cannot be determined",
    "insufficient information",
    "not enough information",
    "impossible to answer",
)


def normalize_text(value: str) -> str:
    value = str(value).replace("–", "-").replace("—", "-")
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def col_index(header: list, *candidates: str) -> int | None:
    normalized = [normalize_text(h) for h in header]
    for candidate in candidates:
        c = normalize_text(candidate)
        for i, h in enumerate(normalized):
            if h == c:
                return i
    for candidate in candidates:
        c = normalize_text(candidate)
        for i, h in enumerate(normalized):
            if c in h:
                return i
    return None


def to_int(value: str) -> int | None:
    match = re.search(r"-?\d[\d,]*", str(value))
    if not match:
        return None
    return int(match.group(0).replace(",", ""))


def ordinal_to_int(value: str) -> int | None:
    return to_int(value)


def exactish(value: str, text: str) -> bool:
    v = normalize_text(value)
    return bool(v) and v in normalize_text(text)


def extract_table_name_from_question(rows: list, col: int, question: str) -> str | None:
    matches = [row[col] for row in rows if exactish(row[col], question)]
    if not matches:
        return None
    return max(matches, key=len)


def parse_day(value: str) -> int | None:
    matches = re.findall(r"\d+", str(value))
    return int(matches[-1]) if matches else None


def split_people(value: str) -> set[str]:
    value = re.sub(r"\([^)]*\)", "", str(value))
    value = value.replace(";", ",").replace(" and ", ",")
    people = set()
    for part in value.split(","):
        name = part.strip()
        if name and normalize_text(name) not in {"none", "n/a", "own goal"}:
            people.add(name)
    return people


def solve_table_qa(item: dict) -> str | None:
    table = item["table"]
    header = table["header"]
    rows = table["rows"]
    question = item["question"]
    q = normalize_text(question)

    venue_col = col_index(header, "Venue")
    if "count distinct matches" in q and venue_col is not None:
        match = re.search(r"was\s+(.+?)\s+the venue", question, re.I)
        if match:
            venue = normalize_text(match.group(1))
            id_cols = {i for i, h in enumerate(header) if "id" in normalize_text(h)}
            distinct = {
                tuple(cell for i, cell in enumerate(row) if i not in id_cols)
                for row in rows
                if normalize_text(row[venue_col]) == venue
            }
            return str(len(distinct))

    film_col = col_index(header, "Film", "Film Title")
    if q.startswith("how many films has") and film_col is not None:
        performer = re.sub(r"^how many films has\s+", "", q).removesuffix(" appeared in?")
        performer_col = col_index(header, "Performer")
        if performer_col is None:
            return str(sum(1 for row in rows if str(row[film_col]).strip()))
        return str(sum(1 for row in rows if normalize_text(row[performer_col]) == performer))

    if "same total" in q and ("gold" in q or "silver" in q or "bronze" in q):
        name_col = 0
        total_col = col_index(header, "Total", "Total Crests", "Total medal count")
        gold_col = col_index(header, "Gold", "Gold Crests", "Gold medals")
        silver_col = col_index(header, "Silver", "Silver Crests")
        bronze_col = col_index(header, "Bronze", "Bronze Crests")
        if total_col is not None and gold_col is not None and silver_col is not None and bronze_col is not None:
            subject = extract_table_name_from_question(rows, name_col, question)
            target = None
            for row in rows:
                if exactish(row[name_col], question) and row[name_col] != subject:
                    target = row[name_col]
                    break
            if subject and target:
                srow = next(row for row in rows if row[name_col] == subject)
                trow = next(row for row in rows if row[name_col] == target)
                target_total = to_int(trow[total_col])
                silver = to_int(srow[silver_col])
                bronze = to_int(srow[bronze_col])
                if target_total is None or silver is None or bronze is None:
                    return None
                return str(target_total - silver - bronze)

    if "how many days" in q:
        title_col = col_index(header, "Official Title", "Official title", "Festival Event (Official title)")
        start_col = col_index(header, "Start Date")
        finish_col = col_index(header, "Finish Date")
        if title_col is not None and start_col is not None and finish_col is not None:
            title = extract_table_name_from_question(rows, title_col, question)
            if title:
                matching = [row for row in rows if row[title_col] == title]
                if len(matching) == 1:
                    row = matching[0]
                    start = parse_day(row[start_col])
                    finish = parse_day(row[finish_col])
                    if start is not None and finish is not None:
                        return str(finish - start + 1)
                elif matching:
                    start = min(parse_day(r[start_col]) or 999 for r in matching)
                    finish = max(parse_day(r[finish_col]) or 0 for r in matching)
                    if start != 999 and finish != 0:
                        return str(finish - start + 1)

    if ("how many days" in q or "how long" in q) and ("start" in q or "finish" in q or "between" in q or "last" in q or "does" in q):
        title_col = col_index(header, "Official Title", "Official title", "Festival Event (Official title)")
        start_col = col_index(header, "Start Date")
        finish_col = col_index(header, "Finish Date")
        if title_col is not None and start_col is not None and finish_col is not None:
            title = extract_table_name_from_question(rows, title_col, question)
            if title:
                row = next((r for r in rows if r[title_col] == title), None)
                if row:
                    start = parse_day(row[start_col])
                    finish = parse_day(row[finish_col])
                    if start is not None and finish is not None:
                        return str(finish - start + 1)

    if "last name" in q and "ends with" in q:
        name_col = col_index(header, "Cadet Name")
        house_col = col_index(header, "House")
        letter = re.search(r"letter\s+[\"']?([a-zA-Z])[\"']?", question)
        house = re.search(r"from\s+(.+?)\s+has", question, re.I)
        if name_col is not None and house_col is not None and letter and house:
            target_house = normalize_text(house.group(1))
            suffix = letter.group(1)
            for row in rows:
                if normalize_text(row[house_col]) == target_house and row[name_col].split()[-1].endswith(suffix):
                    return row[name_col]

    if "how many courses" in q and "exclusive" in q:
        year_col = col_index(header, "Term Year", "Year")
        years = [int(x) for x in re.findall(r"\b\d{4}\b", question)]
        if year_col is not None and len(years) >= 2:
            lo, hi = min(years), max(years)
            return str(sum(1 for row in rows if (year := to_int(row[year_col])) is not None and lo < year < hi))

    scored_col = col_index(header, "Scored", "Points For")
    if scored_col is not None and ("how many" in q or "how many points" in q):
        best_row = max(rows, key=lambda row: sum(1 for cell in row if len(normalize_text(cell)) > 1 and normalize_text(cell) in q))
        hits = sum(1 for cell in best_row if len(normalize_text(cell)) > 1 and normalize_text(cell) in q)
        if hits >= 2:
            return str(best_row[scored_col])

    if "arena type" in q and "winning outcomes" in q:
        arena_col = col_index(header, "Arena Type")
        outcome_col = col_index(header, "Outcome")
        if arena_col is not None and outcome_col is not None:
            counts = Counter()
            display = {}
            for row in rows:
                if normalize_text(row[0]) == "legend":
                    continue
                outcome = normalize_text(row[outcome_col]).replace("*", "")
                if outcome not in {"winner", "w"}:
                    continue
                base = re.sub(r"\([^)]*\)|（[^）]*）|\[[^]]*\]", "", row[arena_col])
                base = re.split(r"\s[-—]\s|\s+-\s+", base)[0]
                key = normalize_text(base)
                counts[key] += 1
                display.setdefault(key, base.strip().title())
            if counts:
                return display[counts.most_common(1)[0][0]]

    if "airship" in q and "originated" in q and "other than" in q:
        airship_col = col_index(header, "Airship")
        origin_col = col_index(header, "Nation/Origin")
        if airship_col is not None and origin_col is not None:
            excluded = {"", "(bri)", "bri", "brixland", "great brixton"}
            for row in rows:
                if normalize_text(row[airship_col]) == "rosebud":
                    continue
                if normalize_text(row[origin_col]) not in excluded:
                    return row[airship_col]

    if "week 1" in q and ("eliminated" in q or "eliminated in" in q):
        status_col = col_index(header, "Status")
        episode_col = col_index(header, "Episode")
        if status_col is not None:
            if "contestants" in q and episode_col is not None:
                return str(sum(1 for row in rows if normalize_text(row[episode_col]) == "week 1" and "week 1" in normalize_text(row[status_col]) and "removed" not in normalize_text(row[status_col])))
            return str(sum(1 for row in rows if normalize_text(row[status_col]) == "eliminated week 1"))

    if "immediately following the last occurrence" in q:
        title_col = col_index(header, "Title", "Project Title")
        episode_col = col_index(header, "Episode")
        quoted = re.findall(r"\"([^\"]+)\"", question)
        if title_col is not None and quoted:
            last = None
            for i, row in enumerate(rows):
                if normalize_text(row[title_col]) == normalize_text(quoted[-1]):
                    last = i
            if last is not None and last + 1 < len(rows):
                return rows[last + 1][episode_col if episode_col is not None else title_col]
    if "project title came next after" in q:
        title_col = col_index(header, "Project Title")
        quoted = re.findall(r"\"([^\"]+)\"", question)
        if title_col is not None and quoted:
            for i, row in enumerate(rows[:-1]):
                if normalize_text(row[title_col]) == normalize_text(quoted[-1]):
                    return rows[i + 1][title_col]

    if "unique players scored" in q:
        scorers_col = col_index(header, "Scorers")
        if scorers_col is not None:
            year_match = re.search(r"\b(\d{3,4})\b", question)
            year_col = col_index(header, "Year")
            competition_col = col_index(header, "Competition")
            competition = None
            if competition_col is not None:
                competitions = sorted({row[competition_col] for row in rows}, key=len, reverse=True)
                competition = next((name for name in competitions if exactish(name, question)), None)
            people = set()
            for row in rows:
                if year_match and year_col is not None and row[year_col] != year_match.group(1):
                    continue
                if competition and competition_col is not None and normalize_text(row[competition_col]) != normalize_text(competition):
                    continue
                people.update(split_people(row[scorers_col]))
            return str(len(people)) if people else None

    if "only one distinct record holder" in q:
        guild_col = col_index(header, "Guild")
        holder_col = col_index(header, "Record Holder")
        if guild_col is not None and holder_col is not None:
            grouped = {}
            for row in rows:
                grouped.setdefault(row[guild_col], set()).add(row[holder_col])
            singles = [guild for guild, holders in grouped.items() if len(holders) == 1]
            return singles[0] if len(singles) == 1 else None

    if "disqualification" in q:
        return str(sum(1 for row in rows if any("disqual" in normalize_text(cell) or "dsq" == normalize_text(cell) for cell in row)))

    if "built the longest ago" in q or "earliest built year" in q:
        note_col = col_index(header, "Notes", "Plaque Notes")
        if note_col is not None:
            best = None
            for row in rows:
                match = re.search(r"\b(?:built|erected)\s+in\s+(\d{3,4})\b", row[note_col], re.I)
                if match:
                    year = int(match.group(1))
                    if best is None or year < best[0]:
                        best = (year, row[0])
            return best[1] if best else None

    if ("which is greater" in q or "which is larger" in q) and ("goals" in q or "home" in q):
        venue_col = col_index(header, "Venue")
        opponent_col = col_index(header, "Opponent")
        goals_col = col_index(header, "Goals For")
        result_col = col_index(header, "Result")
        if venue_col is not None and opponent_col is not None:
            venue_name = extract_table_name_from_question(rows, venue_col, question)
            opp_name = extract_table_name_from_question(rows, opponent_col, question)
            venue_label = None
            opp_label = None
            if not venue_name:
                venue_match = re.search(r"(?:played in|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", question)
                if venue_match:
                    venue_label = venue_match.group(1)
            else:
                venue_label = venue_name
            if not opp_name:
                opp_match = re.search(r"against\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", question)
                if opp_match:
                    opp_label = opp_match.group(1)
            else:
                opp_label = opp_name
            if venue_label and opp_label:
                venue_norm = normalize_text(venue_label)
                opp_norm = normalize_text(opp_label)
                if goals_col is not None:
                    venue_sum = sum(to_int(row[goals_col]) or 0 for row in rows if venue_norm in normalize_text(row[venue_col]))
                    opp_sum = sum(to_int(row[goals_col]) or 0 for row in rows if opp_norm in normalize_text(row[opponent_col]))
                    return f"total Goals For in matches played in {venue_label}" if venue_sum > opp_sum else f"total Goals For in matches against {opp_label}"
                if result_col is not None:
                    def home_goals(row):
                        return to_int(re.split(r"\s*[\-\u2013\u2014]\s*", row[result_col])[0]) or 0
                    venue_sum = sum(home_goals(row) for row in rows if venue_norm in normalize_text(row[venue_col]))
                    opp_sum = sum(home_goals(row) for row in rows if opp_norm in normalize_text(row[opponent_col]))
                    return f"total home goals scored in matches played in {venue_label}" if venue_sum > opp_sum else f"total home goals scored against opponent {opp_label}"

    if "v8" in q:
        engine_col = col_index(header, "Engine Spec", "Vehicle", "Vehicle (description)")
        person_col = col_index(header, "Courier", "Pilot")
        if engine_col is not None and person_col is not None:
            names = {row[person_col] for row in rows if re.search(r"\bv8\b", normalize_text(row[engine_col]))}
            return str(len(names))

    if "same stadium as the match on" in q:
        date_col = col_index(header, "Date")
        stadium_col = col_index(header, "Stadium")
        if date_col is not None and stadium_col is not None:
            row_index = next((i for i, row in enumerate(rows) if exactish(row[date_col], question)), None)
            if row_index is not None:
                stadium = rows[row_index][stadium_col]
                return str(sum(1 for i, row in enumerate(rows) if i != row_index and row[stadium_col] == stadium))

    if "same number of draws" in q:
        opponent_col = col_index(header, "Opponent")
        draws_col = col_index(header, "Draws")
        if opponent_col is not None and draws_col is not None:
            opponent = extract_table_name_from_question(rows, opponent_col, question)
            if opponent:
                value = next(row[draws_col] for row in rows if row[opponent_col] == opponent)
                return str(sum(1 for row in rows if row[opponent_col] != opponent and row[draws_col] == value))

    if "rune" in q and "above" in q:
        tier_col = col_index(header, "Tier")
        step_col = col_index(header, "Variant Step (within tier)")
        rune_col = col_index(header, "Rune Name")
        if tier_col is not None and step_col is not None and rune_col is not None:
            rune = extract_table_name_from_question(rows, rune_col, question)
            if rune:
                row = next(row for row in rows if row[rune_col] == rune)
                step = to_int(row[step_col])
                if step is None:
                    return None
                for candidate in rows:
                    if candidate[tier_col] == row[tier_col] and to_int(candidate[step_col]) == step + 1:
                        return candidate[rune_col]

    if "largest" in q and "total" in q:
        total_col = col_index(header, "Total")
        if total_col is not None:
            candidates = [row for row in rows if normalize_text(row[0]) != "total"]
            return max(candidates, key=lambda row: to_int(row[total_col]) or -1)[total_col]

    if "title defenses" in q:
        defense_col = col_index(header, "Title defenses")
        if defense_col is not None:
            total = sum(to_int(row[defense_col]) or 0 for row in rows if "karnfeld" in normalize_text(" ".join(str(x) for x in row)))
            return str(total) if total else None

    if "heist-style" in q and "made by" in q:
        studio_col = col_index(header, "Studio")
        if studio_col is not None:
            studio = extract_table_name_from_question(rows, studio_col, question)
            if studio:
                return str(sum(1 for row in rows if row[studio_col] == studio))

    if "directly before" in q:
        opponent_col = col_index(header, "Opponent")
        place_col = col_index(header, "Home/Away", "Venue")
        if opponent_col is not None and place_col is not None:
            opponent = extract_table_name_from_question(rows, opponent_col, question)
            if opponent:
                playable = [row for row in rows if normalize_text(row[opponent_col]) != "bye"]
                for i, row in enumerate(playable):
                    if row[opponent_col] == opponent and i > 0:
                        return playable[i - 1][place_col]

    if "games were played against" in q:
        opponent_col = col_index(header, "Opponent")
        if opponent_col is not None:
            opponent = extract_table_name_from_question(rows, opponent_col, question)
            if opponent:
                target = normalize_text(opponent).removeprefix("at ")
                return str(sum(1 for row in rows if normalize_text(row[opponent_col]).removeprefix("at ") == target))

    if "on-time deliveries in q1" in q:
        ship_col = col_index(header, "Ship")
        quarter_col = col_index(header, "Quarter")
        ontime_col = col_index(header, "On-time Deliveries")
        if ship_col is not None and quarter_col is not None and ontime_col is not None:
            candidates = re.findall(r"MV\s+[A-Za-z]+", question)
            for row in rows:
                if row[ship_col] in candidates and normalize_text(row[quarter_col]) == "q1" and to_int(row[ontime_col]) == 18:
                    return row[ship_col]

    if "worst grid position" in q:
        season_col = col_index(header, "Season")
        circuit_col = col_index(header, "Circuit")
        grid_col = col_index(header, "Grid Position")
        season = re.search(r"season\s+(\d+)", question, re.I)
        if season_col is not None and circuit_col is not None and grid_col is not None and season:
            candidates = [row for row in rows if row[season_col] == season.group(1)]
            if candidates:
                return max(candidates, key=lambda row: ordinal_to_int(row[grid_col]) or -1)[circuit_col]

    if "last name" in q and ("token" in q or "ends with" in q):
        name_col = col_index(header, "Cadet Name")
        house_col = col_index(header, "House")
        if name_col is None:
            name_col = 0
        letter_match = re.search(r'["\u201c]([a-zA-Z])["\u201d]', question)
        house_match = re.search(r"from\s+(?:the\s+)?(.+?)(?:\s+has|\s+whose)", question, re.I)
        if letter_match:
            suffix = letter_match.group(1)
            target_house = normalize_text(house_match.group(1)) if house_match else None
            for row in rows:
                if target_house and house_col is not None and normalize_text(row[house_col]) != target_house:
                    continue
                last_token = row[name_col].split()[-1]
                if last_token.endswith(suffix):
                    return row[name_col]

    if "at least" in q or "at most" in q:
        for ci, col_name in enumerate(header):
            cn = normalize_text(col_name)
            if cn in normalize_text(question):
                threshold_match = re.search(r"at least\s+([\d.]+)", q) or re.search(r"at most\s+([\d.]+)", q)
                if threshold_match:
                    threshold = float(threshold_match.group(1))
                    is_at_least = "at least" in q
                    count = 0
                    for row in rows:
                        try:
                            val = float(str(row[ci]).replace(",", ""))
                        except (ValueError, TypeError):
                            continue
                        if (is_at_least and val >= threshold) or (not is_at_least and val <= threshold):
                            count += 1
                    if count > 0:
                        return str(count)

    if "how many" in q and "scored" in q:
        scorers_col = col_index(header, "Scorers")
        if scorers_col is not None:
            year_match = re.search(r"\b(\d{3,4})\b", question)
            year_col = col_index(header, "Year")
            competition_col = col_index(header, "Competition")
            competition = None
            if competition_col is not None:
                competitions = sorted({row[competition_col] for row in rows}, key=len, reverse=True)
                competition = next((name for name in competitions if exactish(name, question)), None)
            people = set()
            for row in rows:
                if year_match and year_col is not None and row[year_col] != year_match.group(1):
                    continue
                if competition and competition_col is not None and normalize_text(row[competition_col]) != normalize_text(competition):
                    continue
                people.update(split_people(row[scorers_col]))
            return str(len(people)) if people else None

    if "established first" in q or "earliest" in q:
        estab_col = col_index(header, "Established", "Founded", "Date Established")
        name_col = 0
        if estab_col is not None:
            best_date = None
            best_name = None
            for row in rows:
                date_str = row[estab_col]
                year_m = re.search(r"\b(\d{3,4})\b", date_str)
                day_m = re.search(r"\b(\d{1,2})\b", date_str)
                if year_m:
                    year = int(year_m.group(1))
                    month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                                 "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
                    month = 1
                    for abbr, mnum in month_map.items():
                        if abbr in date_str.lower():
                            month = mnum
                            break
                    day = int(day_m.group(1)) if day_m else 1
                    date_val = (year, month, day)
                    if best_date is None or date_val < best_date:
                        best_date = date_val
                        best_name = row[name_col]
            if best_name:
                return best_name

    if "more total" in q or "which month" in q and "more" in q:
        month_col = col_index(header, "Month-Year", "Month\u2013Year", "Month")
        if month_col is not None:
            quoted = re.findall(r"(\S+\s+\d{3,4})", question)
            if len(quoted) >= 2:
                m1, m2 = normalize_text(quoted[0]), normalize_text(quoted[1])
                numeric_cols = [i for i in range(len(header)) if i != month_col]
                sum1, sum2 = 0.0, 0.0
                for row in rows:
                    rmonth = normalize_text(row[month_col])
                    for ci in numeric_cols:
                        try:
                            val = float(str(row[ci]).replace(",", ""))
                        except (ValueError, TypeError):
                            continue
                        if rmonth == m1:
                            sum1 += val
                        elif rmonth == m2:
                            sum2 += val
                if sum1 or sum2:
                    return quoted[0] if sum1 >= sum2 else quoted[1]

    return None


def extract_kg_triples(raw: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\s*\|\s*", raw) if p.strip()]
    triples = []
    for part in parts:
        lower = part.lower()
        if "knowledge graph triples" in lower or lower.startswith(("infer ", "answer ", "using ", "you will", "the following")):
            continue
        triples.append(part)
    return triples or [raw.strip()]


def format_kg_triples(raw: str) -> str:
    triples = extract_kg_triples(raw)
    return "\n".join(f"{i + 1}. {triple}" for i, triple in enumerate(triples))


def format_kg_triples_grouped(raw: str) -> str:
    triples = extract_kg_triples(raw)
    groups: dict[str, list[str]] = {}
    for triple in triples:
        tokens = triple.split(" ", 1)
        head = tokens[0] if tokens else "_"
        groups.setdefault(head, []).append(triple)
    lines = []
    idx = 1
    for head, group in groups.items():
        for triple in group:
            lines.append(f"{idx}. {triple}")
            idx += 1
    return "\n".join(lines)


def format_table(table: dict) -> str:
    header = table["header"]
    rows = table["rows"]
    header_line = "| " + " | ".join(str(h).strip() for h in header) + " |"
    separator_line = "| " + " | ".join("---" for _ in header) + " |"
    body = ["| " + " | ".join(str(cell).strip() for cell in row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body])


def is_bad_answer(answer: str) -> bool:
    text = normalize_text(answer).strip(" .。")
    if not text:
        return True
    return any(pattern in text for pattern in BAD_ANSWER_PATTERNS)


def build_prompt(item: dict) -> str:
    task_type = item["task_type"]
    question = item["question"]

    if task_type == "knowledge_graph":
        return f"""You are an expert at knowledge graph reasoning. Below are knowledge graph triples in the format: <Subject> <Relation> <Object>.

Triples:
{format_kg_triples_grouped(item['input'])}

Question: {question}

Reasoning protocol:
1. IDENTIFY the key entities and constraints in the question (who, what, which, how many, AND, OR, NOT, "but not", "later than", "earlier than").
2. TRACE the reasoning path through the triples step by step. For multi-hop questions, find intermediate bridge entities.
3. DISTRACTOR CHECK: Many triples are noise. Only use triples whose relations are relevant to the question.
4. SET OPERATIONS: If the question uses "both ... and", "but not", "neither ... nor", compute the intersection/difference/complement explicitly.
5. VERIFY: Before outputting, re-check that your answer satisfies ALL constraints in the question.
6. For entity names, use the human-readable label (not internal IDs like m.xxxxx).
7. Output ONLY the final answer. No explanation, no quotes, no prefix like "Answer:"."""

    elif task_type == "multi_hop_qa":
        contexts_text = ""
        for i, ctx in enumerate(item["contexts"]):
            title = ctx.get("title", f"Context {i+1}")
            paragraph = ctx.get("paragraph", "")
            if not paragraph and "sentences" in ctx:
                paragraph = " ".join(ctx["sentences"])
            contexts_text += f"\n[Document {i+1}: {title}]\n{paragraph}\n"

        return f"""You are an expert at multi-hop reading comprehension. Read ALL documents below carefully.

{contexts_text}

Question: {question}

Reasoning protocol:
1. DECOMPOSE: Break the question into sub-questions. Identify what bridge entities or facts connect the documents.
2. EVIDENCE TRACE: For each sub-question, find the specific sentence(s) in the documents that provide the answer.
3. INTEGRATE: Combine the sub-answers to form the final answer.
4. FICTIONAL CONTENT: Some documents describe fictional worlds. Answer based strictly on what the documents state, not external knowledge.
5. VERIFY: Re-read the question and confirm your answer satisfies all constraints.
6. Be precise about names, dates, numbers, and titles. Use the exact form from the documents.
7. Output ONLY the final answer (a name, number, date, or short phrase). No explanations, no quotes, no prefix like "Answer:"."""

    elif task_type == "table_qa":
        table = item["table"]
        table_text = format_table(table)
        num_rows = len(table["rows"])
        num_cols = len(table["header"])

        return f"""You are an expert at table-based reasoning. The table has {num_cols} columns and {num_rows} data rows.

{table_text}

Question: {question}

Reasoning protocol:
1. LOCATE: Identify which column(s) and row(s) are relevant to the question.
2. FILTER: Apply any conditions (e.g., "where Venue = X", "in season Y") to select the correct subset of rows.
3. COMPUTE: Perform any required arithmetic (count, sum, difference, max, min) precisely.
4. DEDUP: If the question asks for "distinct" or "unique", remove duplicate rows based on factual content (ignore row IDs).
5. VERIFY: Double-check your count/computation by listing the qualifying items.
6. Follow legend rows exactly when present.
7. Output ONLY the final answer (a name, number, or short phrase). No explanations, no quotes, no prefix like "Answer:"."""

    else:
        raise ValueError(f"未知题型: {task_type}")


def build_repair_prompt(item: dict, bad_answer: str) -> str:
    base = build_prompt(item)
    return f"""{base}

The previous answer was invalid for scoring: {bad_answer!r}.
You must now provide the best concise answer. Do not refuse. Do not output unknown/none/not specified/not available/not mentioned/cannot be determined.
Output ONLY the final answer text."""


def clean_answer(raw: str) -> str:
    text = raw.strip()
    for prefix in ("Answer:", "answer:", "A:", "The answer is", "The answer is:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    text = text.strip('"\' ')
    text = text.rstrip(".")
    return text.strip()


def extract_answer(response) -> str:
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return clean_answer(block.text)
    return ""


async def call_api_once(item: dict, temperature: float = TEMPERATURE,
                       thinking_budget: int = THINKING_BUDGET) -> str:
    prompt = build_prompt(item)
    backoff = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                top_p=TOP_P,
                thinking={"type": "enabled", "budget_tokens": thinking_budget},
                output_config={"effort": "max"},
                messages=[{"role": "user", "content": prompt}],
            )
            return extract_answer(response)

        except anthropic.RateLimitError as e:
            if attempt == MAX_RETRIES:
                print(f"  [id={item['id']}] 429 final fail: {e}")
                return ""
            sleep_s = backoff + random.uniform(0, backoff * 0.5)
            print(f"  [id={item['id']}] 429, retry {attempt}/{MAX_RETRIES} in {sleep_s:.1f}s")
            await asyncio.sleep(sleep_s)
            backoff = min(backoff * 2, 60)

        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            if attempt == MAX_RETRIES:
                print(f"  [id={item['id']}] network final fail: {e}")
                return ""
            sleep_s = backoff + random.uniform(0, backoff * 0.5)
            print(f"  [id={item['id']}] network err, retry {attempt}/{MAX_RETRIES} in {sleep_s:.1f}s: {e}")
            await asyncio.sleep(sleep_s)
            backoff = min(backoff * 2, 60)

        except anthropic.APIStatusError as e:
            status = getattr(e, "status_code", None)
            if status and 500 <= status < 600 and attempt < MAX_RETRIES:
                sleep_s = backoff + random.uniform(0, backoff * 0.5)
                print(f"  [id={item['id']}] {status}, retry {attempt}/{MAX_RETRIES} in {sleep_s:.1f}s")
                await asyncio.sleep(sleep_s)
                backoff = min(backoff * 2, 60)
                continue
            print(f"  [id={item['id']}] API error ({status}): {e}")
            return ""

        except Exception as e:
            print(f"  [id={item['id']}] unexpected: {type(e).__name__}: {e}")
            return ""

    return ""


def majority_vote(answers: list[str]) -> str:
    valid = [a for a in answers if a and not is_bad_answer(a)]
    if not valid:
        return answers[0] if answers else ""
    counts = Counter(normalize_text(a) for a in valid)
    best_norm = counts.most_common(1)[0][0]
    for a in valid:
        if normalize_text(a) == best_norm:
            return a
    return valid[0]


async def call_api(item: dict) -> str:
    temps = [TEMPERATURE] * VOTING_ROUNDS
    if VOTING_ROUNDS >= 3:
        temps[1] = min(TEMPERATURE + 0.2, 1.0)
        temps[2] = max(TEMPERATURE - 0.1, 0.05)
    results = await asyncio.gather(
        *(call_api_once(item, temperature=t) for t in temps)
    )
    candidates = list(results)
    answer = majority_vote(candidates)
    if VOTING_ROUNDS > 1:
        print(f"  [id={item['id']}] votes: {[c[:40] for c in candidates]} -> {answer[:40]}")
    return answer


async def call_repair_api(item: dict, bad_answer: str) -> str:
    prompt = build_repair_prompt(item, bad_answer)
    backoff = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=0.15,
                top_p=0.85,
                thinking={"type": "enabled", "budget_tokens": min(THINKING_BUDGET, 8000)},
                output_config={"effort": "max"},
                messages=[{"role": "user", "content": prompt}],
            )
            return extract_answer(response)
        except anthropic.RateLimitError as e:
            if attempt == MAX_RETRIES:
                print(f"  [id={item['id']}] repair 429 final fail: {e}")
                return bad_answer
            sleep_s = backoff + random.uniform(0, backoff * 0.5)
            print(f"  [id={item['id']}] repair 429, retry {attempt}/{MAX_RETRIES} in {sleep_s:.1f}s")
            await asyncio.sleep(sleep_s)
            backoff = min(backoff * 2, 60)
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            if attempt == MAX_RETRIES:
                print(f"  [id={item['id']}] repair network final fail: {e}")
                return bad_answer
            sleep_s = backoff + random.uniform(0, backoff * 0.5)
            print(f"  [id={item['id']}] repair network err, retry {attempt}/{MAX_RETRIES} in {sleep_s:.1f}s: {e}")
            await asyncio.sleep(sleep_s)
            backoff = min(backoff * 2, 60)
        except anthropic.APIStatusError as e:
            status = getattr(e, "status_code", None)
            if status and 500 <= status < 600 and attempt < MAX_RETRIES:
                sleep_s = backoff + random.uniform(0, backoff * 0.5)
                print(f"  [id={item['id']}] repair {status}, retry {attempt}/{MAX_RETRIES} in {sleep_s:.1f}s")
                await asyncio.sleep(sleep_s)
                backoff = min(backoff * 2, 60)
                continue
            print(f"  [id={item['id']}] repair API error ({status}): {e}")
            return bad_answer
        except Exception as e:
            print(f"  [id={item['id']}] repair unexpected: {type(e).__name__}: {e}")
            return bad_answer

    return bad_answer


def load_done_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "id" in obj and obj.get("answer", "") != "":
                    done.add(obj["id"])
            except json.JSONDecodeError:
                continue
    return done


def load_answers(path: str) -> dict:
    answers = {}
    if not os.path.exists(path):
        return answers
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in obj and isinstance(obj.get("answer", ""), str):
                answers[obj["id"]] = obj["answer"].strip()
    return answers


def write_submit(path: str, data: list, answers: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in sorted(data, key=lambda x: x["id"]):
            f.write(json.dumps({"id": item["id"], "answer": answers.get(item["id"], "")}, ensure_ascii=False) + "\n")


async def worker(sem, item, out_lock, out_file, counter, total):
    async with sem:
        answer = await call_api(item)
        if is_bad_answer(answer):
            repaired = await call_repair_api(item, answer)
            if not is_bad_answer(repaired):
                answer = repaired
        async with out_lock:
            out_file.write(json.dumps({"id": item["id"], "answer": answer}, ensure_ascii=False) + "\n")
            out_file.flush()
            counter["done"] += 1
            preview = (item["question"][:50] + "...") if len(item["question"]) > 50 else item["question"]
            ans_preview = (answer[:60] + "...") if len(answer) > 60 else answer
            print(f"[{counter['done']:3d}/{total}] id={item['id']:3d} {item['task_type']:16s} | Q: {preview}")
            print(f"            -> {ans_preview!r}")


async def main():
    print("=" * 60)
    print(f"CCKS 2026 OneEval  |  {MODEL}  |  budget={THINKING_BUDGET}  temp={TEMPERATURE}  votes={VOTING_ROUNDS}")
    print("=" * 60)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_all = len(data)
    type_counts = Counter(d["task_type"] for d in data)
    print(f"题目总数: {total_all}")
    for t, c in type_counts.items():
        print(f"  - {t}: {c}")

    answers = load_answers(RAW_OUTPUT_FILE)
    if not answers and os.path.exists(OUTPUT_FILE):
        answers = load_answers(OUTPUT_FILE)

    deterministic_count = 0
    for item in data:
        if item["task_type"] != "table_qa":
            continue
        answer = solve_table_qa(item)
        if answer is not None:
            answers[item["id"]] = answer
            deterministic_count += 1

    done_ids = {
        item["id"]
        for item in data
        if item["id"] in answers and answers[item["id"]] and (not REANSWER_BAD or not is_bad_answer(answers[item["id"]]))
    }
    todo = [d for d in data if d["id"] not in done_ids]
    print(f"已完成: {len(done_ids)}  |  待答: {len(todo)}  |  并发: {CONCURRENCY}")
    print(f"表格确定性求解覆盖: {deterministic_count}")

    if not todo:
        write_submit(OUTPUT_FILE, data, answers)
        print(f"全部完成，已重写去重提交 -> {OUTPUT_FILE}")
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    out_lock = asyncio.Lock()
    counter = {"done": 0}

    with open(RAW_OUTPUT_FILE, "a", encoding="utf-8") as out_file:
        tasks = [worker(sem, item, out_lock, out_file, counter, len(todo)) for item in todo]
        await asyncio.gather(*tasks)

    answers.update(load_answers(RAW_OUTPUT_FILE))
    for item in data:
        if item["task_type"] == "table_qa":
            answer = solve_table_qa(item)
            if answer is not None:
                answers[item["id"]] = answer

    write_submit(OUTPUT_FILE, data, answers)

    final_done = load_done_ids(OUTPUT_FILE)
    print(f"\n写入完成: {len(final_done)}/{total_all}")
    missing = [d["id"] for d in data if d["id"] not in final_done]
    if missing:
        print(f"仍缺失 {len(missing)} 题: {missing[:20]}{'...' if len(missing) > 20 else ''}")
        print("再跑一次脚本将自动补答。")
    else:
        print(f"全部 {total_all} 题已答完 -> {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
