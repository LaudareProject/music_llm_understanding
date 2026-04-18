import csv
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib import parse, request

import mido
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.humdrum import kern_to_midi, musicxml_to_midi
from experiments.common.tsv import write_tsv

REPO_ROOT = Path(__file__).resolve().parents[2]
ASAP_REPO_URL = "https://github.com/fosfrancesco/asap-dataset"
ASAP_ROOT = REPO_ROOT / "ASAP"
EWLD_ROOT = REPO_ROOT / "EWLD"
EWLD_DB_PATH = EWLD_ROOT / "EWLD.db"
LIFESPANS_CSV = REPO_ROOT / "data" / "composer_lifespans.csv"
COMPOSITION_DATES_PATH = REPO_ROOT / "data" / "composition_dates.json"
OUTPUT_TSV = REPO_ROOT / "experiments" / "composer_axis" / "songs.tsv"
CACHE_DIR = REPO_ROOT / "cache" / "experiments" / "composer_axis"
DATE_CACHE_PATH = CACHE_DIR / "prepare_tsv_date_cache.json"
COVARIATE_CACHE_PATH = CACHE_DIR / "prepare_tsv_covariate_cache.json"
BIN_DIR = REPO_ROOT / "bin"
FEATURES_MIDI_DIR = CACHE_DIR / "prepare_tsv_midi"
MIN_SONGS_PER_COMPOSER = 4
MAX_SONGS_PER_COMPOSER = 8
SEED = 42

HTTP_USER_AGENT = "Laudare-prepare-tsv/1.0"
IMSLP_API_URL = "https://imslp.org/api.php"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
RISM_SEARCH_URL = "https://rism.online/search"

MUSIC21_COMPOSERS = [
    "bach",
    "beach",
    "beethoven",
    "chopin",
    "ciconia",
    "corelli",
    "cpebach",
    "handel",
    "haydn",
    "johnson_j_r",
    "joplin",
    "josquin",
    "liliuokalani",
    "lusitano",
    "monteverdi",
    "mozart",
    "palestrina",
    "schoenberg",
    "schubert",
    "schumann_clara",
    "schumann_robert",
    "verdi",
    "weber",
    "webern",
]

MUSIC21_NAMES = {
    "bach": "Johann Sebastian Bach",
    "beach": "Amy Beach",
    "beethoven": "Ludwig van Beethoven",
    "chopin": "Frederic Chopin",
    "ciconia": "Johannes Ciconia",
    "corelli": "Arcangelo Corelli",
    "cpebach": "C.P.E. Bach",
    "handel": "George Frideric Handel",
    "haydn": "Joseph Haydn",
    "johnson_j_r": "J. Rosamond Johnson",
    "joplin": "Scott Joplin",
    "josquin": "Josquin des Prez",
    "liliuokalani": "Queen Liliuokalani",
    "lusitano": "Vicente Lusitano",
    "monteverdi": "Claudio Monteverdi",
    "mozart": "Wolfgang Amadeus Mozart",
    "palestrina": "Giovanni Palestrina",
    "schoenberg": "Arnold Schoenberg",
    "schubert": "Franz Schubert",
    "schumann_clara": "Clara Schumann",
    "schumann_robert": "Robert Schumann",
    "verdi": "Giuseppe Verdi",
    "weber": "Carl Maria von Weber",
    "webern": "Anton Webern",
}

MUSIC21_SUFFIXES = {
    ".mid": "midi",
    ".midi": "midi",
    ".xml": "musicxml",
    ".musicxml": "musicxml",
    ".mxl": "musicxml",
    ".krn": "kern",
    ".kern": "kern",
}


def canonicalize(name):
    normalized = unicodedata.normalize("NFKD", name)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_only.lower())


def parse_year(value):
    text = str(value or "").strip()
    match = re.search(r"-?\d{3,4}", text)
    return int(match.group(0)) if match else None


def parse_work_year(value):
    text = str(value or "").strip()
    if not text:
        return None
    normalized = text.replace("\u2013", "-").replace("\u2014", "-")
    pairs = list(re.finditer(r"(\d{4})\s*-\s*(\d{2,4})", normalized))
    if pairs:
        match = pairs[0]
        start = int(match.group(1))
        end_text = match.group(2)
        if len(end_text) == 2:
            end = (start // 100) * 100 + int(end_text)
            if end < start:
                end += 100
        else:
            end = int(end_text)
        return (start + end) / 2.0
    years = [int(token) for token in re.findall(r"\b\d{4}\b", normalized)]
    if not years:
        return None
    if len(years) == 1:
        return float(years[0])
    return (years[0] + years[-1]) / 2.0


def parse_wikidata_time_year(value):
    text = str(value or "").strip()
    match = re.match(r"^([+-])(\d{1,16})-", text)
    if not match:
        return None
    sign, year_text = match.groups()
    year = int(year_text)
    if sign == "-":
        year = -year
    return float(year)


def year_in_lifespan(row, year):
    try:
        value = float(year)
        birth = float(row["birth_year"])
        death = float(row["death_year"])
    except Exception:
        return True
    if np.isnan(value) or np.isnan(birth) or np.isnan(death):
        return True
    return birth <= value <= death


def load_cache(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_cache(path: Path, cache):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True, ensure_ascii=False)


def fetch_text(url, headers=None, accept=None):
    request_headers = {"User-Agent": HTTP_USER_AGENT}
    if headers:
        request_headers.update(headers)
    if accept:
        request_headers["Accept"] = accept
    req = request.Request(url, headers=request_headers)
    with request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8")


def fetch_json(url, headers=None, accept=None):
    return json.loads(fetch_text(url, headers=headers, accept=accept))


def strip_markup(text):
    value = str(text or "")
    value = re.sub(r"\{\{[^{}]*\|([^{}|]+)\}\}", r"\1", value)
    value = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", value)
    value = re.sub(r"\[\[([^\]]+)\]\]", r"\1", value)
    value = value.replace("<br>", "; ").replace("<br/>", "; ").replace("<br />", "; ")
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_title_text(text):
    value = str(text or "").replace("_", " ").strip()
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", value)
    value = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", value)
    value = re.sub(r"[-./]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def title_tokens(text):
    folded = canonicalize(normalize_title_text(text))
    return folded


def extract_catalog_queries(title, path):
    text = f"{title} {path}".lower().replace("_", " ").replace("/", " ")
    patterns = [
        (r"\bbwv\s*([0-9]+[a-z-]*)", "BWV {}"),
        (r"\bk\.?\s*([0-9]+[a-z-]*)", "K.{}"),
        (r"\bd\.?\s*([0-9]+[a-z-]*)", "D.{}"),
        (r"\bop\.?\s*([0-9]+[a-z-]*)", "Op.{}"),
        (r"\bhob\.?\s*([xivlcdm]+[:.]?[0-9a-z-]+)", "Hob.{}"),
        (r"\bs\.?\s*([0-9]+[a-z-]*)", "S.{}"),
        (r"\bm\.?\s*([0-9]+[a-z-]*)", "M.{}"),
    ]
    queries = []
    for pattern, template in patterns:
        for match in re.findall(pattern, text):
            queries.append(template.format(match.upper()))
    return queries


def path_context_queries(path):
    parts = [normalize_title_text(part) for part in Path(path).parts[:-1]]
    queries = []
    for part in reversed(parts):
        lowered = part.lower()
        if not part or lowered in {"asap", "music21", "corpus", "dataset"}:
            continue
        if re.search(r"\b(op|opus|bwv|hob|k\.?|d\.?|s\.?|m\.?)[ .:0-9a-z-]+", lowered):
            queries.append(part)
    return queries[:3]


def build_query_candidates(row):
    composer = row["composer"]
    title = normalize_title_text(row["title"])
    candidates = []

    def add(query):
        value = re.sub(r"\s+", " ", str(query or "")).strip()
        value = re.sub(r"_+", " ", str(value or "")).strip()
        if value and value not in candidates:
            candidates.append(value)

    add(f"{composer} {title}")
    add(title)
    for catalog in extract_catalog_queries(row["title"], row["path"]):
        add(f"{composer} {catalog}")
        add(catalog)
    for context in path_context_queries(row["path"]):
        add(f"{composer} {context}")
        add(context)
    return candidates


def choose_imslp_hit(row, hits):
    composer_key = canonicalize(row["composer"])
    title_key = canonicalize(normalize_title_text(row["title"]))
    catalogs = [
        canonicalize(value)
        for value in extract_catalog_queries(row["title"], row["path"])
    ]
    best_title = None
    best_score = None
    for title in hits:
        hit_key = canonicalize(title)
        score = 0
        if composer_key and composer_key[:12] in hit_key:
            score += 4
        if title_key and title_key[:16] and title_key[:16] in hit_key:
            score += 3
        for catalog in catalogs:
            if catalog and catalog in hit_key:
                score += 5
        if canonicalize(row["composer"].split()[-1]) in hit_key:
            score += 2
        if best_score is None or score > best_score:
            best_score = score
            best_title = title
    return best_title if (best_score or 0) > 0 else None


def fetch_imslp_fields(page_title):
    params = parse.urlencode(
        {
            "action": "query",
            "prop": "revisions",
            "titles": page_title,
            "rvprop": "content",
            "format": "json",
        }
    )
    payload = fetch_json(f"{IMSLP_API_URL}?{params}")
    pages = payload.get("query", {}).get("pages", {})
    page = next(iter(pages.values()), {})
    revisions = page.get("revisions") or []
    if not revisions:
        return {}
    content = revisions[0].get("*") or ""
    fields = {}
    for name in [
        "Work Title",
        "Opus/Catalogue Number",
        "Year/Date of Composition",
        "Year of First Publication",
    ]:
        match = re.search(rf"^\|{re.escape(name)}=(.*)$", content, flags=re.MULTILINE)
        fields[name] = strip_markup(match.group(1)) if match else ""
    return fields


def lookup_impls_query(row, query) -> tuple[float, str, str] | tuple[None, None, None]:
    params = parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
        }
    )
    payload = fetch_json(f"{IMSLP_API_URL}?{params}")
    titles = [
        item.get("title", "") for item in payload.get("query", {}).get("search", [])
    ]
    chosen = choose_imslp_hit(row, titles)
    if not chosen:
        return None, None, None
    fields = fetch_imslp_fields(chosen)
    composition_year = parse_work_year(fields.get("Year/Date of Composition"))
    if composition_year is not None and year_in_lifespan(row, composition_year):
        return composition_year, "imslp_composition", chosen
    publication_year = parse_work_year(fields.get("Year of First Publication"))
    if publication_year is not None and year_in_lifespan(row, publication_year):
        return publication_year, "imslp_first_publication", chosen
    return None, None, None


def lookup_imslp_year(row) -> tuple[float, str, str] | tuple[None, None, None]:
    for query in build_query_candidates(row):
        year, source, hit = lookup_impls_query(row, query)
        if year is not None:
            assert source is not None
            assert hit is not None
            return year, source, hit
    return None, None, None


def choose_wikipedia_hit(row, hits):
    composer_words = [canonicalize(token) for token in row["composer"].split() if token]
    title_words = [
        canonicalize(token)
        for token in normalize_title_text(row["title"]).split()
        if token
    ]
    best = None
    best_score = None
    for hit in hits:
        title = hit.get("title", "")
        snippet = strip_markup(hit.get("snippet", ""))
        haystack = canonicalize(f"{title} {snippet}")
        score = 0
        for word in composer_words:
            if word and word in haystack:
                score += 2
        for word in title_words[:4]:
            if word and word in haystack:
                score += 1
        for catalog in extract_catalog_queries(row["title"], row["path"]):
            if canonicalize(catalog) in haystack:
                score += 4
        if best_score is None or score > best_score:
            best = hit
            best_score = score
    return best if (best_score or 0) > 0 else None


def fetch_wikidata_year(row, qid) -> tuple[float, str, str] | tuple[None, None, None]:
    payload = fetch_json(WIKIDATA_ENTITY_URL.format(qid=qid))
    entity = payload.get("entities", {}).get(qid, {})
    for prop, source in [
        ("P571", "wikipedia_inception"),
        ("P577", "wikipedia_publication"),
        ("P839", "wikipedia_imslp_link"),
    ]:
        claims = entity.get("claims", {}).get(prop) or []
        for claim in claims:
            datavalue = claim.get("mainsnak", {}).get("datavalue", {})
            value = datavalue.get("value", {})

            if prop == "P839":
                query = re.sub(r"_", r" ", str(value or "")).strip()
                year, _, hit = lookup_impls_query(row, query)
                if year is not None and year_in_lifespan(row, year):
                    assert hit is not None
                    return year, "wikipedia_imslp_link", hit
                continue

            year = parse_wikidata_time_year(value.get("time"))
            if year is not None:
                label = entity.get("labels", {}).get("en", {}).get("value", qid)
                label = f"{WIKIDATA_ENTITY_URL.format(qid=qid)} ({label})"
                return year, source, label
    return None, None, None


def lookup_wikipedia_year(row) -> tuple[float, str, str] | tuple[None, None, None]:
    for query in build_query_candidates(row):
        params = parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
            }
        )
        payload = fetch_json(f"{WIKIPEDIA_API_URL}?{params}")
        chosen = choose_wikipedia_hit(row, payload.get("query", {}).get("search", []))
        if not chosen:
            continue
        title = chosen.get("title")
        if not title:
            continue
        detail_params = parse.urlencode(
            {
                "action": "query",
                "prop": "pageprops",
                "titles": title,
                "format": "json",
            }
        )
        detail = fetch_json(f"{WIKIPEDIA_API_URL}?{detail_params}")
        pages = detail.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        qid = page.get("pageprops", {}).get("wikibase_item")
        if not qid:
            continue

        year, source, hit = fetch_wikidata_year(row, qid)
        if year is not None and year_in_lifespan(row, year):
            assert source is not None
            assert hit is not None
            return year, source, hit
    return None, None, None


def choose_rism_item(row, items):
    composer_key = canonicalize(row["composer"])
    best = None
    best_score = None
    for item in items:
        label = ""
        for key in ["none", "en", "de", "fr"]:
            values = (item.get("label") or {}).get(key) or []
            if values:
                label = values[0]
                break
        haystack = canonicalize(label)
        score = 0
        if composer_key[:12] and composer_key[:12] in haystack:
            score += 3
        for catalog in extract_catalog_queries(row["title"], row["path"]):
            if canonicalize(catalog) in haystack:
                score += 5
        for token in normalize_title_text(row["title"]).split()[:4]:
            if canonicalize(token) in haystack:
                score += 1
        if best_score is None or score > best_score:
            best = item
            best_score = score
    return best if (best_score or 0) > 0 else None


def lookup_rism_year(row) -> tuple[float, str, str] | tuple[None, None, None]:
    for query in build_query_candidates(row):
        params = parse.urlencode({"q": query, "rows": 20})
        payload = fetch_json(
            f"{RISM_SEARCH_URL}?{params}", accept="application/ld+json"
        )
        items = payload.get("items") or []
        chosen = choose_rism_item(row, items)
        if not chosen:
            continue
        resource = fetch_json(chosen["id"], accept="application/ld+json")
        creator = (
            resource.get("creator", {})
            .get("relatedTo", {})
            .get("label", {})
            .get("none", [""])[0]
        )
        if canonicalize(row["composer"].split()[-1]) not in canonicalize(creator):
            continue
        dates = resource.get("dates") or {}
        earliest = parse_work_year(dates.get("earliestDate"))
        latest = parse_work_year(dates.get("latestDate"))
        if earliest is not None and latest is not None:
            year = (earliest + latest) / 2.0
            if year_in_lifespan(row, year):
                return year, "rism_source_date", chosen["id"]
        date_statement = parse_work_year(dates.get("dateStatement"))
        if date_statement is not None and year_in_lifespan(row, date_statement):
            return date_statement, "rism_source_date", chosen[""]
    return None, None, None


def lookup_network_year(row) -> tuple[float, str, str] | tuple[None, None, None]:
    for fn in [lookup_wikipedia_year, lookup_imslp_year, lookup_rism_year]:
        try:
            year, source, hit = fn(row)
        except Exception as err:
            lineno = err.__traceback__.tb_lineno if err.__traceback__ else "unknown"
            print(f"Error in {fn.__name__} for row {row}: {err} (Line {lineno})")
            year, source, hit = None, None, None
        if year is not None and year_in_lifespan(row, year):
            assert source is not None
            assert hit is not None
            return year, source, hit
        time.sleep(0.1)
    return None, None, None


def assign_work_years(records):
    with COMPOSITION_DATES_PATH.open("r") as f:
        composition_dates = json.load(f)
    for i, row in enumerate(records):
        cache_key = "|".join(
            [row["dataset"], row["composer"], row["title"], row["path"]]
        )
        print(
            "Checking record {}/{}: {} - {} ({})".format(
                i + 1,
                len(records),
                row["composer"],
                row["title"],
                row["path"],
            )
        )
        cached = composition_dates.get(cache_key)
        if cached is None:
            print(
                f"No year assigned for {row['composer']} - {row['title']}. Using lifespan midpoint."
            )
            row["midpoint_year"] = float(row["midpoint_year"])
            row["year_source"] = "lifespan_midpoint"
            continue
        row["midpoint_year"] = float(cached["year"])
        row["year_source"] = cached["source"]

    # -------------
    #   SCAPPING
    # -------------
    #
    # cache = load_cache(DATE_CACHE_PATH)
    # dirty = False
    # for i, row in enumerate(records):
    #     cache_key = "|".join(
    #         [row["dataset"], row["composer"], row["title"], row["path"]]
    #     )
    #     print(
    #         "Checking record {}/{}: {} - {} ({})".format(
    #             i + 1,
    #             len(records),
    #             row["composer"],
    #             row["title"],
    #             row["path"],
    #         )
    #     )
    #     cached = composition_dates.get(cache_key)
    #     if cached is not None and not year_in_lifespan(row, cached.get("year")):
    #         cache.pop(cache_key, None)
    #         cached = None
    #         dirty = True
    #     if cached is None:
    #         if row["dataset"] == "ewld":
    #             year = parse_work_year(row.get("first_performance_date"))
    #             if year is not None and year_in_lifespan(row, year):
    #                 cached = {"year": year, "source": "ewld_first_performance_date"}
    #             else:
    #                 cached = {
    #                     "year": row["midpoint_year"],
    #                     "source": "lifespan_midpoint",
    #                 }
    #         else:
    #             year, source, hit = lookup_network_year(row)
    #             if year is None:
    #                 year = row["midpoint_year"]
    #                 source = "lifespan_midpoint"
    #             cached = {"year": year, "source": source, "hit": hit}
    #         cache[cache_key] = cached
    #         dirty = True
    #     row["midpoint_year"] = float(cached["year"])
    #     row["year_source"] = cached["source"]
    #     print(
    #         "  Assigned year {} from source '{}'".format(
    #             cached["year"], cached["source"]
    #         )
    #     )
    #     if i % 10 == 0 and dirty:
    #         print("Saving date cache with {} entries...".format(len(cache)))
    #         save_cache(DATE_CACHE_PATH, cache)
    #         dirty = False
    # if dirty:
    #     save_cache(DATE_CACHE_PATH, cache)


def load_lifespans():
    table = {}
    with LIFESPANS_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            composer = (row.get("composer") or "").strip()
            birth = parse_year(row.get("birth_year"))
            last_date = (
                2026  # Year this study was conducted
                if row.get("death_year") == "present"
                else parse_year(row.get("death_year"))
            )
            if composer and birth is not None and last_date is not None:
                table[canonicalize(composer)] = (birth, last_date)
    return table


def resolve_lifespan(composer, table, fallback_birth=None, fallback_death=None):
    if canonicalize(composer) in table:
        return table[canonicalize(composer)]
    if fallback_birth is not None and fallback_death is not None:
        return fallback_birth, fallback_death
    return None


def ensure_asap_dataset():
    if (ASAP_ROOT / "metadata.csv").exists():
        return
    subprocess.run(["git", "clone", ASAP_REPO_URL, str(ASAP_ROOT)], check=True)


def relpath(path):
    return os.path.relpath(Path(path).resolve(), REPO_ROOT)


def downsample(records, rng):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["composer"]].append(row)
    out = []
    for composer in sorted(grouped):
        songs = sorted(
            grouped[composer],
            key=lambda row: (row["title"].lower(), row["path"].lower()),
        )
        if len(songs) > MAX_SONGS_PER_COMPOSER:
            songs = sorted(
                rng.sample(songs, MAX_SONGS_PER_COMPOSER),
                key=lambda row: (row["title"].lower(), row["path"].lower()),
            )
        out.extend(songs)
    return out


def collect_asap(table, rng):
    rows = []
    missing = set()
    seen_paths = set()
    with (ASAP_ROOT / "metadata.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            composer = (row.get("composer") or "").strip()
            if not composer:
                continue
            lifespan = resolve_lifespan(composer, table)
            if lifespan is None:
                missing.add(composer)
                continue
            source_path = None
            input_kind = None
            xml_rel = (row.get("xml_score") or "").strip()
            midi_rel = (row.get("midi_score") or "").strip()
            if xml_rel and (ASAP_ROOT / xml_rel).exists():
                source_path = ASAP_ROOT / xml_rel
                input_kind = "musicxml"
            elif midi_rel and (ASAP_ROOT / midi_rel).exists():
                source_path = ASAP_ROOT / midi_rel
                input_kind = "midi"
            if source_path is None:
                continue
            if str(source_path).lower() in seen_paths:
                continue
            seen_paths.add(str(source_path).lower())
            birth_year, death_year = lifespan
            rows.append(
                {
                    "dataset": "asap",
                    "composer": composer,
                    "title": (row.get("title") or "").strip() or source_path.stem,
                    "birth_year": birth_year,
                    "death_year": death_year,
                    "midpoint_year": (birth_year + death_year) / 2.0,
                    "input_kind": input_kind,
                    "path": relpath(source_path),
                }
            )
    return rows, missing


def collect_music21(table, rng):
    from music21 import corpus

    rows = []
    missing = set()
    for selector in MUSIC21_COMPOSERS:
        composer = MUSIC21_NAMES.get(selector, selector)
        lifespan = resolve_lifespan(composer, table)
        if lifespan is None:
            missing.add(composer)
            continue
        work_paths = sorted(
            {
                Path(item)
                for item in corpus.getComposer(selector)
                if Path(item).suffix.lower() in MUSIC21_SUFFIXES
            }
        )
        birth_year, death_year = lifespan
        for work_path in work_paths:
            if not work_path.exists():
                continue
            rows.append(
                {
                    "dataset": "music21",
                    "composer": composer,
                    "title": work_path.stem,
                    "birth_year": birth_year,
                    "death_year": death_year,
                    "midpoint_year": (birth_year + death_year) / 2.0,
                    "input_kind": MUSIC21_SUFFIXES[work_path.suffix.lower()],
                    "path": relpath(work_path),
                }
            )
    return rows, missing


def collect_ewld(table, rng):
    rows = []
    missing = set()
    conn = sqlite3.connect(EWLD_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        WITH single_author_works AS (
          SELECT
            w.id AS work_id,
            w.title AS title,
            w.path_leadsheet AS path_leadsheet,
            MIN(wa.author) AS author,
            COUNT(*) AS author_count
          FROM works w
          JOIN work_author wa ON wa.id = w.id
          GROUP BY w.id
          HAVING author_count = 1
        )
        SELECT
          s.work_id,
          s.title,
          s.path_leadsheet,
          w.first_performance_date,
          a.common_name,
          a.birth,
          a.death
        FROM single_author_works s
        JOIN works w ON w.id = s.work_id
        JOIN authors a ON a.common_name = s.author
        WHERE a.common_name NOT IN ('[Unknown]', '[Traditional]')
        """
    )
    fetched = cursor.fetchall()
    conn.close()
    for (
        work_id,
        title,
        path_leadsheet,
        first_performance_date,
        composer,
        birth_raw,
        death_raw,
    ) in fetched:
        if not path_leadsheet:
            continue
        source_path = (EWLD_ROOT / str(path_leadsheet)).resolve()
        if not source_path.exists():
            continue
        lifespan = resolve_lifespan(
            str(composer), table, parse_year(birth_raw), parse_year(death_raw)
        )
        if lifespan is None:
            missing.add(str(composer))
            continue
        birth_year, death_year = lifespan
        rows.append(
            {
                "dataset": "ewld",
                "composer": str(composer),
                "title": (str(title).strip() if title is not None else "")
                or source_path.stem,
                "birth_year": birth_year,
                "death_year": death_year,
                "midpoint_year": (birth_year + death_year) / 2.0,
                "first_performance_date": first_performance_date,
                "input_kind": "musicxml",
                "path": relpath(source_path),
                "work_id": str(work_id),
            }
        )
    return rows, missing


def keep_composers_with_enough_songs(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["composer"]].append(row)
    out = []
    for composer, songs in grouped.items():
        if len(songs) >= MIN_SONGS_PER_COMPOSER:
            out.extend(songs)
    return out


def assign_query_ids(records):
    counters = defaultdict(int)
    for row in sorted(
        records,
        key=lambda row: (
            row["dataset"],
            canonicalize(row["composer"]),
            row["title"].lower(),
            row["path"].lower(),
        ),
    ):
        key = (row["dataset"], canonicalize(row["composer"]))
        counters[key] += 1
        row["query_id"] = (
            f"{row['dataset']}_{canonicalize(row['composer'])[:20]}_{counters[key]:04d}"
        )


def quantiles5(values):
    if not values:
        return [float("nan")] * 5
    return [
        float(v)
        for v in np.quantile(
            np.asarray(values, dtype=np.float64), [0, 0.25, 0.5, 0.75, 1.0]
        )
    ]


def prepare_feature_midi(row, midi_dir):
    source_path = (REPO_ROOT / row["path"]).resolve()
    midi_path = midi_dir / f"{row['query_id']}.mid"
    if row["input_kind"] == "midi":
        return source_path
    if row["input_kind"] == "musicxml":
        musicxml_to_midi(source_path, midi_path, BIN_DIR)
        return midi_path
    if row["input_kind"] == "kern":
        kern_to_midi(source_path, midi_path, BIN_DIR)
        return midi_path
    raise ValueError(row["input_kind"])


def midi_covariates(midi_path):
    midi = mido.MidiFile(str(midi_path))
    note_pitches = []
    note_durations = []
    active_notes = defaultdict(list)
    tempo = 500000
    elapsed_seconds = 0.0
    for msg in mido.merge_tracks(midi.tracks):
        if msg.time:
            elapsed_seconds += mido.tick2second(msg.time, midi.ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
            continue
        if msg.type not in {"note_on", "note_off"}:
            continue
        channel = getattr(msg, "channel", 0)
        note_key = (channel, msg.note)
        if msg.type == "note_on" and msg.velocity > 0:
            note_pitches.append(float(msg.note))
            active_notes[note_key].append(elapsed_seconds)
            continue
        if active_notes[note_key]:
            start_time = active_notes[note_key].pop()
            note_durations.append(max(0.0, elapsed_seconds - start_time))
    pitch_q = quantiles5(note_pitches)
    duration_q = quantiles5(note_durations)
    return {
        "midi_note_count": len(note_pitches),
        "midi_duration_seconds": float(midi.length),
        "midi_track_count": len(midi.tracks),
        "pitch_q0": pitch_q[0],
        "pitch_q25": pitch_q[1],
        "pitch_q50": pitch_q[2],
        "pitch_q75": pitch_q[3],
        "pitch_q100": pitch_q[4],
        "note_duration_q0": duration_q[0],
        "note_duration_q25": duration_q[1],
        "note_duration_q50": duration_q[2],
        "note_duration_q75": duration_q[3],
        "note_duration_q100": duration_q[4],
    }


def assign_midi_covariates(records):
    FEATURES_MIDI_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache(COVARIATE_CACHE_PATH)
    dirty = False
    for i, row in enumerate(records):
        cache_key = "|".join(
            [row["dataset"], row["composer"], row["title"], row["path"]]
        )
        cached = cache.get(cache_key)
        if cached is not None:
            row.update(cached)
            continue
        print(
            f"Computing MIDI covariates for record {i + 1}/{len(records)}: {row['query_id']}"
        )
        try:
            midi_path = prepare_feature_midi(row, FEATURES_MIDI_DIR)
            stats = midi_covariates(midi_path)
        except Exception as exc:
            print(f"  midi covariates failed for {row['query_id']}: {exc}")
            stats = {
                "midi_note_count": float("nan"),
                "midi_duration_seconds": float("nan"),
                "midi_track_count": float("nan"),
                "pitch_q0": float("nan"),
                "pitch_q25": float("nan"),
                "pitch_q50": float("nan"),
                "pitch_q75": float("nan"),
                "pitch_q100": float("nan"),
                "note_duration_q0": float("nan"),
                "note_duration_q25": float("nan"),
                "note_duration_q50": float("nan"),
                "note_duration_q75": float("nan"),
                "note_duration_q100": float("nan"),
            }
        cache[cache_key] = stats
        dirty = True
        row.update(stats)

        if i % 10 == 0 and dirty:
            print(f"  saving covariate cache with {len(cache)} entries...")
            dirty = False
            save_cache(COVARIATE_CACHE_PATH, cache)

    if dirty:
        save_cache(COVARIATE_CACHE_PATH, cache)


def format_float(value):
    number = float(value)
    if np.isnan(number):
        return "nan"
    return f"{number:.6f}"


def format_int(value):
    number = float(value)
    if np.isnan(number):
        return "nan"
    return str(int(number))


def main():
    rng = random.Random(SEED)
    print("[1/6] collecting datasets")
    lifespans = load_lifespans()
    ensure_asap_dataset()
    asap_rows, missing_asap = collect_asap(lifespans, rng)
    music21_rows, missing_music21 = collect_music21(lifespans, rng)
    ewld_rows, missing_ewld = collect_ewld(lifespans, rng)

    print("[2/6] filtering composers")
    asap_rows = downsample(asap_rows, rng)
    music21_rows = downsample(music21_rows, rng)
    ewld_rows = downsample(ewld_rows, rng)

    by_dataset = {
        "asap": keep_composers_with_enough_songs(asap_rows),
        "music21": keep_composers_with_enough_songs(music21_rows),
        "ewld": keep_composers_with_enough_songs(ewld_rows),
    }
    rows = by_dataset["asap"] + by_dataset["music21"] + by_dataset["ewld"]
    print("[3/6] assigning work years")
    assign_work_years(rows)
    assign_query_ids(rows)
    print("[4/6] computing midi covariates")
    assign_midi_covariates(rows)

    print("[5/6] writing tsv")
    out_rows = []
    for row in sorted(rows, key=lambda row: row["query_id"]):
        out_rows.append(
            {
                "query_id": row["query_id"],
                "dataset": row["dataset"],
                "composer": row["composer"],
                "title": row["title"],
                "birth_year": row["birth_year"],
                "death_year": row["death_year"],
                "midpoint_year": f"{row['midpoint_year']:.6f}",
                "year_source": row["year_source"],
                "input_kind": row["input_kind"],
                "path": row["path"],
                "midi_note_count": format_int(row["midi_note_count"]),
                "midi_duration_seconds": format_float(row["midi_duration_seconds"]),
                "midi_track_count": format_int(row["midi_track_count"]),
                "midi_pitch_q0": format_float(row["pitch_q0"]),
                "midi_pitch_q25": format_float(row["pitch_q25"]),
                "midi_pitch_q50": format_float(row["pitch_q50"]),
                "midi_pitch_q75": format_float(row["pitch_q75"]),
                "midi_pitch_q100": format_float(row["pitch_q100"]),
                "midi_note_duration_q0": format_float(row["note_duration_q0"]),
                "midi_note_duration_q25": format_float(row["note_duration_q25"]),
                "midi_note_duration_q50": format_float(row["note_duration_q50"]),
                "midi_note_duration_q75": format_float(row["note_duration_q75"]),
                "midi_note_duration_q100": format_float(row["note_duration_q100"]),
            }
        )
    write_tsv(
        OUTPUT_TSV,
        out_rows,
        [
            "query_id",
            "dataset",
            "composer",
            "title",
            "birth_year",
            "death_year",
            "midpoint_year",
            "year_source",
            "input_kind",
            "path",
            "midi_note_count",
            "midi_duration_seconds",
            "midi_track_count",
            "midi_pitch_q0",
            "midi_pitch_q25",
            "midi_pitch_q50",
            "midi_pitch_q75",
            "midi_pitch_q100",
            "midi_note_duration_q0",
            "midi_note_duration_q25",
            "midi_note_duration_q50",
            "midi_note_duration_q75",
            "midi_note_duration_q100",
        ],
    )

    print("[6/6] done")
    print(f"wrote {OUTPUT_TSV}")
    print(
        {dataset: len(by_dataset[dataset]) for dataset in ["asap", "music21", "ewld"]}
    )
    missing = set().union(missing_asap, missing_music21, missing_ewld)
    if missing:
        print(f"skipped missing lifespans: {len(missing)}")


if __name__ == "__main__":
    main()
