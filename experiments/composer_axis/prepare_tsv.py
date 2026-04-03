from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import subprocess
import sys
import time
import unicodedata
from urllib import parse, request

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.tsv import write_tsv


REPO_ROOT = Path(__file__).resolve().parents[2]
ASAP_REPO_URL = "https://github.com/fosfrancesco/asap-dataset"
ASAP_ROOT = REPO_ROOT / "ASAP"
EWLD_ROOT = REPO_ROOT / "EWLD"
EWLD_DB_PATH = EWLD_ROOT / "EWLD.db"
LIFESPANS_CSV = REPO_ROOT / "data" / "composer_lifespans.csv"
OUTPUT_TSV = REPO_ROOT / "experiments" / "composer_axis" / "songs.tsv"
CACHE_DIR = REPO_ROOT / "cache" / "experiments" / "composer_axis"
DATE_CACHE_PATH = CACHE_DIR / "prepare_tsv_date_cache.json"
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


def load_date_cache():
    if not DATE_CACHE_PATH.exists():
        return {}
    with DATE_CACHE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_date_cache(cache):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with DATE_CACHE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)


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
        if value and value not in candidates:
            candidates.append(value)

    add(f"{title} {composer}")
    add(title)
    for catalog in extract_catalog_queries(row["title"], row["path"]):
        add(f"{catalog} {composer}")
        add(catalog)
    for context in path_context_queries(row["path"]):
        add(f"{context} {composer}")
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


def lookup_imslp_year(row):
    for query in build_query_candidates(row):
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
            continue
        fields = fetch_imslp_fields(chosen)
        composition_year = parse_work_year(fields.get("Year/Date of Composition"))
        if composition_year is not None:
            return composition_year, "imslp_composition"
        publication_year = parse_work_year(fields.get("Year of First Publication"))
        if publication_year is not None:
            return publication_year, "imslp_first_publication"
    return None, None


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


def fetch_wikidata_year(qid):
    payload = fetch_json(WIKIDATA_ENTITY_URL.format(qid=qid))
    entity = payload.get("entities", {}).get(qid, {})
    for prop, source in [
        ("P571", "wikipedia_inception"),
        ("P577", "wikipedia_publication"),
    ]:
        claims = entity.get("claims", {}).get(prop) or []
        for claim in claims:
            datavalue = claim.get("mainsnak", {}).get("datavalue", {})
            value = datavalue.get("value", {})
            year = parse_work_year(value.get("time"))
            if year is not None:
                return year, source
    return None, None


def lookup_wikipedia_year(row):
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
        year, source = fetch_wikidata_year(qid)
        if year is not None:
            return year, source
    return None, None


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


def lookup_rism_year(row):
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
            return (earliest + latest) / 2.0, "rism_source_date"
        date_statement = parse_work_year(dates.get("dateStatement"))
        if date_statement is not None:
            return date_statement, "rism_source_date"
    return None, None


def lookup_network_year(row):
    for fn in [lookup_imslp_year, lookup_wikipedia_year, lookup_rism_year]:
        try:
            year, source = fn(row)
        except Exception:
            year, source = None, None
        if year is not None:
            return year, source
        time.sleep(0.1)
    return None, None


def assign_work_years(records):
    cache = load_date_cache()
    dirty = False
    for row in records:
        cache_key = "|".join(
            [row["dataset"], row["composer"], row["title"], row["path"]]
        )
        cached = cache.get(cache_key)
        if cached is None:
            if row["dataset"] == "ewld":
                year = parse_work_year(row.get("first_performance_date"))
                if year is not None:
                    cached = {"year": year, "source": "ewld_first_performance_date"}
                else:
                    cached = {
                        "year": row["midpoint_year"],
                        "source": "lifespan_midpoint",
                    }
            else:
                year, source = lookup_network_year(row)
                if year is None:
                    year = row["midpoint_year"]
                    source = "lifespan_midpoint"
                cached = {"year": year, "source": source}
            cache[cache_key] = cached
            dirty = True
        row["midpoint_year"] = float(cached["year"])
        row["year_source"] = cached["source"]
    if dirty:
        save_date_cache(cache)


def load_lifespans():
    table = {}
    with LIFESPANS_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            composer = (row.get("composer") or "").strip()
            birth = parse_year(row.get("birth_year"))
            death = parse_year(row.get("death_year"))
            if composer and birth is not None and death is not None:
                table[canonicalize(composer)] = (birth, death)
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
    return downsample(rows, rng), missing


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
        if len(work_paths) > MAX_SONGS_PER_COMPOSER:
            work_paths = sorted(rng.sample(work_paths, MAX_SONGS_PER_COMPOSER))
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
    return downsample(rows, rng), missing


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


def main():
    rng = random.Random(SEED)
    print("[1/5] collecting datasets")
    lifespans = load_lifespans()
    ensure_asap_dataset()
    asap_rows, missing_asap = collect_asap(lifespans, rng)
    music21_rows, missing_music21 = collect_music21(lifespans, rng)
    ewld_rows, missing_ewld = collect_ewld(lifespans, rng)

    print("[2/5] filtering composers")
    by_dataset = {
        "asap": keep_composers_with_enough_songs(asap_rows),
        "music21": keep_composers_with_enough_songs(music21_rows),
        "ewld": keep_composers_with_enough_songs(ewld_rows),
    }
    rows = by_dataset["asap"] + by_dataset["music21"] + by_dataset["ewld"]
    print("[2.5/5] assigning work years")
    assign_work_years(rows)
    assign_query_ids(rows)

    print("[3/5] writing tsv")
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
        ],
    )

    print("[4/5] done")
    print(f"wrote {OUTPUT_TSV}")
    print(
        {dataset: len(by_dataset[dataset]) for dataset in ["asap", "music21", "ewld"]}
    )
    missing = set().union(missing_asap, missing_music21, missing_ewld)
    if missing:
        print(f"skipped missing lifespans: {len(missing)}")


if __name__ == "__main__":
    main()
