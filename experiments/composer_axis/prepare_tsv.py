from collections import defaultdict
import csv
import os
from pathlib import Path
import random
import re
import sqlite3
import subprocess
import sys
import unicodedata

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
MIN_SONGS_PER_COMPOSER = 4
MAX_SONGS_PER_COMPOSER = 8
SAMPLE_SIZE_PER_SIDE = 4
OVERALL_SAMPLE_SIZE = 8
SEED = 42

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
          a.common_name,
          a.birth,
          a.death
        FROM single_author_works s
        JOIN authors a ON a.common_name = s.author
        WHERE a.common_name NOT IN ('[Unknown]', '[Traditional]')
        """
    )
    fetched = cursor.fetchall()
    conn.close()
    for work_id, title, path_leadsheet, composer, birth_raw, death_raw in fetched:
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


def assign_composer_ranks(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[row["dataset"]][row["composer"]].append(row)
    for dataset in grouped:
        ordered = sorted(
            grouped[dataset],
            key=lambda composer: (
                grouped[dataset][composer][0]["midpoint_year"],
                canonicalize(composer),
            ),
        )
        for rank, composer in enumerate(ordered):
            for row in grouped[dataset][composer]:
                row["composer_rank"] = rank


def remove_year_outliers(composers, grouped):
    """Return composers with midpoint_year outliers removed using IQR."""
    years = np.array(
        [grouped[c][0]["midpoint_year"] for c in composers], dtype=np.float64
    )
    q1, q3 = np.percentile(years, [25, 75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    return [c for c, y in zip(composers, years) if lo <= y <= hi]


def choose_dataset_anchors(records, rng):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["composer"]].append(row)
    ordered = sorted(
        grouped,
        key=lambda composer: (
            grouped[composer][0]["midpoint_year"],
            canonicalize(composer),
        ),
    )
    ordered = remove_year_outliers(ordered, grouped)
    early = sorted(grouped[ordered[0]], key=lambda row: row["query_id"])
    late = sorted(grouped[ordered[-1]], key=lambda row: row["query_id"])
    anchor_a = sorted(
        rng.sample(early, SAMPLE_SIZE_PER_SIDE), key=lambda row: row["query_id"]
    )
    anchor_b = sorted(
        rng.sample(late, SAMPLE_SIZE_PER_SIDE), key=lambda row: row["query_id"]
    )
    return anchor_a, anchor_b


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
    assign_query_ids(rows)
    assign_composer_ranks(rows)
    for row in rows:
        row["axis_side"] = "."
        row["overall_side"] = "."

    print("[3/5] choosing anchors")
    dataset_anchor_ids = set()
    for dataset in ["asap", "music21", "ewld"]:
        anchor_a, anchor_b = choose_dataset_anchors(by_dataset[dataset], rng)
        for row in anchor_a:
            row["axis_side"] = "a"
            dataset_anchor_ids.add(row["query_id"])
        for row in anchor_b:
            row["axis_side"] = "b"
            dataset_anchor_ids.add(row["query_id"])

    overall_pool = sorted(
        [row for row in rows if row["query_id"] not in dataset_anchor_ids],
        key=lambda row: row["query_id"],
    )
    overall_sample = sorted(
        rng.sample(overall_pool, OVERALL_SAMPLE_SIZE), key=lambda row: row["query_id"]
    )
    for row in overall_sample[: OVERALL_SAMPLE_SIZE // 2]:
        row["overall_side"] = "a"
    for row in overall_sample[OVERALL_SAMPLE_SIZE // 2 :]:
        row["overall_side"] = "b"

    print("[4/5] writing tsv")
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
                "composer_rank": row["composer_rank"],
                "input_kind": row["input_kind"],
                "path": row["path"],
                "axis_side": row["axis_side"],
                "overall_side": row["overall_side"],
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
            "composer_rank",
            "input_kind",
            "path",
            "axis_side",
            "overall_side",
        ],
    )

    print("[5/5] done")
    print(f"wrote {OUTPUT_TSV}")
    print(
        {dataset: len(by_dataset[dataset]) for dataset in ["asap", "music21", "ewld"]}
    )
    missing = set().union(missing_asap, missing_music21, missing_ewld)
    if missing:
        print(f"skipped missing lifespans: {len(missing)}")


if __name__ == "__main__":
    main()
