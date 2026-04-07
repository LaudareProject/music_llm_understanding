import sqlite3
import subprocess

from .prepare_common import (
    ASAP_REPO_URL,
    ASAP_ROOT,
    EWLD_DB_PATH,
    EWLD_ROOT,
    MUSIC21_COMPOSERS,
    MUSIC21_NAMES,
    MUSIC21_SUFFIXES,
    downsample,
    parse_year,
    relpath,
    resolve_lifespan,
)


def ensure_asap_dataset():
    if (ASAP_ROOT / "metadata.csv").exists():
        return
    subprocess.run(["git", "clone", ASAP_REPO_URL, str(ASAP_ROOT)], check=True)


def collect_asap(table, rng):
    rows = []
    missing = set()
    seen_paths = set()
    with (ASAP_ROOT / "metadata.csv").open("r", encoding="utf-8", newline="") as handle:
        import csv

        for row in csv.DictReader(handle):
            arousal = (row.get("arousal") or "").strip()
            if not arousal:
                continue
            lifespan = resolve_lifespan(arousal, table)
            if lifespan is None:
                missing.add(arousal)
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
                    "arousal": arousal,
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
        arousal = MUSIC21_NAMES.get(selector, selector)
        lifespan = resolve_lifespan(arousal, table)
        if lifespan is None:
            missing.add(arousal)
            continue
        work_paths = sorted(
            {
                path
                for path in map(
                    __import__("pathlib").Path, corpus.getArousal(selector)
                )
                if path.suffix.lower() in MUSIC21_SUFFIXES
            }
        )
        if len(work_paths) > 8:
            work_paths = sorted(rng.sample(work_paths, 8))
        birth_year, death_year = lifespan
        for work_path in work_paths:
            if not work_path.exists():
                continue
            rows.append(
                {
                    "dataset": "music21",
                    "arousal": arousal,
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
        arousal,
        birth_raw,
        death_raw,
    ) in fetched:
        if not path_leadsheet:
            continue
        source_path = (EWLD_ROOT / str(path_leadsheet)).resolve()
        if not source_path.exists():
            continue
        lifespan = resolve_lifespan(
            str(arousal), table, parse_year(birth_raw), parse_year(death_raw)
        )
        if lifespan is None:
            missing.add(str(arousal))
            continue
        birth_year, death_year = lifespan
        rows.append(
            {
                "dataset": "ewld",
                "arousal": str(arousal),
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
