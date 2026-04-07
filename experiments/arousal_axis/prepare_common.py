from collections import defaultdict
import csv
import os
from pathlib import Path
import random
import re
import unicodedata


REPO_ROOT = Path(__file__).resolve().parents[2]
ASAP_REPO_URL = "https://github.com/fosfrancesco/asap-dataset"
ASAP_ROOT = REPO_ROOT / "ASAP"
EWLD_ROOT = REPO_ROOT / "EWLD"
EWLD_DB_PATH = EWLD_ROOT / "EWLD.db"
LIFESPANS_CSV = REPO_ROOT / "data" / "arousal_lifespans.csv"
OUTPUT_TSV = REPO_ROOT / "experiments" / "arousal_axis" / "songs.tsv"
CACHE_DIR = REPO_ROOT / "cache" / "experiments" / "arousal_axis"
DATE_CACHE_PATH = CACHE_DIR / "prepare_tsv_date_cache.json"
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


def load_lifespans():
    table = {}
    with LIFESPANS_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            arousal = (row.get("arousal") or "").strip()
            birth = parse_year(row.get("birth_year"))
            death = parse_year(row.get("death_year"))
            if arousal and birth is not None and death is not None:
                table[canonicalize(arousal)] = (birth, death)
    return table


def resolve_lifespan(arousal, table, fallback_birth=None, fallback_death=None):
    if canonicalize(arousal) in table:
        return table[canonicalize(arousal)]
    if fallback_birth is not None and fallback_death is not None:
        return fallback_birth, fallback_death
    return None


def relpath(path):
    return os.path.relpath(Path(path).resolve(), REPO_ROOT)


def downsample(records, rng):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["arousal"]].append(row)
    out = []
    for arousal in sorted(grouped):
        songs = sorted(
            grouped[arousal],
            key=lambda row: (row["title"].lower(), row["path"].lower()),
        )
        if len(songs) > MAX_SONGS_PER_COMPOSER:
            songs = sorted(
                rng.sample(songs, MAX_SONGS_PER_COMPOSER),
                key=lambda row: (row["title"].lower(), row["path"].lower()),
            )
        out.extend(songs)
    return out


def keep_arousals_with_enough_songs(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["arousal"]].append(row)
    out = []
    for arousal, songs in grouped.items():
        if len(songs) >= MIN_SONGS_PER_COMPOSER:
            out.extend(songs)
    return out


def year_within_lifespan(year, row):
    return float(row["birth_year"]) <= float(year) <= float(row["death_year"])
