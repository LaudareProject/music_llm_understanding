from __future__ import annotations

import re


NOTE_TO_PITCH_CLASS: dict[str, int] = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}

PITCH_CLASS_TO_NAME = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
KEYCOR_ROW_RE = re.compile(
    r"^\s*([A-G](?:-flat|-sharp|#|b)?)\s+(Major|Minor)\s+(-?\d+(?:\.\d+)?)\s*$"
)
KEYCOR_INDEX_RE = re.compile(r"\b(Major|Minor)\[(\d{1,2})\]\s*=\s*(-?\d+(?:\.\d+)?)")


def parse_keycor_scores(output: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for line in output.splitlines():
        match = KEYCOR_ROW_RE.match(line)
        if match is not None:
            note_name = normalize_note_name(match.group(1))
            mode = match.group(2).lower()
            score = float(match.group(3))
            pitch_class = NOTE_TO_PITCH_CLASS.get(note_name)
            if pitch_class is not None:
                rows.append((f"{PITCH_CLASS_TO_NAME[pitch_class]}_{mode}", score))
            continue
        for indexed_match in KEYCOR_INDEX_RE.finditer(line):
            mode = indexed_match.group(1).lower()
            pitch_class = int(indexed_match.group(2)) % 12
            score = float(indexed_match.group(3))
            rows.append((f"{PITCH_CLASS_TO_NAME[pitch_class]}_{mode}", score))
    return rows


def ensemble_vote(results) -> str:
    if not results:
        raise ValueError("Cannot vote with empty key-correlation results.")
    votes: dict[str, int] = {}
    scores: dict[str, float] = {}
    for result in results:
        votes[result.tonality] = votes.get(result.tonality, 0) + 1
        scores[result.tonality] = scores.get(result.tonality, 0.0) + result.score
    best_vote = max(votes.values())
    candidates = [tonality for tonality, count in votes.items() if count == best_vote]
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda tonality: scores.get(tonality, float("-inf")))


def normalize_note_name(name: str) -> str:
    value = name.strip().replace("-flat", "b").replace("-sharp", "#")
    if not value:
        return value
    return value[0].upper() + value[1:]
