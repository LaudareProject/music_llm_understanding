from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:
    from music_latents.ewld import TonalityAnalysis


def analyze_files(
    files: List[Path],
    *,
    analyzer: Callable[[Path], TonalityAnalysis],
    workers: int,
    progress_callback: Callable[[int, int], None] | None,
) -> List[TonalityAnalysis]:
    from music_latents.ewld import TonalityAnalysis

    total = len(files)
    if total == 0:
        return []

    analyses_buffer: List[TonalityAnalysis | None] = [None for _ in range(total)]
    completed = 0

    def _tick() -> None:
        nonlocal completed
        completed += 1
        if progress_callback is not None:
            progress_callback(completed, total)

    def _safe_analyze(path: Path) -> TonalityAnalysis:
        try:
            return analyzer(path)
        except Exception as exc:  # pragma: no cover
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            return TonalityAnalysis(
                file_path=path,
                file_size=size,
                tonality=None,
                accepted=False,
                reason=f"analyzer_exception:{type(exc).__name__}",
            )

    if workers == 1:
        for idx, file_path in enumerate(files):
            analyses_buffer[idx] = _safe_analyze(file_path)
            _tick()
        return [item for item in analyses_buffer if item is not None]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(_safe_analyze, file_path): idx
            for idx, file_path in enumerate(files)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            analyses_buffer[idx] = future.result()
            _tick()

    return [item for item in analyses_buffer if item is not None]
