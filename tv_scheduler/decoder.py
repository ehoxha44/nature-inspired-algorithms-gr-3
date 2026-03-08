from __future__ import annotations

import bisect
from typing import List, Tuple

from .models import Program, ProblemInstance


class _ScheduleTracker:

    def __init__(self) -> None:
        self.starts:   List[int] = []
        self.ends:     List[int] = []
        self.genres:   List[str] = []
        self.channels: List[int] = []
        self.programs: List[Program] = []

    def overlaps(self, start: int, end: int) -> bool:
        idx = bisect.bisect_left(self.starts, start)

        if idx > 0 and self.ends[idx - 1] > start:
            return True

        if idx < len(self.starts) and self.starts[idx] < end:
            return True

        return False

    def violates_genre_limit(self, start: int, genre: str, R: int) -> bool:
        if R >= len(self.starts) + 1:
            return False

        pos = bisect.bisect_left(self.starts, start)

        before = 0
        for i in range(pos - 1, -1, -1):
            if self.genres[i] == genre:
                before += 1
            else:
                break
            if before >= R:
                break

        after = 0
        for i in range(pos, len(self.starts)):
            if self.genres[i] == genre:
                after += 1
            else:
                break
            if after >= R:
                break

        return before + 1 + after > R

    def add(self, prog: Program) -> None:
        idx = bisect.bisect_left(self.starts, prog.start)
        self.starts.insert(idx, prog.start)
        self.ends.insert(idx, prog.end)
        self.genres.insert(idx, prog.genre)
        self.channels.insert(idx, prog.channel_id)
        self.programs.insert(idx, prog)

    def as_list(self) -> List[Program]:
        return list(self.programs)

    def __len__(self) -> int:
        return len(self.programs)


def decode_chromosome(
    chromosome: List[int],
    programs: List[Program],
    instance: ProblemInstance,
) -> List[Program]:
    tracker = _ScheduleTracker()

    for prog_idx in chromosome:
        prog = programs[prog_idx]
        if _can_add(prog, tracker, instance):
            tracker.add(prog)

    return tracker.as_list()


def _can_add(
    prog: Program,
    tracker: _ScheduleTracker,
    instance: ProblemInstance,
) -> bool:
    if prog.start < instance.opening_time or prog.end > instance.closing_time:
        return False

    if tracker.overlaps(prog.start, prog.end):
        return False

    for block in instance.priority_blocks:
        if block.overlaps_program(prog.start, prog.end):
            if not block.allows(prog.channel_id):
                return False

    if tracker.violates_genre_limit(
        prog.start, prog.genre, instance.max_consecutive_genre
    ):
        return False

    return True
