from __future__ import annotations

from typing import List

from .models import Program, ProblemInstance


def compute_score(schedule: List[Program], instance: ProblemInstance) -> float:
    if not schedule:
        return 0.0

    D = instance.min_duration
    S = instance.switch_penalty

    total: float = 0.0

    for prog in schedule:
        total += prog.score

    for prog in schedule:
        min_overlap = min(D, prog.duration)
        for pref in instance.time_preferences:
            if pref.genre != prog.genre:
                continue
            overlap = pref.overlap_with(prog.start, prog.end)
            if overlap >= min_overlap:
                total += pref.bonus

    for i in range(1, len(schedule)):
        if schedule[i].channel_id != schedule[i - 1].channel_id:
            total -= S

    return total


def compute_attractiveness(prog: Program, instance: ProblemInstance) -> float:
    min_overlap = min(instance.min_duration, prog.duration)
    bonus = sum(
        pref.bonus
        for pref in instance.time_preferences
        if pref.genre == prog.genre
        and pref.overlap_with(prog.start, prog.end) >= min_overlap
    )
    return prog.score + bonus
