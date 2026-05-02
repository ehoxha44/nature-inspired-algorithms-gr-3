from __future__ import annotations

import bisect
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


def relaxed_interval_upper_bound(programs: List[Program], instance: ProblemInstance) -> float:
    """
    Maximum total attractiveness over pairwise non-overlapping programs
    (interval scheduling DP, priority-block violations get weight 0).

    Any feasible decoded schedule has score ≤ this value (switch penalties
    only decrease the objective vs. raw attractiveness sum).
    """
    n = len(programs)
    if n == 0:
        return 0.0

    eff: List[float] = []
    for p in programs:
        violates_block = any(
            blk.overlaps_program(p.start, p.end) and not blk.allows(p.channel_id)
            for blk in instance.priority_blocks
        )
        eff.append(0.0 if violates_block else compute_attractiveness(p, instance))

    sort_order = sorted(range(n), key=lambda i: programs[i].end)
    s_end = [programs[sort_order[k]].end for k in range(n)]
    s_start = [programs[sort_order[k]].start for k in range(n)]
    s_eff = [eff[sort_order[k]] for k in range(n)]

    def pred(k: int) -> int:
        return bisect.bisect_right(s_end, s_start[k], lo=0, hi=k) - 1

    dp = [0.0] * (n + 1)
    for k in range(n):
        p_k = pred(k)
        dp[k + 1] = max(dp[k], s_eff[k] + dp[p_k + 1])

    return float(dp[n])
