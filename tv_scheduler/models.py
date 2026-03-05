from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class Program:

    program_id: str
    channel_id: int
    start: int
    end: int
    genre: str
    score: int

    @property
    def duration(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:
        return (f"Program({self.program_id!r}, ch={self.channel_id}, "
                f"{self.start}-{self.end}, {self.genre}, score={self.score})")


@dataclass
class PriorityBlock:

    start: int
    end: int
    allowed_channels: Set[int]

    def overlaps_program(self, prog_start: int, prog_end: int) -> bool:
        return prog_start < self.end and prog_end > self.start

    def allows(self, channel_id: int) -> bool:
        return channel_id in self.allowed_channels


@dataclass
class PreferenceInterval:

    start: int
    end: int
    genre: str
    bonus: int

    def overlap_with(self, prog_start: int, prog_end: int) -> int:
        return max(0, min(self.end, prog_end) - max(self.start, prog_start))


@dataclass
class ProblemInstance:

    opening_time: int
    closing_time: int
    min_duration: int
    max_consecutive_genre: int
    channels_count: int
    switch_penalty: int
    termination_penalty: int
    priority_blocks: List[PriorityBlock]
    time_preferences: List[PreferenceInterval]
    programs: List[Program]
