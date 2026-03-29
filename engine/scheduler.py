"""
Continuous batching scheduler.

Manages request lifecycle and batching:
- New requests arrive asynchronously via add_request()
- Scheduler forms batches mixing prefill + decode operations
- Handles preemption when memory is exhausted
- Tracks per-sequence state: WAITING → RUNNING → FINISHED
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from .sampler import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class SequenceState:
    seq_id: int
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    status: SequenceStatus = SequenceStatus.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    prefill_done: bool = False
    arrival_time: float = 0.0
    first_token_time: float = 0.0
    finish_time: float = 0.0

    @property
    def total_len(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.generated_token_ids)

    def is_finished(self) -> bool:
        if self.num_generated >= self.sampling_params.max_tokens:
            return True
        if self.generated_token_ids and self.generated_token_ids[-1] in self.sampling_params.stop_token_ids:
            return True
        return False


@dataclass
class SchedulerBatch:
    """A batch of sequences to process in one step."""
    prefill_seqs: list[SequenceState] = field(default_factory=list)
    decode_seqs: list[SequenceState] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefill_seqs and not self.decode_seqs

    @property
    def total_seqs(self) -> int:
        return len(self.prefill_seqs) + len(self.decode_seqs)


class Scheduler:
    """
    Continuous batching scheduler.

    Each step:
    1. Move waiting sequences to running if cache space available
    2. Form batch: prefill new sequences + decode running sequences
    3. If out of memory, preempt longest running sequence
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len

        self.waiting: list[SequenceState] = []
        self.running: list[SequenceState] = []
        self.finished: list[SequenceState] = []

        self._next_seq_id = 0

    def add_request(
        self,
        prompt_token_ids: list[int],
        sampling_params: Optional[SamplingParams] = None,
    ) -> int:
        """Add a new request. Returns sequence ID."""
        seq_id = self._next_seq_id
        self._next_seq_id += 1

        if sampling_params is None:
            sampling_params = SamplingParams()

        seq = SequenceState(
            seq_id=seq_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            arrival_time=time.time(),
        )
        self.waiting.append(seq)
        return seq_id

    def schedule(self, num_free_blocks: int, block_size: int) -> SchedulerBatch:
        """
        Schedule the next batch.

        Args:
            num_free_blocks: available cache blocks
            block_size: tokens per block
        Returns:
            SchedulerBatch with prefill and decode sequences
        """
        batch = SchedulerBatch()
        available_blocks = num_free_blocks

        # 1. Continue decoding running sequences (each needs at most 1 new block)
        still_running = []
        for seq in self.running:
            # Check if this decode might need a new block
            blocks_needed = 0
            if seq.total_len % block_size == 0:
                blocks_needed = 1

            if available_blocks >= blocks_needed and batch.total_seqs < self.max_batch_size:
                batch.decode_seqs.append(seq)
                available_blocks -= blocks_needed
                still_running.append(seq)
            else:
                # Preempt: move back to waiting
                seq.status = SequenceStatus.WAITING
                self.waiting.insert(0, seq)

        self.running = still_running

        # 2. Admit waiting sequences for prefill
        remaining_waiting = []
        for seq in self.waiting:
            if batch.total_seqs >= self.max_batch_size:
                remaining_waiting.append(seq)
                continue

            # Calculate blocks needed for this prompt
            blocks_needed = (len(seq.prompt_token_ids) + block_size - 1) // block_size

            if available_blocks >= blocks_needed and seq.total_len <= self.max_seq_len:
                seq.status = SequenceStatus.RUNNING
                batch.prefill_seqs.append(seq)
                available_blocks -= blocks_needed
            else:
                remaining_waiting.append(seq)

        self.waiting = remaining_waiting

        return batch

    def update_after_step(self, finished_seq_ids: set[int]):
        """Move finished sequences out of running."""
        still_running = []
        for seq in self.running:
            if seq.seq_id in finished_seq_ids:
                seq.status = SequenceStatus.FINISHED
                seq.finish_time = time.time()
                self.finished.append(seq)
            else:
                still_running.append(seq)
        self.running = still_running

    def mark_running(self, seq: SequenceState):
        """Mark a prefilled sequence as running."""
        if seq not in self.running:
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

    def get_finished(self) -> list[SequenceState]:
        """Pop and return all finished sequences."""
        finished = self.finished
        self.finished = []
        return finished

    def has_pending(self) -> bool:
        return bool(self.waiting or self.running)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def num_running(self) -> int:
        return len(self.running)
