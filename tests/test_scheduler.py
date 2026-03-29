"""Tests for continuous batching scheduler."""

from engine.sampler import SamplingParams
from engine.scheduler import Scheduler, SequenceStatus


class TestScheduler:
    def test_add_request(self):
        sched = Scheduler()
        seq_id = sched.add_request([1, 2, 3])
        assert seq_id == 0
        assert sched.num_waiting == 1

    def test_schedule_prefill(self):
        sched = Scheduler()
        sched.add_request([1, 2, 3, 4])

        batch = sched.schedule(num_free_blocks=10, block_size=4)
        assert len(batch.prefill_seqs) == 1
        assert len(batch.decode_seqs) == 0

    def test_schedule_respects_block_limit(self):
        sched = Scheduler()
        sched.add_request([1] * 100)  # needs 100/4 = 25 blocks

        batch = sched.schedule(num_free_blocks=5, block_size=4)
        assert len(batch.prefill_seqs) == 0  # not enough blocks

    def test_multiple_requests(self):
        sched = Scheduler(max_batch_size=4)
        for _ in range(3):
            sched.add_request([1, 2, 3])

        batch = sched.schedule(num_free_blocks=100, block_size=4)
        assert len(batch.prefill_seqs) == 3

    def test_max_batch_size(self):
        sched = Scheduler(max_batch_size=2)
        for _ in range(5):
            sched.add_request([1, 2])

        batch = sched.schedule(num_free_blocks=100, block_size=4)
        assert batch.total_seqs == 2
        assert sched.num_waiting == 3

    def test_decode_after_prefill(self):
        sched = Scheduler()
        sched.add_request([1, 2, 3])

        # First schedule: prefill
        batch = sched.schedule(num_free_blocks=10, block_size=4)
        assert len(batch.prefill_seqs) == 1

        # Mark as running
        seq = batch.prefill_seqs[0]
        sched.mark_running(seq)

        # Second schedule: decode
        batch = sched.schedule(num_free_blocks=10, block_size=4)
        assert len(batch.decode_seqs) == 1

    def test_finish_sequence(self):
        sched = Scheduler()
        seq_id = sched.add_request([1, 2])

        batch = sched.schedule(num_free_blocks=10, block_size=4)
        sched.mark_running(batch.prefill_seqs[0])

        sched.update_after_step({seq_id})
        assert sched.num_running == 0

        finished = sched.get_finished()
        assert len(finished) == 1
        assert finished[0].seq_id == seq_id

    def test_has_pending(self):
        sched = Scheduler()
        assert not sched.has_pending()

        sched.add_request([1])
        assert sched.has_pending()

    def test_sequence_is_finished_max_tokens(self):
        sched = Scheduler()
        params = SamplingParams(max_tokens=3)
        sched.add_request([1, 2], params)

        batch = sched.schedule(num_free_blocks=10, block_size=4)
        seq = batch.prefill_seqs[0]
        seq.generated_token_ids = [10, 11, 12]
        assert seq.is_finished()

    def test_sequence_is_finished_stop_token(self):
        sched = Scheduler()
        params = SamplingParams(stop_token_ids=[999])
        sched.add_request([1, 2], params)

        batch = sched.schedule(num_free_blocks=10, block_size=4)
        seq = batch.prefill_seqs[0]
        seq.generated_token_ids = [10, 999]
        assert seq.is_finished()
