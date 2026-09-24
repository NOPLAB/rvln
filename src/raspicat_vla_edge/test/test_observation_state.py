"""Goal generation and camera freshness remain safe across callback races."""
import numpy as np

from raspicat_vla_edge.observation_state import CameraFrameStore, ObservationLedger


def test_goal_change_rejects_observation_preprocessed_for_old_goal():
    ledger = ObservationLedger()
    floors = []
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    ledger.change_goal('first', floors.append)
    _, old_generation = ledger.snapshot_goal()
    first_id = ledger.record_sent(old_generation, frame)
    ledger.change_goal('second', floors.append)

    assert floors == [0, first_id]
    assert ledger.record_sent(old_generation, frame) is None
    assert ledger.take_reply_frame(first_id) is None
    _, new_generation = ledger.snapshot_goal()
    assert ledger.record_sent(new_generation, frame) == first_id + 1


def test_camera_frame_expires_and_clears():
    frames = CameraFrameStore()
    frame = np.ones((2, 2, 3), dtype=np.uint8)
    frames.put(frame, stamp_ns=100)
    np.testing.assert_array_equal(frames.fresh(10, now_ns=105), frame)
    assert frames.has_fresh(10, now_ns=105)
    assert frames.fresh(10, now_ns=111) is None
    assert not frames.has_fresh(10, now_ns=111)
    frames.clear()
    assert frames.fresh(10, now_ns=105) is None
