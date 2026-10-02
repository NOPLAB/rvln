"""Checks for joining PhysX contacts to a live ROS episode."""

import json

import pytest

from isaac_rvln.live_results import attach_contacts


def test_attach_contacts_counts_only_during_episode(tmp_path):
    result = tmp_path / "episode.json"
    contact = tmp_path / "episode.contacts.json"
    result.write_text(
        json.dumps(
            {
                "pose_source": "isaac_ground_truth",
                "wall_started_at": 20.0,
                "wall_ended_at": 30.0,
                "contact_log": None,
            }
        ),
        encoding="utf-8",
    )
    contact.write_text(
        json.dumps(
            {
                "schema": 1,
                "simulator": "isaac_sim",
                "samples": {"physics_steps": 100},
                "receive_errors": 0,
                "collisions": 3,
                "events": [
                    {"wall_monotonic_sec": 19.0},
                    {"wall_monotonic_sec": 24.0},
                    {"wall_monotonic_sec": 31.0},
                ],
            }
        ),
        encoding="utf-8",
    )

    row = attach_contacts(result, contact)

    assert row["collisions"] == 1
    assert row["contact_events"][0]["t_sec"] == 4.0
    assert row["contact_samples"]["physics_steps"] == 100
    assert row["contact_log"] == contact.name
    with pytest.raises(ValueError, match="already has contact data"):
        attach_contacts(result, contact)


def test_attach_contacts_rejects_mismatched_simulator(tmp_path):
    result = tmp_path / "episode.json"
    contact = tmp_path / "episode.contacts.json"
    result.write_text(
        json.dumps(
            {
                "pose_source": "isaac_ground_truth",
                "wall_started_at": 20.0,
                "wall_ended_at": 30.0,
                "contact_log": None,
            }
        ),
        encoding="utf-8",
    )
    contact.write_text(
        json.dumps(
            {
                "schema": 1,
                "simulator": "gazebo",
                "samples": {"physics_steps": 100},
                "receive_errors": 0,
                "collisions": 0,
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not an Isaac Sim report"):
        attach_contacts(result, contact)
