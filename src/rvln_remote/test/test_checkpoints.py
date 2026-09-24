"""Tests for the shared checkpoint helpers (no real weights)."""
import pytest
import torch

from rvln_remote.backends._checkpoints import (
    load_checkpoint,
    remove_ddp_prefix,
)


def test_remove_ddp_prefix_strips_module():
    sd = {'module.layer.weight': torch.zeros(1), 'fc.bias': torch.zeros(1)}
    out = remove_ddp_prefix(sd)
    assert set(out.keys()) == {'layer.weight', 'fc.bias'}


def test_remove_ddp_prefix_handles_no_prefix():
    sd = {'layer.weight': torch.zeros(1)}
    out = remove_ddp_prefix(sd)
    assert set(out.keys()) == {'layer.weight'}


def test_load_checkpoint_strips_ddp(tmp_path):
    cp = tmp_path / 'edge_adapter--42_checkpoint.pt'
    torch.save({'module.foo.weight': torch.zeros(2, 3)}, cp)
    loaded = load_checkpoint('edge_adapter', str(tmp_path), step=42)
    assert 'foo.weight' in loaded
    assert loaded['foo.weight'].shape == (2, 3)


def test_load_checkpoint_pose_projector_falls_back_to_proprio_projector(tmp_path):
    """OmniVLA quirk: filename on disk is proprio_projector but referenced as pose_projector."""
    cp = tmp_path / 'proprio_projector--120000_checkpoint.pt'
    torch.save({'foo.weight': torch.zeros(1)}, cp)
    loaded = load_checkpoint('pose_projector', str(tmp_path), step=120000)
    assert 'foo.weight' in loaded


def test_load_checkpoint_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint('nonexistent', str(tmp_path), step=42)
