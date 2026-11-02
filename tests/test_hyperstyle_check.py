"""hyperstyle.same_generator compares checkpoint files; must report 0 for identical and raise on layout mismatch."""
import pytest
import torch

from medsg.hyperstyle import same_generator


def _ck(path, d):
    torch.save({"state_dict": {f"decoder.{k}": v for k, v in d.items()} | {"encoder.x": torch.ones(2)}}, path)


def test_same_generator(tmp_path):
    w = {"conv.weight": torch.randn(4, 3), "to_rgb.bias": torch.randn(3)}
    _ck(tmp_path / "a.pt", w)
    _ck(tmp_path / "b.pt", {k: v.clone() for k, v in w.items()})
    assert same_generator(tmp_path / "a.pt", tmp_path / "b.pt") == 0.0
    _ck(tmp_path / "c.pt", {"conv.weight": torch.randn(4, 3)})
    with pytest.raises(ValueError):
        same_generator(tmp_path / "a.pt", tmp_path / "c.pt")
