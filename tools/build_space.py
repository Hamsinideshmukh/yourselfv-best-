"""Assemble the runnable app folder (build/space) from app/ + the minimal medsg subset + assets + HyperStyle code.

Only these files are copied, so no dataset image, private output or model weight can end up in the Space.
Weights are downloaded at run time (MediaPipe task file from Google; HyperStyle weights from a Hugging Face mirror,
loaded only if their SHA-256 equals the official release).
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "build" / "space"
MODULES = ["__init__.py", "config.py", "landmarks.py", "metrics2d.py", "deform.py", "transfer.py", "measure.py",
           "editspec.py", "editor.py", "align.py", "gan.py", "judges.py", "quality.py", "hyperstyle.py"]
ASSETS = ["patterns_v1.npz", "editor_v1.npz", "quality_v1.json"]


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "medsg" / "assets").mkdir(parents=True)
    for f in ("app.py", "requirements.txt", "README.md"):
        shutil.copy(ROOT / "app" / f, DST / f)
    for m in MODULES:
        shutil.copy(ROOT / "src" / "medsg" / m, DST / "medsg" / m)
    for a in ASSETS:
        shutil.copy(ROOT / "src" / "medsg" / "assets" / a, DST / "medsg" / "assets" / a)
    hs = ROOT / "repos" / "hyperstyle"                   # MIT; code only (weights are downloaded and hash-checked)
    if not hs.exists():
        raise FileNotFoundError("clone it first: git clone https://github.com/yuval-alaluf/hyperstyle repos/hyperstyle")
    for pkg in ("models", "configs", "utils"):
        shutil.copytree(hs / pkg, DST / "vendor" / "hyperstyle" / pkg,
                        ignore=shutil.ignore_patterns("__pycache__", "*.cu", "*.cpp", "mtcnn"))
    shutil.copy(hs / "LICENSE", DST / "vendor" / "hyperstyle" / "LICENSE")
    ex = ROOT / "app" / "examples"
    if ex.exists():
        shutil.copytree(ex, DST / "examples")
    (DST / ".gitignore").write_text("models/\n__pycache__/\n")
    bad = [p for p in DST.rglob("*") if p.suffix.lower() in {".pt", ".pth", ".pkl", ".onnx", ".task"}
           or "flattened_images" in str(p) or "private" in str(p)]
    assert not bad, f"forbidden files in Space build: {bad}"
    print("built", DST, "|", sum(1 for _ in DST.rglob("*") if _.is_file()), "files,",
          round(sum(p.stat().st_size for p in DST.rglob("*") if p.is_file()) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
