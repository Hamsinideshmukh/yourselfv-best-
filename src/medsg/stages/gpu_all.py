"""One unattended GPU run: Stage 2 (inversion benchmark) then Stage 3 (synthetic latent<->anatomy data).

Stage 3 is resumable (chunk files), so a re-run after interruption continues where it stopped.
At the end two zips are written to /kaggle/working/outputs:
  gpu_results.zip         tables, metadata, synthetic chunks (no real faces) -> shareable
  gpu_private_hda.zip     HDA latent codes + example images (research-only data) -> keep private
"""
from __future__ import annotations

import time
import zipfile

from ..config import get_config
from ..runmeta import write_run_meta
from . import stage2_invert, stage3_synth


def _zip(paths, root, dst) -> None:
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as z:
        for f in paths:
            z.write(f, f.relative_to(root))
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    cfg = get_config()
    t0 = time.time()
    if not (cfg.out_dir / "stage2" / "pti.csv").exists():
        stage2_invert.main()
    print(f"== stage 2 done at {time.time() - t0:.0f}s", flush=True)
    stage3_synth.main()
    print(f"== stage 3 done at {time.time() - t0:.0f}s", flush=True)

    write_run_meta(cfg.out_dir / "gpu_run", cfg.as_dict(),
                   {"e4e_ffhq_encode.pt": cfg.models_dir / "e4e_ffhq_encode.pt",
                    "face_landmarker.task": cfg.models_dir / "face_landmarker.task"},
                   {"stage": "gpu_all", "seconds": time.time() - t0,
                    "e4e_source": (cfg.models_dir / "e4e_source.txt").read_text().strip()})
    o = cfg.out_dir
    public = [*(o / "stage2").glob("*.csv"), *(o / "stage3").glob("*.json"), *(o / "stage3").glob("*.npy"),
              *(o / "stage3" / "chunks").glob("*.npz"), o / "gpu_run" / "run_meta.json", o / "splits.json"]
    _zip(public, o, o / "gpu_results.zip")
    _zip(list((o / "stage2" / "private").glob("*")), o, o / "gpu_private_hda.zip")


if __name__ == "__main__":
    main()
