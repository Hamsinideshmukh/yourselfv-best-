"""Pre-declared edit specification (one source).

Everything here is fixed *before* looking at any edit result (metric set and primary endpoint approved 2026-10-31).
"""
from __future__ import annotations

from .metrics2d import METRICS_2D

EDIT_METRICS = list(METRICS_2D)            # 10 frontal 2D ratios (IPD units); the in-loop measurement

# E3 primary targets: the four with real HDA change beyond noise (Stage 1b).
E3_TARGETS = ["brow_height", "palpebral_height", "nose_width", "mouth_width"]

# Clinically / geometrically coupled pairs: NOT protected when editing the key metric (still reported).
#  - brow <-> upper eyelid: brow position changes after blepharoplasty (Liu et al. 2023, Aesthetic Plast Surg).
#  - palpebral height <-> width, width <-> intercanthal: same eyelid/canthal landmarks (canthoplasty).
#  - nose_length <-> lower_face_height: share the subnasale landmark (Stage 1b, rho = -0.55 artefact).
#  - face_width <-> jaw_width: both lower-face width landmarks (facial-bone procedures move both).
COUPLED = {
    "brow_height": ["palpebral_height"],
    "palpebral_height": ["brow_height", "palpebral_width"],
    "palpebral_width": ["palpebral_height", "intercanthal"],
    "intercanthal": ["palpebral_width"],
    "nose_length": ["lower_face_height"],
    "lower_face_height": ["nose_length"],
    "face_width": ["jaw_width"],
    "jaw_width": ["face_width"],
    "nose_width": [],
    "mouth_width": [],
}

# Nuisance quantities protected during edits, with tolerances (no synthetic noise floor exists for them):
#  pose: 2 deg (HDA real yaw SD 1.6 deg; pair filter 10 deg);
#  expression: 0.05 blendshape units (size of the real HDA pre->post expression shift, Stage 1b);
#  generator-frame IPD at 512 px: 0.5 px (~0.4 %, the Stage-1b common-mode IPD shift).
NUISANCE = ["yaw", "pitch", "roll", "bs_smile", "bs_jawOpen", "bs_blink", "bs_squint", "bs_browInnerUp",
            "bs_browDown"]
NUISANCE_TOL = {"yaw": 2.0, "pitch": 2.0, "roll": 2.0, "bs_smile": 0.05, "bs_jawOpen": 0.05,
                "bs_blink": 0.05, "bs_squint": 0.05, "bs_browInnerUp": 0.05, "bs_browDown": 0.05}
# Expression read-outs that ARE the anatomy change (Stage 1b: a lifted brow is read as browInnerUp; a more
# open eye as less blink/squint), so protecting them would forbid the edit itself. Not protected; reported.
COUPLED_NUISANCE = {"brow_height": ["bs_browInnerUp", "bs_browDown"],
                    "palpebral_height": ["bs_blink", "bs_squint"]}
# Derived blendshapes: mean of left/right
BS_DERIVED = {"bs_smile": ["bs_mouthSmileLeft", "bs_mouthSmileRight"], "bs_blink": ["bs_eyeBlinkLeft", "bs_eyeBlinkRight"],
              "bs_squint": ["bs_eyeSquintLeft", "bs_eyeSquintRight"], "bs_browDown": ["bs_browDownLeft", "bs_browDownRight"],
              "bs_jawOpen": ["bs_jawOpen"], "bs_browInnerUp": ["bs_browInnerUp"]}
IPD_TOL_PX = 0.5
CORRECTION_NUISANCE = ["yaw", "pitch", "bs_smile"]   # extra closed-loop correction directions

# Closed loop (D6)
LOOP_MAX_ITERS = 5
LOOP_TARGET_TOL_MDC = 0.5        # stop when |achieved - requested| <= 0.5 MDC_t
LOOP_PROTECT_TOL_MDC = 1.0       # ... and every protected metric moved <= 1 MDC
LOOP_K_AVG = 4                   # measurement averaging variants: orig, flip, +1 % shift, -1 % shift
LOOP_DAMPING = 0.1
LOOP_N_CORRECT = 3               # most-leaking protected metrics added to the correction subspace

# Controllability criteria (Stage 4, success definition)
CTRL_R2_FRAC = 0.5               # held-out R² >= 0.5 R²_max
CTRL_SLOPE = (0.8, 1.2)          # open-loop achieved/requested slope
CTRL_LEAK_MDC = 0.5              # median measured leakage < 0.5 MDC on each protected metric
