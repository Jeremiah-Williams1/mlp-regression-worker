"""
Model-specific inference for the regression worker.

This is the ONLY file in this repo that base.py / worker.py care about
the contents of: it loads the trained model once, and exposes a single
make_prediction(dict) -> float function. Everything else (Redis, BRPOP,
Pushgateway, status writes) lives in orchestrator-worker-base.

Run `python smoke_test.py` after training to sanity-check this file in
isolation, before wiring it into worker.py / Redis / k8s.
"""

import json
import os

import numpy as np
import tensorflow as tf

MODEL_PATH = os.getenv("MODEL_PATH", "model/auto_mpg_mlp.keras")
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "model/feature_columns.json")

# --- Load once at import time -----------------------------------------
# The BRPOP loop in base.py calls make_prediction() once per job. Loading
# the SavedModel here (module import time, i.e. once per worker process)
# instead of inside make_prediction() avoids reloading it on every job.

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f"No SavedModel found at '{MODEL_PATH}'. Train it with "
        f"notebooks/train_model.ipynb first (it writes here), or set "
        f"MODEL_PATH to point at the right directory."
    )

_model = tf.keras.models.load_model(MODEL_PATH)

if not os.path.isfile(FEATURE_COLUMNS_PATH):
    raise FileNotFoundError(
        f"No feature_columns.json found at '{FEATURE_COLUMNS_PATH}'. "
        f"This file is written by the training notebook alongside the "
        f"model and records the exact input column order the model "
        f"expects -- it has to ship with the model, not be hand-typed here."
    )

with open(FEATURE_COLUMNS_PATH) as f:
    FEATURE_COLUMNS = json.load(f)

# --- Catch model/columns drift at startup, not at request time --------
# If someone retrains the model with a different feature set but forgets
# to regenerate feature_columns.json (or vice versa), this fails loudly
# the moment the worker process starts, instead of quietly returning
# wrong predictions for every job.
_expected_width = _model.input_shape[-1]
if _expected_width != len(FEATURE_COLUMNS):
    raise RuntimeError(
        f"Model/feature mismatch: the model at '{MODEL_PATH}' expects "
        f"{_expected_width} input features, but '{FEATURE_COLUMNS_PATH}' "
        f"lists {len(FEATURE_COLUMNS)} columns: {FEATURE_COLUMNS}. "
        f"The model and the column list were saved from different "
        f"training runs -- retrain, or re-save feature_columns.json from "
        f"the run that produced this model."
    )

ORIGIN_VALUES = ("USA", "Europe", "Japan")
REQUIRED_RAW_FIELDS = [
    "Cylinders", "Displacement", "Horsepower",
    "Weight", "Acceleration", "Model Year", "Origin",
]


def _encode_origin(origin: str) -> dict:
    if origin not in ORIGIN_VALUES:
        raise ValueError(
            f"'Origin' must be one of {list(ORIGIN_VALUES)}, got {origin!r}"
        )
    return {f"Origin_{name}": int(name == origin) for name in ORIGIN_VALUES}


def make_prediction(input_data: dict) -> float:
    """
    input_data: dict with keys Cylinders, Displacement, Horsepower, Weight,
                Acceleration, Model Year (numeric), and Origin (one of
                "USA" / "Europe" / "Japan").
    Returns: predicted MPG as a plain float.
    """
    missing = [f for f in REQUIRED_RAW_FIELDS if f not in input_data]
    if missing:
        raise ValueError(f"missing required field(s): {missing}")

    row = {k: v for k, v in input_data.items() if k != "Origin"}
    row.update(_encode_origin(input_data["Origin"]))

    try:
        ordered = [float(row[col]) for col in FEATURE_COLUMNS]
    except KeyError as e:
        # Should be unreachable given the drift check above + REQUIRED_RAW_FIELDS,
        # but kept as a clear error rather than a raw KeyError if it ever happens.
        raise ValueError(f"internal feature column mismatch: {e}") from e
    except (TypeError, ValueError) as e:
        raise ValueError(f"non-numeric value in input_data: {e}") from e

    x = np.array([ordered], dtype=np.float32)
    # verbose=0: Keras prints a progress bar per .predict() call by default,
    # which floods worker logs (and Loki) with noise on every single job.
    prediction = _model.predict(x, verbose=0)
    return float(prediction[0][0])
