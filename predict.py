import json
import os

import tensorflow as tf

MODEL_PATH = os.getenv("MODEL_PATH", "model/auto_mpg_mlp")
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "model/feature_columns.json")

if not os.path.isdir(MODEL_PATH):
    raise FileNotFoundError(
        f"No SavedModel found at '{MODEL_PATH}'. Train it with "
        f"notebooks/train_model.ipynb first, or set MODEL_PATH."
    )

_loaded = tf.saved_model.load(MODEL_PATH)

if "serving_default" not in _loaded.signatures:
    raise RuntimeError(
        f"SavedModel at '{MODEL_PATH}' has no 'serving_default' signature. "
        f"Found: {list(_loaded.signatures.keys())}. "
        f"Make sure you used model.export(), not model.save()."
    )

_infer = _loaded.signatures["serving_default"]
_input_name = list(_infer.structured_input_signature[1].keys())[0]
_output_name = list(_infer.structured_outputs.keys())[0]

if not os.path.isfile(FEATURE_COLUMNS_PATH):
    raise FileNotFoundError(
        f"No feature_columns.json found at '{FEATURE_COLUMNS_PATH}'."
    )

with open(FEATURE_COLUMNS_PATH) as f:
    FEATURE_COLUMNS = json.load(f)

_expected_width = _infer.structured_input_signature[1][_input_name].shape[-1]
if _expected_width != len(FEATURE_COLUMNS):
    raise RuntimeError(
        f"Model/feature mismatch: model expects {_expected_width} features "
        f"but feature_columns.json has {len(FEATURE_COLUMNS)}: {FEATURE_COLUMNS}"
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
    missing = [f for f in REQUIRED_RAW_FIELDS if f not in input_data]
    if missing:
        raise ValueError(f"missing required field(s): {missing}")

    row = {k: v for k, v in input_data.items() if k != "Origin"}
    row.update(_encode_origin(input_data["Origin"]))

    try:
        ordered = [float(row[col]) for col in FEATURE_COLUMNS]
    except KeyError as e:
        raise ValueError(f"internal feature column mismatch: {e}") from e
    except (TypeError, ValueError) as e:
        raise ValueError(f"non-numeric value in input_data: {e}") from e

    x = tf.constant([ordered], dtype=tf.float32)
    result = _infer(**{_input_name: x})
    prediction = result[_output_name].numpy()
    return float(prediction[0][0])