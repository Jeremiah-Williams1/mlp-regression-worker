"""
Standalone sanity check for predict.py -- no Redis, no worker.py, no k8s.

Run this right after training, and again any time you touch predict.py
or retrain the model, before wiring anything into the orchestrator:

    python smoke_test.py

If this prints a sane-looking MPG number, predict.py is good. If it
raises, the error messages in predict.py are written to tell you exactly
what's wrong (missing model dir, missing feature_columns.json, mismatch
between them, missing/non-numeric fields in the sample input).
"""

from predict import make_prediction

# Real values for a 1970 Buick Skylark 320 (V8) from the Auto MPG dataset --
# actual MPG for this car is 18, so this is also a rough "does this look
# right" gut check, not just a "did it crash" check.
sample_input = {
    "Cylinders": 8,
    "Displacement": 307.0,
    "Horsepower": 130.0,
    "Weight": 3504.0,
    "Acceleration": 12.0,
    "Model Year": 70,
    "Origin": "USA",
}

if __name__ == "__main__":
    result = make_prediction(sample_input)
    print(f"Predicted MPG: {result:.2f} (actual for this car: ~18)")
