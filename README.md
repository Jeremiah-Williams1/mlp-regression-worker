# mlp-regression-worker

Second worker for the inference orchestrator. Predicts car fuel economy
(MPG) from a small set of features using a Keras MLP — a `regression`
job type, alongside the existing XGBoost `classification` worker.

## Repo layout

```
mlp-regression-worker/
├── notebooks/
│   └── train_model.ipynb      # run this first — produces model/
├── model/
│   ├── auto_mpg_mlp/          # SavedModel, written by the notebook
│   └── feature_columns.json   # input feature order, written by the notebook
├── predict.py                 # the only file with model-specific logic
├── worker.py                  # 5-line glue: base.run_worker(job_type, predict_fn)
├── smoke_test.py              # local check, no Redis/k8s needed
├── requirements.txt           # runtime deps (what ships in the Docker image)
├── requirements-train.txt     # extra deps for the notebook only
├── Dockerfile
├── .dockerignore
└── .gitignore
```

`model/` is intentionally **not** gitignored — train once locally, commit
the SavedModel + `feature_columns.json`, and the Docker build just copies
them in. There's no training step in the container or in CI for this repo.

## 1. Set up a local environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
```

(`requirements-train.txt` alone is enough for the notebook — it pulls in
plain `tensorflow`, pandas, matplotlib, jupyter. You don't need
`requirements.txt`, which is the *runtime* image's deps, including the
git-installed `orchestrator-worker-base`.)

## 2. Train

```bash
jupyter notebook notebooks/train_model.ipynb
```

Run all cells top to bottom. The last cell reloads the saved model fresh
and asserts it matches `feature_columns.json` — if that cell errors, fix
it before going further; don't commit inconsistent artifacts.

Expect a test MAE somewhere in the 2–3 MPG range; if it's wildly off,
something upstream (data cleaning, column order) is probably wrong rather
than needing more epochs — this is a tiny, easy dataset.

## 3. Smoke test predict.py directly (no Redis, no Docker)

Once you also have the base package available locally (see step 4), or
even before that — `smoke_test.py` only imports `predict`, not `worker`:

```bash
pip install tensorflow-cpu numpy   # if you haven't already via requirements-train.txt
python smoke_test.py
```

Expect output like:

```
Predicted MPG: 17.84 (actual for this car: ~18)
```

If this fails, the error message tells you which of these it is:
missing `model/auto_mpg_mlp/`, missing `model/feature_columns.json`,
a mismatch between the two, or a bad/missing field in the sample input.

## 4. Install the shared base, point requirements.txt at your fork

Before this repo can actually run as a worker, you need
`orchestrator-worker-base` pushed to your own GitHub, and
`requirements.txt` here needs to point at it:

```
git+https://github.com/<you>/orchestrator-worker-base.git
```

Replace `<you>` with your GitHub username — same pattern as the existing
classification worker.

## 5. Build and run the container

```bash
docker build -t mlp-regression-worker:dev .
docker run --rm \
  -e REDIS_URL=redis://redis-svc:6379 \
  -e PUSHGATEWAY_URL=http://pushgateway-svc:9091 \
  mlp-regression-worker:dev
```

On startup you should see (via `base.py`'s logging):

```
regression worker started, polling queue:regression...
```

## 6. Wire into the infra repo

This repo only produces an image — the orchestrator (Go API, Redis, k8s
manifests, Helm chart) lives in the separate infra repo. From there:

- `k8s/worker-regression/{deployment,configmap,scaledobject}.yaml`,
  copied from an existing worker folder and renamed.
- `ScaledObject` for this worker points at the queue depth endpoint for
  `regression` (the Go API's `key := fmt.Sprintf("queue:%s", job.Type)`
  pattern already covers this — no Go changes needed).
- Same `PUSHGATEWAY_URL` Pushgateway, same Loki/Promtail — both workers'
  metrics and logs show up in the same Grafana dashboard, labeled by
  `type`.

## Sample job payload

For testing against the API / load-test script:

```json
{
  "type": "regression",
  "input": {
    "Cylinders": 8,
    "Displacement": 307.0,
    "Horsepower": 130.0,
    "Weight": 3504.0,
    "Acceleration": 12.0,
    "Model Year": 70,
    "Origin": "USA"
  }
}
```

Expected result shape: `{"predicted_mpg": <float>}`.
