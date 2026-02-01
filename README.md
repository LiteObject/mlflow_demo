# MLflow Demo (Iris)

The main purpose of this repo is to **demo MLflow** (experiment tracking + model logging) with a tiny, repeatable example.

`main.py` trains a simple scikit-learn Logistic Regression classifier on the Iris dataset and logs to MLflow:

- Parameters (model hyperparameters)
- Metrics (accuracy, precision, recall, F1)
- The trained model artifact (with an inferred signature)

This gives you something you can run in ~seconds and immediately inspect in the MLflow UI.

## What is MLflow?

MLflow is an open source platform for the machine learning lifecycle. It is designed to tackle four primary functions:

- **MLflow Tracking**: Recording and querying experiments: code, data, config, and results.
- **MLflow Models**: Packaging data science code in a format to reproduce runs on any platform.
- **MLflow Model Registry**: Managing the full lifecycle of an MLflow Model.
- **MLflow Projects**: A standard format for packaging reusable data science code.

In this demo, we use the **Tracking** component to log our Logistic Regression experiments.

### Alternatives to MLflow

While MLflow is a standard choice, there are other excellent tools in the MLOps ecosystem:

- **Weights & Biases (W&B)**: Very popular in the Deep Learning community; excellent visualization and collaboration features.
- **Comet ML**: Clear dashboarding and comparison tools.
- **Neptune.ai**: Lightweight metadata store for MLOps.
- **DVC (Data Version Control)**: Heavy focus on data versioning (git for data) and reproducible pipelines.
- **ClearML**: An open-source platform that automates the ML lifecycle.

## What `main.py` does

1. Configures MLflow tracking:
   - Tries to use an MLflow server at `http://localhost:5000`
   - If that fails, it falls back to local file-based tracking in `./mlruns`
2. Loads the Iris dataset from scikit-learn.
3. Splits into train/test.
4. Trains a `LogisticRegression` model.
5. Computes evaluation metrics.
6. Starts an MLflow run and logs params/metrics/model.
7. Loads the logged model back and runs predictions (sanity check).
8. Prints a small preview of predictions vs actuals.

## What to demo in the MLflow UI

After running the script, open the MLflow UI and show:

- **Experiment**: `MLflow Tutorial`
- **Runs table**: start time, metrics columns, params columns
- **Run details**:
   - Params: `solver`, `max_iter`, `random_state`
   - Metrics: `accuracy`, `precision`, `recall`, `f1`
   - Artifacts: the logged sklearn model and its `MLmodel` metadata
   - Signature: input/output schema inferred by `infer_signature`

## Prerequisites

- Python 3.9+ recommended
- A virtual environment (recommended)

## Install

Create and activate a virtual environment (example using `venv`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

MLflow is split into two parts:

- `mlflow server`: the **tracking server** (and UI) that receives logs over HTTP.
- `mlflow ui`: the **UI only**, reading runs from a local backend store.

### Option A (recommended): Run with an MLflow server

In one terminal, start the MLflow server:

```powershell
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri ./mlruns --default-artifact-root ./mlruns
```

Leave this running (it keeps the server alive).

In a second terminal, run the script:

```powershell
python main.py
```

Open the MLflow UI in your browser:

- http://127.0.0.1:5000

### Option B: Run with local file-based tracking

If you don’t start an MLflow server, the script will automatically fall back to local tracking in `./mlruns`.

Run the script:

```powershell
python main.py
```

To view the local runs in an MLflow UI, start the UI pointing at the local folder:

```powershell
mlflow ui --backend-store-uri ./mlruns
```

Then open:

- http://127.0.0.1:5000

## Troubleshooting

- **Port already in use**: change the port, e.g. `--port 5001` and open `http://127.0.0.1:5001`.
- **PowerShell venv activation blocked**: you may need `Set-ExecutionPolicy -Scope Process Bypass` and then re-run `./.venv/Scripts/Activate.ps1`.

## Notes

- The script uses weighted precision/recall/F1 and sets `zero_division=0` to avoid metric errors in edge cases.
- The experiment name is set to `MLflow Tutorial`.
- By default, MLflow artifacts and runs will appear either in your tracking server’s configured backend store, or in the local `./mlruns` folder when falling back.
