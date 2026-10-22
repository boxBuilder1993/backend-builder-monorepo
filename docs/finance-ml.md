# ML in the finance app

Status: proposal, 2026-10-22. Machine learning on the market data as part of the finance app: the
definitions, runs, models and inferences are entities, the work is reactions, the artifacts are files in
the same store as the data. This replaces the earlier plan for a standalone `mlkit` library; what that
design settled about semantics (one vector target, the schema as the inference contract, metrics per
split, the artifact layout) is kept, in entity terms. [finance-modelling.md](finance-modelling.md) says
what we fit first; this document says how fitting works.

## The idea in one paragraph

A **Fit** is a definition: a SQL query that produces the rows to learn from, the features and their
kinds, the target, an optional validation query, and a search over algorithms and parameters. Running
it is a **FitRun**, with the query parameters actually used, and it produces one **Model** per point of
the search, each with its own id, its artifact uploaded to the store and its validation metrics stored
on the row. An **InferRun** takes a model id and a query, checks the query against the model's schema,
scores the rows and writes a table of key plus predictions as a file in the store. All of it is visible
in the console as runs and rows, and queryable from DuckDB next to the prices.

## Entities

```python
class Fit(Entity):
    """A definition. Immutable once run; a change is a new Fit."""
    name: str
    description: str = ""
    query: str                    # SQL over the DuckDB views; yields key + features + target columns
    params: dict[str, Any]        # the query's named parameters and their defaults, e.g. {"end_date": "2026-09-30"}
    key: list[str]                # columns identifying a row, e.g. ["market", "symbol", "date"]
    features: list[dict]          # [{"name": "ret_5d", "kind": "numeric"}, {"name": "sector", "kind": "categorical"}]
    target: dict                  # {"columns": ["fwd_ret_20d"], "task": "regression" | "binary" | "multiclass"}
    validation_query: str | None  # same shape and params as query; rows the models are scored on
    search: list[dict]            # [{"algorithm": "xgboost", "params": {"max_depth": [3, 6], "n_estimators": [300]}}]
    metrics: list[str]            # names from the metric registry; default per task when empty
    seed: int = 42
    content_hash: str             # of everything above except name and description

class FitRun(Entity):
    """One execution of a Fit."""
    fit_id: str
    params: dict[str, Any]        # the query parameters used, defaults filled in
    status: str                   # queued | running | done | failed
    specs: int                    # models this run will produce
    rows: int = 0                 # fit rows loaded
    validation_rows: int = 0
    data_hash: str | None = None  # of the fit frame, for the record
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

class Model(Entity):
    """One fitted model. The id is the model id."""
    fit_run_id: str
    fit_id: str
    algorithm: str
    params: dict[str, Any]        # the resolved point of the search
    schema: dict                  # key, features with kinds and categorical levels, target; the inference contract
    status: str                   # queued | fitting | fitted | failed
    metrics: dict | None = None   # {"<metric>": value} on the validation rows
    fit_seconds: float | None = None
    device: str | None = None
    path: str | None = None       # models/<model id>/ in the data dir
    bytes: int = 0
    sha256: str | None = None
    published_at: datetime | None = None
    error: str | None = None

class InferRun(Entity):
    """One scoring of a model."""
    model_id: str
    query: str
    params: dict[str, Any]
    status: str                   # running | done | failed
    rows: int = 0
    unknown_levels: dict | None = None   # per categorical, how many values were not in the fitted levels
    metrics: dict | None = None          # when the query also yields the target columns
    path: str | None = None              # parquet/inference/<infer run id>.parquet
    published_at: datetime | None = None
    error: str | None = None
```

Indexes and uniqueness are migrations: `Fit.name` unique, `(fit_run_id)` on `Model`, `(model_id)` on
`InferRun`, `content_hash` on `Fit` so registering the same definition twice returns the existing row.

**The schema is the contract.** `Model.schema` records the key, every feature with its kind and, for
categoricals, the levels seen at fit time, and the target. Inference coerces each column to the kind,
maps categorical values to the fitted levels (an unseen value becomes the null level and is counted in
`unknown_levels`), and refuses a missing feature or a type it cannot coerce. The estimator only ever
sees the matrix it was trained on, whatever the SQL looked like.

**One vector target.** `target.columns` is one or more columns predicted jointly by every model of the
Fit; one task for the whole vector; multiclass only with a single column. Adapters that cannot fit a
vector fit one estimator per column inside the one model and say so in the artifact.

**Artifacts are files.** A model is a directory `models/<model id>/` with the serialised estimator,
`schema.json` and `metrics.json`; an inference output is one Parquet file. Both are `DataFile` rows
(kinds `model` and `inference`) and live in the private Hugging Face store with the data, so a model id
resolves from any machine. Postgres holds the path, size and hash, never the bytes.

## Reactions

Each sub-reaction is there because it is reused or because the run record needs the boundary.

```
RunFit(fit_id, params?)                         one FitRun; LoadRows; one deferred FitModel per spec
  LoadRows(query, params, schema?)              DuckDB -> frame, coerced; saves rows and data hash
  FitModel(fit_run_id, spec)      [deferred]    fit in a subprocess, score validation, write artifact, Model row
    PublishArtifact(paths)                      upload to the store, DataFile rows stamped published_at
  CloseFitRun                     [OnChange(Model, ["status"])]   FitRun done when its last model is terminal
Infer(model_id, query, params?)                 LoadRows against the model's schema; score; write; InferRun row
  LoadRows
  PublishArtifact
```

**`RunFit`** resolves the params (defaults overlaid with the input), expands `search` into concrete
specs (a Cartesian product per entry; a list value is an axis, a scalar is fixed), writes the `FitRun`
as `queued` with `specs`, runs `LoadRows` for the fit and validation queries to record row counts and
the data hash, and defers one `FitModel` per spec with a budget from the algorithm's registry entry.
It does not fit anything itself, so it returns in seconds and the console shows the fan-out.

**`FitModel`** is the long one. It loads the rows again (the frames are not passed through the outbox;
the query and params are, and the data hash is checked against the `FitRun`'s), trains in a subprocess
so a CUDA fault or an out-of-memory kills the child and not the worker, scores the validation rows with
the Fit's metrics, writes the artifact directory, and fills the `Model` row: `fitted`, metrics, path,
hash. `PublishArtifact` then uploads the directory; a failed upload leaves `published_at` null and the
model `fitted`, and the nightly `Publish` completes it. A training failure is the model's outcome,
`failed` with the error; the other models of the run are unaffected.

**`CloseFitRun`** runs after any `Model` changes status: it loads the run's models and, when every one
is `fitted` or `failed`, marks the `FitRun` `done` (or `failed` when all failed) with `finished_at`. This
is the one data-change trigger in the app.

**`Infer`** loads the model, runs `LoadRows` with the model's schema, scores, and writes a Parquet of
the key columns, `pred_<col>` per target column (`pred_<col>_<class>` probabilities for
classification), and `model_id`, `infer_run_id`, `scored_at`. If the query also yields the target
columns, the Fit's metrics are computed and stored on the run, so re-scoring an old model on new months
is an inference run, not a new entity. Immediate, since scoring is seconds to minutes; `deferred=True`
on its trigger if a scoring ever is not.

**Deferral and state.** Fits are deferred because they are long. The state is built for it: a `FitRun`
is `queued`, then `running` from the first `FitModel`, and `done` only through `CloseFitRun`; a `Model` is
`queued` from the moment its outbox row exists, so the console shows the whole sweep before a single
fit starts; a worker that dies leaves its outbox row to be reclaimed after its lease, and the `Model`
row's `fitting` status with no `finished` is how a stuck fit is found. Nothing is inferred from
timing. A `FitRun` whose models never arrive is visible as `running` with `specs` larger than its
terminal models, and can be requeued from the console.

**Budgets.** `RunFit` and `Infer` minutes; `FitModel` from the algorithm registry (gradient boosting
on the GPU 1800 s, linear models 600 s), overridable per `Fit` in `search`.

## Data

Rows come from DuckDB over the Parquet in `FINANCE_DATA_DIR`, through the same views the research
code uses (`equity`, `equity_adj`, `us_equity`, `index_daily`, `equity_1m`, the reference tables). A
Fit's query is any SQL over them; the only rules are that it yields the key, features and target
columns, and that its parameters are named `:param` and declared in `Fit.params`. Features that need
lookbacks are written as window functions in the query, not declared separately; the query is the
feature definition.

`LoadRows` runs the query, checks the columns, coerces by kind, encodes categoricals, and returns a
frame plus the encoding used; at fit time the encoding becomes part of the schema, at inference time
the schema dictates the encoding. Row order is the key's order, so the data hash is stable.

## Estimators

One protocol, adapters behind it: `fit(X, Y, weight, categorical) -> FittedEstimator`, `predict(X) ->
dict[column, ndarray]`, `save(dir)`, `load(dir)`, a `Params` dataclass per adapter that validates a
search point at registration, `multi_output: bool`, `devices: {"cpu", "cuda"}`. First adapters:
`xgboost` (GPU), `lightgbm`, `sklearn.linear` (ridge and logistic as the honest baseline). Registry
entries carry the default budget. `search[*].algorithm` must be a registry key; `params` are validated
against the adapter's `Params` when the Fit is registered, so a typo fails at registration, not an hour
into a sweep.

## Metrics

A registry of named functions `(y_true, y_pred, task) -> float`, applied per target column and
reported as `{"<metric>.<column>": value}`; defaults per task (regression: mae, rmse, r2; binary:
log_loss, auc, precision_at_0.5; multiclass: log_loss, accuracy). The finance-specific ones, net
return per trade after costs and hit rate above a threshold, are registered the same way and are what
[finance-modelling.md](finance-modelling.md) uses.

## Endpoints

| Route | Reaction | Does |
| --- | --- | --- |
| `POST /ml/fits` | `RegisterFit` | validate and store a definition; returns the existing row on an identical hash |
| `POST /ml/fits/{id}/runs` | `RunFit` | start a run with optional params; returns the fit run id |
| `GET /ml/runs/{id}` | builder only | run status, its models with metrics |
| `POST /ml/models/{id}/infer` | `Infer` | score a query; returns the infer run id and output path |

The CLI's `trigger` covers all of them without HTTP, and the console shows every run. A Fit can live as
a JSON file in this repo and be registered with one command; the row is the record.

## Layout

```
apps/finance/src/finance/ml/
  entities.py       Fit, FitRun, Model, InferRun
  reactions.py      RunFit, LoadRows, FitModel, PublishArtifact, CloseFitRun, Infer, RegisterFit
  api.py            the four endpoints
  data.py           DuckDB connection over the data dir, query execution, coercion and encoding, data hash
  schema.py         the Model.schema shape, its validation against a frame
  search.py         search expansion, Params validation
  estimators/       protocol, registry, xgboost.py, lightgbm.py, linear.py
  metrics.py        registry and the finance metrics
  artifacts.py      the model directory: save, load, schema.json, metrics.json
```

`data.py`, `schema.py`, `search.py`, `estimators/`, `metrics.py` and `artifacts.py` know nothing about
the runtime and are tested on frames; the reactions write rows.

## Acceptance

- Unit: search expansion, Params validation, coercion and encoding round trip, unknown-level handling,
  metric values on known inputs, artifact save and load, each adapter on a small frame on CPU.
- Postgres: `RunFit` fans out the right specs and rows; a `FitModel` with a fake estimator fills `Model`
  and triggers `CloseFitRun`; `Infer` refuses a schema mismatch and writes the output table; a failed
  fit is an outcome.
- The first experiment of [finance-modelling.md](finance-modelling.md) run end to end on the box, with
  the GPU, as a `FitRun` with its models in the store.

## Not in the first version

Cross-validation folds (a validation query is one split; folds are several Fits for now), Optuna or
other sequential search, per-row weights, model promotion or aliases, a feature store. Each has a place
in the entities above when it is needed.
