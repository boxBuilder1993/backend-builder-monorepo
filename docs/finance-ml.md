# ML in the finance app

Status: proposal, 2026-10-22. Machine learning on the market data as part of the finance app: the
definitions, runs, models and inferences are entities, the work is reactions, the artifacts live in the
same private store as the data. This replaces the earlier plan for a standalone `mlkit` library; what
that design settled about semantics (one vector target, the schema as the inference contract, the
artifact layout) is kept, in entity terms. [finance-modelling.md](finance-modelling.md) says what we
fit first; this document says how fitting works.

## The idea in one paragraph

A **Fit** is a definition: a SQL query that produces the rows to learn from, the features and their
kinds, the target, and a search over algorithms and parameters. Running
it is a **FitRun**, with the query parameters actually used, and it produces one **Model** per point of
the search, each with its own id and its artifact uploaded to the store before it is called fitted. An
**InferRun** takes a model id and a query, checks the query against the model's schema, scores the rows
and writes a table of key plus predictions. Validation is inference under a label, and an **Evaluation**
is a set of metrics computed over one inference run's prediction table; a model can have any number of
either. All of it is visible in the console as runs and rows, and queryable from DuckDB
next to the prices.

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
    search: list[dict]            # [{"algorithm": "xgboost", "params": {"max_depth": [3, 6], "n_estimators": [300]}}]
    content_hash: str             # of everything above except name and description

class FitRun(Entity):
    """One execution of a Fit."""
    fit_id: str
    params: dict[str, Any]        # the query parameters used, defaults filled in
    status: str                   # queued | running | done | failed
    specs: int                    # models this run will produce
    rows: int = 0                 # fit rows loaded
    data_hash: str | None = None  # of the fit frame, for the record
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

class Model(Entity):
    """One fitted model. The id is the model id; the store is where it lives."""
    fit_run_id: str
    fit_id: str
    algorithm: str
    params: dict[str, Any]        # the resolved point of the search
    schema: dict                  # key, features with kinds and categorical levels, target; the inference contract
    status: str                   # queued | fitting | uploading | fitted | failed
    repo: str | None = None       # the Hugging Face dataset repo
    path: str | None = None       # models/<model id>/ inside it
    revision: str | None = None   # the commit the upload produced; (repo, path, revision) is exact bytes
    sha256: str | None = None     # of the archived directory
    bytes: int = 0
    cache_path: str | None = None # where the directory sits on the fitting host until uploaded
    fit_seconds: float | None = None
    device: str | None = None
    error: str | None = None

class InferRun(Entity):
    """One scoring of a model: a model id and a query, nothing else."""
    model_id: str
    label: str                    # "validation", "validation 2026", "live 2026-10-22", free text
    query: str
    params: dict[str, Any]
    status: str                   # running | done | failed
    rows: int = 0
    unknown_levels: dict | None = None   # per categorical, how many values were not in the fitted levels
    path: str | None = None              # parquet/inference/<infer run id>.parquet, a DataFile
    error: str | None = None

class StoreSync(Entity):
    """One pull of the store into the local mirror. The store is the source of truth; FINANCE_DATA_DIR
    is a mirror that is never read before it has been brought to the store's current revision."""
    repo: str
    revision: str                 # the store's commit the mirror was brought to
    patterns: list[str]           # what was pulled, e.g. ["parquet/**"]
    files: int = 0                # fetched this time; zero when the mirror was already current
    bytes: int = 0
    seconds: float = 0
    status: str                   # current | pulled | failed
    error: str | None = None

class Evaluation(Entity):
    """Metrics over one InferRun's prediction table. Several may sit on one run."""
    infer_run_id: str
    metrics: list[str]            # names from the metric registry, e.g. ["net_return_per_trade", "mae"]
    params: dict[str, Any]        # what the metrics need: costs, principal, threshold
    results: dict | None = None   # {"<metric>.<column>": value}
    status: str                   # done | failed
    error: str | None = None
```

Indexes and uniqueness are migrations: `Fit.name` unique, `content_hash` unique on `Fit` so registering
the same definition twice returns the existing row, `fit_run_id` on `Model`, `model_id` on `InferRun`, `infer_run_id` on `Evaluation`, `created_at` on
`StoreSync`.

**The schema is the contract.** `Model.schema` records the key, every feature with its kind and, for
categoricals, the levels seen at fit time, and the target. Inference coerces each column to the kind,
maps categorical values to the fitted levels (an unseen value becomes the null level and is counted in
`unknown_levels`), and refuses a missing feature or a type it cannot coerce. The estimator only ever
sees the matrix it was trained on, whatever the SQL looked like.

**One vector target.** `target.columns` is one or more columns predicted jointly by every model of the
Fit; one task for the whole vector; multiclass only with a single column. Adapters that cannot fit a
vector fit one estimator per column inside the one model and say so in the artifact.

**Models live in the store.** A model is a directory, the serialised estimator plus `schema.json`
and `fit.json` (algorithm, params, fit run, timings), uploaded to `models/<model id>/` in the private
Hugging Face dataset repo by the run that fitted it. `(repo, path, revision)` names exact bytes, so a
model id resolves from any machine; the local directory is a cache. Models are not `DataFile` rows:
they are not data, and the data ledger stays about data. Inference outputs are data, one Parquet each,
and are `DataFile` rows published with the rest.

**A Fit says how a model is made, nothing about judging it.** Validation is an `InferRun` with a label
over whatever query you choose, and metrics are an `Evaluation` over that run. A model's quality is
never a number frozen at fit time; it is as many evaluations, over as many validation runs, as the
question needs, and comparing models is a query across the three tables.

## Reactions

Each sub-reaction is there because it is reused or because the run record needs the boundary.

```
PullStore(patterns?)                            mirror <- store at its current revision; StoreSync row
RunFit(fit_id, params?)                         one FitRun; LoadRows; one deferred FitModel per spec
  LoadRows(query, params, schema?)              PullStore, then DuckDB -> frame, coerced; rows and data hash
    PullStore
  FitModel(fit_run_id, spec)      [deferred]    fit in a subprocess, write the directory, upload, Model row
    UploadModel(model_id)                       the directory to the store; records revision and sha256
  CloseFitRun                     [OnChange(Model, ["status"])]   FitRun done when its last model is terminal
Infer(model_id, label, query, params?)          LoadRows against the model's schema; score; write; InferRun row
  LoadRows
  PublishArtifact(paths)                        the output Parquet to the store, its DataFile row stamped
Evaluate(infer_run_id, metrics, params?)        the prediction table -> named metrics; Evaluation row
```

**`PullStore`** asks the store for its current revision and compares it with the latest `StoreSync`. If
the mirror is at that revision and the files are present it saves "current" and returns: one API call.
Otherwise it runs `snapshot_download` into `FINANCE_DATA_DIR` for the patterns, `parquet/**` by
default since the ML never needs raw, which fetches only the files whose hash differs, and writes a
`StoreSync` row with the revision. The first pull on a fresh box is the whole Parquet tree; after that
it is the previous night's files. Nothing is deleted locally, as nothing is deleted from the store; a
pull that fails halfway leaves a mirror that is behind, a `failed` row, and the next `LoadRows` tries
again. On the poller box the mirror is the poller's own data directory, so the pull finds every file
present and is a no-op; the legs and the pull only ever write the same bytes for the same path.

**`LoadRows`** triggers `PullStore` first, every time, so a fit or an inference never reads stale data
and never reads at all on a box that has not synced. Then it runs the query through DuckDB over the
mirror, checks the columns, coerces by kind, encodes categoricals, and returns the frame with the
encoding used. `(repo, revision)` on the `StoreSync` and `data_hash` on the `FitRun` together record
the exact store state a fit read.

**`RunFit`** resolves the params (defaults overlaid with the input), expands `search` into concrete
specs (a Cartesian product per entry; a list value is an axis, a scalar is fixed), writes the `FitRun`
as `queued` with `specs`, runs `LoadRows` for the fit query to record the row count and the data hash,
and defers one `FitModel` per spec with a budget from the algorithm's registry entry. It does not fit
anything itself, so it returns in seconds and the console shows the fan-out.

**`FitModel`** is the long one. It loads the rows again (frames are not passed through the outbox; the
query and params are, and the data hash is checked against the `FitRun`'s), trains in a subprocess so a
CUDA fault or an out-of-memory kills the child and not the worker, writes the directory into the cache
(`fitting` → `uploading`, `cache_path` set), then `UploadModel` pushes it to the store and records
`repo`, `path`, `revision`, `sha256`; only then is the model `fitted`. A training failure is the model's
outcome, `failed` with the error; a store outage leaves it `uploading` with its cache path, and the
upload is retried by the next `FitModel` attempt or by hand. The other models of the run are unaffected.

**`CloseFitRun`** runs after any `Model` changes status: it loads the run's models and, when every one
is `fitted` or `failed`, marks the `FitRun` `done` (or `failed` when all failed) with `finished_at`. This
is the one data-change trigger in the app.

**`Infer`** takes a model id, a label, a query and params. It fetches the model by `(repo, path,
revision)` into the cache, runs `LoadRows` with the model's schema, scores, and writes a Parquet of the
key columns, the target columns when the query yields them, `pred_<col>` per target column
(`pred_<col>_<class>` probabilities for classification), and `model_id`, `infer_run_id`, `scored_at`.
`PublishArtifact` uploads it and writes its `DataFile` row. Immediate, since scoring is seconds to
minutes; `deferred=True` on its trigger if a scoring ever is not. Validating a model against several
datasets is several `Infer` runs on the same model id, each its own table and label.

**`Evaluate`** takes an infer run id and metric names with their params, reads the prediction table,
which carries the true target columns when the query yielded them, computes each metric per target
column, and writes an `Evaluation`. Metrics are a registry of named functions over (truth, prediction,
params): the generic ones, mae, rmse, log loss, auc; and the finance ones, net return per trade after
costs at a principal, hit rate above a threshold. Several evaluations on one run, with different costs
or different metric sets, are the normal case. Judging a sweep is `Infer` on every model with the same
validation query and label, `Evaluate` on each, and one query over the three tables; a small
orchestrating reaction can wrap that once the pattern settles.

**Deferral and state.** Fits are deferred because they are long. The state is built for it: a `FitRun`
is `queued`, then `running` from the first `FitModel`, and `done` only through `CloseFitRun`; a `Model` is
`queued` from the moment its outbox row exists, so the console shows the whole sweep before a single
fit starts; a worker that dies leaves its outbox row to be reclaimed after its lease, and a `Model` left
in `fitting` or `uploading` is how a stuck fit is found. Nothing is inferred from timing. A `FitRun`
whose models never arrive is visible as `running` with `specs` larger than its terminal models, and the
row can be requeued from the console.

**Budgets.** `PullStore` an hour for a first pull, `RunFit`, `Infer` and `Evaluate` minutes; `FitModel` from the algorithm registry (gradient boosting on
the GPU 1800 s, linear models 600 s), overridable per `Fit` in `search`.

## Data

The store is the source of truth and `FINANCE_DATA_DIR` is its local mirror, brought up to date by
`PullStore` before every read. DuckDB reads the mirror's Parquet; it could read `hf://` paths directly,
and does for an ad hoc look, but one file per day means a thousand footer fetches per table per query
with nothing cached, so anything that scans history reads locally. Rows come through the same views the
research code uses (`equity`, `equity_adj`, `us_equity`, `index_daily`, `equity_1m`, the reference
tables). A Fit's query is any SQL over them; the only rules are that it yields the key, features and target
columns, and that its parameters are named `:param` and declared in `Fit.params`. Features that need
lookbacks are written as window functions in the query, not declared separately; the query is the
feature definition.

At fit time the encoding `LoadRows` produces becomes part of the schema; at inference time the schema
dictates the encoding. Row order is the key's order, so the data hash is stable.

## Estimators

One protocol, adapters behind it: `fit(X, Y, categorical) -> FittedEstimator`, `predict(X) ->
dict[column, ndarray]`, `save(dir)`, `load(dir)`, a `Params` dataclass per adapter that validates a
search point at registration, `multi_output: bool`, `devices: {"cpu", "cuda"}`, a default seed. First
adapters: `xgboost` (GPU), `lightgbm`, `sklearn.linear` (ridge and logistic as the honest baseline).
Registry entries carry the default budget. `search[*].algorithm` must be a registry key; `params` are
validated against the adapter's `Params` when the Fit is registered, so a typo fails at registration,
not an hour into a sweep.

## Endpoints

| Route | Reaction | Does |
| --- | --- | --- |
| `POST /ml/fits` | `RegisterFit` | validate and store a definition; returns the existing row on an identical hash |
| `POST /ml/fits/{id}/runs` | `RunFit` | start a run with optional params; returns the fit run id |
| `GET /ml/runs/{id}` | builder only | run status and its models |
| `POST /ml/models/{id}/infer` | `Infer` | score a query under a label; returns the infer run id and output path |
| `POST /ml/infer-runs/{id}/evaluate` | `Evaluate` | metrics over a run's prediction table; returns the evaluation |

The CLI's `trigger` covers all of them without HTTP, and the console shows every run. A Fit can live as
a JSON file in this repo and be registered with one command; the row is the record.

## Layout

```
apps/finance/src/finance/ml/
  entities.py       Fit, FitRun, Model, InferRun, Evaluation, StoreSync
  reactions.py      PullStore, RegisterFit, RunFit, LoadRows, FitModel, UploadModel, CloseFitRun, Infer,
                    PublishArtifact, Evaluate
  api.py            the five endpoints
  data.py           DuckDB connection over the data dir, query execution, coercion and encoding, data hash
  schema.py         the Model.schema shape, its validation against a frame
  search.py         search expansion, Params validation
  estimators/       protocol, registry, xgboost.py, lightgbm.py, linear.py
  artifacts.py      the model directory: save, load, schema.json, fit.json; archive and hash
  metrics.py        the metric registry: generic ones and the finance ones (costs, principal)
  modelstore.py     upload a model directory to the store and fetch one by (repo, path, revision)
  mirror.py         the store's current revision, snapshot_download into the mirror, what is present
```

`data.py`, `schema.py`, `search.py`, `estimators/`, `artifacts.py`, `metrics.py`, `modelstore.py` and
`mirror.py` know nothing
about the runtime and are tested on frames and temp directories; the reactions write rows.

## Acceptance

- Unit: search expansion, Params validation, coercion and encoding round trip, unknown-level handling,
  artifact save and load, archive hash stable, each adapter on a small frame on CPU, each metric on
  known inputs.
- Postgres: `PullStore` with a fake store is "current" the second time and `pulled` after a change;
  `RunFit` fans out the right specs and rows; a `FitModel` with a fake estimator and a fake
  store fills `Model` through `uploading` to `fitted` and triggers `CloseFitRun`; `Infer` refuses a schema
  mismatch and writes the output table; `Evaluate` reads it back and writes the row; a failed fit and
  a store outage are outcomes with the right statuses.
- The first Fit of [finance-modelling.md](finance-modelling.md) run end to end on the box, with the GPU,
  its models fetched back from the store by id and scored.

## Not in the first version

Cross-validation folds (several Fits for now), Optuna or other sequential search, per-row
weights, model promotion or aliases, a feature store. Each has a place in the entities above when it is
needed.
