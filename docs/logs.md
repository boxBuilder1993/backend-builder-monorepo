# Logs

Status: 2026-10-24. Sections 1 to 5 are built on `main`: structured lines, the compose `logs` profile with one day of retention, and the console panel. Section 6 step 4, Kubernetes collection, waits for the pools work. Companion to [design.md](design.md) §11, which describes what the runtime logs today, and to [pools.md](pools.md), whose Kubernetes manifests carry the collection pieces described here.

## Intent

Logs can be large, so the runtime never stores them. Every process writes one structured line per event to standard output; the platform the process runs on collects the stream into a log store with a fixed retention; the admin console queries that store to show the lines of one run or one invocation. The same shape holds on a developer machine under compose, on a host under systemd, and in Kubernetes: only the collector changes, never the app, and the console reads from one place.

The store is Grafana Loki. It indexes labels, not text, so it is cheap to run; its shipper, Alloy, runs identically as a compose service and a Kubernetes DaemonSet; Grafana already sits on the Prometheus metrics every process exports, so metrics and logs share one front end; and its HTTP query API is small enough for the console to call directly.

## 1. What a line carries

Every record, from the runtime or from an app's own `logging.getLogger(__name__)`, is written as one JSON object:

```json
{"time": "2026-10-24T09:12:03.418Z", "level": "INFO", "logger": "shop.reactions",
 "message": "notify requested order=…", "app": "shop", "process": "worker", "pool": "default",
 "root_id": "…", "reaction_id": "…", "reaction": "shop.reactions.NotifyOrder", "source": "outbox"}
```

`app`, `process` and `pool` identify where the line came from and are constant for a process. `root_id`, `reaction_id`, `reaction` and `source` identify what was running and are present only inside a run. `invoke` sets a context variable on entry and restores it on exit; a logging filter copies the variable's fields onto every record emitted while it is set. A child's lines therefore carry the child's `reaction_id` and the run's `root_id`, with no change to how reactions log. Secrets read through `ctx.secret` are redacted from `message` before the line is written, as they are today.

`RUNTIME_ENV=dev` keeps the human-readable one-line format for terminals, with `reaction_id` appended, since a developer reads stderr directly. `test` and `prod` write JSON. The format is the only thing that differs between environments.

The runtime's own lifecycle lines (reaction started and finished, flush, worker delivered, scheduler seeded) are unchanged and carry the same fields, so a run's complete story, runtime and app lines interleaved, is one query.

## 2. Collection

Processes write to stdout and nothing else. The container runtime captures the stream to its standard log file; Alloy tails those files, parses the JSON, attaches the labels `app`, `process` and `pool`, and pushes to Loki. Nothing else is a label: `root_id` and `reaction_id` have unbounded cardinality and would ruin Loki's index, so they stay inside the line and are filtered at query time with the `json` stage, which is fast enough for a run's time window.

Where it runs:

- **Compose.** `loki`, `alloy` and `grafana` services behind a `logs` profile. Alloy reads Docker's log files for the compose project. `docker compose --profile shop --profile logs up` gives the whole picture; without the profile, logs stay in `docker compose logs` and the console's panel says so.
- **Host under systemd.** Alloy reads the journal with the unit's `app` and `process` from the unit name. Same labels, same Loki.
- **Kubernetes.** Alloy as a DaemonSet reading `/var/log/pods`, with the labels from the pod's `app` and `pool` labels, which the manifests already set. Loki as a single-binary Deployment with a persistent volume for one cluster; object storage when there is more than one.

## 3. Retention: one day

Loki deletes what is older than a day:

```yaml
limits_config:
  retention_period: 24h
compactor:
  retention_enabled: true
  retention_delete_delay: 1h
  delete_request_store: filesystem
```

The compactor marks chunks past the period and deletes them after the delay, so lines are gone roughly 25 hours after they were written and the volume is bounded by one day's output. The same block appears in the compose service and the manifest. Per-stream overrides exist if an app ever needs longer; the default is one number.

Two settings keep the day honest:

- **Container files rotate by size, not age.** Docker's json-file driver does not rotate by default; every compose service gets `logging: {driver: json-file, options: {max-size: 50m, max-file: 3}}`. The kubelet rotates at 10 MB keeping 5 files by default, which the manifests leave alone. Alloy ships lines as they are written, so rotation loses nothing Loki wanted.
- **Run records outlive logs.** `BACKEND_BUILDER_RUN_RETENTION_DAYS` is 7, so a run page can exist whose logs have expired. The console says "logs expired after 24h" for such a run instead of showing nothing. Records stay longer by design: they are small and are the audit trail; the logs are the ephemeral detail.

## 4. The console

A `BACKEND_BUILDER_LOGS_URL` setting names the Loki base URL. When it is set, the run page gains a **Logs** panel. For a seed, it queries

```
{app="shop"} | json | root_id="<root_id>"
```

over the window from the run's start to its end plus a margin, and renders time, level, logger and message, with the runtime's lifecycle lines in muted text and the invocation each line belongs to shown by its `reaction_id`. For a child's page, the filter is `reaction_id="<id>"`. A link opens the same query in Grafana for anything beyond a page of lines. The panel is capped at a few hundred lines with "open in Grafana" for the rest, and the query is bounded by the run's own window, so a run that logged a million lines costs the console one request and a truncated page.

When the setting is unset, the panel is one sentence saying logs are not wired and which variable wires them. The console never stores a line and never writes to Loki.

Grafana itself is the fleet-wide view: search across runs, a reaction's error lines over a week, a pool's log volume. The console's panel is deliberately per run; it answers "what did this run say" and hands everything else to Grafana.

## 5. Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `BACKEND_BUILDER_LOGS_URL` | unset | Loki base URL the console queries; unset disables the panel |
| `BACKEND_BUILDER_GRAFANA_URL` | unset | Grafana base URL for the panel's "open in Grafana" link |
| `BACKEND_BUILDER_LOG_LEVEL` | `INFO` | unchanged |
| `RUNTIME_ENV` | `dev` | `dev` writes the human format; `test` and `prod` write JSON |

Retention is Loki's configuration, not the app's, so it is not a runtime setting.

## 6. Order of work

1. **Structured logging.** The JSON formatter, the context variable set by `invoke`, the filter that stamps records, `dev` keeping the human format. Useful on its own: `docker compose logs api | jq 'select(.root_id=="…")'` works before anything else exists. Tests assert the fields on runtime and app lines, on children, and redaction.
2. **Compose `logs` profile.** Loki with the retention block, Alloy, Grafana with Loki and Prometheus as data sources, size-based rotation on every service. An e2e scenario brings the profile up and finds a run's lines in Loki by `root_id`.
3. **Console panel.** `BACKEND_BUILDER_LOGS_URL`, the query, the rendering, the expired and unset states, the Grafana link. Tested against a stub Loki in unit tests and the real one in e2e.
4. **Kubernetes.** Alloy DaemonSet and Loki Deployment in the manifests from [pools.md](pools.md) §7, as part of that work.

## 7. Considered and set aside

- **Storing lines on the run record.** Simple and query-free, and wrong at scale: a chatty reaction would bloat `_run`, retention would be tied to records, and the limit would be ours to invent. Logs are a stream; the store is a log store.
- **SigNoz.** Datadog in one box: logs, traces and metrics on ClickHouse, OpenTelemetry-native. More pleasant than Grafana and heavier to run. Worth revisiting when `call` makes cross-pod traces of a run interesting; Loki plus Tempo is the equivalent path on the stack chosen here.
- **OpenSearch or Elastic.** Full-text indexing of every line: the strongest search and the most expensive to run; more than this needs.
- **A log viewer built into the console.** The per-run panel is that, scoped to one query. Search across the fleet, dashboards and alerting are Grafana's, and duplicating them would be a second product.
- **Labels for `root_id` or `reaction_id`.** Unbounded cardinality destroys Loki's index. They live inside the line and are filtered at query time.
- **Shipping from the process** (a logging handler pushing to Loki). Couples the app to the collector, loses lines when the collector is down, and duplicates what the platform does for free from stdout.
