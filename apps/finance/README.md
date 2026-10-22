# finance

The market-data poller as a Backend Builder app. Design and status: [docs/finance-app.md](../../docs/finance-app.md).
Run it on a host with `make up APP=finance` at the repository root ([docs/running.md](../../docs/running.md)).

Environment, in `~/.config/backend-builder/finance.env`:

| Variable | Meaning |
| --- | --- |
| `FINANCE_DATA_DIR` | working copy of the store: `parquet/`, `raw/` (default `~/finance-data`) |
| `FINANCE_HF_REPO` | the private Hugging Face dataset repo (default `shr22avan/finance-data`) |
| `SECRET_HF_TOKEN` | fine-grained token with write access to that repo |

One run a night at 20:00 and 22:00 IST: the four legs in order, then the publish. Legs not yet implemented
record `implemented=false` and write nothing. Run a piece by hand:

```sh
uv run backend-builder trigger finance finance.reactions.SyncLeg --input '{"leg": "nse"}'
uv run backend-builder trigger finance finance.reactions.Publish --input '{}'
```
