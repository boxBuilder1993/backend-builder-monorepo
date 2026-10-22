# finance

The market-data poller as a Backend Builder app. Design and status: [docs/finance-app.md](../../docs/finance-app.md).
Run it on a host with `make up APP=finance` at the repository root ([docs/running.md](../../docs/running.md)).

Environment, in `~/.config/backend-builder/finance.env`:

| Variable | Meaning |
| --- | --- |
| `XDG_DATA_HOME` | optional; the mirror of the store lives at `$XDG_DATA_HOME/finance/mirror/<owner>--<repo>/`, default `~/.local/share/...`. Not a finance setting: the path is the app's convention |
| `FINANCE_HF_REPO` | the private Hugging Face dataset repo (default `shr22avan/finance-data`) |
| `SECRET_HF_TOKEN` | fine-grained token with write access to that repo |
| `FINANCE_PUBLISH` | `off` runs the legs but uploads nothing: the shadow phase beside the old poller (default `on`) |

One run a night at 20:00 and 22:00 IST: the four legs in order, then the publish. Legs not yet implemented
record `implemented=false` and write nothing. Run a piece by hand:

```sh
uv run backend-builder trigger finance finance.reactions.SyncLeg --input '{"leg": "nse"}'
uv run backend-builder trigger finance finance.reactions.Publish --input '{}'
```
