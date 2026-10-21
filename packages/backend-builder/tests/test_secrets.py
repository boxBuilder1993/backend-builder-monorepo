import logging

import pytest
from backend_builder import (
    Context,
    InMemoryStore,
    Reaction,
    SecretError,
    Secrets,
    configure_logging,
    run,
)
from backend_builder.store import Session

KEY = "sk_live_0123456789abcdef"


class UseKey(Reaction):
    async def calculate(self, ctx: Context) -> None:
        key = ctx.secret("payments_api_key")
        ctx.save("header", f"Bearer {key}")
        child = await ctx.trigger(Child, {"token": key})
        assert child is not None


class Child(Reaction):
    async def calculate(self, ctx: Context) -> None:
        assert ctx.secret("payments_api_key") == ctx.input["token"]


class Boom(Reaction):
    async def calculate(self, ctx: Context) -> None:
        raise RuntimeError(f"provider rejected {ctx.secret('payments_api_key')}")


def test_reads_the_prefixed_upper_cased_variable() -> None:
    secrets = Secrets({"SECRET_PAYMENTS_API_KEY": KEY})
    assert secrets.get("payments_api_key") == KEY
    assert secrets.seen == ("payments_api_key",)


def test_missing_or_empty_names_the_variable_to_export() -> None:
    secrets = Secrets({"SECRET_EMPTY": ""})
    with pytest.raises(SecretError, match="export SECRET_PAYMENTS_API_KEY"):
        secrets.get("payments_api_key")
    with pytest.raises(SecretError, match="export SECRET_EMPTY"):
        secrets.get("empty")


@pytest.mark.parametrize("name", ["Stripe", "stripe-key", "1key", "", "a b"])
def test_rejects_names_that_are_not_snake_case(name: str) -> None:
    with pytest.raises(SecretError, match="bad secret name"):
        Secrets({}).get(name)


def test_redacts_values_read_so_far_by_name() -> None:
    secrets = Secrets({"SECRET_A": "aaaaaaaaaa", "SECRET_B": "bbbbbbbbbb"})
    assert secrets.redact("aaaaaaaaaa bbbbbbbbbb") == "aaaaaaaaaa bbbbbbbbbb"  # nothing read yet
    secrets.get("a")
    assert secrets.redact("x aaaaaaaaaa y") == "x [secret:a] y"
    assert secrets.redact("bbbbbbbbbb") == "bbbbbbbbbb"
    secrets.get("b")
    assert secrets.redact_jsonable({"k": ["aaaaaaaaaa", 1, {"n": "bbbbbbbbbb!"}]}) == {
        "k": ["[secret:a]", 1, {"n": "[secret:b]!"}]
    }


def test_short_values_are_not_redacted_and_warn(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="backend_builder")
    secrets = Secrets({"SECRET_PIN": "1234"})
    assert secrets.get("pin") == "1234"
    assert secrets.redact("pin 1234") == "pin 1234"
    assert "secret pin is shorter than 8 characters" in caplog.text


async def test_context_reads_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECRET_PAYMENTS_API_KEY", KEY)
    ctx = Context(Session(), {})
    assert ctx.secret("payments_api_key") == KEY
    monkeypatch.delenv("SECRET_PAYMENTS_API_KEY")
    with pytest.raises(SecretError):
        ctx.secret("payments_api_key")


async def test_children_share_the_parents_secrets() -> None:
    secrets = Secrets({"SECRET_PAYMENTS_API_KEY": KEY})
    ctx = Context(Session(), {}, secrets=secrets)
    await UseKey().calculate(ctx)
    assert ctx.load("header") == f"Bearer {KEY}"


async def test_run_records_and_logs_mask_the_value(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("SECRET_PAYMENTS_API_KEY", KEY)
    caplog.set_level(logging.DEBUG, logger="backend_builder")
    store = InMemoryStore()
    await run(store, UseKey, {"token": KEY})
    with pytest.raises(RuntimeError, match=r"provider rejected sk_live"):
        await run(store, Boom, {}, max_attempts=1)

    ok, failed = store.records
    assert ok.vars == {"header": "Bearer [secret:payments_api_key]"}
    assert ok.input == {"token": "[secret:payments_api_key]"}
    assert failed.error == "RuntimeError: provider rejected [secret:payments_api_key]"

    # The configured stream handler redacts; the raw records caplog sees do not.
    import io

    stream = io.StringIO()
    configure_logging(logging.DEBUG, stream=stream)
    try:
        logging.getLogger("backend_builder.test").debug("sending %s", KEY)
    finally:
        configure_logging(logging.INFO)
    assert KEY not in stream.getvalue()
    assert "sending [secret:payments_api_key]" in stream.getvalue()
