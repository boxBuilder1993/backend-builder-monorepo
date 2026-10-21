# One image per app. The entrypoint is the backend-builder CLI; the process
# is chosen by the command: serve, worker, scheduler, migrate.
#
#   docker build --build-arg APP=shop -t backend-builder-shop .
#   docker run backend-builder-shop serve shop --host 0.0.0.0

FROM python:3.13-slim AS base
COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1
WORKDIR /app

# Third-party dependencies from the lock file only, as a cached layer.
ARG APP=shop
COPY pyproject.toml uv.lock ./
COPY packages/backend-builder/pyproject.toml packages/backend-builder/
COPY packages/backend-builder-testing/pyproject.toml packages/backend-builder-testing/
COPY apps/${APP}/pyproject.toml apps/${APP}/
RUN uv sync --frozen --no-dev --package ${APP} --no-install-workspace

# The workspace sources, then the members themselves.
COPY packages packages
COPY apps apps
RUN uv sync --frozen --no-dev --package ${APP}

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app
ENV PATH="/app/.venv/bin:${PATH}"
ENTRYPOINT ["backend-builder"]
CMD ["--help"]
