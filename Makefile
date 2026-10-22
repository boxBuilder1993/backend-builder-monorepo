# Run any app in this repo on this host: one shared Postgres in Docker, the app's processes as systemd
# user units that start at boot. `make up APP=finance` is the whole thing. `make` lists the targets.
APP        ?=
REPO       := $(CURDIR)
UV         ?= uv
ENV_DIR    := $(HOME)/.config/backend-builder
ENV_FILE   := $(ENV_DIR)/$(APP).env
UNIT_DIR   := $(HOME)/.config/systemd/user
# Processes per app. An app with endpoints adds `serve` to BB_ROLES in its env file.
ROLES      ?= worker scheduler admin
PG_USER    := backend_builder
PSQL       := docker compose exec -T postgres psql -U $(PG_USER) -v ON_ERROR_STOP=1 -q
need_app    = @test -n "$(APP)" || { echo "usage: make $@ APP=<package>"; exit 2; }
roles       = $(shell test -f $(ENV_FILE) && . $(ENV_FILE) && echo $${BB_ROLES:-$(ROLES)} || echo $(ROLES))
units       = $(foreach r,$(roles),bb@$(APP):$(r).service)

.PHONY: help setup lint test pg pg-down db env migrate up down restart status logs deploy apps graph

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-14s %s\n", $$1, $$2}'

setup:           ## install every workspace member into one env
	$(UV) sync --all-packages

lint:            ## ruff + pyright, the way CI runs them
	$(UV) run ruff format --check . && $(UV) run ruff check . && $(UV) run pyright

test:            ## unit suites (e2e excluded, as in CI)
	$(UV) run pytest -q

pg:              ## start the shared Postgres (Docker, restarts on boot)
	docker compose up -d --wait postgres

pg-down:         ## stop Postgres (data stays in the volume)
	docker compose stop postgres

db:              ## create the app's database if missing          APP=
	$(need_app)
	@$(PSQL) -tAc "select 1 from pg_database where datname='$(APP)'" | grep -q 1 \
	  || { $(PSQL) -c 'create database "$(APP)"'; echo "created database $(APP)"; }

env:             ## write ~/.config/backend-builder/APP.env if missing (ports, database url, admin token)
	$(need_app)
	@mkdir -p $(ENV_DIR)
	@test -f $(ENV_FILE) || { \
	  base=$$(cat $(ENV_DIR)/*.env 2>/dev/null | sed -n 's/^BB_PORT_BASE=//p' | sort -n | tail -1); \
	  base=$${base:-9200}; base=$$((base + 100)); \
	  umask 077; printf '%s\n' \
	    "# $(APP) on this host. Read by bin/bb-run for every role. Secrets go here as SECRET_<NAME>=..." \
	    "BACKEND_BUILDER_DATABASE_URL=postgresql://$(PG_USER):$(PG_USER)@127.0.0.1:5432/$(APP)" \
	    "# BB_PORT_BASE: admin console; +1 API; +2..5 metrics of worker, scheduler, admin, serve" \
	    "BB_PORT_BASE=$$base" \
	    "BB_ROLES=\"$(strip $(ROLES))\"" \
	    "BACKEND_BUILDER_ADMIN_TOKEN=$$(head -c 24 /dev/urandom | base64 | tr -d '/+=')" \
	    "BACKEND_BUILDER_LOG_LEVEL=INFO" > $(ENV_FILE); \
	  echo "wrote $(ENV_FILE) (admin console on http://127.0.0.1:$$base)"; }

migrate:         ## apply the app's migration chain                 APP=
	$(need_app)
	@set -a; . $(ENV_FILE); set +a; RUNTIME_ENV=prod $(UV) run --frozen backend-builder migrate $(APP)

up: pg db env migrate   ## bring the whole app up and enable it at boot   APP=
	@mkdir -p $(UNIT_DIR)
	@sed 's#@REPO@#$(REPO)#g' systemd/bb@.service > $(UNIT_DIR)/bb@.service
	@systemctl --user daemon-reload
	@systemctl --user enable --now $(units)
	@loginctl show-user $(USER) -p Linger --value | grep -q yes || loginctl enable-linger $(USER)
	@$(MAKE) --no-print-directory status APP=$(APP)

down:            ## stop and disable the app's processes (Postgres and data stay)   APP=
	$(need_app)
	-@systemctl --user disable --now $(units) 2>/dev/null

restart:         ## restart the app's processes                     APP=
	$(need_app)
	@systemctl --user restart $(units)

status:          ## is it running, on which ports                   APP=
	$(need_app)
	@for u in $(units); do printf '  %-32s %s\n' $$u "$$(systemctl --user is-active $$u)"; done
	@. $(ENV_FILE) 2>/dev/null && echo "  admin console http://127.0.0.1:$$BB_PORT_BASE  (token in $(ENV_FILE))" || true

logs:            ## follow the app's journal                        APP=
	$(need_app)
	@journalctl --user -f $(foreach u,$(units),-u $(u))

deploy:          ## pull main, sync, migrate, restart              APP=
	$(need_app)
	git pull --ff-only
	$(MAKE) --no-print-directory setup migrate restart APP=$(APP)

apps:            ## every app configured on this host and its state
	@for f in $(ENV_DIR)/*.env; do [ -f "$$f" ] || continue; a=$$(basename $$f .env); \
	  printf '%-12s' $$a; for r in $$(. $$f; echo $${BB_ROLES:-$(ROLES)}); do \
	  printf ' %s=%s' $$r "$$(systemctl --user is-active bb@$$a:$$r.service)"; done; echo; done

graph:           ## print the app's entities, reactions, endpoints and triggers   APP=
	$(need_app)
	$(UV) run backend-builder graph $(APP)
