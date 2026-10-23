"""A Fit's search expanded into concrete specs, validated against each algorithm's Params."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, cast

from finance.ml.estimators import REGISTRY, resolve_params


@dataclass(frozen=True)
class Spec:
    algorithm: str
    params: dict[str, Any]
    device: str = "auto"


def expand(search: list[dict[str, Any]]) -> list[Spec]:
    """Per entry, a list value is an axis and a scalar is fixed; the entry expands to the Cartesian
    product of its axes with the adapter's defaults filled in. Entries do not cross. A name the
    adapter does not accept, or a wrong type, raises ValueError naming the entry and the field."""
    if not search:
        raise ValueError("search must have at least one entry")
    specs: list[Spec] = []
    for i, entry in enumerate(search):
        algorithm = entry.get("algorithm")
        if algorithm not in REGISTRY:
            raise ValueError(
                f"search[{i}]: unknown algorithm {algorithm!r}; known: {sorted(REGISTRY)}"
            )
        raw: dict[str, Any] = dict(entry.get("params") or {})
        device = str(entry.get("device", "auto"))
        axes: dict[str, list[Any]] = {
            k: (list(cast("list[Any]", v)) if isinstance(v, list) else [v]) for k, v in raw.items()
        }
        names = list(axes)
        for values in itertools.product(*(axes[n] for n in names)) if names else [()]:
            point = dict(zip(names, values, strict=True))
            try:
                resolved = resolve_params(algorithm, point)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"search[{i}] ({algorithm}): {exc}") from None
            specs.append(Spec(algorithm, resolved, device))
    return specs
