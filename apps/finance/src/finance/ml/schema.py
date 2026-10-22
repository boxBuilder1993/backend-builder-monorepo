"""The schema: what a model was fitted on, and therefore what inference must provide. Built from a
Fit's declaration at registration (kinds, no levels), completed at fit time (levels, classes), and
stored on the Model as the inference contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Kind = Literal["numeric", "categorical"]
Task = Literal["regression", "binary", "multiclass"]
KINDS: tuple[str, ...] = ("numeric", "categorical")
TASKS: tuple[str, ...] = ("regression", "binary", "multiclass")


@dataclass(frozen=True)
class Feature:
    name: str
    kind: Kind
    levels: tuple[str, ...] | None = None
    """Categorical values seen at fit time, sorted; None before fitting or for numerics."""


@dataclass(frozen=True)
class Target:
    columns: tuple[str, ...]
    task: Task
    classes: tuple[str, ...] | None = None
    """Class labels seen at fit time for binary and multiclass, sorted."""


@dataclass(frozen=True)
class Schema:
    key: tuple[str, ...]
    features: tuple[Feature, ...]
    target: Target
    version: int = field(default=1)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    @property
    def categorical(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features if f.kind == "categorical")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["key"] = list(self.key)
        d["features"] = [
            {"name": f.name, "kind": f.kind, "levels": list(f.levels) if f.levels else None}
            for f in self.features
        ]
        d["target"] = {
            "columns": list(self.target.columns),
            "task": self.target.task,
            "classes": list(self.target.classes) if self.target.classes else None,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Schema:
        return cls(
            key=tuple(d["key"]),
            features=tuple(
                Feature(f["name"], f["kind"], tuple(f["levels"]) if f.get("levels") else None)
                for f in d["features"]
            ),
            target=Target(
                tuple(d["target"]["columns"]),
                d["target"]["task"],
                tuple(d["target"]["classes"]) if d["target"].get("classes") else None,
            ),
            version=int(d.get("version", 1)),
        )


def declare(key: list[str], features: list[dict[str, Any]], target: dict[str, Any]) -> Schema:
    """A Fit's declaration as a Schema without levels. Raises ValueError naming what is wrong."""
    if not key:
        raise ValueError("key must name at least one column")
    names = [str(f.get("name", "")) for f in features]
    if not names:
        raise ValueError("features must name at least one column")
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")
    feats: list[Feature] = []
    for f in features:
        kind = f.get("kind")
        if kind not in KINDS:
            raise ValueError(
                f"feature {f.get('name')!r}: kind must be one of {KINDS}, not {kind!r}"
            )
        feats.append(Feature(str(f["name"]), kind))
    cols = [str(c) for c in target.get("columns", [])]
    task = target.get("task")
    if not cols:
        raise ValueError("target.columns must name at least one column")
    if task not in TASKS:
        raise ValueError(f"target.task must be one of {TASKS}, not {task!r}")
    if task == "multiclass" and len(cols) != 1:
        raise ValueError("a multiclass target has exactly one column")
    overlap = (set(key) & set(names)) | (set(cols) & set(names)) | (set(key) & set(cols))
    if overlap:
        raise ValueError(f"columns must be key, feature or target, not several: {sorted(overlap)}")
    return Schema(tuple(key), tuple(feats), Target(tuple(cols), task))
