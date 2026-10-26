"""Įrankio aprašas. Kiekvienas įrankis = modulis app/tools/<id>/ su kintamuoju TOOL."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable


@dataclass
class Param:
    key: str
    label: str
    type: str = "float"            # float | int | bool | str | choice
    default: Any = None
    unit: str | None = None
    group: str = "Parametrai"
    help: str | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list | None = None    # [[reikšmė, pavadinimas], ...]
    enabled_by: str | None = None  # bool parametro raktas, kuris įjungia šį laukelį
    show_if: str | None = None     # rodomas tik kai sąlyga tenkinama: "mode=retinti" arba "mode!=retinti"

    def coerce(self, v):
        if v is None or v == "":
            return self.default
        if self.type == "float":
            v = float(str(v).replace(",", "."))
        elif self.type == "int":
            v = int(float(str(v).replace(",", ".")))
        elif self.type == "bool":
            v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "on", "taip", "yes")
        elif self.type == "choice":
            allowed = [c[0] for c in (self.choices or [])]
            if v not in allowed:
                raise ValueError(f"{self.label}: neleistina reikšmė {v!r}")
        else:
            v = str(v)
        if self.type in ("float", "int"):
            if self.min is not None and v < self.min:
                raise ValueError(f"{self.label}: mažiausia reikšmė {self.min}")
            if self.max is not None and v > self.max:
                raise ValueError(f"{self.label}: didžiausia reikšmė {self.max}")
        return v


@dataclass
class FileInput:
    key: str = "files"
    label: str = "Failai"
    accept: str = ""               # pvz. ".dxf,.csv"
    multiple: bool = True
    required: bool = True


class Tool:
    id: str = ""
    title: str = ""
    group: str = "Bendra"
    icon: str = "⚙"
    description: str = ""
    help: str = ""                 # ilgesnis aprašas (HTML leidžiamas)
    inputs: list[FileInput] = []
    params: list[Param] = []

    def schema(self) -> dict:
        return {"id": self.id, "title": self.title, "group": self.group, "icon": self.icon,
                "description": self.description, "help": self.help,
                "inputs": [asdict(i) for i in self.inputs],
                "params": [asdict(p) for p in self.params]}

    def parse_params(self, raw: dict) -> dict:
        return {p.key: p.coerce(raw.get(p.key)) for p in self.params}

    def run(self, ctx: "JobContext") -> None:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Output:
    name: str
    label: str
    kind: str = "file"             # file | image | text


@dataclass
class JobContext:
    job_id: str
    workdir: Path
    inputs: dict[str, list[Path]]
    params: dict
    log: Callable[[str], None]
    progress: Callable[[float], None]
    outputs: list[Output] = field(default_factory=list)
    preview: dict | None = None

    @property
    def outdir(self) -> Path:
        d = self.workdir / "out"
        d.mkdir(exist_ok=True)
        return d

    def add_output(self, path, label, kind="file"):
        p = Path(path)
        if p.parent != self.outdir:
            raise ValueError("Išvesties failas turi būti outdir kataloge")
        self.outputs.append(Output(p.name, label, kind))
