"""Parser/normalizer for the mlearn (Zuo et al. 2019) Cu benchmark JSONs.

Source facts and hashes: docs/research/dataset_due_diligence_cu.{md,json}.
Parsing is strict: any record that does not match the documented schema or
whose description cannot be assigned a fine-grained group aborts
qualification — no silent fallbacks.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from mlip_research_agent.data.manifests import LabeledConfiguration

MLEARN_LEVEL_OF_THEORY = "VASP-5.4.1_PBE_PAW_520eV (paper-stated, arXiv:1906.08888)"

# AIMD-correlated snapshots are split in contiguous time blocks so that
# temporally adjacent frames never straddle a partition boundary
# (plan_phase_2.md §4.5; preregistration grouping rule).
AIMD_TIME_BLOCK = 10
VACANCY_TIME_BLOCK = 5

_AIMD_RE = re.compile(r"^Snapshot (\d+) of \d+ of AIMD NVT simulation at (\d+) K$")
_VACANCY_RE = re.compile(r"^Snapshot (\d+) of \d+ of Vacancy AIMD NVT simulation at (\d+) K$")
_ELASTIC_RE = re.compile(r"mode (\d+), strain (-?\d+\.\d+)")
_SURFACE_RE = re.compile(r"Cu_mp-30_\(([\d_]+)\)")


class MlearnParseError(Exception):
    """A record violated the documented mlearn schema. Fail-closed."""


def classify_record(group: str, description: str) -> tuple[str, str]:
    """Return (group_id, split_unit_id) for one record.

    group_id is the fine-grained physical subgroup; split_unit_id is the
    atomic splitting unit (trajectory time block, strain mode, or slab).
    """
    if group == "AIMD-NVT":
        m = _AIMD_RE.match(description)
        if not m:
            raise MlearnParseError(f"unrecognized AIMD description: {description!r}")
        snapshot, temp = int(m.group(1)), int(m.group(2))
        block = (snapshot - 1) // AIMD_TIME_BLOCK
        return f"aimd_nvt_{temp}k", f"aimd_nvt_{temp}k_block{block}"
    if group == "Vacancy":
        m = _VACANCY_RE.match(description)
        if not m:
            raise MlearnParseError(f"unrecognized Vacancy description: {description!r}")
        snapshot, temp = int(m.group(1)), int(m.group(2))
        block = (snapshot - 1) // VACANCY_TIME_BLOCK
        return f"vacancy_{temp}k", f"vacancy_{temp}k_block{block}"
    if group == "Elastic":
        m = _ELASTIC_RE.search(description)
        if m:
            mode = int(m.group(1))
            return f"elastic_mode_{mode}", f"elastic_mode_{mode}"
        # The single ground-state crystal record has no mode/strain text.
        return "elastic_ground_state", "elastic_ground_state"
    if group == "Surface":
        m = _SURFACE_RE.search(description)
        if not m:
            raise MlearnParseError(f"unrecognized Surface description: {description!r}")
        miller = m.group(1)
        return f"surface_{miller}", f"surface_{miller}"
    raise MlearnParseError(f"unknown mlearn group: {group!r}")


def _parse_record(
    raw: dict[str, Any], *, dataset_id: str, source_file: str, index: int
) -> LabeledConfiguration:
    try:
        group = raw["group"]
        description = raw["description"]
        structure = raw["structure"]
        lattice = structure["lattice"]["matrix"]
        sites = structure["sites"]
        outputs = raw["outputs"]
        energy = float(outputs["energy"])
        forces = [[float(x) for x in row] for row in outputs["forces"]]
        stress_raw = outputs.get("virial_stress")
        stress = [float(x) for x in stress_raw] if stress_raw is not None else None
    except (KeyError, TypeError, ValueError) as exc:
        raise MlearnParseError(f"{source_file}[{index}]: malformed record: {exc}") from exc

    symbols: list[str] = []
    positions: list[list[float]] = []
    for site in sites:
        species = site["species"]
        if len(species) != 1 or species[0].get("occu") != 1:
            raise MlearnParseError(
                f"{source_file}[{index}]: non-trivial site occupancy is unsupported"
            )
        symbols.append(str(species[0]["element"]))
        positions.append([float(x) for x in site["xyz"]])

    group_id, split_unit_id = classify_record(group, description)
    source_record_sha256 = hashlib.sha256(
        json.dumps(raw, sort_keys=True).encode()
    ).hexdigest()

    cfg = LabeledConfiguration(
        config_id="pending",  # replaced below; the pattern requires non-empty
        source_id=f"{source_file}[{index}]",
        top_group=group,
        group_id=group_id,
        split_unit_id=split_unit_id,
        symbols=symbols,
        positions=positions,
        cell=[[float(x) for x in row] for row in lattice],
        pbc=[True, True, True],
        energy_ev=energy,
        forces_ev_per_a=forces,
        virial_stress_kbar=stress,
        level_of_theory=MLEARN_LEVEL_OF_THEORY,
        source_record_sha256=source_record_sha256,
    )
    # Identity is content-derived so duplicates collide by construction.
    return cfg.model_copy(update={"config_id": f"{dataset_id}-{cfg.content_hash()[:16]}"})


def parse_mlearn_file(path: Path, *, dataset_id: str) -> list[LabeledConfiguration]:
    raw: Any = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise MlearnParseError(f"{path}: expected a JSON list of records")
    return [
        _parse_record(record, dataset_id=dataset_id, source_file=path.name, index=i)
        for i, record in enumerate(raw)
    ]
