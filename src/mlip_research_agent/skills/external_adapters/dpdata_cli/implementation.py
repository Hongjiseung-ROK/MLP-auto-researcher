"""NormalizedDataset → deepmd/npy conversion via dpdata; fail-closed without it."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from mlip_research_agent.artifacts.registry import sha256_file
from mlip_research_agent.skills.base import Skill, SkillContext, expect_inputs, register_skill
from mlip_research_agent.skills.external_adapters.common import (
    adapter_error,
    require_module,
    upstream_provenance,
    write_registered_json,
)
from mlip_research_agent.skills.external_adapters.dpdata_cli.schema import (
    DpdataConvertInput,
    DpdataConvertOutput,
)
from mlip_research_agent.skills.external_adapters.dpdata_cli.validators import (
    finite_labeled_configurations,
)

UPSTREAM_PATH = "data-processing/dpdata-cli"
MANIFEST_FILE = "dpdata_conversion_manifest.json"
OUTPUT_DIR = "deepmd_npy"


@register_skill
class ExternalDpdataConvertSkill(Skill):
    """Convert a first-party labeled dataset to deepmd/npy systems."""

    name = "external_dpdata_cli"
    input_model = DpdataConvertInput
    output_model = DpdataConvertOutput
    cost_class = "cheap"

    def run(self, inputs: BaseModel, ctx: SkillContext) -> BaseModel:
        params = expect_inputs(inputs, DpdataConvertInput)
        dpdata = require_module("dpdata", package="dpdata", extra="dpdata")
        import numpy as np

        from mlip_research_agent.data.manifests import NormalizedDataset

        dataset_path = (ctx.run_dir / params.dataset_path).resolve()
        try:
            dataset_path.relative_to(ctx.run_dir.resolve())
        except ValueError as exc:
            raise adapter_error("dataset path escapes the run directory") from exc
        try:
            dataset = NormalizedDataset.load(dataset_path)
        except (OSError, ValueError) as exc:
            raise adapter_error(f"dataset file is invalid: {exc}") from exc
        finite_labeled_configurations(dataset.configurations)

        # dpdata systems require one fixed atom ordering; group by symbol tuple.
        groups: dict[tuple[str, ...], list[int]] = {}
        for position, configuration in enumerate(dataset.configurations):
            groups.setdefault(tuple(configuration.symbols), []).append(position)

        output_root = ctx.step_dir / OUTPUT_DIR
        written: list[dict[str, object]] = []
        for system_index, (symbols, members) in enumerate(
            sorted(groups.items(), key=lambda item: (len(item[0]), item[0]))
        ):
            atom_names = sorted(set(symbols))
            atom_types = np.array([atom_names.index(symbol) for symbol in symbols])
            frames = [dataset.configurations[position] for position in members]
            data = {
                "atom_names": atom_names,
                "atom_numbs": [int((atom_types == i).sum()) for i in range(len(atom_names))],
                "atom_types": atom_types,
                "cells": np.array([frame.cell for frame in frames], dtype=float),
                "coords": np.array([frame.positions for frame in frames], dtype=float),
                "energies": np.array([frame.energy_ev for frame in frames], dtype=float),
                "forces": np.array([frame.forces_ev_per_a for frame in frames], dtype=float),
                "orig": np.zeros(3),
            }
            system = dpdata.LabeledSystem(data=data)
            system_dir = output_root / f"system-{system_index:03d}"
            system.to_deepmd_npy(str(system_dir))
            for path in sorted(system_dir.rglob("*")):
                if path.is_file():
                    written.append(
                        {
                            "relative_path": path.relative_to(ctx.step_dir).as_posix(),
                            "sha256": sha256_file(path),
                            "size_bytes": path.stat().st_size,
                        }
                    )
        if not written:
            raise adapter_error("dpdata conversion produced no files")

        manifest_artifact = write_registered_json(
            ctx,
            MANIFEST_FILE,
            {
                **upstream_provenance(UPSTREAM_PATH),
                "tool": "dpdata",
                "input": params.model_dump(mode="json"),
                "dataset_content_sha256": dataset.content_hash(),
                "units": {"energy": "eV", "forces": "eV/angstrom", "length": "angstrom"},
                "files": written,
            },
            "external_dpdata_manifest",
        )
        self._register_files(ctx, written)
        return DpdataConvertOutput(
            output_manifest_artifact=manifest_artifact.artifact_id,
            output_manifest_path=manifest_artifact.relative_path,
            n_configurations=len(dataset.configurations),
            n_systems=len(groups),
            n_files=len(written),
        )

    @staticmethod
    def _register_files(ctx: SkillContext, written: list[dict[str, object]]) -> None:
        for entry in written:
            ctx.registry.register(
                Path(ctx.step_dir / str(entry["relative_path"])),
                "external_deepmd_npy",
                ctx.step_id,
            )
