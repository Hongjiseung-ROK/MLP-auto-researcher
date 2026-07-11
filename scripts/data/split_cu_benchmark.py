"""Generate the deterministic WP2 split manifest from the qualified WP1 manifest."""

from __future__ import annotations

import sys
from math import ceil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mlip_research_agent.data.registry import NormalizedManifest  # noqa: E402
from mlip_research_agent.data.split import SplitSpec, build_split_manifest  # noqa: E402
from mlip_research_agent.research import load_pilot_config  # noqa: E402


def main() -> int:
    dataset_dir = REPO_ROOT / "data_registry" / "datasets" / "cu_phase2"
    manifest_path = dataset_dir / "normalized_manifest.json"
    manifest = NormalizedManifest.load(manifest_path)
    pilot = load_pilot_config(REPO_ROOT / "configs" / "research" / "cu_mace_al_pilot.yaml")
    split_config = pilot.preregistration.split
    frozen_test = ceil(manifest.n_configurations * split_config.test_fraction_min)
    validation = ceil(manifest.n_configurations * split_config.validation_fraction_min)
    acquisition_pool = (
        manifest.n_configurations
        - split_config.initial_labeled_count
        - validation
        - frozen_test
    )
    split = build_split_manifest(
        manifest,
        SplitSpec(
            initial_labeled=split_config.initial_labeled_count,
            acquisition_pool=acquisition_pool,
            validation=validation,
            frozen_test=frozen_test,
            seed=split_config.split_seed,
            initial_labeled_min=split_config.initial_labeled_count,
            acquisition_pool_min=split_config.pool_min,
            validation_min=validation,
            frozen_test_min=frozen_test,
        ),
        qualified_manifest_path=manifest_path,
        qualified_manifest_reference=manifest_path.relative_to(REPO_ROOT).as_posix(),
    )
    output = dataset_dir / "split_manifest.json"
    file_hash = split.save(output)
    print(f"split semantic sha256: {split.semantic_hash()}")
    print(f"split file sha256:     {file_hash}")
    print(f"actual sizes:          {split.actual_partition_sizes}")
    print(f"warnings:              {split.warnings}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
