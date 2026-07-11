"""WP2 hidden-label, budget, idempotency, and transaction gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlip_research_agent.artifacts.registry import ArtifactRegistry
from mlip_research_agent.data.manifests import LabeledConfiguration, NormalizedDataset
from mlip_research_agent.data.oracle import (
    OracleAccessError,
    OracleBudgetError,
    OracleConflictError,
    SimulatedLabelOracle,
    build_acquisition_view,
)
from mlip_research_agent.data.registry import NormalizedManifest, sha256_file
from mlip_research_agent.data.split import (
    PartitionName,
    SplitManifest,
    SplitSpec,
    _partition_hash,
    build_split_manifest,
)
from mlip_research_agent.skills.active_learning.oracle_reveal.implementation import (
    TRANSACTION_DIR,
    OracleRevealSkill,
)
from mlip_research_agent.skills.active_learning.oracle_reveal.schema import (
    OracleRevealInput,
    OracleRevealOutput,
)
from mlip_research_agent.skills.base import SkillContext, SkillError


def make_dataset() -> NormalizedDataset:
    records: list[LabeledConfiguration] = []
    for index in range(12):
        records.append(
            LabeledConfiguration(
                config_id=f"oracle-fixture-{index:02d}",
                source_id=f"fixture.json[{index}]",
                top_group=f"family-{index // 4}",
                group_id=f"trajectory-{index // 2}",
                split_unit_id=f"unit-{index:02d}",
                symbols=["Cu", "Cu"],
                positions=[[0.0, 0.0, 0.0], [1.0 + index / 100, 1.0, 1.0]],
                cell=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                pbc=[True, True, True],
                energy_ev=-7.0 - index / 10,
                forces_ev_per_a=[[index / 100, 0.0, 0.0], [0.0, -index / 100, 0.0]],
                virial_stress_kbar=None,
                level_of_theory="synthetic-test",
                source_record_sha256=f"{index + 100:064x}",
            )
        )
    return NormalizedDataset(
        dataset_id="oracle_fixture",
        level_of_theory="synthetic-test",
        configurations=records,
    )


def prepare(tmp_path: Path) -> tuple[NormalizedDataset, SplitManifest, Path, Path, Path]:
    inputs = tmp_path / "inputs"
    dataset = make_dataset()
    dataset_path = inputs / "normalized_dataset.json"
    dataset_file_sha = dataset.save(dataset_path)
    manifest = NormalizedManifest.from_dataset(dataset, dataset_file_sha)
    manifest_path = inputs / "normalized_manifest.json"
    manifest.save(manifest_path)
    split = build_split_manifest(
        manifest,
        SplitSpec(
            initial_labeled=2,
            acquisition_pool=6,
            validation=2,
            frozen_test=2,
            seed=55,
        ),
        qualified_manifest_path=manifest_path,
        qualified_manifest_reference="inputs/normalized_manifest.json",
    )
    split_path = inputs / "split_manifest.json"
    split.save(split_path)
    return dataset, split, dataset_path, manifest_path, split_path


def pool_ids(split: SplitManifest) -> list[str]:
    return split.record_ids[PartitionName.ACQUISITION_POOL.value]


def oracle(
    tmp_path: Path,
    dataset: NormalizedDataset,
    split: SplitManifest,
    *,
    state_name: str = "state.json",
    previous: Path | None = None,
    total_budget: int = 4,
    round_budgets: dict[str, int] | None = None,
) -> SimulatedLabelOracle:
    return SimulatedLabelOracle(
        dataset,
        split,
        state_path=tmp_path / state_name,
        campaign_id="pilot-random",
        total_budget=total_budget,
        round_budgets=round_budgets or {"round-1": 2, "round-2": 2},
        event_reference="test:oracle",
        previous_state_path=previous,
        allow_new_campaign=previous is None and not (tmp_path / state_name).exists(),
    )


def test_acquisition_view_serializes_no_hidden_labels(tmp_path: Path) -> None:
    dataset, split, _, _, _ = prepare(tmp_path)
    view = build_acquisition_view(dataset, split)
    path = tmp_path / "acquisition_view.json"
    view.save(path)
    payload = json.loads(path.read_text())
    serialized = path.read_text()
    assert len(payload["candidates"]) == 6
    for forbidden in (
        "energy_ev",
        "forces_ev_per_a",
        "virial_stress_kbar",
        "source_record_sha256",
        "normalized_record_sha256",
        "source_id",
        "level_of_theory",
    ):
        assert forbidden not in serialized


def test_oracle_reveals_only_selected_pool_ids_with_source_hashes(tmp_path: Path) -> None:
    dataset, split, _, _, _ = prepare(tmp_path)
    selected = pool_ids(split)[:2]
    result = oracle(tmp_path, dataset, split).reveal_result(
        selected, "pilot-random", "round-1"
    )
    assert [record.record_id for record in result.label_batch.records] == sorted(selected)
    assert result.ledger.charged_total == 2
    assert result.lineage.training_record_ids == sorted(
        split.record_ids[PartitionName.INITIAL_LABELED.value] + selected
    )
    assert all(len(record.source_record_sha256) == 64 for record in result.label_batch.records)
    assert all(len(record.normalized_record_sha256) == 64 for record in result.label_batch.records)


@pytest.mark.parametrize("partition", [PartitionName.FROZEN_TEST, PartitionName.VALIDATION])
def test_test_and_validation_reveal_rejected(
    tmp_path: Path, partition: PartitionName
) -> None:
    dataset, split, _, _, _ = prepare(tmp_path)
    protected_id = split.record_ids[partition.value][0]
    state = tmp_path / "state.json"
    with pytest.raises(OracleAccessError, match=partition.value):
        oracle(tmp_path, dataset, split).reveal([protected_id], "pilot-random", "round-1")
    assert not state.exists()


def test_unknown_and_duplicate_ids_rejected_without_state(tmp_path: Path) -> None:
    dataset, split, _, _, _ = prepare(tmp_path)
    one = pool_ids(split)[0]
    for candidates, match in [(["unknown-id"], "unknown"), ([one, one], "duplicate")]:
        state = tmp_path / f"{match}.json"
        with pytest.raises(OracleAccessError, match=match):
            oracle(tmp_path, dataset, split, state_name=state.name).reveal(
                candidates, "pilot-random", "round-1"
            )
        assert not state.exists()


def test_round_and_campaign_budget_overrun_rejected_atomically(tmp_path: Path) -> None:
    dataset, split, _, _, _ = prepare(tmp_path)
    selected = pool_ids(split)[:2]
    with pytest.raises(OracleBudgetError, match="round"):
        oracle(
            tmp_path,
            dataset,
            split,
            total_budget=4,
            round_budgets={"round-1": 1},
        ).reveal(selected, "pilot-random", "round-1")
    assert not (tmp_path / "state.json").exists()

    with pytest.raises(OracleBudgetError, match="campaign"):
        oracle(
            tmp_path,
            dataset,
            split,
            state_name="campaign.json",
            total_budget=1,
            round_budgets={"round-1": 2},
        ).reveal(selected, "pilot-random", "round-1")
    assert not (tmp_path / "campaign.json").exists()


def test_identical_reveal_is_idempotent_and_conflict_fails(tmp_path: Path) -> None:
    dataset, split, _, _, _ = prepare(tmp_path)
    selected = pool_ids(split)[:2]
    first_oracle = oracle(tmp_path, dataset, split)
    first = first_oracle.reveal_result(selected, "pilot-random", "round-1")
    state_bytes = (tmp_path / "state.json").read_bytes()
    repeated = oracle(tmp_path, dataset, split).reveal_result(
        list(reversed(selected)), "pilot-random", "round-1"
    )
    assert repeated.idempotent_replay
    assert repeated.label_batch == first.label_batch
    assert repeated.ledger.charged_total == 2
    assert (tmp_path / "state.json").read_bytes() == state_bytes

    with pytest.raises(OracleConflictError, match="different immutable"):
        oracle(tmp_path, dataset, split).reveal(
            pool_ids(split)[1:3], "pilot-random", "round-1"
        )


def test_resume_and_cross_round_reselection_do_not_double_charge(tmp_path: Path) -> None:
    dataset, split, _, _, _ = prepare(tmp_path)
    selected = pool_ids(split)[:2]
    first_state = tmp_path / "round1.json"
    first = oracle(
        tmp_path, dataset, split, state_name=first_state.name
    ).reveal_result(selected, "pilot-random", "round-1")
    assert first.ledger.charged_total == 2

    second = oracle(
        tmp_path,
        dataset,
        split,
        state_name="round2.json",
        previous=first_state,
    ).reveal_result([selected[0]], "pilot-random", "round-2")
    assert second.decision.charged_record_ids == []
    assert second.decision.previously_revealed_record_ids == [selected[0]]
    assert second.ledger.charged_total == 2
    assert len(second.lineage.training_record_ids) == len(first.lineage.training_record_ids)


def test_failed_state_commit_reveals_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, split, _, _, _ = prepare(tmp_path)
    selected = pool_ids(split)[:1]

    def fail_write(path: Path, payload: object) -> str:
        raise OSError("injected commit failure")

    monkeypatch.setattr("mlip_research_agent.data.oracle.write_json_atomic", fail_write)
    with pytest.raises(OSError, match="injected"):
        oracle(tmp_path, dataset, split).reveal(selected, "pilot-random", "round-1")
    assert not (tmp_path / "state.json").exists()


def test_skill_registers_complete_atomic_transaction_and_resumes(tmp_path: Path) -> None:
    _, split, dataset_path, manifest_path, split_path = prepare(tmp_path)
    params = OracleRevealInput(
        dataset_path=dataset_path.relative_to(tmp_path).as_posix(),
        qualified_manifest_path=manifest_path.relative_to(tmp_path).as_posix(),
        split_manifest_path=split_path.relative_to(tmp_path).as_posix(),
        expected_split_manifest_sha256=sha256_file(split_path),
        candidate_ids=pool_ids(split)[:2],
        campaign_id="pilot-random",
        round_id="round-1",
        total_budget=4,
        round_budgets={"round-1": 2, "round-2": 2},
    )
    registry = ArtifactRegistry(tmp_path)
    ctx = SkillContext(
        run_dir=tmp_path,
        step_id="oracle-round-1",
        seed=0,
        attempt=0,
        registry=registry,
    )
    output = OracleRevealSkill().run(params, ctx)
    assert isinstance(output, OracleRevealOutput)
    assert output.n_newly_charged == 2
    assert len(registry.all()) == 6
    assert all(registry.verify(artifact.artifact_id) for artifact in registry.all())
    assert (ctx.step_dir / TRANSACTION_DIR).is_dir()
    decision = json.loads((tmp_path / output.decision_path).read_text())
    assert decision["event_reference"].endswith(":oracle-round-1:STEP_COMPLETED")

    repeated = OracleRevealSkill().run(params, ctx)
    assert isinstance(repeated, OracleRevealOutput)
    assert repeated.idempotent_replay
    assert repeated.remaining_budget == output.remaining_budget


def test_skill_failure_leaves_no_partial_transaction(tmp_path: Path) -> None:
    _, split, dataset_path, manifest_path, split_path = prepare(tmp_path)
    params = OracleRevealInput(
        dataset_path=dataset_path.relative_to(tmp_path).as_posix(),
        qualified_manifest_path=manifest_path.relative_to(tmp_path).as_posix(),
        split_manifest_path=split_path.relative_to(tmp_path).as_posix(),
        expected_split_manifest_sha256=sha256_file(split_path),
        candidate_ids=pool_ids(split)[:2],
        campaign_id="pilot-random",
        round_id="round-1",
        total_budget=1,
        round_budgets={"round-1": 1},
    )
    ctx = SkillContext(
        run_dir=tmp_path,
        step_id="oracle-fail",
        seed=0,
        attempt=0,
        registry=ArtifactRegistry(tmp_path),
    )
    with pytest.raises(SkillError, match="budget exceeded"):
        OracleRevealSkill().run(params, ctx)
    assert not (ctx.step_dir / TRANSACTION_DIR).exists()
    assert ctx.registry.all() == []


def test_stale_oracle_writer_cannot_overwrite_new_state(tmp_path: Path) -> None:
    dataset, split, _, _, _ = prepare(tmp_path)
    state_path = tmp_path / "shared-state.json"
    first = SimulatedLabelOracle(
        dataset,
        split,
        state_path=state_path,
        campaign_id="pilot-random",
        total_budget=4,
        round_budgets={"round-1": 2, "round-2": 2},
        allow_new_campaign=True,
    )
    stale = SimulatedLabelOracle(
        dataset,
        split,
        state_path=state_path,
        campaign_id="pilot-random",
        total_budget=4,
        round_budgets={"round-1": 2, "round-2": 2},
        allow_new_campaign=True,
    )
    first.reveal(pool_ids(split)[:2], "pilot-random", "round-1")
    with pytest.raises(OracleConflictError, match="changed since"):
        stale.reveal(pool_ids(split)[2:4], "pilot-random", "round-2")
    state = json.loads(state_path.read_text())
    assert set(state["requests_by_round"]) == {"round-1"}
    assert state["ledger"]["charged_total"] == 2


def test_stale_lock_inode_does_not_block_interrupted_resume(tmp_path: Path) -> None:
    dataset, split, _, _, _ = prepare(tmp_path)
    state_path = tmp_path / "interrupted-state.json"
    state_path.with_name(f".{state_path.name}.lock").write_text("")
    result = SimulatedLabelOracle(
        dataset,
        split,
        state_path=state_path,
        campaign_id="pilot-random",
        total_budget=2,
        round_budgets={"round-1": 2},
        allow_new_campaign=True,
    ).reveal_result(pool_ids(split)[:2], "pilot-random", "round-1")
    assert result.ledger.charged_total == 2


def test_skill_requires_registered_campaign_head_for_continuation(tmp_path: Path) -> None:
    _, split, dataset_path, manifest_path, split_path = prepare(tmp_path)
    registry = ArtifactRegistry(tmp_path)
    common = {
        "dataset_path": dataset_path.relative_to(tmp_path).as_posix(),
        "qualified_manifest_path": manifest_path.relative_to(tmp_path).as_posix(),
        "split_manifest_path": split_path.relative_to(tmp_path).as_posix(),
        "expected_split_manifest_sha256": sha256_file(split_path),
        "campaign_id": "pilot-random",
        "total_budget": 4,
        "round_budgets": {"round-1": 2, "round-2": 2},
    }
    ctx1 = SkillContext(
        run_dir=tmp_path, step_id="oracle-round-1", seed=0, attempt=0, registry=registry
    )
    first = OracleRevealSkill().run(
        OracleRevealInput.model_validate(
            {**common, "candidate_ids": pool_ids(split)[:2], "round_id": "round-1"}
        ),
        ctx1,
    )
    assert isinstance(first, OracleRevealOutput)

    fork_ctx = SkillContext(
        run_dir=tmp_path, step_id="oracle-fork", seed=0, attempt=0, registry=registry
    )
    with pytest.raises(SkillError, match="must supply its registered head"):
        OracleRevealSkill().run(
            OracleRevealInput.model_validate(
                {**common, "candidate_ids": pool_ids(split)[2:4], "round_id": "round-2"}
            ),
            fork_ctx,
        )

    registered_state_path = tmp_path / first.oracle_state_path
    original_state = registered_state_path.read_bytes()
    registered_state_path.write_text("{}")
    with pytest.raises(SkillError, match="failed artifact verification"):
        OracleRevealSkill().run(
            OracleRevealInput.model_validate(
                {
                    **common,
                    "candidate_ids": pool_ids(split)[2:4],
                    "round_id": "round-2",
                    "previous_state_path": first.oracle_state_path,
                }
            ),
            fork_ctx,
        )
    registered_state_path.write_bytes(original_state)

    ctx2 = SkillContext(
        run_dir=tmp_path, step_id="oracle-round-2", seed=0, attempt=0, registry=registry
    )
    second = OracleRevealSkill().run(
        OracleRevealInput.model_validate(
            {
                **common,
                "candidate_ids": pool_ids(split)[2:4],
                "round_id": "round-2",
                "previous_state_path": first.oracle_state_path,
            }
        ),
        ctx2,
    )
    assert isinstance(second, OracleRevealOutput)
    assert second.remaining_budget == 0


def test_skill_rejects_unpinned_or_forged_split(tmp_path: Path) -> None:
    _, split, dataset_path, manifest_path, split_path = prepare(tmp_path)
    base = {
        "dataset_path": dataset_path.relative_to(tmp_path).as_posix(),
        "qualified_manifest_path": manifest_path.relative_to(tmp_path).as_posix(),
        "split_manifest_path": split_path.relative_to(tmp_path).as_posix(),
        "candidate_ids": pool_ids(split)[:1],
        "campaign_id": "pilot-random",
        "round_id": "round-1",
        "total_budget": 2,
        "round_budgets": {"round-1": 2},
    }
    ctx = SkillContext(
        run_dir=tmp_path,
        step_id="oracle-bad-split",
        seed=0,
        attempt=0,
        registry=ArtifactRegistry(tmp_path),
    )
    with pytest.raises(SkillError, match="pinned expected"):
        OracleRevealSkill().run(
            OracleRevealInput.model_validate(
                {**base, "expected_split_manifest_sha256": "0" * 64}
            ),
            ctx,
        )

    payload = split.model_dump(mode="json")
    pool = PartitionName.ACQUISITION_POOL.value
    test = PartitionName.FROZEN_TEST.value
    payload["record_ids"][pool][0], payload["record_ids"][test][0] = (
        payload["record_ids"][test][0],
        payload["record_ids"][pool][0],
    )
    for partition in (pool, test):
        payload["record_ids"][partition].sort()
        payload["partition_hashes"][partition] = _partition_hash(
            payload["record_ids"][partition], payload["source_group_ids"][partition]
        )
    forged = SplitManifest.model_validate(payload)
    forged.save(split_path)
    forged_id = forged.record_ids[pool][0]
    forged_params = {**base, "candidate_ids": [forged_id]}
    with pytest.raises(SkillError, match="record/group mapping mismatch"):
        OracleRevealSkill().run(
            OracleRevealInput.model_validate(
                {
                    **forged_params,
                    "expected_split_manifest_sha256": sha256_file(split_path),
                }
            ),
            ctx,
        )


def test_resume_rejects_changed_inputs_and_transaction_files(tmp_path: Path) -> None:
    dataset, split, dataset_path, manifest_path, split_path = prepare(tmp_path)
    params = OracleRevealInput(
        dataset_path=dataset_path.relative_to(tmp_path).as_posix(),
        qualified_manifest_path=manifest_path.relative_to(tmp_path).as_posix(),
        split_manifest_path=split_path.relative_to(tmp_path).as_posix(),
        expected_split_manifest_sha256=sha256_file(split_path),
        candidate_ids=pool_ids(split)[:1],
        campaign_id="pilot-random",
        round_id="round-1",
        total_budget=2,
        round_budgets={"round-1": 2},
    )
    registry = ArtifactRegistry(tmp_path)
    ctx = SkillContext(
        run_dir=tmp_path, step_id="oracle-resume", seed=0, attempt=0, registry=registry
    )
    output = OracleRevealSkill().run(params, ctx)
    assert isinstance(output, OracleRevealOutput)
    decision_path = tmp_path / output.decision_path
    original_decision = decision_path.read_text()
    decision = json.loads(original_decision)
    decision["reason"] = "tampered"
    decision_path.write_text(json.dumps(decision))
    with pytest.raises(SkillError, match="standalone oracle artifacts"):
        OracleRevealSkill().run(params, ctx)
    decision_path.write_text(original_decision)

    changed = dataset.model_copy(deep=True)
    changed.configurations[0].energy_ev += 1.0
    changed.save(dataset_path)
    with pytest.raises(SkillError, match="dataset and manifest lineage"):
        OracleRevealSkill().run(params, ctx)
