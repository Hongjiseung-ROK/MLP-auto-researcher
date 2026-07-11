"""Typed, fail-closed mutation contracts and the transactional config store.

A mutation names one configuration key, its exact old and new values, the
bounds it must respect, and both its scientific and engineering intent. The
policy classifies every key: anything not explicitly listed is ``immutable``
(fail-closed). Bounded application is transactional — the store keeps every
previously accepted state and rolls back rejected mutations without rewriting
history.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlip_research_agent.research.auto_research.validators import (
    ConfigValue,
    ContentAddressedModel,
    canonical_json,
    sha256_of_text,
)


class MutationClass(StrEnum):
    IMMUTABLE = "immutable"
    OWNER_GATED = "owner_gated"
    BOUNDED_MUTABLE = "bounded_mutable"
    FREELY_MUTABLE_FIXTURE_ONLY = "freely_mutable_fixture_only"


class AllowedRange(BaseModel):
    """Bounds for one bounded-mutable key: numeric interval or choice set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value_type: str = Field(pattern=r"^(float|int|choice)$")
    minimum: float | None = None
    maximum: float | None = None
    choices: list[str] | None = None

    @model_validator(mode="after")
    def _consistent(self) -> AllowedRange:
        if self.value_type in {"float", "int"}:
            if self.minimum is None or self.maximum is None:
                raise ValueError("numeric ranges need both minimum and maximum")
            if self.maximum < self.minimum:
                raise ValueError("maximum must be >= minimum")
            if self.choices is not None:
                raise ValueError("numeric ranges cannot carry choices")
        else:
            if not self.choices:
                raise ValueError("choice ranges need a non-empty choice list")
            if self.minimum is not None or self.maximum is not None:
                raise ValueError("choice ranges cannot carry numeric bounds")
        return self

    def permits(self, value: ConfigValue) -> bool:
        if self.value_type == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                return False
            assert self.minimum is not None and self.maximum is not None
            return self.minimum <= value <= self.maximum
        if self.value_type == "float":
            if isinstance(value, bool) or not isinstance(value, int | float):
                return False
            assert self.minimum is not None and self.maximum is not None
            return self.minimum <= float(value) <= self.maximum
        assert self.choices is not None
        return isinstance(value, str) and value in self.choices


class Mutation(ContentAddressedModel):
    """One declared change to one configuration key."""

    model_config = ConfigDict(extra="forbid")

    target_type: str = Field(pattern=r"^(config)$", description="Only config mutations exist yet")
    target_path: str = Field(min_length=1, description="Which configuration document is mutated")
    target_key: str = Field(min_length=1, description="Dotted key inside the target document")
    old_value: ConfigValue
    new_value: ConfigValue
    allowed_range: AllowedRange | None = None
    reversibility: str = Field(pattern=r"^(reversible|irreversible)$")
    scientific_effect: str = Field(min_length=5, max_length=1000)
    engineering_effect: str = Field(min_length=5, max_length=1000)

    @model_validator(mode="after")
    def _real_change(self) -> Mutation:
        if self.old_value == self.new_value:
            raise ValueError("a mutation must change the value")
        return self


class LegalityResult(ContentAddressedModel):
    """Fail-closed legality verdict for one proposal's mutations."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    policy_id: str
    policy_content_sha256: str
    legal: bool
    per_mutation_class: list[MutationClass]
    violations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _closed(self) -> LegalityResult:
        if self.legal and self.violations:
            raise ValueError("a legal result cannot carry violations")
        if not self.legal and not self.violations:
            raise ValueError("an illegal result must say why")
        return self


class MutationPolicy(BaseModel):
    """configs/research/ralphthon_mutation_policy.yaml. Unlisted keys are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "1.0.0"
    policy_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    bounded_mutable: dict[str, AllowedRange]
    owner_gated: list[str] = Field(default_factory=list)
    freely_mutable_fixture_only: list[str] = Field(default_factory=list)
    immutable: list[str] = Field(
        min_length=1,
        description="Explicitly protected keys; everything unlisted is also immutable",
    )

    @model_validator(mode="after")
    def _disjoint(self) -> MutationPolicy:
        groups: dict[str, set[str]] = {
            "bounded_mutable": set(self.bounded_mutable),
            "owner_gated": set(self.owner_gated),
            "freely_mutable_fixture_only": set(self.freely_mutable_fixture_only),
            "immutable": set(self.immutable),
        }
        names = list(groups)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                overlap = groups[a] & groups[b]
                if overlap:
                    raise ValueError(f"keys in both {a} and {b}: {sorted(overlap)}")
        return self

    def content_hash(self) -> str:
        return sha256_of_text(canonical_json(self.model_dump(mode="json")))

    def classify(self, key: str) -> MutationClass:
        if key in self.bounded_mutable:
            return MutationClass.BOUNDED_MUTABLE
        if key in self.owner_gated:
            return MutationClass.OWNER_GATED
        if key in self.freely_mutable_fixture_only:
            return MutationClass.FREELY_MUTABLE_FIXTURE_ONLY
        # Fail-closed: explicit immutable list and every unknown key.
        return MutationClass.IMMUTABLE

    def check_mutations(
        self,
        proposal_id: str,
        mutations: list[Mutation],
        current_config: dict[str, ConfigValue],
        *,
        allowed_classes: list[MutationClass],
    ) -> LegalityResult:
        violations: list[str] = []
        classes: list[MutationClass] = []
        seen_keys: set[str] = set()
        for m in mutations:
            cls = self.classify(m.target_key)
            classes.append(cls)
            if m.target_key in seen_keys:
                violations.append(f"{m.target_key}: duplicate mutation of one key")
            seen_keys.add(m.target_key)
            if cls is MutationClass.IMMUTABLE:
                violations.append(f"{m.target_key}: immutable under policy {self.policy_id}")
                continue
            if cls is MutationClass.OWNER_GATED:
                violations.append(
                    f"{m.target_key}: owner-gated; requires an explicit approval record"
                )
                continue
            if cls not in allowed_classes:
                violations.append(
                    f"{m.target_key}: class {cls.value} not permitted by the objective"
                )
                continue
            declared = self.bounded_mutable.get(m.target_key)
            if declared is not None:
                if m.allowed_range != declared:
                    violations.append(
                        f"{m.target_key}: proposal's allowed_range does not match policy"
                    )
                    continue
                if not declared.permits(m.new_value):
                    violations.append(
                        f"{m.target_key}: new value {m.new_value!r} outside declared bounds"
                    )
                    continue
            if m.target_key not in current_config:
                violations.append(f"{m.target_key}: key absent from the current configuration")
                continue
            if current_config[m.target_key] != m.old_value:
                violations.append(
                    f"{m.target_key}: old_value {m.old_value!r} does not match "
                    f"current {current_config[m.target_key]!r}"
                )
        return LegalityResult(
            proposal_id=proposal_id,
            policy_id=self.policy_id,
            policy_content_sha256=self.content_hash(),
            legal=not violations,
            per_mutation_class=classes,
            violations=violations,
        ).sealed()

    @classmethod
    def load(cls, path: Path) -> MutationPolicy:
        raw: Any = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"mutation policy {path} must be a YAML mapping")
        return cls.model_validate(raw)


class MutationDiff(ContentAddressedModel):
    """Canonical diff artifact for one applied (or rolled-back) mutation set."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    base_config_sha256: str
    result_config_sha256: str
    changes: list[Mutation]
    applied: bool
    rolled_back: bool = False


class TransactionalConfigStore:
    """Applies mutations transactionally over a JSON config history.

    Layout under ``root``: ``state-000.json`` is the base configuration;
    every accepted mutation appends ``state-<n>.json``. ``current.json``
    names the accepted head. Historical states are never rewritten.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def _state_path(self, index: int) -> Path:
        return self.root / f"state-{index:03d}.json"

    def _pointer_path(self) -> Path:
        return self.root / "current.json"

    def initialize(self, base_config: dict[str, ConfigValue]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self._pointer_path().is_file():
            return  # resume: never rewrite history
        self._write_state(0, base_config)
        self._write_pointer(current_index=0, next_index=1, candidate_index=None)

    def _write_state(self, index: int, config: dict[str, ConfigValue]) -> None:
        path = self._state_path(index)
        if path.is_file():
            raise FileExistsError(f"config history is append-only: {path} already exists")
        path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    def _write_pointer(
        self, *, current_index: int, next_index: int, candidate_index: int | None
    ) -> None:
        self._pointer_path().write_text(
            json.dumps(
                {
                    "current_index": current_index,
                    "next_index": next_index,
                    "candidate_index": candidate_index,
                },
                sort_keys=True,
            )
            + "\n"
        )

    def _read_pointer(self) -> dict[str, int | None]:
        raw = json.loads(self._pointer_path().read_text())
        if not isinstance(raw, dict):
            raise ValueError("corrupt config-store pointer")
        return raw

    @property
    def current_index(self) -> int:
        index = self._read_pointer()["current_index"]
        if not isinstance(index, int):
            raise ValueError("corrupt config-store pointer")
        return index

    def read_state(self, index: int) -> dict[str, ConfigValue]:
        raw = json.loads(self._state_path(index).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"corrupt config state {index}")
        return raw

    def current_config(self) -> dict[str, ConfigValue]:
        return self.read_state(self.current_index)

    def config_sha256(self, config: dict[str, ConfigValue]) -> str:
        return sha256_of_text(canonical_json(config))

    def apply(self, proposal_id: str, mutations: list[Mutation]) -> MutationDiff:
        """Write the candidate state without moving the accepted pointer."""
        pointer = self._read_pointer()
        current = pointer["current_index"]
        next_index = pointer["next_index"]
        if pointer["candidate_index"] is not None:
            raise ValueError("a candidate is already pending; commit or roll back first")
        if not isinstance(current, int) or not isinstance(next_index, int):
            raise ValueError("corrupt config-store pointer")
        base = self.read_state(current)
        candidate = dict(base)
        for m in mutations:
            if m.target_key not in candidate or candidate[m.target_key] != m.old_value:
                raise ValueError(
                    f"transactional apply failed: {m.target_key} no longer matches old_value"
                )
            candidate[m.target_key] = m.new_value
        self._write_state(next_index, candidate)
        self._write_pointer(
            current_index=current, next_index=next_index + 1, candidate_index=next_index
        )
        return MutationDiff(
            proposal_id=proposal_id,
            base_config_sha256=self.config_sha256(base),
            result_config_sha256=self.config_sha256(candidate),
            changes=mutations,
            applied=True,
        ).sealed()

    def has_pending_candidate(self) -> bool:
        return self._read_pointer()["candidate_index"] is not None

    def _candidate_index(self) -> int:
        candidate = self._read_pointer()["candidate_index"]
        if not isinstance(candidate, int):
            raise ValueError("no pending candidate configuration")
        return candidate

    def candidate_config(self) -> dict[str, ConfigValue]:
        return self.read_state(self._candidate_index())

    def commit(self) -> None:
        """Accept the pending candidate state (advance the accepted pointer)."""
        pointer = self._read_pointer()
        candidate = self._candidate_index()
        next_index = pointer["next_index"]
        assert isinstance(next_index, int)
        self._write_pointer(
            current_index=candidate, next_index=next_index, candidate_index=None
        )

    def rollback(self) -> None:
        """Reject the pending candidate: the accepted pointer stays; the
        candidate file remains on disk as immutable history of the attempt."""
        pointer = self._read_pointer()
        current = pointer["current_index"]
        next_index = pointer["next_index"]
        self._candidate_index()  # asserts a candidate is pending
        assert isinstance(current, int) and isinstance(next_index, int)
        self._write_pointer(
            current_index=current, next_index=next_index, candidate_index=None
        )
