"""Version-gated MACE training loop with owned early-stop and resume semantics."""

from __future__ import annotations

import importlib.metadata
import math
import os
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

from mlip_research_agent.data.manifests import LabeledConfiguration

SUPPORTED_MACE_VERSION = "0.3.16"
MONITOR = "validation_force_component_mae_ev_per_a"


@dataclass(frozen=True)
class ControlledTrainingResult:
    completed_epochs: int
    optimizer_steps: int
    best_validation_force_mae_ev_per_a: float
    patience_count: int
    stopped_early: bool
    records: list[dict[str, float | int | str]]
    changed_parameter_tensors: int
    max_abs_parameter_change: float


def require_supported_mace() -> None:
    installed = importlib.metadata.version("mace-torch")
    if installed != SUPPORTED_MACE_VERSION:
        raise RuntimeError(
            f"controlled fine-tune runner requires mace-torch=={SUPPORTED_MACE_VERSION}; "
            f"found {installed}"
        )


@contextmanager
def _default_dtype(torch: Any, dtype: Any) -> Iterator[None]:
    """MACE data builders create tensors at torch's process default dtype, so the
    requested dtype must be the default for the whole loop — and restored after."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _atoms(record: LabeledConfiguration) -> Atoms:
    atoms = Atoms(
        symbols=record.symbols,
        positions=np.asarray(record.positions),
        cell=np.asarray(record.cell),
        pbc=record.pbc,
    )
    atoms.info["REF_energy"] = record.energy_ev
    atoms.arrays["REF_forces"] = np.asarray(record.forces_ev_per_a)
    return atoms


def _loader(
    records: list[LabeledConfiguration],
    model: Any,
    batch_size: int,
    *,
    shuffle: bool,
    generator: Any,
) -> Any:
    from mace.data import AtomicData, KeySpecification, config_from_atoms
    from mace.tools.torch_geometric.dataloader import DataLoader
    from mace.tools.utils import AtomicNumberTable

    keys = KeySpecification(
        info_keys={"energy": "REF_energy"},
        arrays_keys={"forces": "REF_forces"},
    )
    atomic_numbers = [int(value) for value in model.atomic_numbers.tolist()]
    z_table = AtomicNumberTable(atomic_numbers)
    cutoff = float(model.r_max.detach().cpu())
    data: Any = [
        AtomicData.from_config(
            config_from_atoms(_atoms(record), key_specification=keys),
            z_table=z_table,
            cutoff=cutoff,
        )
        for record in records
    ]
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def _atomic_torch_save(torch: Any, payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _parameter_change_stats(base: dict[str, Any], model: Any) -> tuple[int, float]:
    import torch

    changed = 0
    maximum = 0.0
    for name, parameter in model.named_parameters():
        reference = base.get(name)
        if reference is None or reference.shape != parameter.shape:
            continue
        current = parameter.detach().cpu()
        if not bool(torch.isfinite(current).all()):
            raise ValueError(f"fine-tuned parameter is non-finite: {name}")
        local = float((current - reference).abs().max()) if current.numel() else 0.0
        if local > 0.0:
            changed += 1
            maximum = max(maximum, local)
    if changed == 0:
        raise ValueError("fine-tuning did not change any comparable parameter tensor")
    return changed, maximum


def run_controlled_training(
    *,
    foundation_path: Path,
    checkpoint_path: Path,
    model_path: Path,
    train_records: list[LabeledConfiguration],
    validation_records: list[LabeledConfiguration],
    resume_checkpoint_path: Path | None,
    resume_contract_sha256: str,
    seed: int,
    optimizer_name: str,
    learning_rate: float,
    gradient_clip: float,
    batch_size: int,
    valid_batch_size: int,
    max_epochs: int,
    max_optimizer_steps: int,
    patience: int,
    device_name: str,
    default_dtype: str,
    max_wall_seconds: int,
) -> ControlledTrainingResult:
    """Train only at full-epoch boundaries and checkpoint the complete continuation state."""
    require_supported_mace()
    import torch
    from mace.modules.loss import WeightedEnergyForcesLoss
    from mace.tools.train import evaluate, take_step

    if batch_size < len(train_records):
        raise ValueError(
            "controlled boundary runner requires one full training batch per epoch"
        )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    dtype = torch.float32 if default_dtype == "float32" else torch.float64
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    with _default_dtype(torch, dtype):
        model: Any = torch.load(foundation_path, map_location="cpu", weights_only=False)
        model = model.to(device=device, dtype=dtype)
        base_parameters = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        }
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer_class = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
        optimizer = optimizer_class(parameters, lr=learning_rate, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(1, patience // 2)
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        start_epoch = 0
        optimizer_steps = 0
        best_force_mae = math.inf
        patience_count = 0
        records: list[dict[str, float | int | str]] = []
        if resume_checkpoint_path is not None:
            # Always deserialize to CPU: RNG states must stay CPU ByteTensors,
            # and load_state_dict moves model/optimizer state to the live device.
            state: dict[str, Any] = torch.load(
                resume_checkpoint_path, map_location="cpu", weights_only=False
            )
            if state.get("schema_version") != "1.0.0":
                raise ValueError("unsupported controlled MACE checkpoint schema")
            if state.get("resume_contract_sha256") != resume_contract_sha256:
                raise ValueError("controlled checkpoint resume contract mismatch")
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            start_epoch = int(state["next_epoch"])
            optimizer_steps = int(state["optimizer_steps"])
            best_force_mae = float(state["best_validation_force_mae_ev_per_a"])
            patience_count = int(state["patience_count"])
            records = list(state["metric_records"])
            random.setstate(state["python_rng_state"])
            np.random.set_state(state["numpy_rng_state"])
            torch.set_rng_state(state["torch_rng_state"])
            generator.set_state(state["loader_generator_state"])

        train_loader = _loader(
            train_records, model, batch_size, shuffle=True, generator=generator
        )
        validation_loader = _loader(
            validation_records,
            model,
            valid_batch_size,
            shuffle=False,
            generator=generator,
        )
        loss_fn = WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=100.0)
        output_args = {"energy": True, "forces": True, "virials": False, "stress": False}
        started = time.monotonic()
        stopped_early = False
        completed_epochs = start_epoch
        for epoch in range(start_epoch, max_epochs):
            if optimizer_steps >= max_optimizer_steps:
                break
            model.train()
            batches = list(train_loader)
            if len(batches) != 1:
                raise ValueError("controlled runner expected exactly one training batch")
            # mace annotates take_step as returning float, but it returns a tensor.
            loss: Any = take_step(
                model,
                loss_fn,
                batches[0],
                optimizer,
                None,
                output_args,
                gradient_clip,
                device,
            )[0]
            train_loss = float(loss.detach().cpu())
            optimizer_steps += 1
            model.eval()
            valid_loss_raw, aux = evaluate(
                model, loss_fn, validation_loader, output_args, device
            )
            valid_loss = float(valid_loss_raw)
            force_mae = float(aux["mae_f"])
            if not all(math.isfinite(value) for value in (train_loss, valid_loss, force_mae)):
                raise ValueError("non-finite training or validation metric")
            scheduler.step(force_mae)
            if force_mae < best_force_mae:
                best_force_mae = force_mae
                patience_count = 0
            else:
                patience_count += 1
            completed_epochs = epoch + 1
            records.append(
                {
                    "epoch": epoch,
                    "optimizer_step": optimizer_steps,
                    "training_loss": train_loss,
                    "validation_loss": valid_loss,
                    MONITOR: force_mae,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            state = {
                "schema_version": "1.0.0",
                "mace_torch_version": SUPPORTED_MACE_VERSION,
                "resume_contract_sha256": resume_contract_sha256,
                "next_epoch": completed_epochs,
                "optimizer_steps": optimizer_steps,
                "best_validation_force_mae_ev_per_a": best_force_mae,
                "patience_count": patience_count,
                "monitor": MONITOR,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "loader_generator_state": generator.get_state(),
                "metric_records": records,
            }
            _atomic_torch_save(torch, state, checkpoint_path)
            if patience_count >= patience:
                stopped_early = True
                break
            if time.monotonic() - started > max_wall_seconds:
                raise TimeoutError("controlled MACE training exceeded max_wall_seconds")

        if completed_epochs == start_epoch:
            raise ValueError("fine-tune request performed no optimizer step")
        _atomic_torch_save(torch, model.to("cpu"), model_path)
        changed, maximum = _parameter_change_stats(base_parameters, model)
    return ControlledTrainingResult(
        completed_epochs=completed_epochs,
        optimizer_steps=optimizer_steps,
        best_validation_force_mae_ev_per_a=best_force_mae,
        patience_count=patience_count,
        stopped_early=stopped_early,
        records=records,
        changed_parameter_tensors=changed,
        max_abs_parameter_change=maximum,
    )
