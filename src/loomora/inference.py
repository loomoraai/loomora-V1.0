"""Minimal tensor inference and evaluation for binary Loomora weights."""

from __future__ import annotations

from pathlib import Path

import torch
from sklearn.metrics import log_loss, roc_auc_score

from .model import LoomoraConfig, LoomoraModel, build_model
from .model.factory import validate_checkpoint
from .inference_constants import MemorySavingMode
from .inference_loading import load_checkpoint_cached


def load_stage_model(path: str | Path, device: str | torch.device = "cpu", *, model_config: LoomoraConfig | None = None) -> LoomoraModel:
    """Load V1.0 weights and return an evaluation model."""
    resolved = Path(path).expanduser().resolve()
    identity = resolved.stat()
    payload = load_checkpoint_cached(str(resolved), (identity.st_mtime_ns, identity.st_size))
    validate_checkpoint(payload, model_config)
    model = build_model(payload.get("config", model_config))
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


@torch.inference_mode()
def predict_proba(
    model: LoomoraModel,
    context_features: torch.Tensor,
    context_targets: torch.Tensor,
    query_features: torch.Tensor,
    *,
    n_estimators: int = 8,
    random_state: int = 42,
    memory_saving_mode: MemorySavingMode = "auto",
    fit_mode: str = "fit_with_cache",
) -> torch.Tensor:
    """Tensor convenience API; defaults to the same ensemble as the classifier."""
    if context_features.ndim != 2 or query_features.ndim != 2:
        raise ValueError("Feature tensors must be two-dimensional")
    if context_features.shape[1] != query_features.shape[1]:
        raise ValueError("Context and query feature counts must match")
    if context_targets.ndim != 1 or len(context_targets) != len(context_features):
        raise ValueError("Context targets must have one value per context row")
    device = next(model.parameters()).device
    from .classifier import LoomoraClassifier

    classifier = LoomoraClassifier(model, device=device, n_estimators=n_estimators, random_state=random_state, memory_saving_mode=memory_saving_mode, fit_mode=fit_mode)
    classifier.fit(context_features.detach().cpu().numpy(), context_targets.detach().cpu().numpy())
    output = classifier.predict_proba(query_features.detach().cpu().numpy())
    return torch.from_numpy(output).to(device)


def evaluate_binary(probabilities: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Compute ROC AUC and log loss on held-out binary query labels."""
    y_true = targets.detach().cpu().numpy()
    y_prob = probabilities.detach().cpu().numpy()
    return {"roc_auc": float(roc_auc_score(y_true, y_prob[:, 1])), "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1]))}
