"""WER/CER metrics with Vietnamese normalization."""

from __future__ import annotations

from jiwer import wer, cer

from src.evaluation.normalize_vi import normalize_vietnamese


def compute_wer(
    references: list[str],
    hypotheses: list[str],
    normalize: bool = True,
) -> float:
    """Compute Word Error Rate with optional Vietnamese normalization."""
    if normalize:
        references = [normalize_vietnamese(r) for r in references]
        hypotheses = [normalize_vietnamese(h) for h in hypotheses]

    # Filter empty references
    pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    if not pairs:
        return 0.0

    refs, hyps = zip(*pairs)
    return wer(list(refs), list(hyps))


def compute_cer(
    references: list[str],
    hypotheses: list[str],
    normalize: bool = True,
) -> float:
    """Compute Character Error Rate with optional Vietnamese normalization."""
    if normalize:
        references = [normalize_vietnamese(r) for r in references]
        hypotheses = [normalize_vietnamese(h) for h in hypotheses]

    pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    if not pairs:
        return 0.0

    refs, hyps = zip(*pairs)
    return cer(list(refs), list(hyps))
