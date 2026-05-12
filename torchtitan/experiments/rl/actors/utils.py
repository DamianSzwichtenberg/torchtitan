# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchtitan.observability import structured_logger as sl


@sl.log_trace_span("compute_logprobs")
def compute_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Compute per-token logprobs from logits.

    Returns logprobs for positions 1..N (the predicted tokens).
    Output shape is ``[batch, seq_len - 1]``.

    Args:
        logits: Model output logits, shape [batch, seq_len, vocab_size].
        token_ids: Input token IDs, shape [batch, seq_len].
        chunk_size: If set, process log_softmax in chunks of this many tokens
            along the sequence dimension to reduce peak memory. When None
            (default), the full sequence is computed at once.
    """
    from torch.distributed.tensor import DTensor

    # Config-based TP returns logits as a Replicate DTensor. Downstream RL
    # code (gather with plain-tensor indices, slicing per-sample) expects a
    # plain tensor - materialize once here.
    dtensor_placements = None
    if isinstance(logits, DTensor):
        # TODO: pass `grad_placements=[Replicate(), ...]` to make the autograd
        # contract explicit (see .claude/rules/distributed.md).
        dtensor_placements = logits.placements  # Save for chunking guard
        logits = logits.to_local()

    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    # Shift: logits[:-1] predict token_ids[1:]
    shift_logits = logits[:, :-1, :]
    shift_targets = token_ids[:, 1:]

    if chunk_size is None or shift_logits.shape[1] <= chunk_size:
        # Full computation — simple and fast, but uses more memory.
        # Also covers shift_logits.shape[1] == 0 (single-token input),
        # keeping grad_fn intact.
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        return log_probs.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)

    # Chunked log_softmax + gather to avoid materializing full [seq, vocab] fp32
    # Guard: chunking with vocab-sharded logits would issue one collective per chunk
    if dtensor_placements is not None:
        if not all(p.is_replicate() for p in dtensor_placements):
            raise ValueError(
                f"Chunked logprobs incompatible with sharded DTensor: {dtensor_placements}. "
                "Would issue one collective per chunk. Use chunk_size=None with loss_parallel."
            )
    seq_len = shift_logits.shape[1]
    chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_logits = shift_logits[:, start:end, :].float()
        chunk_lp = F.log_softmax(chunk_logits, dim=-1)
        chunk_targets = shift_targets[:, start:end].unsqueeze(-1)
        chunks.append(chunk_lp.gather(2, chunk_targets).squeeze(-1))

    return torch.cat(chunks, dim=1)


@sl.log_trace_span("extract_response_logprobs")
def extract_response_logprobs(
    packed_logprobs: torch.Tensor,
    seq_lens: list[int],
    prompt_lens: list[int],
    response_lens: list[int],
) -> list[torch.Tensor]:
    """Extract per-sample response logprobs from packed logprobs."""
    seq_start = 0
    result = []
    for i in range(len(seq_lens)):
        # Logprobs are shifted: position j holds logprob of token j+1,
        # so response start (seq_start + prompt_len) maps to index
        # (seq_start + prompt_len - 1) in the logprobs tensor.
        s = seq_start + prompt_lens[i] - 1
        e = s + response_lens[i]
        result.append(packed_logprobs[0, s:e])
        seq_start += seq_lens[i]
    return result


@dataclass(frozen=True, slots=True)
class PartialLogprobDrift:
    """Per-rank generator-vs-trainer logprob drift awaiting reduction across the loss-mesh.

    Args:
        logprob_diff_mean: Scalar tensor; To be sum-reduced.
        logprob_diff_max: Scalar tensor; To be max-reduced.
        ratio_tokens_different: Scalar tensor; To be sum-reduced.
    """

    logprob_diff_mean: torch.Tensor
    logprob_diff_max: torch.Tensor
    ratio_tokens_different: torch.Tensor


@torch.no_grad()
@sl.log_trace_span("verify_logprob_identity")
def verify_logprob_identity(
    generator_token_logprobs: list[list[float]],
    trainer_token_logprobs: list[torch.Tensor],
    *,
    num_global_valid_tokens: torch.Tensor,
    device: torch.device,
) -> PartialLogprobDrift:
    """Compute per-rank drift between generator and trainer logprobs.

    Args:
        generator_token_logprobs (list[list[float]]): generator-side per-token logprobs, shaped
            `[num_episodes_local][response_len_i]`.
        trainer_token_logprobs (list[torch.Tensor]): Trainer-side per-token logprobs, one
            GPU tensor per episode, each of shape `[response_len_i]`.
        num_global_valid_tokens (torch.Tensor): Scalar tensor holding global token count
             across DP ranks. Used to normalize the output metrics.
        device: Device to use for tensor allocation, so metrics are ready for
            reduction across loss_mesh.

    Returns:
        PartialLogprobDrift.
    """
    # Each tensor has a different number of tokens, so we flatten them.
    generator_flat = torch.as_tensor(
        [v for sample in generator_token_logprobs for v in sample],
        dtype=torch.float32,
        device=device,
    )
    trainer_flat = torch.cat(trainer_token_logprobs).to(
        device=device, dtype=torch.float32
    )

    if generator_flat.numel() == 0:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return PartialLogprobDrift(zero, zero, zero)

    # 1e-6 threshold ignores bf16-quantization-level diffs
    diff = trainer_flat - generator_flat
    return PartialLogprobDrift(
        logprob_diff_mean=diff.sum() / num_global_valid_tokens,
        logprob_diff_max=diff.abs().max(),
        ratio_tokens_different=(diff.abs() > 1e-6).sum() / num_global_valid_tokens,
    )
