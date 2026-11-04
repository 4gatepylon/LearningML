# okc: problem=muse_attention_decode
"""`fused_attention.py` computes muse_attention_decode, moving its steps into CUDA one at a time.

The step runs in two stages. The prologue RMS-normalizes the queries and the block's new keys,
scales the queries by `qk_scale_factor`, applies RoPE on sliding layers, rounds the queries to bf16
as `reference.py` does, and writes the new keys and values to the FP8 cache. The attention stage
then attends over each request's cache and applies the per-head RMSNorm and the sigmoid gate. Both
stages are still PyTorch; later versions move them into one CUDA kernel.

Interface:
    kernel_fn(data): `TASK.md`'s timed call; returns bfloat16 [B, L, Hq * 64].

Example:
    >>> out = kernel_fn(reference.generate_input(1, 2048, "tp8", "sliding", seed=1))

Must know:
    Performance:
        - The PyTorch attention gathers every position a request's page table can hold.
"""

import torch

HEAD_DIM: int = 64
ROPE_PAIR_STRIDE: int = 2
RMS_EPS: float = 1e-5
_inv_freq_cache: dict[tuple[float, torch.device], torch.Tensor] = {}


def _get_inv_freq(theta: float, device: torch.device) -> torch.Tensor:
    """`_get_inv_freq` returns RoPE's inverse frequencies the way `reference._rope` computes them.

    Args:
        theta: RoPE base; 0 means no RoPE.
        device: Device for the tensor.

    Returns:
        float32 [32] inverse frequencies, or an empty tensor when `theta` is 0.

    Example:
        >>> inv_freq = _get_inv_freq(500_000.0, torch.device("cuda"))
    """
    key: tuple[float, torch.device] = (theta, device)
    if key not in _inv_freq_cache:
        exponents: torch.Tensor = torch.arange(0, HEAD_DIM, ROPE_PAIR_STRIDE, device=device).float()
        _inv_freq_cache[key] = (
            1.0 / (theta ** (exponents / HEAD_DIM)) if theta else torch.empty(0, device=device)
        )
    return _inv_freq_cache[key]


def _prologue(data: dict, inv_freq: torch.Tensor, q: torch.Tensor) -> None:
    """`_prologue` prepares the queries and writes the block's new keys and values to the cache.

    Args:
        data: The inputs; `k_cache` and `v_cache` receive the new FP8 rows.
        inv_freq: float32 [32] RoPE inverse frequencies, or empty for no RoPE.
        q: bf16 [B, L, Hq, 64] output: normalized, scaled, rotated queries.

    Example:
        >>> _prologue(data, _get_inv_freq(500_000.0, device), q)
    """
    config: dict = data["config"]
    Hq, Hkv = config["num_q_heads"], config["num_kv_heads"]
    q_in, k_in, v_in, _ = data["qkv"].split(
        [Hq * HEAD_DIM, Hkv * HEAD_DIM, Hkv * HEAD_DIM, Hq * HEAD_DIM], -1
    )
    shape: tuple[int, int, int] = (config["batch"], config["q_len"], -1)
    x: torch.Tensor = torch.cat(
        [q_in.view(*shape, HEAD_DIM), k_in.view(*shape, HEAD_DIM)], 2
    ).float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    x[:, :, :Hq] *= config["qk_scale_factor"]
    if inv_freq.numel():
        angle: torch.Tensor = data["positions"].view(*shape)[..., None].float() * inv_freq
        cos, sin = angle.cos(), angle.sin()
        even, odd = x[..., 0::ROPE_PAIR_STRIDE], x[..., 1::ROPE_PAIR_STRIDE]
        x = torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)
    q.copy_(x[:, :, :Hq])
    rows: torch.Tensor = data["cache_loc"]
    k_new: torch.Tensor = (x[:, :, Hq:] / data["k_scale"]).flatten(0, 1)
    data["k_cache"].view(-1, Hkv, HEAD_DIM)[rows] = k_new.to(torch.float8_e4m3fn)
    v_new: torch.Tensor = v_in.float().view(-1, Hkv, HEAD_DIM) / data["v_scale"]
    data["v_cache"].view(-1, Hkv, HEAD_DIM)[rows] = v_new.to(torch.float8_e4m3fn)


def _attend_and_gate(data: dict, q: torch.Tensor) -> torch.Tensor:
    """`_attend_and_gate` attends the prepared queries over the cache and gates the result.

    Args:
        data: The inputs; the caches already hold the block's new keys and values.
        q: bf16 [B, L, Hq, 64] prepared queries.

    Returns:
        bfloat16 [B, L, Hq * 64]: rmsnorm(attention) * sigmoid(gate).

    Example:
        >>> out = _attend_and_gate(data, q)
    """
    config: dict = data["config"]
    B, L, Hq = config["batch"], config["q_len"], config["num_q_heads"]
    Hkv, page_size, window = config["num_kv_heads"], config["page_size"], config["window"]
    k_cache: torch.Tensor = data["k_cache"].view(-1, Hkv, HEAD_DIM)
    v_cache: torch.Tensor = data["v_cache"].view(-1, Hkv, HEAD_DIM)
    positions: torch.Tensor = data["positions"].view(B, L)
    t: torch.Tensor = torch.arange(data["page_table"].shape[1] * page_size, device=q.device)
    out: torch.Tensor = torch.empty((B, L, Hq, HEAD_DIM), device=q.device)
    for b in range(B):
        rows: torch.Tensor = data["page_table"][b].long().clamp(min=0)[t // page_size] * page_size
        rows = rows + t % page_size
        K: torch.Tensor = k_cache[rows].float() * data["k_scale"]
        V: torch.Tensor = v_cache[rows].float() * data["v_scale"]
        is_visible: torch.Tensor = t[None, :] <= positions[b][:, None]
        if window:
            is_visible &= t[None, :] > positions[b][:, None] - window
        S: torch.Tensor = torch.einsum("lkgd,skd->lkgs", q[b].view(L, Hkv, -1, HEAD_DIM).float(), K)
        S = (S * config["sm_scale"]).masked_fill(~is_visible[:, None, None, :], float("-inf"))
        out[b] = torch.einsum("lkgs,skd->lkgd", S.softmax(-1), V).reshape(L, Hq, HEAD_DIM)
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    gate: torch.Tensor = data["qkv"][..., (Hq + 2 * Hkv) * HEAD_DIM :].float().view(B, L, Hq, -1)
    return (out * torch.sigmoid(gate)).reshape(B, L, Hq * HEAD_DIM).to(torch.bfloat16)


def kernel_fn(data: dict) -> torch.Tensor:
    """`kernel_fn` runs one decode or verify step of gated attention (see `TASK.md`).

    Args:
        data: The inputs `reference.generate_input` builds.

    Returns:
        bfloat16 [B, L, Hq * 64]; the block's new keys and values are written to the caches.

    Example:
        >>> out = kernel_fn(data)
    """
    config: dict = data["config"]
    q: torch.Tensor = torch.empty(
        (config["batch"], config["q_len"], config["num_q_heads"], HEAD_DIM),
        dtype=torch.bfloat16,
        device=data["qkv"].device,
    )
    _prologue(data, _get_inv_freq(config["rope_theta"], data["qkv"].device), q)
    return _attend_and_gate(data, q)
