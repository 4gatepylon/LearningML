# okc: problem=muse_attention_decode
"""`003-packed-decode.py` attends decode tokens with one block-diagonal dot per 128-byte key tile.

Kernel 003's parent is 002. It computes, for each request b and block token j:
    q, k = rmsnorm(q) * qk_scale_factor, rmsnorm(k), with RoPE on sliding layers
    cache[cache_loc] = fp8(k), fp8(v)
    out = rmsnorm(softmax(q k^T / 8) v) * sigmoid(gate)

Interface:
    kernel_fn(data): Returns bfloat16 [B, L, Hq * 64] and writes the block's keys and values.

Example:
    $ okc check kernels/003-packed-decode.py

Must know:
    Performance:
        - `_attend_split` gives each program two KV heads (128 contiguous cache bytes per token).
        - `_attend_split` (verify) runs separate Q K^T and P V dots per head, [64, 64] each.
        - `_attend_split_packed` (decode) packs both heads' 8 rows into one [16, 128] query.
        - Q K^T runs in FP8 (queries rounded to E4M3, as FA3 does); P V runs in bf16.
        - `_combine_splits` gives each query row its own program and reads splits in blocks.
"""

import torch
import triton
import triton.language as tl

HEADS_PER_PROGRAM: int = 2
MIN_DOT_ROWS: int = 16
BLOCK_N: int = 64
SPLIT_BLOCK: int = 64
PROGRAMS_PER_SM: int = 4
PACKED_BLOCK_N_PER_LAYER: dict[str, int] = {"sliding": 64, "full": 128}
PACKED_PROGRAMS_PER_SM_PER_LAYER: dict[str, int] = {"sliding": 4, "full": 8}
ATTEND_WARPS: int = 4
ATTEND_STAGES: int = 3
RMS_EPS: float = 1e-5
LOG2_E: float = 1.4426950408889634

_inv_freq_per_theta: dict = {}


@triton.jit
def _write_new_kv(
    qkv_ptr, positions_ptr, cache_loc_ptr, inv_freq_ptr, k_cache_ptr, v_cache_ptr,
    k_scale_ptr, v_scale_ptr,
    QKV_STRIDE: tl.constexpr, Q_WIDTH: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
    HAS_ROPE: tl.constexpr, EPS: tl.constexpr,
):
    """`_write_new_kv` normalizes, rotates and stores one block token's keys and values as FP8."""
    token = tl.program_id(0)
    h = tl.arange(0, HKV)[:, None]
    c = tl.arange(0, D)[None, :]
    k_ptrs = qkv_ptr + token * QKV_STRIDE + Q_WIDTH + h * D
    k = tl.load(k_ptrs + c).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(k * k, 1) / D + EPS)[:, None]
    k = k * inv_rms
    if HAS_ROPE:
        k_partner = tl.load(k_ptrs + (c ^ 1)).to(tl.float32) * inv_rms
        angle = tl.load(positions_ptr + token).to(tl.float32) * tl.load(inv_freq_ptr + c // 2)
        sign = tl.where(c % 2 == 0, -1.0, 1.0)
        k = k * tl.cos(angle) + sign * k_partner * tl.sin(angle)
    v = tl.load(k_ptrs + HKV * D + c).to(tl.float32)
    row = tl.load(cache_loc_ptr + token)
    tl.store(k_cache_ptr + row * HKV * D + h * D + c, (k / tl.load(k_scale_ptr)).to(tl.float8e4nv))
    tl.store(v_cache_ptr + row * HKV * D + h * D + c, (v / tl.load(v_scale_ptr)).to(tl.float8e4nv))


@triton.jit
def _load_head_query(
    qkv_ptr, inv_freq_ptr, b, kv_head, query_pos, rh, d,
    L: tl.constexpr, G: tl.constexpr, D: tl.constexpr, HAS_ROPE: tl.constexpr,
    QKV_STRIDE: tl.constexpr, EPS: tl.constexpr,
):
    """`_load_head_query` returns one KV head's normalized, rotated queries as FP8 [RH, D]."""
    is_real_row = rh < L * G
    q_ptrs = qkv_ptr + (b * L + rh // G) * QKV_STRIDE + (kv_head * G + rh % G) * D
    q = tl.load(q_ptrs + d, mask=is_real_row, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(q * q, 1, keep_dims=True) / D + EPS)
    q = q * inv_rms
    if HAS_ROPE:
        q_partner = tl.load(q_ptrs + (d ^ 1), mask=is_real_row, other=0.0).to(tl.float32)
        angle = query_pos.to(tl.float32) * tl.load(inv_freq_ptr + d // 2)
        sign = tl.where(d % 2 == 0, -1.0, 1.0)
        q = q * tl.cos(angle) + sign * q_partner * inv_rms * tl.sin(angle)
    return q.to(tl.float8e4nv)


@triton.jit
def _attend_head_tile(q8, k, v, s_scale, is_allowed, m, l_sum, acc):
    """`_attend_head_tile` folds one key tile into one head's running max, sum and output."""
    s = tl.dot(q8, tl.trans(k)) * s_scale
    s = tl.where(is_allowed, s, float("-inf"))
    m_new = tl.maximum(m, tl.max(s, 1))
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.exp2(m - m_safe)
    p = tl.exp2(s - m_safe[:, None])
    l_sum = l_sum * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
    return m_new, l_sum, acc


@triton.jit
def _store_head_part(part_acc_ptr, part_ml_ptr, row0, rh, d, m, l_sum, acc, L: tl.constexpr, G: tl.constexpr, D: tl.constexpr):
    """`_store_head_part` writes one head's real rows' max, sum and output for the combine."""
    is_real_row = rh < L * G
    row = row0 + rh
    tl.store(part_acc_ptr + row * D + d, acc, mask=is_real_row)
    row_1d = tl.reshape(row, [row.shape[0]])
    is_real_1d = tl.reshape(is_real_row, [row.shape[0]])
    tl.store(part_ml_ptr + row_1d * 2, m, mask=is_real_1d)
    tl.store(part_ml_ptr + row_1d * 2 + 1, l_sum, mask=is_real_1d)


@triton.jit
def _attend_split(
    qkv_ptr, k_cache_ptr, v_cache_ptr, page_table_ptr, seq_lens_ptr, inv_freq_ptr,
    k_scale_ptr, part_acc_ptr, part_ml_ptr, page_table_stride, chunk,
    L: tl.constexpr, G: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
    RH: tl.constexpr, PAGE: tl.constexpr, WINDOW: tl.constexpr, HAS_ROPE: tl.constexpr,
    QKV_STRIDE: tl.constexpr, QK_LOG2_SCALE: tl.constexpr, EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """`_attend_split` attends one request's KV-head pair over one chunk of its keys.

    Each head has RH query rows (block token j, query head g), L * G of them real and the rest
    padding up to the dot's minimum. It writes each real row's running max and sum (log2 domain)
    and unnormalized output.
    """
    split = tl.program_id(0)
    b = tl.program_id(1)
    hg = tl.program_id(2)
    seq = tl.load(seq_lens_ptr + b)
    end_excl = seq + L
    if WINDOW > 0:
        begin = tl.maximum(seq - WINDOW + 1, 0)
    else:
        begin = seq * 0
    start = begin + split * chunk
    stop_excl = tl.minimum(start + chunk, end_excl)

    rh = tl.arange(0, RH)[:, None]
    d = tl.arange(0, D)[None, :]
    query_pos = seq + rh // G
    head0 = 2 * hg
    q0 = _load_head_query(qkv_ptr, inv_freq_ptr, b, head0, query_pos, rh, d, L, G, D, HAS_ROPE, QKV_STRIDE, EPS)
    q1 = _load_head_query(qkv_ptr, inv_freq_ptr, b, head0 + 1, query_pos, rh, d, L, G, D, HAS_ROPE, QKV_STRIDE, EPS)
    s_scale = QK_LOG2_SCALE * tl.load(k_scale_ptr)

    m0 = tl.full([RH], float("-inf"), tl.float32)
    m1 = tl.full([RH], float("-inf"), tl.float32)
    l0 = tl.zeros([RH], tl.float32)
    l1 = tl.zeros([RH], tl.float32)
    acc0 = tl.zeros([RH, D], tl.float32)
    acc1 = tl.zeros([RH, D], tl.float32)
    for tile_start in range(start, stop_excl, BLOCK_N):
        t = tile_start + tl.arange(0, BLOCK_N)
        is_valid = t < stop_excl
        page = tl.load(page_table_ptr + b * page_table_stride + t // PAGE, mask=is_valid, other=0)
        kv_ptrs = (page.to(tl.int64) * PAGE + t % PAGE)[:, None] * (HKV * D) + head0 * D + d
        k0 = tl.load(k_cache_ptr + kv_ptrs, mask=is_valid[:, None], other=0.0)
        k1 = tl.load(k_cache_ptr + kv_ptrs + D, mask=is_valid[:, None], other=0.0)
        v0 = tl.load(v_cache_ptr + kv_ptrs, mask=is_valid[:, None], other=0.0)
        v1 = tl.load(v_cache_ptr + kv_ptrs + D, mask=is_valid[:, None], other=0.0)
        is_allowed = is_valid[None, :] & (t[None, :] <= query_pos)
        if WINDOW > 0:
            is_allowed = is_allowed & (t[None, :] > query_pos - WINDOW)
        m0, l0, acc0 = _attend_head_tile(q0, k0, v0, s_scale, is_allowed, m0, l0, acc0)
        m1, l1, acc1 = _attend_head_tile(q1, k1, v1, s_scale, is_allowed, m1, l1, acc1)

    row0 = ((split * tl.num_programs(1) + b) * HKV + head0) * (L * G)
    _store_head_part(part_acc_ptr, part_ml_ptr, row0, rh, d, m0, l0, acc0, L, G, D)
    _store_head_part(part_acc_ptr, part_ml_ptr, row0 + L * G, rh, d, m1, l1, acc1, L, G, D)


@triton.jit
def _attend_split_packed(
    qkv_ptr, k_cache_ptr, v_cache_ptr, page_table_ptr, seq_lens_ptr, inv_freq_ptr,
    k_scale_ptr, part_acc_ptr, part_ml_ptr, page_table_stride, chunk,
    G: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, WINDOW: tl.constexpr,
    HAS_ROPE: tl.constexpr, QKV_STRIDE: tl.constexpr, QK_LOG2_SCALE: tl.constexpr,
    EPS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """`_attend_split_packed` attends one decode token's KV-head pair over one chunk of its keys.

    It packs both heads' G query rows block-diagonally into one [2G, 2D] query, so each key tile is
    one [BLOCK_N, 2D] load of 128-byte cache rows and one dot. It writes the same partials as
    `_attend_split` with L = 1.
    """
    split, b, hg = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq = tl.load(seq_lens_ptr + b)
    begin = tl.maximum(seq - WINDOW + 1, 0) if WINDOW > 0 else seq * 0
    start = begin + split * chunk
    stop_excl = tl.minimum(start + chunk, seq + 1)

    r = tl.arange(0, 2 * G)
    local_head = r // G
    c = tl.arange(0, 2 * D)
    col_dim = c % D
    is_own_head = (c // D)[None, :] == local_head[:, None]
    q_ptrs = qkv_ptr + b * QKV_STRIDE + ((2 * hg + local_head) * G + r % G)[:, None] * D
    q = tl.load(q_ptrs + col_dim[None, :], mask=is_own_head, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(q * q, 1) / D + EPS)[:, None]
    q = q * inv_rms
    if HAS_ROPE:
        q_partner = tl.load(q_ptrs + (col_dim ^ 1)[None, :], mask=is_own_head, other=0.0)
        angle = seq.to(tl.float32) * tl.load(inv_freq_ptr + col_dim // 2)[None, :]
        sign = tl.where(col_dim % 2 == 0, -1.0, 1.0)[None, :]
        q = q * tl.cos(angle) + sign * q_partner.to(tl.float32) * inv_rms * tl.sin(angle)
    q8 = q.to(tl.float8e4nv)
    s_scale = QK_LOG2_SCALE * tl.load(k_scale_ptr)

    m, l_sum = tl.full([2 * G], float("-inf"), tl.float32), tl.zeros([2 * G], tl.float32)
    acc = tl.zeros([2 * G, 2 * D], tl.float32)
    for tile_start in range(start, stop_excl, BLOCK_N):
        t = tile_start + tl.arange(0, BLOCK_N)
        is_valid = t < stop_excl
        page = tl.load(page_table_ptr + b * page_table_stride + t // PAGE, mask=is_valid, other=0)
        kv_ptrs = (page.to(tl.int64) * PAGE + t % PAGE)[:, None] * (HKV * D) + 2 * hg * D + c[None, :]
        k = tl.load(k_cache_ptr + kv_ptrs, mask=is_valid[:, None], other=0.0)
        s = tl.where(is_valid[None, :], tl.dot(q8, tl.trans(k)) * s_scale, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha, p = tl.exp2(m - m_safe), tl.exp2(s - m_safe[:, None])
        l_sum = l_sum * alpha + tl.sum(p, 1)
        v = tl.load(v_cache_ptr + kv_ptrs, mask=is_valid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        m = m_new

    own = tl.sum(tl.where(is_own_head, acc, 0.0).reshape(2 * G, 2, D), 1)
    row = ((split * tl.num_programs(1) + b) * HKV + 2 * hg) * G + r
    tl.store(part_acc_ptr + row[:, None] * D + tl.arange(0, D)[None, :], own)
    tl.store(part_ml_ptr + row * 2, m)
    tl.store(part_ml_ptr + row * 2 + 1, l_sum)


@triton.jit
def _combine_splits(
    qkv_ptr, part_acc_ptr, part_ml_ptr, out_ptr, split_count, rows_per_split,
    L: tl.constexpr, G: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr,
    QKV_STRIDE: tl.constexpr, GATE_OFFSET: tl.constexpr, EPS: tl.constexpr,
    SPLIT_BLOCK: tl.constexpr,
):
    """`_combine_splits` merges one query row's splits, then norms and gates it."""
    row = tl.program_id(0)
    s = tl.arange(0, SPLIT_BLOCK)
    d = tl.arange(0, D)
    m_max = tl.full([SPLIT_BLOCK], float("-inf"), tl.float32)
    for split_start in range(0, split_count, SPLIT_BLOCK):
        is_split = split_start + s < split_count
        part = (split_start + s) * rows_per_split + row
        m_max = tl.maximum(m_max, tl.load(part_ml_ptr + part * 2, mask=is_split, other=float("-inf")))
    m_all = tl.max(m_max, 0)
    l_total = tl.zeros([SPLIT_BLOCK], tl.float32)
    acc = tl.zeros([D], tl.float32)
    for split_start in range(0, split_count, SPLIT_BLOCK):
        is_split = split_start + s < split_count
        part = (split_start + s) * rows_per_split + row
        weight = tl.exp2(tl.load(part_ml_ptr + part * 2, mask=is_split, other=float("-inf")) - m_all)
        l_total += weight * tl.load(part_ml_ptr + part * 2 + 1, mask=is_split, other=0.0)
        part_acc = tl.load(part_acc_ptr + part[:, None] * D + d[None, :], mask=is_split[:, None], other=0.0)
        acc += tl.sum(weight[:, None] * part_acc, 0)
    o = acc / tl.sum(l_total, 0)
    o = o * tl.rsqrt(tl.sum(o * o, 0) / D + EPS)
    b = row // (HKV * L * G)
    head_row = row % (HKV * L * G)
    kv_head = head_row // (L * G)
    j = (head_row // G) % L
    q_head = kv_head * G + head_row % G
    token = b * L + j
    gate = tl.load(qkv_ptr + token * QKV_STRIDE + GATE_OFFSET + q_head * D + d).to(tl.float32)
    tl.store(out_ptr + token * (HQ * D) + q_head * D + d, (o * tl.sigmoid(gate)).to(tl.bfloat16))


def _get_inv_freq(theta: float, head_dim: int, device: torch.device) -> torch.Tensor:
    """`_get_inv_freq` returns RoPE's inverse frequencies, computed as the reference does.

    Args:
        theta: RoPE base; 0 means no RoPE, and a dummy tensor comes back.
        head_dim: Head size.
        device: Where the tensor lives.

    Returns:
        float32 [head_dim / 2], cached per (theta, head_dim, device).

    Example:
        >>> inv_freq = _get_inv_freq(500000.0, 64, torch.device("cuda"))
    """
    key: tuple = (theta, head_dim, device)
    if key not in _inv_freq_per_theta:
        exponents: torch.Tensor = torch.arange(0, head_dim, 2, device=device).float() / head_dim
        _inv_freq_per_theta[key] = 1.0 / (max(theta, 1.0) ** exponents)
    return _inv_freq_per_theta[key]


def kernel_fn(data: dict) -> torch.Tensor:
    """`kernel_fn` writes the block's keys and values to the cache and returns the gated attention.

    Args:
        data: The judge's inputs (see TASK.md).

    Returns:
        bfloat16 [B, L, Hq * 64].

    Example:
        >>> out = kernel_fn(reference.generate_input(1, 80000, "tp8", "full", seed=1))
    """
    cfg: dict = data["config"]
    B, L = cfg["batch"], cfg["q_len"]
    hq, hkv, D, page = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"], cfg["page_size"]
    window, theta = cfg["window"], cfg["rope_theta"]
    qkv: torch.Tensor = data["qkv"]
    qkv_stride: int = qkv.shape[-1]
    inv_freq: torch.Tensor = _get_inv_freq(theta, D, qkv.device)
    _write_new_kv[(B * L,)](
        qkv, data["positions"], data["cache_loc"], inv_freq, data["k_cache"], data["v_cache"],
        data["k_scale"], data["v_scale"],
        QKV_STRIDE=qkv_stride, Q_WIDTH=hq * D, HKV=hkv, D=D, HAS_ROPE=theta > 0, EPS=RMS_EPS,
    )
    group_count: int = hkv // HEADS_PER_PROGRAM
    G: int = hq // hkv
    is_packed: bool = L * G < MIN_DOT_ROWS
    layer: str = "sliding" if window > 0 else "full"
    block_n: int = PACKED_BLOCK_N_PER_LAYER[layer] if is_packed else BLOCK_N
    programs_per_sm: int = PACKED_PROGRAMS_PER_SM_PER_LAYER[layer] if is_packed else PROGRAMS_PER_SM
    page_table: torch.Tensor = data["page_table"]
    span_tokens: int = window + L - 1 if window > 0 else page_table.shape[1] * page
    sm_count: int = torch.cuda.get_device_properties(qkv.device).multi_processor_count
    split_count: int = max(1, min(triton.cdiv(sm_count * programs_per_sm, B * group_count),
                                  triton.cdiv(span_tokens, block_n)))
    chunk: int = triton.cdiv(triton.cdiv(span_tokens, split_count), block_n) * block_n
    rows_per_split: int = B * hkv * L * G
    part_acc: torch.Tensor = torch.empty((split_count, rows_per_split, D), device=qkv.device)
    part_ml: torch.Tensor = torch.empty((split_count, rows_per_split, 2), device=qkv.device)
    common_args: tuple = (qkv, data["k_cache"], data["v_cache"], page_table, data["seq_lens"], inv_freq,
                          data["k_scale"], part_acc, part_ml, page_table.stride(0), chunk)
    qk_log2_scale: float = cfg["qk_scale_factor"] * cfg["sm_scale"] * LOG2_E
    if is_packed:
        _attend_split_packed[(split_count, B, group_count)](
            *common_args, G=G, HKV=hkv, D=D, PAGE=page, WINDOW=window, HAS_ROPE=theta > 0,
            QKV_STRIDE=qkv_stride, QK_LOG2_SCALE=qk_log2_scale, EPS=RMS_EPS, BLOCK_N=block_n,
            num_warps=ATTEND_WARPS, num_stages=ATTEND_STAGES,
        )
    else:
        _attend_split[(split_count, B, group_count)](
            *common_args, L=L, G=G, HKV=hkv, D=D, RH=L * G, PAGE=page, WINDOW=window,
            HAS_ROPE=theta > 0, QKV_STRIDE=qkv_stride, QK_LOG2_SCALE=qk_log2_scale, EPS=RMS_EPS,
            BLOCK_N=block_n, num_warps=ATTEND_WARPS, num_stages=ATTEND_STAGES,
        )
    out: torch.Tensor = torch.empty((B, L, hq * D), dtype=torch.bfloat16, device=qkv.device)
    _combine_splits[(rows_per_split,)](
        qkv, part_acc, part_ml, out, split_count, rows_per_split,
        L=L, G=G, HKV=hkv, D=D, HQ=hq, QKV_STRIDE=qkv_stride,
        GATE_OFFSET=(hq + 2 * hkv) * D, EPS=RMS_EPS, SPLIT_BLOCK=SPLIT_BLOCK,
    )
    return out
