# okc: problem=muse_attention_decode
"""`001-triton-splitkv.py` runs the attention step as three Triton kernels: new KV, split-KV, combine.

Kernel 001 has no parent. It computes, for each request b and block token j:
    q, k = rmsnorm(q) * qk_scale_factor, rmsnorm(k), with RoPE on sliding layers
    cache[cache_loc] = fp8(k), fp8(v)
    out = rmsnorm(softmax(q k^T / 8) v) * sigmoid(gate)

Interface:
    kernel_fn(data): Returns bfloat16 [B, L, Hq * 64] and writes the block's keys and values.

Example:
    $ okc check kernels/001-triton-splitkv.py

Must know:
    Performance:
        - `_attend_split` gives each program two KV heads (128 contiguous cache bytes per token).
        - `_attend_split` packs both heads' queries block-diagonally, so one `tl.dot` serves both.
        - Q K^T runs in FP8 (queries rounded to E4M3, as FA3 does); P V runs in bf16.
"""

import torch
import triton
import triton.language as tl

HEADS_PER_PROGRAM: int = 2
BLOCK_N: int = 64
PROGRAMS_PER_SM: int = 4
RMS_EPS: float = 1e-5
LOG2_E: float = 1.4426950408889634

_inv_freq_per_theta: dict = {}


@triton.jit
def _write_new_kv(
    qkv_ptr, positions_ptr, cache_loc_ptr, inv_freq_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr,
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
def _attend_split(
    qkv_ptr, k_cache_ptr, v_cache_ptr, page_table_ptr, seq_lens_ptr, inv_freq_ptr,
    k_scale_ptr, part_acc_ptr, part_ml_ptr, page_table_stride, chunk,
    L: tl.constexpr, G: tl.constexpr, HB: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
    PAGE: tl.constexpr, WINDOW: tl.constexpr, HAS_ROPE: tl.constexpr, QKV_STRIDE: tl.constexpr,
    QK_LOG2_SCALE: tl.constexpr, EPS: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """`_attend_split` attends one request's head group over one chunk of its keys.

    It writes the chunk's running max and sum (log2 domain) and unnormalized output per query row.
    """
    split, b, hg = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    ROWS: tl.constexpr = L * HB * G
    COLS: tl.constexpr = HB * D
    seq = tl.load(seq_lens_ptr + b)
    begin = tl.maximum(seq - WINDOW + 1, 0) if WINDOW > 0 else seq * 0
    start = begin + split * chunk
    stop_excl = tl.minimum(start + chunk, seq + L)

    r = tl.arange(0, ROWS)
    j = r // (HB * G)
    local_head = (r // G) % HB
    q_head = (hg * HB + local_head) * G + r % G
    c = tl.arange(0, COLS)
    col_dim = c % D
    is_own_head = (c // D)[None, :] == local_head[:, None]
    q_ptrs = qkv_ptr + (b * L + j)[:, None] * QKV_STRIDE + q_head[:, None] * D
    q = tl.load(q_ptrs + col_dim[None, :], mask=is_own_head, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(q * q, 1) / D + EPS)[:, None]
    q = q * inv_rms
    query_pos = seq + j
    if HAS_ROPE:
        q_partner = tl.load(q_ptrs + (col_dim ^ 1)[None, :], mask=is_own_head, other=0.0)
        angle = query_pos[:, None].to(tl.float32) * tl.load(inv_freq_ptr + col_dim // 2)[None, :]
        sign = tl.where(col_dim % 2 == 0, -1.0, 1.0)[None, :]
        q = q * tl.cos(angle) + sign * q_partner.to(tl.float32) * inv_rms * tl.sin(angle)
    q8 = q.to(tl.float8e4nv)
    qk_scale = QK_LOG2_SCALE * tl.load(k_scale_ptr)

    m, l_sum = tl.full([ROWS], float("-inf"), tl.float32), tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, COLS], tl.float32)
    for tile_start in range(start, stop_excl, BLOCK_N):
        t = tile_start + tl.arange(0, BLOCK_N)
        is_valid = t < stop_excl
        page = tl.load(page_table_ptr + b * page_table_stride + t // PAGE, mask=is_valid, other=0)
        kv_row = page.to(tl.int64) * PAGE + t % PAGE
        kv_ptrs = kv_row[:, None] * (HKV * D) + hg * COLS + c[None, :]
        k = tl.load(k_cache_ptr + kv_ptrs, mask=is_valid[:, None], other=0.0)
        s = tl.dot(q8, tl.trans(k)) * qk_scale
        is_allowed = is_valid[None, :] & (t[None, :] <= query_pos[:, None])
        if WINDOW > 0:
            is_allowed = is_allowed & (t[None, :] > query_pos[:, None] - WINDOW)
        s = tl.where(is_allowed, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha, p = tl.exp2(m - m_safe), tl.exp2(s - m_safe[:, None])
        l_sum = l_sum * alpha + tl.sum(p, 1)
        v = tl.load(v_cache_ptr + kv_ptrs, mask=is_valid[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        m = m_new

    own = tl.sum(tl.where(is_own_head, acc, 0.0).reshape(ROWS, HB, D), 1)
    part = ((split * tl.num_programs(1) + b) * tl.num_programs(2) + hg) * ROWS + r
    tl.store(part_acc_ptr + part[:, None] * D + tl.arange(0, D)[None, :], own)
    tl.store(part_ml_ptr + part * 2, m)
    tl.store(part_ml_ptr + part * 2 + 1, l_sum)


@triton.jit
def _combine_splits(
    qkv_ptr, part_acc_ptr, part_ml_ptr, out_ptr, SPLITS,
    L: tl.constexpr, G: tl.constexpr, HB: tl.constexpr, D: tl.constexpr, HQ: tl.constexpr,
    QKV_STRIDE: tl.constexpr, GATE_OFFSET: tl.constexpr, EPS: tl.constexpr,
):
    """`_combine_splits` merges the splits of one request's head group, then norms and gates."""
    b, hg = tl.program_id(0), tl.program_id(1)
    ROWS: tl.constexpr = L * HB * G
    r, d = tl.arange(0, ROWS), tl.arange(0, D)
    m_max = tl.full([ROWS], float("-inf"), tl.float32)
    for split in range(SPLITS):
        part = ((split * tl.num_programs(0) + b) * tl.num_programs(1) + hg) * ROWS + r
        m_max = tl.maximum(m_max, tl.load(part_ml_ptr + part * 2))
    l_total = tl.zeros([ROWS], tl.float32)
    acc = tl.zeros([ROWS, D], tl.float32)
    for split in range(SPLITS):
        part = ((split * tl.num_programs(0) + b) * tl.num_programs(1) + hg) * ROWS + r
        weight = tl.exp2(tl.load(part_ml_ptr + part * 2) - m_max)
        l_total += weight * tl.load(part_ml_ptr + part * 2 + 1)
        acc += weight[:, None] * tl.load(part_acc_ptr + part[:, None] * D + d[None, :])
    o = acc / l_total[:, None]
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + EPS)[:, None]
    j = r // (HB * G)
    q_head = (hg * HB + (r // G) % HB) * G + r % G
    gate = tl.load(qkv_ptr + (b * L + j)[:, None] * QKV_STRIDE + GATE_OFFSET + q_head[:, None] * D + d[None, :])
    o = o * tl.sigmoid(gate.to(tl.float32))
    tl.store(out_ptr + (b * L + j)[:, None] * (HQ * D) + q_head[:, None] * D + d[None, :], o.to(tl.bfloat16))


def _get_inv_freq(theta: float, head_dim: int, device: torch.device) -> torch.Tensor:
    """`_get_inv_freq` returns RoPE's inverse frequencies [head_dim / 2], as the reference has them.

    Args:
        theta: RoPE base; 0 means no RoPE, and a dummy tensor comes back.
        head_dim: Head size.
        device: Where the tensor lives.

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
        data: The judge's inputs (see TASK.md); the output is bfloat16 [B, L, Hq * 64].

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
    rows: int = L * HEADS_PER_PROGRAM * G
    page_table: torch.Tensor = data["page_table"]
    span_tokens: int = window + L - 1 if window > 0 else page_table.shape[1] * page
    sm_count: int = torch.cuda.get_device_properties(qkv.device).multi_processor_count
    split_count: int = max(1, min(triton.cdiv(sm_count * PROGRAMS_PER_SM, B * group_count),
                                  triton.cdiv(span_tokens, BLOCK_N)))
    chunk: int = triton.cdiv(triton.cdiv(span_tokens, split_count), BLOCK_N) * BLOCK_N
    part_acc: torch.Tensor = torch.empty((split_count, B, group_count, rows, D), device=qkv.device)
    part_ml: torch.Tensor = torch.empty((split_count, B, group_count, rows, 2), device=qkv.device)
    _attend_split[(split_count, B, group_count)](
        qkv, data["k_cache"], data["v_cache"], page_table, data["seq_lens"], inv_freq,
        data["k_scale"], part_acc, part_ml, page_table.stride(0), chunk,
        L=L, G=G, HB=HEADS_PER_PROGRAM, HKV=hkv, D=D, PAGE=page, WINDOW=window,
        HAS_ROPE=theta > 0, QKV_STRIDE=qkv_stride,
        QK_LOG2_SCALE=cfg["qk_scale_factor"] * cfg["sm_scale"] * LOG2_E, EPS=RMS_EPS, BLOCK_N=BLOCK_N,
    )
    out: torch.Tensor = torch.empty((B, L, hq * D), dtype=torch.bfloat16, device=qkv.device)
    _combine_splits[(B, group_count)](
        qkv, part_acc, part_ml, out, split_count,
        L=L, G=G, HB=HEADS_PER_PROGRAM, D=D, HQ=hq, QKV_STRIDE=qkv_stride,
        GATE_OFFSET=(hq + 2 * hkv) * D, EPS=RMS_EPS,
    )
    return out
