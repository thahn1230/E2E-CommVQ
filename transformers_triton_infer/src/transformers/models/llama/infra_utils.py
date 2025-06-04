import torch
import triton
import triton.language as tl
import time

# @torch.compile
def calculate_QC_T2(query_states, centers):
    centers = centers.unsqueeze(0)
    query_states = query_states.squeeze(2)
    query_states_expand = query_states.unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)
    QC_T = torch.matmul(query_states_expand.contiguous(), centers.contiguous())
    QC_T = QC_T.permute(1, 2, 0, 3, 4, 5, 6)
    return QC_T


@triton.autotune(
  configs=[triton.Config(kwargs={'BLOCK_SIZE': bs}) for bs in [4, 8, 16, 32]],
  key=["B"],
)
@triton.jit
def calculate_QC_T_kernel(
    query_states_ptr,
    centers_ptr,
    output_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    DIM: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    hx = tl.program_id(0)
    batch_i = hx // H
    hx = hx % H
    dx = tl.program_id(1)
    bx = tl.program_id(2)
    query_start = (hx * DIM * B + dx * B + batch_i) * 2
    q1 = tl.load(query_states_ptr + query_start)
    q2 = tl.load(query_states_ptr + query_start + 1)
    rc = tl.arange(0, BLOCK_SIZE) + bx * BLOCK_SIZE
    mask = rc < (R * C)
    centers_start = (hx * DIM * R * C + dx * R * C + rc) * 4
    c1 = tl.load(centers_ptr + centers_start, mask=mask)
    c2 = tl.load(centers_ptr + centers_start + 1, mask=mask)
    c3 = tl.load(centers_ptr + centers_start + 2, mask=mask)
    c4 = tl.load(centers_ptr + centers_start + 3, mask=mask)
    output_start = (hx * DIM * B * R * C + dx * B * R * C + batch_i * R * C + rc) * 2
    tl.store(output_ptr + output_start, q1 * c1 + q2 * c3, mask=mask)
    tl.store(output_ptr + output_start + 1, q1 * c2 + q2 * c4, mask=mask)


def calculate_QC_T_triton(query_states, centers):
    query_states = query_states.squeeze(2).permute(1, 2, 0, 3).contiguous()
    # query_states: [32, 64, 80, 2]
    # centers: [32, 64, 16, 64, 2, 2]
    H, DIM, R, C, _, _ = centers.shape
    B = query_states.shape[2]
    output = torch.zeros(H, DIM, B, R, C, 1, 2, device=query_states.device, dtype=query_states.dtype)
    grid = lambda META: (B * H, DIM, triton.cdiv(R * C, META["BLOCK_SIZE"]))
    # BLOCK_SIZE = 16
    # grid = (H * B, DIM, triton.cdiv(R * C, BLOCK_SIZE))
    calculate_QC_T_kernel[grid](
        query_states_ptr=query_states,
        centers_ptr=centers,
        output_ptr=output,
        B=B, H=H, DIM=DIM, R=R, C=C,
        # BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


@torch.compile
def gatherC1C2_and_sum_over_residual(key_code, QC_T):
    key_code = key_code.permute(1, 0, 2)
    C1_x = key_code // 64
    C2_x = key_code % 64
    QC_T = QC_T.permute(0, 2, 1, 3, 4, 5)
    QC_T = QC_T.unsqueeze(0).unsqueeze(3)
    C1_x = C1_x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    C2_x = C2_x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    QC_T_expand = QC_T.expand(-1, -1, -1, C1_x.shape[3], -1, -1, -1, -1)
    C1_x_expand = C1_x.expand(-1, -1, -1, -1, QC_T.shape[4], -1, QC_T.shape[6], QC_T.shape[7])
    C2_x_expand = C2_x.expand(-1, -1, -1, -1, QC_T.shape[4], -1, QC_T.shape[6], QC_T.shape[7])
    C1_x_expand = C1_x_expand.repeat_interleave(4, dim=1)
    C2_x_expand = C2_x_expand.repeat_interleave(4, dim=1)
    X = torch.gather(QC_T_expand, 5, C1_x_expand).squeeze(-2).squeeze(-2)
    Y = torch.gather(QC_T_expand, 5, C2_x_expand).squeeze(-2).squeeze(-2)
    X = X.sum(2)
    Y = Y.sum(2)
    return X, Y






@triton.jit
def gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_kernel_slower(
    key_code_ptr,
    QC_T_ptr,
    sin_ptr,
    cos_ptr,
    prescale_ptr,
    output_ptr,
    N: int,
    H_RATIO: tl.constexpr, KV_H: tl.constexpr, DIM: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    hx = tl.program_id(0)
    bx = tl.program_id(1)
    token_offset = tl.arange(0, BLOCK_SIZE) + bx * BLOCK_SIZE
    mask = token_offset < N
    prescale = tl.load(prescale_ptr + token_offset, mask=mask)
    dx = tl.arange(0, DIM)
    # sin_start: [BLOCK_SIZE, DIM]
    sin_start = token_offset[:, None] * DIM + dx[None, :]
    cos_start = token_offset[:, None] * DIM + dx[None, :]
    sin = tl.load(sin_ptr + sin_start)
    cos = tl.load(cos_ptr + cos_start)
    for h_i in range(0, H_RATIO):
        codebook_start = (hx * DIM + dx) * (H_RATIO * R * C * 2)
        acc_1 = tl.zeros([BLOCK_SIZE, DIM, 2], dtype=tl.bfloat16)
        acc_2 = tl.zeros([BLOCK_SIZE, DIM, 2], dtype=tl.bfloat16)
        # for r_i in range(0, R):  # TODO: change to tl.static_range
        for r_i in tl.static_range(0, R):
            c_start = hx * N * R + token_offset * R + r_i
            c = tl.load(key_code_ptr + c_start)
            c1x = c // 64
            c2x = c % 64
            c1_offset = codebook_start[None, :, None] + h_i * R * C * 2 + r_i * C * 2 + c1x[:, None, None] * 2 + tl.arange(0, 2)[None, None, :]
            c2_offset = codebook_start[None, :, None] + h_i * R * C * 2 + r_i * C * 2 + c2x[:, None, None] * 2 + tl.arange(0, 2)[None, None, :]
            c1 = tl.load(QC_T_ptr + c1_offset)
            c2 = tl.load(QC_T_ptr + c2_offset)
            acc_1 += c1
            acc_2 += c2
        acc_1 = tl.sum(tl.reshape(acc_1 * tl.join( cos, sin), (BLOCK_SIZE, DIM * 2)), axis=-1)
        acc_2 = tl.sum(tl.reshape(acc_2 * tl.join(-sin, cos), (BLOCK_SIZE, DIM * 2)), axis=-1)
        acc = (acc_1 + acc_2) * prescale
        output_offset = (hx * H_RATIO + h_i) * N + token_offset
        tl.store(output_ptr + output_offset, acc, mask=mask)


@triton.jit
def gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_inv_freq_kernel(
    key_code_ptr,
    QC_T_ptr,
    inv_freq_ptr,
    prescale_ptr,
    output_ptr,
    N: int,
    H_RATIO: tl.constexpr, KV_H: tl.constexpr, DIM: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    hx = tl.program_id(0)
    h_i = tl.program_id(1)
    bx = tl.program_id(2)
    token_offset = tl.arange(0, BLOCK_SIZE) + bx * BLOCK_SIZE
    mask = token_offset < N
    dx = tl.arange(0, DIM)
    inv_freq = tl.load(inv_freq_ptr + dx)
    prescale = tl.load(prescale_ptr + token_offset, mask=mask)
    codebook_start = (hx * DIM + dx) * (H_RATIO * R * C * 2)
    acc_1X = tl.zeros([BLOCK_SIZE, DIM], dtype=tl.bfloat16)
    acc_1Y = tl.zeros([BLOCK_SIZE, DIM], dtype=tl.bfloat16)
    acc_2X = tl.zeros([BLOCK_SIZE, DIM], dtype=tl.bfloat16)
    acc_2Y = tl.zeros([BLOCK_SIZE, DIM], dtype=tl.bfloat16)
    # for r_i in range(0, R):  # TODO: change to tl.static_range
    for r_i in tl.static_range(0, R):
        c_start = hx * R * N + r_i * N + token_offset
        c = tl.load(key_code_ptr + c_start)
        c1x = c // 64
        c2x = c % 64
        c1_offset = codebook_start[None, :] + h_i * R * C * 2 + r_i * C * 2 + c1x[:, None] * 2
        c2_offset = codebook_start[None, :] + h_i * R * C * 2 + r_i * C * 2 + c2x[:, None] * 2
        c1x = tl.load(QC_T_ptr + c1_offset)
        c1y = tl.load(QC_T_ptr + c1_offset + 1)
        c2x = tl.load(QC_T_ptr + c2_offset)
        c2y = tl.load(QC_T_ptr + c2_offset + 1)
        acc_1X += c1x
        acc_1Y += c1y
        acc_2X += c2x
        acc_2Y += c2y
    # sin: [BLOCK_SIZE, DIM]
    inv_freq_array = (inv_freq[None, :] * (token_offset[:, None]))
    sin = tl.sin(inv_freq_array).cast(tl.bfloat16)
    cos = tl.cos(inv_freq_array).cast(tl.bfloat16)
    acc = tl.sum(acc_1X * cos + acc_1Y * sin + acc_2Y * cos - acc_2X * sin, axis=1)
    output_offset = (hx * H_RATIO + h_i) * N + token_offset
    acc = acc * prescale
    tl.store(output_ptr + output_offset, acc, mask=mask)


def gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_inv_freq_triton(key_code, QC_T, inv_freq, prescale, factor, H_RATIO=4):
    prescale = prescale.unsqueeze(1).permute(0, 1, 3, 2) / factor
    QC_T = QC_T.unflatten(0, (-1, H_RATIO)).transpose(1, 2).squeeze(3)  # squeeze Batch dim
    key_code = key_code.transpose(0, 1)
    # QC_T: [KV_H, DIM, H_RATIO, R, C, 1, 2]
    # key_code: [KV_H, R, N]
    # inv_freq: [DIM]
    # output X: [KV_H * H_RATIO, N, DIM, 2]
    # output Y: [KV_H * H_RATIO, N, DIM, 2]
    # output: [1, KV_H * H_RATIO, N]
    assert key_code.shape[0] == 8 and key_code.shape[1] == 11
    assert QC_T.shape[0] == 8 and QC_T.shape[1] == 64 and QC_T.shape[2] == H_RATIO and QC_T.shape[3] == 11 and QC_T.shape[4] == 64 and QC_T.shape[5] == 1 and QC_T.shape[6] == 2
    KV_H, _, N = key_code.shape
    _, DIM, _, R, C, _, _ = QC_T.shape
    BLOCK_SIZE = 4
    # assert N % BLOCK_SIZE == 0
    assert H_RATIO == 4
    output = torch.zeros(1, KV_H * H_RATIO, N, device=QC_T.device, dtype=QC_T.dtype)
    grid = (KV_H, H_RATIO, triton.cdiv(N, BLOCK_SIZE))
    gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_inv_freq_kernel[grid](
        key_code_ptr=key_code.contiguous(),
        QC_T_ptr=QC_T.contiguous(),
        inv_freq_ptr=inv_freq.contiguous(),
        prescale_ptr=prescale.contiguous(),
        output_ptr=output,
        N=N, 
        H_RATIO=H_RATIO, KV_H=KV_H, DIM=DIM, R=R, C=C, BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_inv_freq(key_code, QC_T, inv_freq, prescale, factor):
    attn_weights1 = gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_inv_freq_triton(key_code, QC_T, inv_freq, prescale, factor)
    attn_weights1 = attn_weights1.unsqueeze(-2)
    return attn_weights1



@triton.jit
def gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_not_vectorized_kernel(
    key_code_ptr,
    QC_T_ptr,
    sin_ptr,
    cos_ptr,
    prescale_ptr,
    output_ptr,
    N: int,
    H_RATIO: tl.constexpr, KV_H: tl.constexpr, DIM: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    hx = tl.program_id(0)
    bx = tl.program_id(1)
    for token_i in range(0, N, BLOCK_SIZE):
        prescale_start = token_i + bx
        prescale = tl.load(prescale_ptr + prescale_start)
        for h_i in range(0, H_RATIO):
            acc = 0.0
            for dx in range(0, DIM):
                codebook_start = (hx * DIM + dx) * (H_RATIO * R * C * 2)
                acc_1X = 0.0
                acc_1Y = 0.0
                acc_2X = 0.0
                acc_2Y = 0.0
                for r_i in range(0, R):
                    c_start = hx * N * R + (token_i + bx) * R + r_i
                    c = tl.load(key_code_ptr + c_start)
                    c1x = c // 64
                    c2x = c % 64
                    c1_offset = codebook_start + h_i * R * C * 2 + r_i * C * 2 + c1x * 2
                    c2_offset = codebook_start + h_i * R * C * 2 + r_i * C * 2 + c2x * 2
                    c1x = tl.load(QC_T_ptr + c1_offset)
                    c1y = tl.load(QC_T_ptr + c1_offset + 1)
                    c2x = tl.load(QC_T_ptr + c2_offset)
                    c2y = tl.load(QC_T_ptr + c2_offset + 1)
                    acc_1X += c1x
                    acc_1Y += c1y
                    acc_2X += c2x
                    acc_2Y += c2y
                sin_start = (token_i + bx) * DIM + dx
                cos_start = (token_i + bx) * DIM + dx
                sin = tl.load(sin_ptr + sin_start)
                cos = tl.load(cos_ptr + cos_start)
                acc += acc_1X * cos + acc_1Y * sin + acc_2Y * cos - acc_2X * sin
            output_offset = (hx * H_RATIO + h_i) * N + (token_i + bx)
            acc = acc * prescale
            acc_bf16 = acc.cast(tl.bfloat16)
            tl.store(output_ptr + output_offset, acc_bf16)

def gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_not_vectorized_triton(key_code, QC_T, qsin, qcos, prescale, factor, H_RATIO=4):
    prescale = prescale.unsqueeze(1).permute(0, 1, 3, 2) / factor
    QC_T = QC_T.unflatten(0, (-1, H_RATIO)).transpose(1, 2).squeeze(3)  # squeeze Batch dim
    key_code = key_code.permute(1, 2, 0)
    # prescale: [1, 1, 1, N]
    # QC_T: [KV_H, DIM, H_RATIO, R, C, 1, 2]
    # key_code: [KV_H, N, R]
    # qsin: [1, 1, N, DIM, 1]
    # qcos: [1, 1, N, DIM, 1]
    # output X: [KV_H * H_RATIO, N, DIM, 2]
    # output Y: [KV_H * H_RATIO, N, DIM, 2]
    # output: [1, KV_H * H_RATIO, N]
    assert key_code.shape[0] == 8 and key_code.shape[2] == 11
    assert QC_T.shape[0] == 8 and QC_T.shape[1] == 64 and QC_T.shape[2] == H_RATIO and QC_T.shape[3] == 11 and QC_T.shape[4] == 64 and QC_T.shape[5] == 1 and QC_T.shape[6] == 2
    KV_H, N, _ = key_code.shape
    _, DIM, _, R, C, _, _ = QC_T.shape
    BLOCK_SIZE = 128
    assert N % BLOCK_SIZE == 0
    assert H_RATIO == 4
    output = torch.zeros(1, KV_H * H_RATIO, N, device=QC_T.device, dtype=QC_T.dtype)
    grid = (KV_H, BLOCK_SIZE)
    gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_not_vectorized_kernel[grid](
        key_code_ptr=key_code.contiguous(),
        QC_T_ptr=QC_T.contiguous(),
        sin_ptr=qsin.contiguous(),
        cos_ptr=qcos.contiguous(),
        prescale_ptr=prescale.contiguous(),
        output_ptr=output,
        N=N, 
        H_RATIO=H_RATIO, KV_H=KV_H, DIM=DIM, R=R, C=C, BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_not_vectorized(key_code, QC_T, qsin, qcos, prescale, factor):
    attn_weights1 = gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_not_vectorized_triton(key_code, QC_T, qsin, qcos, prescale, factor)
    attn_weights1 = attn_weights1.unsqueeze(-2)
    return attn_weights1



@triton.autotune(
  configs=[triton.Config(kwargs={'BLOCK_SIZE': bs}) for bs in [4, 8, 16, 32]],
  key=['N']
)
@triton.jit
def gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_kernel(
    key_code_ptr,
    QC_T_ptr,
    sin_ptr,
    cos_ptr,
    prescale_ptr,
    output_ptr,
    N: int,
    H_RATIO: tl.constexpr, KV_H: tl.constexpr, DIM: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    hx = tl.program_id(0)
    h_i = tl.program_id(1)
    bx = tl.program_id(2)
    token_offset = tl.arange(0, BLOCK_SIZE) + bx * BLOCK_SIZE
    mask = token_offset < N
    dx = tl.arange(0, DIM)
    # sin_start: [BLOCK_SIZE, DIM]
    sin_start = token_offset[:, None] * DIM + dx[None, :]
    cos_start = token_offset[:, None] * DIM + dx[None, :]
    prescale = tl.load(prescale_ptr + token_offset, mask=mask)
    codebook_start = (hx * DIM + dx) * (H_RATIO * R * C * 2)
    acc_1X = tl.zeros([BLOCK_SIZE, DIM], dtype=tl.bfloat16)
    acc_1Y = tl.zeros([BLOCK_SIZE, DIM], dtype=tl.bfloat16)
    acc_2X = tl.zeros([BLOCK_SIZE, DIM], dtype=tl.bfloat16)
    acc_2Y = tl.zeros([BLOCK_SIZE, DIM], dtype=tl.bfloat16)
    for r_i in tl.static_range(0, R):
        c_start = hx * R * N + r_i * N + token_offset
        c = tl.load(key_code_ptr + c_start)  # FIXME: do I need mask?
        c1x = c // 64
        c2x = c % 64
        c1_offset = codebook_start[None, :] + h_i * R * C * 2 + r_i * C * 2 + c1x[:, None] * 2
        c2_offset = codebook_start[None, :] + h_i * R * C * 2 + r_i * C * 2 + c2x[:, None] * 2
        c1x = tl.load(QC_T_ptr + c1_offset)
        c1y = tl.load(QC_T_ptr + c1_offset + 1)
        c2x = tl.load(QC_T_ptr + c2_offset)
        c2y = tl.load(QC_T_ptr + c2_offset + 1)
        acc_1X += c1x
        acc_1Y += c1y
        acc_2X += c2x
        acc_2Y += c2y
    sin = tl.load(sin_ptr + sin_start)
    cos = tl.load(cos_ptr + cos_start)
    acc = tl.sum(acc_1X * cos + acc_1Y * sin + acc_2Y * cos - acc_2X * sin, axis=1)
    output_offset = (hx * H_RATIO + h_i) * N + token_offset
    acc = acc * prescale
    tl.store(output_ptr + output_offset, acc, mask=mask)

def gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_triton(key_code, QC_T, qsin, qcos, prescale, factor, H_RATIO=4):
    B = prescale.shape[0]
    QC_T = QC_T.unflatten(0, (-1, H_RATIO)).permute(3, 0, 2, 1, 4, 5, 6, 7).flatten(0, 1).contiguous()
    key_code =key_code.flatten(0, 1).contiguous()
    # key_code = key_code.transpose(0, 1)
    # QC_T: [B * KV_H, DIM, H_RATIO, R, C, 1, 2]
    # key_code: [B * KV_H, R, N]
    # qsin: [1, 1, N, DIM, 1]
    # qcos: [1, 1, N, DIM, 1]
    # output X: [KV_H * H_RATIO, N, DIM, 2]
    # output Y: [KV_H * H_RATIO, N, DIM, 2]
    # output: [1, KV_H * H_RATIO, N]
    assert key_code.shape[0] == B * 8 and key_code.shape[1] == 11
    assert QC_T.shape[0] == B * 8 and QC_T.shape[1] == 64 and QC_T.shape[2] == H_RATIO and QC_T.shape[3] == 11 and QC_T.shape[4] == 64 and QC_T.shape[5] == 1 and QC_T.shape[6] == 2
    KV_H, _, N = key_code.shape
    _, DIM, _, R, C, _, _ = QC_T.shape
    # assert N % BLOCK_SIZE == 0
    assert H_RATIO == 4
    output = torch.zeros(KV_H * H_RATIO, N, device=QC_T.device, dtype=QC_T.dtype)
    # BLOCK_SIZE = 32
    # grid = (KV_H, H_RATIO, triton.cdiv(N, BLOCK_SIZE))
    grid = lambda META: (KV_H, H_RATIO, triton.cdiv(N, META["BLOCK_SIZE"]))
    gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_kernel[grid](
        key_code_ptr=key_code,
        QC_T_ptr=QC_T,
        sin_ptr=qsin,
        cos_ptr=qcos,
        prescale_ptr=prescale,
        output_ptr=output,
        N=N, 
        H_RATIO=H_RATIO, KV_H=KV_H, DIM=DIM, R=R, C=C,
        # BLOCK_SIZE=BLOCK_SIZE,
    )
    output = output.unflatten(0, (B, -1))
    return output

def gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum(key_code, QC_T, qsin, qcos, prescale, factor):
    attn_weights1 = gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_triton(key_code, QC_T, qsin, qcos, prescale, factor)
    attn_weights1 = attn_weights1.unsqueeze(-2)
    return attn_weights1


@triton.jit
def gatherC1C2_and_sum_over_residual_kernel(
    key_code_ptr,
    QC_T_ptr,
    output_X_ptr,
    output_Y_ptr,
    N: int,
    H_RATIO: tl.constexpr, KV_H: tl.constexpr, DIM: tl.constexpr, R: tl.constexpr, C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    hx = tl.program_id(0)
    dx = tl.program_id(1)
    bx = tl.program_id(2)
    codebook_start = (hx * DIM + dx) * (H_RATIO * R * C * 2)
    for token_i in range(0, N, BLOCK_SIZE):
        for h_i in range(0, H_RATIO):
            acc1 = tl.zeros([2], dtype=tl.bfloat16)
            acc2 = tl.zeros([2], dtype=tl.bfloat16)
            for r_i in range(0, R):
                c_start = hx * N * R + (token_i + bx) * R + r_i
                c = tl.load(key_code_ptr + c_start)
                c1x = c // 64
                c2x = c % 64
                c1_offset = codebook_start + h_i * R * C * 2 + r_i * C * 2 + c1x * 2 + tl.arange(0, 2)
                c2_offset = codebook_start + h_i * R * C * 2 + r_i * C * 2 + c2x * 2 + tl.arange(0, 2)
                c1 = tl.load(QC_T_ptr + c1_offset)
                c2 = tl.load(QC_T_ptr + c2_offset)
                acc1 += c1
                acc2 += c2
            output_offset = (hx * H_RATIO + h_i) * N * DIM * 2 + (token_i + bx) * DIM * 2 + dx * 2 + tl.arange(0, 2)
            tl.store(output_X_ptr + output_offset, acc1)
            tl.store(output_Y_ptr + output_offset, acc2)


def gatherC1C2_and_sum_over_residual_triton(key_code, QC_T):
    H_RATIO = 4
    QC_T = QC_T.unflatten(0, (-1, H_RATIO)).transpose(1, 2).squeeze(3)  # squeeze Batch dim
    key_code = key_code.permute(1, 2, 0)
    # QC_T: [KV_H, DIM, H_RATIO, R, C, 1, 2]
    # key_code: [KV_H, N, R]
    # output X: [KV_H * H_RATIO, N, DIM, 2]
    # output Y: [KV_H * H_RATIO, N, DIM, 2]
    assert key_code.shape[0] == 8 and key_code.shape[2] == 11
    assert QC_T.shape[0] == 8 and QC_T.shape[1] == 64 and QC_T.shape[2] == H_RATIO and QC_T.shape[3] == 11 and QC_T.shape[4] == 64 and QC_T.shape[5] == 1 and QC_T.shape[6] == 2
    KV_H, N, _ = key_code.shape
    _, DIM, _, R, C, _, _ = QC_T.shape
    BLOCK_SIZE = 128
    assert N % BLOCK_SIZE == 0
    assert H_RATIO == 4
    output_X = torch.zeros(KV_H * H_RATIO, N, DIM, 2, device=QC_T.device, dtype=QC_T.dtype)
    output_Y = torch.zeros(KV_H * H_RATIO, N, DIM, 2, device=QC_T.device, dtype=QC_T.dtype)
    grid = (KV_H, DIM, BLOCK_SIZE)
    gatherC1C2_and_sum_over_residual_kernel[grid](
        key_code_ptr=key_code.contiguous(),
        QC_T_ptr=QC_T.contiguous(),
        output_X_ptr=output_X,
        output_Y_ptr=output_Y,
        N=N, 
        H_RATIO=H_RATIO, KV_H=KV_H, DIM=DIM, R=R, C=C, BLOCK_SIZE=BLOCK_SIZE,
    )
    return output_X, output_Y


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def gatherC1C2_and_sum_over_residual_and_apply_rope_and_sum_not_fused_rope(key_code, QC_T, qsin, qcos, prescale, factor):
    # QC_T: [H, DIM, B, R, C, 1, 2] -> [B, H, R, DIM, C, 1, 2]
    # B = prescale.shape[0]
    # key_code = key_code.long().unflatten(-1, (B, -1)).permute(2, 1, 0, 3)
    # C1_x = key_code // 64
    # C2_x = key_code % 64
    # QC_T = QC_T.permute(2, 0, 3, 1, 4, 5, 6)
    # QC_T = QC_T.unsqueeze(3)
    # C1_x = C1_x.unsqueeze(2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # C2_x = C2_x.unsqueeze(2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # QC_T_expand = QC_T.unflatten(1, (C1_x.shape[1], -1)).expand(-1, -1, -1, -1, C1_x.shape[4], -1, -1, -1, -1)
    # C1_x_expand = C1_x.expand(-1, -1, QC_T_expand.shape[2], -1, -1, QC_T_expand.shape[5], -1, QC_T_expand.shape[7], QC_T_expand.shape[8])
    # C2_x_expand = C2_x.expand(-1, -1, QC_T_expand.shape[2], -1, -1, QC_T_expand.shape[5], -1, QC_T_expand.shape[7], QC_T_expand.shape[8])
    # X = torch.gather(QC_T_expand, 6, C1_x_expand).flatten(1, 2).flatten(-3).sum(2)
    # Y = torch.gather(QC_T_expand, 6, C2_x_expand).flatten(1, 2).flatten(-3).sum(2)
    X, Y = gatherC1C2_and_sum_over_residual_triton(key_code, QC_T)
    qcos = qcos.unsqueeze(1).unsqueeze(-1)
    qsin = qsin.unsqueeze(1).unsqueeze(-1)
    X = X * qcos - rotate_half(X) * qsin
    Y = Y * qcos - rotate_half(Y) * qsin
    attn_weights1 = (X[..., 0] + Y[..., 1]).sum(-1).unsqueeze(-2) * prescale.unsqueeze(1).permute(0, 1, 3, 2) / factor
    return attn_weights1


@torch.compile
def attn_weights_mul_value(attn_weights, value_code, prescale, learnable_scale, eps, codebook):
    attn_weights1 = attn_weights[..., :prescale.shape[1]] * prescale.unsqueeze(1).transpose(-2, -1)
    SA = torch.matmul(attn_weights1, value_code.unsqueeze(1))
    SA = SA * (learnable_scale + eps)
    attn_output1 = torch.matmul(SA, codebook.unsqueeze(0))
    return attn_output1



@triton.jit
def attn_weights_mul_value_kernel_v1(
    attn_weights_ptr,
    value_code_ptr,
    SA_ptr,
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    N: int,
):
    bx = tl.program_id(0)
    hx = tl.program_id(1)
    dx = tl.program_id(2)
    acc = 0.0
    for i in range(0, N):
        code_start = bx * N * D + i * D + dx
        code = tl.load(value_code_ptr + code_start).to(tl.int32)
        attn_weight_start = bx * H * N * 8 + hx * N * 8 + i * 8
        offset = tl.arange(0, 8)
        attn_weight = tl.load(attn_weights_ptr + attn_weight_start + offset)
        bit = ((code & (1 << offset)) != 0)
        acc += tl.sum(attn_weight * bit)
    output_start = bx * H * D + hx * D + dx
    tl.store(SA_ptr + output_start, acc)


def attn_weights_mul_value_triton_v1(attn_weights, value_code, prescale, learnable_scale, eps, codebook):
    attn_weights = (attn_weights[..., :prescale.shape[1]] * prescale.unsqueeze(1).transpose(-2, -1)).contiguous()
    value_code = value_code.unsqueeze(1).contiguous()
    # triton code to impl the following operation
    # SA = torch.matmul(attn_weights, value_code)
    B, H, N, D = attn_weights.shape[0], attn_weights.shape[1], value_code.shape[2], value_code.shape[3]
    grid = (B, H, D)
    SA = torch.zeros(B, H, D, device=attn_weights.device, dtype=attn_weights.dtype)
    attn_weights_mul_value_kernel_v1[grid](
        attn_weights_ptr=attn_weights,
        value_code_ptr=value_code,
        SA_ptr=SA,
        B=B, H=H, D=D, N=N,
    )
    SA = SA.unsqueeze(2)
    SA = SA * (learnable_scale + eps)
    attn_output1 = torch.matmul(SA, codebook.unsqueeze(0))
    return attn_output1

configs = []
for bs in [4, 8, 16, 32, 64, 128, 256]:
    for hs in [16, 32, 64, 128, 256]:
        for ds in [16, 32, 64, 128, 256]:
            configs.append(triton.Config(kwargs={
                'BLOCK_SIZE': bs,
                "H_SIZE": hs,
                "D_SIZE": ds,
            }))

@triton.autotune(
  configs=configs,
  key=['N']
)
@triton.jit
def attn_weights_mul_value_kernel_v2(
    attn_weights_ptr,
    value_code_ptr,
    SA_ptr,
    B: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    N: int,
    BLOCK_SIZE: tl.constexpr, H_SIZE: tl.constexpr, D_SIZE: tl.constexpr,
):
    # attn_weights: [B, H, 1, N * 8]
    # value_code: [B, 1, N, D]
    # SA:[B, H, D]
    hx = tl.program_id(0)
    batch_i = hx >> 1
    hx = hx & 1
    dx = tl.program_id(1)
    bx = tl.program_id(2)
    hx_i = tl.arange(0, H_SIZE) + hx * H_SIZE
    dx_i = tl.arange(0, D_SIZE) + dx * D_SIZE
    i = tl.arange(0, BLOCK_SIZE) + bx * BLOCK_SIZE
    mask = i < N
    # code: [BLOCK_SIZE, D_SIZE]
    code_start = batch_i * N * D + i[:, None] * D + dx_i[None, :]
    code = tl.load(value_code_ptr + code_start, mask=mask[:, None])
    # attn_weight_start: [H_SIZE, BLOCK_SIZE]
    attn_weight_start = batch_i * H * N * 8 + hx_i[:, None] * N * 8 + i[None, :] * 8
    # offset: [8]
    offset = tl.arange(0, 8)
    # attn_weight: [H_SIZE, BLOCK_SIZE, 8]
    attn_weight = tl.load(attn_weights_ptr + attn_weight_start[:, :, None] + offset[None, None, :], mask=mask[None, :, None])
    # bit: [BLOCK_SIZE, 8, D_SIZE]
    bit = (((code[:, None, :] & (1 << offset[None, :, None])) != 0) * 1.0).cast(tl.bfloat16)
    # attn_weight: [H_SIZE, BLOCK_SIZE * 8]
    attn_weight = tl.reshape(attn_weight, (H_SIZE, attn_weight.shape[1] * attn_weight.shape[2]))
    # bit: [BLOCK_SIZE * 8, D_SIZE]
    bit = tl.reshape(bit, (bit.shape[0] * bit.shape[1], D_SIZE))
    acc_fp16 = tl.dot(attn_weight, bit).cast(tl.float16)
    # output_start: [H_SIZE, D_SIZE]
    output_start = batch_i * H * D + hx_i[:, None] * D + dx_i[None, :]
    tl.atomic_add(SA_ptr + output_start, acc_fp16)


def attn_weights_mul_value_triton_v2(attn_weights, value_code, prescale, learnable_scale, eps, codebook):
    attn_weights = (attn_weights[..., :prescale.shape[1]] * prescale.unsqueeze(1).transpose(-2, -1)).contiguous()
    value_code = value_code.unsqueeze(1).contiguous()
    # triton code to impl the following operation
    # SA = torch.matmul(attn_weights, value_code)
    B, H, N, D = attn_weights.shape[0], attn_weights.shape[1], value_code.shape[2], value_code.shape[3]
    # assert B == 1
    # BLOCK_SIZE = 128
    # H_SIZE = 32
    # D_SIZE = 32
    # assert (B * H) % H_SIZE == 0
    # assert D % D_SIZE == 0
    # assert H == H_SIZE * 2
    # grid = ((B * H) // H_SIZE, D // D_SIZE, triton.cdiv(N, BLOCK_SIZE))
    grid = lambda META: (triton.cdiv(B * H, META["H_SIZE"]), triton.cdiv(D, META["D_SIZE"]), triton.cdiv(N, META["BLOCK_SIZE"]))
    SA = torch.zeros(B, H, D, device=attn_weights.device, dtype=torch.float16)
    attn_weights_mul_value_kernel_v2[grid](
        attn_weights_ptr=attn_weights,
        value_code_ptr=value_code,
        SA_ptr=SA,
        B=B, H=H, D=D, N=N,
        # BLOCK_SIZE=BLOCK_SIZE,
        # H_SIZE=H_SIZE,
        # D_SIZE=D_SIZE,
    )
    SA = SA.unsqueeze(2).to(torch.bfloat16)
    SA = SA * (learnable_scale + eps)
    attn_output1 = torch.matmul(SA, codebook.unsqueeze(0))
    return attn_output1



if __name__ == "__main__":
    import time
    from tqdm import trange
    import warnings
    # ignore all warnings
    warnings.filterwarnings("ignore")
    # query_states = torch.randn(1, 32, 1, 64, 2).cuda().bfloat16()
    # centers = torch.randn(32, 64, 16, 64, 2, 2).cuda().bfloat16()
    # result1 = calculate_QC_T2(query_states, centers)
    # result2 = calculate_QC_T_triton(query_states, centers)
    # error = torch.abs(result1 - result2)
    # print(error.mean(), error.max())


    # check correctness
    # key_code = torch.load("/proj/long-multi/zfchen/long-kv-cache/niah/key_code.pt")
    # QC_T = torch.load("/proj/long-multi/zfchen/long-kv-cache/niah/QC_T.pt")
    # X, Y = gatherC1C2_and_sum_over_residual(key_code, QC_T)
    # X = X.squeeze(0)
    # Y = Y.squeeze(0)
    # key_code = key_code.permute(1, 2, 0)
    # key_code = key_code.contiguous()
    # QC_T = QC_T.unflatten(0, (8, 4)).transpose(1, 2)
    # QC_T = QC_T.contiguous()
    # X2, Y2 = gatherC1C2_and_sum_over_residual_triton(key_code, QC_T)
    # error = torch.abs(X - X2)
    # print(error.mean(), error.max())
    # print(X.flatten()[:10])
    # print(X2.flatten()[:10])
    # exit()


    N = 1000
    key_code = torch.load("/proj/long-multi/zfchen/long-kv-cache/niah/key_code.pt")
    QC_T = torch.load("/proj/long-multi/zfchen/long-kv-cache/niah/QC_T.pt")
    torch.cuda.synchronize()
    s1 = time.time()
    for _ in trange(N):
        torch.cuda.synchronize()
        X, Y = gatherC1C2_and_sum_over_residual(key_code, QC_T)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    e1 = time.time()
    key_code = torch.load("/proj/long-multi/zfchen/long-kv-cache/niah/key_code.pt")
    QC_T = torch.load("/proj/long-multi/zfchen/long-kv-cache/niah/QC_T.pt")
    key_code = key_code.permute(1, 2, 0)
    key_code = key_code.contiguous()
    QC_T = QC_T.unflatten(0, (8, 4)).transpose(1, 2)
    QC_T = QC_T.contiguous()
    torch.cuda.synchronize()
    s2 = time.time()
    for _ in trange(N):
        torch.cuda.synchronize()
        X2, Y2 = gatherC1C2_and_sum_over_residual_triton(key_code, QC_T)
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    e2 = time.time()
    error = torch.abs(X - X2)
    print(error.mean(), error.max())
    error = torch.abs(Y - Y2)
    print(error.mean(), error.max())
    print("PyTorch time:", (e1 - s1) / N)
    print("Triton  time:", (e2 - s2) / N)
