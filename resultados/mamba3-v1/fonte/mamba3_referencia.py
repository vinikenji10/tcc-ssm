"""Bloco Mamba-3 SISO em PyTorch puro, para CPU e GPU Apple (MPS).

Origem: https://github.com/state-spaces/mamba, commit e9594ce1c732d97440f0332fdc43170a2294dbfa
(22/07/2026), licença Apache-2.0. Copyright (c) 2024-2026, Tri Dao, Dao AI Lab, Goombalab.

Copiadas sem alteração, entre as marcas "INÍCIO/FIM DO CÓDIGO DOS AUTORES":
- `_segsum`, `mamba3_siso_step_ref` e `mamba3_siso_fwd_ref`, de tests/ops/triton/test_mamba3_siso.py;
- `rms_norm_ref`, de mamba_ssm/ops/triton/layernorm_gated.py;
- `heavy_tail_activation`, de mamba_ssm/modules/mamba3.py.

Escrito para o TCC: `RMSNormRef` e `Mamba3SISO`. `Mamba3SISO` reproduz o caminho SISO de
`mamba_ssm.modules.mamba3.Mamba3`, trocando os kernels Triton/CuTe pelas referências acima.
Os nomes e formatos dos parâmetros são os do módulo oficial, para que o `state_dict` possa ser
carregado nele numa GPU NVIDIA e as saídas comparadas.

`mamba3_siso_fwd_ref` calcula a sequência inteira de forma quadrática no comprimento (usado no
treino); `mamba3_siso_step_ref` avança uma amostra por vez com estado (usado na simulação).
Não é o kernel otimizado dos autores: tempos medidos com este módulo não representam aquele desempenho.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ==================== INÍCIO DO CÓDIGO DOS AUTORES (sem alteração) ====================
def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Segment sum helper for attention computation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def mamba3_siso_step_ref(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    Input_States: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Reference implementation of Mamba-3 in recurrent (step) mode.
    
    Args:
        Input_States: Optional tuple of (Angle_State, SSM_State, K_State, V_State)
    
    Returns:
        out: Output tensor (batch, seqlen, nheads, headdim_v)
        Final_States: Tuple of (Angle_State, SSM_State, K_State, V_State)
    """
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    headdim_angles = Angles.shape[-1]
    device = Q.device
    assert seqlen > 0
    Angles = torch.tanh(Angles) * math.pi

    # Expand Q/K for GQA
    if Q.shape[2] != V.shape[2]:
        Q = repeat(Q, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // Q.shape[2])
    if K.shape[2] != V.shape[2]:
        K = repeat(K, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // K.shape[2])

    def apply_rotary_emb(tensor, cos, sin):
        tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
        tensor_0 = tensor_reshaped[..., 0]
        tensor_1 = tensor_reshaped[..., 1]
        if cos.shape[-1] < tensor_0.shape[-1]:
            pad_size = tensor_0.shape[-1] - cos.shape[-1]
            cos = F.pad(cos, (0, pad_size), value=1.0)
            sin = F.pad(sin, (0, pad_size), value=0.0)
        rotated_0 = tensor_0 * cos - tensor_1 * sin
        rotated_1 = tensor_0 * sin + tensor_1 * cos
        rotated = torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)
        return rotated
    
    # Initialize states
    if Input_States is not None:
        Angle_State, SSM_State, K_State, V_State = Input_States
        Angle_State = Angle_State.clone()
        SSM_State = SSM_State.clone().to(torch.float32)
        K_State = K_State.clone()
        V_State = V_State.clone()
    else:
        Angle_State = torch.zeros((batch, nheads, headdim_angles), dtype=torch.float32, device=device)
        SSM_State = torch.zeros((batch, nheads, headdim_v, headdim_qk), dtype=torch.float32, device=device)
        K_State = torch.zeros((batch, nheads, headdim_qk), dtype=Q.dtype, device=device)
        V_State = torch.zeros((batch, nheads, headdim_v), dtype=V.dtype, device=device)
    
    TWO_PI = 2 * math.pi
    out_arr = []

    for idx in range(seqlen):
        q = Q[:, idx, :, :] + Q_bias.unsqueeze(0)
        k = K[:, idx, :, :] + K_bias.unsqueeze(0)
        v = V[:, idx, :, :]
        adt = ADT[:, :, idx]
        dt = DT[:, :, idx]
        trap = Trap[:, :, idx]
        z = Z[:, idx, :, :] if Z is not None else None
        angles = Angles[:, idx, :, :]

        # Update angle state with cumsum: Angle_State = (Angle_State + Angles * DT) mod 2π
        Angle_State = Angle_State + angles * dt.unsqueeze(-1)
        Angle_State = Angle_State - TWO_PI * torch.floor(Angle_State / TWO_PI)

        # Apply rotary embeddings to Q and K using cumulative angles
        cos_angles = torch.cos(Angle_State)
        sin_angles = torch.sin(Angle_State)
        q_rot = apply_rotary_emb(q, cos_angles, sin_angles)
        k_rot = apply_rotary_emb(k, cos_angles, sin_angles)

        trap = torch.sigmoid(trap)
        alpha = torch.exp(adt)
        beta = (1 - trap) * dt * alpha
        gamma = trap * dt

        # Update SSM state using previous K_State and V_State
        SSM_State = alpha.unsqueeze(-1).unsqueeze(-1) * SSM_State 
        SSM_State = SSM_State + beta.unsqueeze(-1).unsqueeze(-1) * (K_State.unsqueeze(-2) * V_State.unsqueeze(-1))
        SSM_State = SSM_State + gamma.unsqueeze(-1).unsqueeze(-1) * (k_rot.unsqueeze(-2) * v.unsqueeze(-1))

        # Compute output
        out = torch.einsum("bhdD, bhD -> bhd", SSM_State, q_rot.to(SSM_State.dtype))
        
        if D is not None:
            out = out + D[None, :, None] * v
        
        if Z is not None:
            out = out * z * torch.sigmoid(z)
        
        out_arr.append(out)
        
        # Update K and V states for next step
        K_State = k_rot
        V_State = v
    
    out = torch.stack(out_arr, dim=1)
    Final_States = (Angle_State, SSM_State, K_State, V_State)
    return out, Final_States


def mamba3_siso_fwd_ref(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    Initial_States: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    chunk_size: int = 64,
    dtype: torch.dtype = torch.float32,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """Reference implementation of Mamba-3 forward pass.
    
    Args:
        Initial_States: Optional tuple of (Angle_State, SSM_State, K_State, V_State)
    
    Returns:
        out_z: Output with Z gating applied
        final_states: (Final_Angle_State, Final_SSM_State, Final_K_State, Final_V_State)
    """
    batch, total_seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    headdim_angles = Angles.shape[-1]
    device = Q.device
    
    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert batch == 1
    
    # Cast inputs
    Q = Q.to(dtype)
    K = K.to(dtype)
    V = V.to(dtype)
    ADT = ADT.to(torch.float32)
    DT = DT.to(torch.float32)
    Trap = Trap.to(dtype)
    Q_bias = Q_bias.to(dtype)
    K_bias = K_bias.to(dtype)
    Angles = Angles.to(dtype)
    if D is not None:
        D = D.to(dtype)
    if Z is not None:
        Z = Z.to(dtype)
    if Initial_States is not None:
        Initial_Angle_State, Initial_SSM_State, Initial_K_State, Initial_V_State = Initial_States

    Angles = torch.tanh(Angles) * math.pi
    # Expand Q/K for GQA
    if Q.shape[2] != V.shape[2]:
        Q = repeat(Q, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // Q.shape[2])
    if K.shape[2] != V.shape[2]:
        K = repeat(K, "b s h_bc d -> b s (h_bc g) d", g=V.shape[2] // K.shape[2])

    out_zs = []
    Final_Angle_States = []
    Final_SSM_States = []
    Final_K_States = []
    Final_V_States = []

    TWO_PI = 2 * math.pi

    def _rotary(tensor, cos, sin):
        tensor_reshaped = tensor.view(*tensor.shape[:-1], -1, 2)
        tensor_0 = tensor_reshaped[..., 0]
        tensor_1 = tensor_reshaped[..., 1]
        if cos.shape[-1] < tensor_0.shape[-1]:
            pad_size = tensor_0.shape[-1] - cos.shape[-1]
            cos = F.pad(cos, (0, pad_size), value=1.0)
            sin = F.pad(sin, (0, pad_size), value=0.0)
        rotated_0 = tensor_0 * cos - tensor_1 * sin
        rotated_1 = tensor_0 * sin + tensor_1 * cos
        return torch.stack([rotated_0, rotated_1], dim=-1).view_as(tensor)

    def compute_one_sequence(seq_idx):
        if is_varlen:
            start_idx, end_idx = cu_seqlens[seq_idx].item(), cu_seqlens[seq_idx + 1].item()
            Q_curr = Q[0, start_idx:end_idx, :, :]
            K_curr = K[0, start_idx:end_idx, :, :]
            V_curr = V[0, start_idx:end_idx, :, :]
            ADT_curr = ADT[0, :, start_idx:end_idx]
            DT_curr = DT[0, :, start_idx:end_idx]
            Trap_curr = Trap[0, :, start_idx:end_idx]
            Angles_curr = Angles[0, start_idx:end_idx, :, :]
            Z_curr = Z[0, start_idx:end_idx, :, :] if Z is not None else None
        else:
            Q_curr = Q[seq_idx]
            K_curr = K[seq_idx]
            V_curr = V[seq_idx]
            ADT_curr = ADT[seq_idx]
            DT_curr = DT[seq_idx]
            Trap_curr = Trap[seq_idx]
            Angles_curr = Angles[seq_idx]
            Z_curr = Z[seq_idx] if Z is not None else None

        Trap_curr = torch.sigmoid(Trap_curr)
        seqlen_curr = Q_curr.shape[0]

        Angles_scaled = Angles_curr.float() * DT_curr.transpose(0, 1).unsqueeze(-1)
        Angles_Cumsum = torch.cumsum(Angles_scaled, dim=0)
        if Initial_States is not None:
            Initial_Angle_State_curr = Initial_Angle_State[seq_idx]
            Angles_Cumsum = Angles_Cumsum + Initial_Angle_State_curr.unsqueeze(0)
        Angles_Cumsum = Angles_Cumsum - TWO_PI * torch.floor(Angles_Cumsum / TWO_PI)
        Final_Angle_States.append(Angles_Cumsum[-1])

        # Initialize acc_states
        if Initial_States is not None:
            Initial_SSM_State_curr = Initial_SSM_State[seq_idx]
            Initial_K_State_curr = Initial_K_State[seq_idx]
            Initial_V_State_curr = Initial_V_State[seq_idx]

            scalar = DT_curr[:, 0] * (1 - Trap_curr[:, 0])
            acc_states = Initial_SSM_State_curr + Initial_V_State_curr[:, :, None] * Initial_K_State_curr[:, None, :] * scalar[:, None, None]
        else:
            acc_states = torch.zeros((nheads, headdim_v, headdim_qk), device=device, dtype=torch.float32)

        # Compute shifted gamma and scale
        DT_shifted = F.pad(DT_curr[:, 1:], (0, 1))
        Trap_shifted = F.pad(Trap_curr[:, 1:], (0, 1))
        shifted_gamma = DT_shifted * (1 - Trap_shifted)
        scale = DT_curr * Trap_curr + DT_shifted * (1 - Trap_shifted)

        # Add biases
        Q_curr = Q_curr + Q_bias.unsqueeze(0)
        K_curr = K_curr + K_bias.unsqueeze(0)

        # Compute QK dot for skip connection
        QK_dot = torch.sum(K_curr * Q_curr, dim=-1) * shifted_gamma.transpose(0, 1)

        # Rotary embeddings using Angles_Cumsum
        cos_angles_curr = torch.cos(Angles_Cumsum).to(Q_curr.dtype)
        sin_angles_curr = torch.sin(Angles_Cumsum).to(Q_curr.dtype)
        Q_curr = _rotary(Q_curr, cos_angles_curr, sin_angles_curr)
        K_curr = _rotary(K_curr, cos_angles_curr, sin_angles_curr)

        Final_K_States.append(K_curr[-1])
        Final_V_States.append(V_curr[-1])

        K_curr_scaled = K_curr * scale.transpose(0, 1).unsqueeze(-1).to(K_curr.dtype)

        # Compute output via quadratic attention
        QK = torch.einsum("thd,shd->hts", Q_curr, K_curr_scaled)
        QK_causal = torch.tril(QK)
        QK_causal = (QK_causal * torch.exp(_segsum(ADT_curr))).to(QK_causal.dtype)
        out = torch.einsum("hts,shd->thd", QK_causal, V_curr)

        if Initial_States is not None:
            da_cs = torch.cumsum(ADT_curr, dim=-1)
            exp_da_cs = torch.exp(da_cs)
            out = out + torch.einsum("hDd,thd,ht->thD", acc_states.to(Q_curr.dtype), Q_curr, exp_da_cs.to(Q_curr.dtype))

        if D is not None:
            out = out + D[None, :, None] * V_curr

        out = out - V_curr * QK_dot.unsqueeze(-1)

        if Z_curr is not None:
            out = out * Z_curr * torch.sigmoid(Z_curr)
        out_zs.append(out)

        # Compute final state
        da_cs_last = torch.exp(torch.sum(ADT_curr, dim=-1))
        da_cs_rev = torch.exp(torch.sum(ADT_curr, dim=-1, keepdim=True) - torch.cumsum(ADT_curr, dim=-1))
        V_curr_scaled = V_curr * da_cs_rev.permute(1, 0).unsqueeze(-1).to(V_curr.dtype)
        final_acc_states = acc_states * da_cs_last.unsqueeze(-1).unsqueeze(-1) + torch.einsum(
            "thd,thD->hDd", K_curr_scaled, V_curr_scaled.to(K_curr_scaled.dtype))
        Final_SSM_States.append(final_acc_states)

    num_sequences = cu_seqlens.size(0) - 1 if is_varlen else batch
    for seq_idx in range(num_sequences):
        compute_one_sequence(seq_idx)

    if not is_varlen:
        out_zs = torch.stack(out_zs, dim=0)
        Final_Angle_States = torch.stack(Final_Angle_States, dim=0)
        Final_SSM_States = torch.stack(Final_SSM_States, dim=0)
        Final_K_States = torch.stack(Final_K_States, dim=0)
        Final_V_States = torch.stack(Final_V_States, dim=0)
    else:
        out_zs = torch.cat(out_zs, dim=0).unsqueeze(0)
        Final_Angle_States = torch.stack(Final_Angle_States, dim=0)
        Final_SSM_States = torch.stack(Final_SSM_States, dim=0)
        Final_K_States = torch.stack(Final_K_States, dim=0)
        Final_V_States = torch.stack(Final_V_States, dim=0)

    return out_zs, (Final_Angle_States, Final_SSM_States, Final_K_States, Final_V_States)


def rms_norm_ref(x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True):
    dtype = x.dtype
    N = x.shape[-1]
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


def heavy_tail_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Heavy-tail activation for data-dependent A.

    Using this activation can improve stability during WSD training and at
    higher learning rates.

        f(x) = 1 + x        if x >= 0
            = 1 / (1 - x)  if x < 0

    The function is positive, continuous, and differentiable at x = 0.
    """
    neg = x.clamp_max(0)
    pos = x.clamp_min(0)
    return pos + torch.reciprocal(1 - neg)


# ==================== FIM DO CÓDIGO DOS AUTORES ====================


class RMSNormRef(nn.Module):
    """Mesmos parâmetros de `mamba_ssm.ops.triton.layernorm_gated.RMSNorm`, calculado por `rms_norm_ref`."""

    def __init__(self, hidden_size, eps=1e-5, group_size=None, norm_before_gate=True, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate

    def forward(self, x, z=None):
        return rms_norm_ref(x, self.weight, self.bias, z=z, eps=self.eps, group_size=self.group_size,
                            norm_before_gate=self.norm_before_gate)


class Mamba3SISO(nn.Module):
    """Caminho SISO de `Mamba3` (mamba_ssm, commit e9594ce) sobre as referências em PyTorch.

    O estado é a tupla (ângulo, SSM, K, V) devolvida pelas referências. `forward` e `step`
    aceitam e devolvem o mesmo formato; assim é possível aquecer com `forward` numa janela
    de medições e continuar com `step`, realimentando previsões.
    """

    def __init__(self, d_model, d_state=128, expand=2, headdim=64, ngroups=1, rope_fraction=0.5,
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, A_floor=1e-4, is_outproj_norm=False,
                 chunk_size=64, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size  # só o kernel usa; mantido para registrar a configuração
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm
        self.mimo_rank = 1

        self.d_inner = int(self.expand * self.d_model)
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = ngroups

        assert rope_fraction in [0.5, 1.0]
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        # Ordem da projeção, como no módulo oficial: [z, x, B, C, dd_dt, dd_A, trap, angle]
        d_in_proj = 2 * self.d_inner + 2 * self.d_state * self.num_bc_heads + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(_dt + torch.log(-torch.expm1(-_dt)))
        self.dt_bias._no_weight_decay = True

        self.B_bias = nn.Parameter(1 + torch.zeros((self.nheads, 1, self.d_state), dtype=torch.float32, device=device))
        self.C_bias = nn.Parameter(1 + torch.zeros((self.nheads, 1, self.d_state), dtype=torch.float32, device=device))
        self.B_norm = RMSNormRef(self.d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormRef(self.d_state, eps=1e-5, **factory_kwargs)

        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.is_outproj_norm:
            self.norm = RMSNormRef(self.d_inner, eps=1e-5, norm_before_gate=True, group_size=self.headdim,
                                   **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

    def _project(self, u):
        """Mesmo pré-processamento de `Mamba3.forward` antes do kernel SISO."""
        z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            self.in_proj(u),
            [self.d_inner, self.d_inner,
             self.d_state * self.num_bc_heads, self.d_state * self.num_bc_heads,
             self.nheads, self.nheads, self.nheads, self.num_rope_angles],
            dim=-1)
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.num_bc_heads)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        _A = -heavy_tail_activation(dd_A.to(torch.float32))
        _A = torch.clamp(_A, max=-self.A_floor)
        DT = F.softplus(dd_dt + self.dt_bias)
        ADT = rearrange(_A * DT, "b l h -> b h l")
        DT = rearrange(DT, "b l h -> b h l")
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1).to(torch.float32)

        return dict(Q=self.C_norm(C), K=self.B_norm(B), V=x, ADT=ADT, DT=DT, Trap=trap,
                    Q_bias=self.C_bias.squeeze(1), K_bias=self.B_bias.squeeze(1), Angles=angles,
                    D=self.D, Z=None if self.is_outproj_norm else z), z

    def _output(self, y, z, dtype):
        y = rearrange(y, "b l h p -> b l (h p)")
        if self.is_outproj_norm:
            y = self.norm(y, rearrange(z, "b l h p -> b l (h p)"))
        return self.out_proj(y.to(dtype))

    def forward(self, u, state=None, return_state=False):
        """Sequência inteira, u: (batch, L, d_model). Diferenciável; usado no treino."""
        args, z = self._project(u)
        y, final_state = mamba3_siso_fwd_ref(**args, Initial_States=state, chunk_size=self.chunk_size,
                                             dtype=args["V"].dtype)
        out = self._output(y, z, u.dtype)
        return (out, final_state) if return_state else out

    def step(self, u, state=None):
        """Uma amostra por vez, u: (batch, L, d_model) com L >= 1. Devolve (saída, estado final)."""
        args, z = self._project(u)
        y, final_state = mamba3_siso_step_ref(**args, Input_States=state)
        return self._output(y, z, u.dtype), final_state
