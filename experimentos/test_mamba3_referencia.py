"""Verificações do Mamba-3 SISO em PyTorch: parâmetros, gradientes, causalidade e step_ref contra fwd_ref.

Rodam na CPU por padrão. Para a GPU do Mac: TCC_DEVICE=mps.
"""
import math
import os
import unittest

import torch
import torch.nn as nn

from mamba3_referencia import Mamba3SISO

DEVICE = os.environ.get("TCC_DEVICE", "cpu")
CONFIG = {"d_model": 64, "d_state": 64, "expand": 2, "headdim": 64}  # configuração inicial do plano
TOLERANCE = 1e-5  # float32; as diferenças medidas em 17/10 ficaram abaixo de 4e-6


def relative_max_error(value, reference):
    return float((value - reference).abs().max() / reference.abs().max())


def angle_max_error(value, reference):
    delta = torch.remainder(value - reference + math.pi, 2 * math.pi) - math.pi
    return float(delta.abs().max())


class Mamba3Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261017)
        self.block = Mamba3SISO(**CONFIG, device=DEVICE)
        self.u = torch.randn(3, 256, CONFIG["d_model"], device=DEVICE)

    def test_parameters_match_official_module(self):
        # Formatos de mamba_ssm/modules/mamba3.py (commit e9594ce) nesta configuração: duas cabeças,
        # in_proj = 2*128 (z, x) + 2*64 (B, C) + 3*2 (dt, A, trap) + 16 ângulos = 406 saídas.
        expected = {"in_proj.weight": (406, 64), "dt_bias": (2,), "B_bias": (2, 1, 64), "C_bias": (2, 1, 64),
                    "B_norm.weight": (64,), "C_norm.weight": (64,), "D": (2,), "out_proj.weight": (64, 128)}
        self.assertEqual({name: tuple(p.shape) for name, p in self.block.named_parameters()}, expected)
        self.assertEqual(sum(p.numel() for p in self.block.parameters()), 34_564)

    def test_forward_shape_and_finite_values(self):
        y = self.block(self.u)
        self.assertEqual(y.shape, self.u.shape)
        self.assertTrue(torch.isfinite(y).all())

    def test_gradients_are_finite_and_reach_every_parameter(self):
        u = self.u.clone().requires_grad_()
        self.block(u).square().mean().backward()
        self.assertTrue(torch.isfinite(u.grad).all())
        for name, p in self.block.named_parameters():
            with self.subTest(parametro=name):
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().max().item(), 0.0)

    def test_step_ref_matches_fwd_ref(self):
        with torch.no_grad():
            y_fwd, state_fwd = self.block(self.u, return_state=True)
            y_step, state_step = self.block.step(self.u)
        self.assertLess(relative_max_error(y_step, y_fwd), TOLERANCE)
        self.assertLess(angle_max_error(state_step[0], state_fwd[0]), TOLERANCE)
        for name, a, b in zip(("SSM", "K", "V"), state_step[1:], state_fwd[1:]):
            with self.subTest(estado=name):
                self.assertLess(relative_max_error(a, b), TOLERANCE)

    def test_warm_start_with_forward_then_step_sample_by_sample(self):
        # Regime da avaliação: aquecer com a sequência de medições e seguir uma amostra por vez.
        k = 192
        with torch.no_grad():
            y_full = self.block(self.u)
            _, state = self.block(self.u[:, :k], return_state=True)
            outputs = []
            for t in range(k, self.u.shape[1]):
                y_t, state = self.block.step(self.u[:, t:t + 1], state)
                outputs.append(y_t)
        self.assertLess(relative_max_error(torch.cat(outputs, dim=1), y_full[:, k:]), TOLERANCE)

    def test_future_inputs_do_not_change_past_outputs(self):
        k = 100
        changed = self.u.clone()
        changed[:, k:] = torch.randn_like(changed[:, k:])
        with torch.no_grad():
            y, y_changed = self.block(self.u), self.block(changed)
            s, s_changed = self.block.step(self.u)[0], self.block.step(changed)[0]
        self.assertTrue(torch.equal(s_changed[:, :k], s[:, :k]))
        self.assertTrue(torch.equal(y_changed[:, :k - 1], y[:, :k - 1]))
        # fwd_ref soma e subtrai na saída t um termo com dt e trap de t+1: em t = k-1 sobra arredondamento.
        self.assertLess(relative_max_error(y_changed[:, k - 1], y[:, k - 1]), TOLERANCE)
        self.assertGreater(relative_max_error(y_changed[:, k:], y[:, k:]), 1e-2)

    def test_training_updates_reduce_loss_on_a_linear_filter(self):
        # Alvo y[t] = 0,9 y[t-1] + 0,1 u[t-1]: a rede precisa de memória para acertá-lo.
        torch.manual_seed(0)
        u = torch.randn(16, 128, 1, device=DEVICE)
        y = torch.zeros_like(u)
        for t in range(1, u.shape[1]):
            y[:, t] = 0.9 * y[:, t - 1] + 0.1 * u[:, t - 1]
        y = y / y.std()
        model = nn.Sequential(nn.Linear(1, 64), Mamba3SISO(**CONFIG), nn.Linear(64, 1)).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        losses = []
        for _ in range(150):
            loss = (model(u) - y).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertTrue(all(math.isfinite(value) for value in losses))
        self.assertLess(losses[-1], 0.1 * losses[0])


if __name__ == "__main__":
    unittest.main()
