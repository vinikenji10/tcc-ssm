"""Plantas e excitações do E008 sobre o protocolo v2, sem editar `protocolo_v2.py`.

Ao importar este módulo, são registradas:

- em `protocolo_v2.PLANTS`, a planta `saturation_friction` da seção 3.2 do pré-registro:

      m p'' = sat(u) - c p' - k p - Fc tanh(p'/vε),   sat(u) = clip(u, -umax, umax),

  integrada por RK4 com subpassos sob segurador de ordem zero (a entrada é constante em cada período de
  amostragem, então a saturação também é), vetorizada sobre as trajetórias;
- em `protocolo_v2.EXCITATIONS`, a excitação `multisine`: multisseno de fase aleatória com os harmônicos
  `first_harmonic` a `last_harmonic` de f0 = 1/(período × Ts), amplitudes iguais, fases uniformes em
  [0, 2π) sorteadas por trajetória e RMS dado, com exatamente um período por trajetória.

A ordem de sorteio por trajetória continua a de `protocolo_v2.generate_dataset`: estado inicial, excitação,
ruído. A planta sorteia o estado inicial como a de Duffing (posição e depois velocidade, uniformes).

Uso, igual ao de `protocolo_v2.py gerar` (o módulo é importado antes de `generate_from_config`):

    PYTHONUTF8=1 .venv\\Scripts\\python.exe experimentos/plantas_v2.py gerar --config <config> --output <pasta>

`ruido` calcula, só com as trajetórias de treino de uma geração sem ruído em memória, o desvio do ruído do
E008 pela regra do pré-registro (0,07 vezes o desvio-padrão da posição, com dois algarismos significativos).
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import protocolo_v2
from protocolo_v2 import EXCITATIONS, PLANTS, SimulatedPlant, generate_dataset, generate_from_config

PLANT_NAME = "saturation_friction"
EXCITATION_NAME = "multisine"
NOISE_TO_SIGNAL = 0.07  # relação do E006 (seção 3.2 do pré-registro)


# ---------------------------------------------------------------------------------------------------
# Planta com saturação e atrito

def saturation_friction_step(x, u, plant, substeps=None):
    """Um período de amostragem com u constante (ZOH), RK4 com subpassos. x: (..., 2) = [posição, velocidade].

    A saturação é aplicada uma vez por período, porque a entrada é constante nele. `substeps` substitui
    `plant["rk4_substeps"]` (só para o teste de convergência).
    """
    m, c, k = plant["mass"], plant["damping"], plant["stiffness"]
    limit, fc, v_eps = plant["saturation_limit"], plant["coulomb_friction"], plant["friction_velocity_scale"]
    steps = plant["rk4_substeps"] if substeps is None else substeps
    h = plant["sample_time"] / steps
    force = np.clip(u, -limit, limit)

    def f(state):
        p, v = state[..., 0], state[..., 1]
        friction = fc * np.tanh(v / v_eps) if fc != 0 else 0.0
        return np.stack([v, (force - c * v - k * p - friction) / m], axis=-1)

    for _ in range(steps):
        k1 = f(x)
        k2 = f(x + 0.5 * h * k1)
        k3 = f(x + 0.5 * h * k2)
        k4 = f(x + h * k3)
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def simulate_saturation_friction(u, x0, plant, substeps=None):
    """u (N, T), x0 (N, 2) -> x (N, T+1, 2); todas as trajetórias em paralelo."""
    u = np.asarray(u, dtype=np.float64)
    x = np.empty((len(u), u.shape[1] + 1, 2), dtype=np.float64)
    x[:, 0] = x0
    for t in range(u.shape[1]):
        x[:, t + 1] = saturation_friction_step(x[:, t], u[:, t], plant, substeps)
    return x


def saturation_friction_plant(config):
    """Planta da seção 3.2 como `SimulatedPlant`; o estado inicial é sorteado como o do E005."""
    plant, limits = config["plant"], config["initial_state"]
    for key in ("mass", "sample_time", "friction_velocity_scale", "saturation_limit"):
        if plant[key] <= 0:
            raise ValueError(f"`plant.{key}` deve ser positivo.")
    if plant["rk4_substeps"] < 1 or plant["coulomb_friction"] < 0:
        raise ValueError("`rk4_substeps` deve ser positivo e `coulomb_friction`, não negativo.")

    def initial_state(rng):
        return np.array([rng.uniform(-limits["position_limit"], limits["position_limit"]),
                         rng.uniform(-limits["velocity_limit"], limits["velocity_limit"])])

    return SimulatedPlant(PLANT_NAME, initial_state, lambda u, x0: simulate_saturation_friction(u, x0, plant),
                          lambda x: x[:, :, 0])


# ---------------------------------------------------------------------------------------------------
# Multisseno de fase aleatória

def multisine_harmonics(settings):
    harmonics = np.arange(settings["first_harmonic"], settings["last_harmonic"] + 1)
    if harmonics.size == 0 or harmonics[0] < 1 or 2 * harmonics[-1] >= settings["period"]:
        raise ValueError("Os harmônicos do multisseno devem estar entre 1 e período/2 (exclusive).")
    return harmonics


def multisine_values(phases, n, settings):
    """u[n] = A Σ cos(2π k n / período + φ_k), com A tal que o RMS sobre um período seja `settings["rms"]`.

    Os harmônicos são distintos e menores que período/2, então os cossenos são ortogonais no período e o RMS
    é exatamente A·√(K/2). O produto k·n é reduzido módulo o período em inteiros, sem perda de precisão.
    """
    harmonics = multisine_harmonics(settings)
    amplitude = settings["rms"] * np.sqrt(2.0 / harmonics.size)
    angle = 2 * np.pi * (np.outer(np.asarray(n, dtype=np.int64), harmonics) % settings["period"]) / settings["period"]
    return amplitude * np.cos(angle + phases).sum(axis=1)


def multisine(rng, steps, settings):
    """Um período por trajetória. Sorteia só as fases (uma por harmônico, em ordem crescente de k)."""
    if steps != settings["period"]:
        raise ValueError(f"O multisseno tem um período por trajetória: steps ({steps}) deve ser igual a "
                         f"period ({settings['period']}).")
    harmonics = multisine_harmonics(settings)
    phases = rng.uniform(0.0, 2 * np.pi, harmonics.size)
    return multisine_values(phases, np.arange(steps), settings)


PLANTS[PLANT_NAME] = saturation_friction_plant
EXCITATIONS[EXCITATION_NAME] = multisine


# ---------------------------------------------------------------------------------------------------
# Regra do ruído do E008 e diagnóstico do treino

def two_significant_digits(value):
    return float(f"{value:.2g}")


def clean_training(config):
    """Geração sem ruído com a mesma semente, em memória; devolve SÓ o treino (entrada, saída e estados).

    As trajetórias de validação e de teste final são simuladas (é o gerador do protocolo, que sorteia todas)
    e descartadas aqui, sem leitura. O ruído é sorteado por último em cada trajetória, então as trajetórias
    sem ruído são as mesmas da geração com ruído.
    """
    quiet = copy.deepcopy(config)
    quiet["data"]["measurement_noise_std"] = 0.0
    plant = PLANTS[quiet["plant"]["model"]](quiet)
    data, states, _ = generate_dataset(plant, EXCITATIONS[quiet["excitation"]["type"]], quiet)
    train = {"u": data["treino_u"], "y": data["treino_y_limpo"], "x": states["treino_x"]}
    del data, states
    return train


def noise_rule(config, train=None):
    """Desvio do ruído do E008: 0,07 × desvio-padrão (populacional) da posição sem ruído do treino."""
    train = clean_training(config) if train is None else train
    std = float(np.std(train["y"]))
    return {"desvio_posicao_treino_m": std, "fator": NOISE_TO_SIGNAL, "produto_m": NOISE_TO_SIGNAL * std,
            "desvio_ruido_m": two_significant_digits(NOISE_TO_SIGNAL * std),
            "amostras": int(train["y"].size), "trajetorias_treino": int(len(train["y"]))}


def training_activity(config, train=None):
    """Quanto a saturação e o atrito atuam no treino: frações de amostras e de tempo."""
    train = clean_training(config) if train is None else train
    plant = config["plant"]
    v = train["x"][:, :, 1]
    v_eps = plant["friction_velocity_scale"]
    return {"fracao_entrada_saturada": float(np.mean(np.abs(train["u"]) > plant["saturation_limit"])),
            "fracao_v_abaixo_1_veps": float(np.mean(np.abs(v) < v_eps)),
            "fracao_v_abaixo_3_veps": float(np.mean(np.abs(v) < 3 * v_eps)),
            "rms_entrada_N": float(np.sqrt(np.mean(train["u"] ** 2))),
            "max_abs_entrada_N": float(np.max(np.abs(train["u"]))),
            "desvio_posicao_m": float(np.std(train["y"])),
            "max_abs_posicao_m": float(np.max(np.abs(train["y"]))),
            "max_abs_velocidade_m_s": float(np.max(np.abs(v)))}


# ---------------------------------------------------------------------------------------------------
# Comando

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("gerar", help="gera e grava os dados de um cenário (pasta nova)")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    noise = commands.add_parser("ruido", help="desvio do ruído do E008 pela regra do pré-registro (só o treino)")
    noise.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "gerar":
        print(f"Dados gravados em {generate_from_config(args.config, args.output).resolve()}")
    elif args.command == "ruido":
        config = json.loads(args.config.read_bytes())
        protocolo_v2.validate_config(config)
        train = clean_training(config)
        rule, activity = noise_rule(config, train), training_activity(config, train)
        print(json.dumps({"regra_do_ruido": rule, "atuacao_no_treino": activity}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
