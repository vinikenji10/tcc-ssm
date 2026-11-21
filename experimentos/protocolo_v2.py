"""Protocolo v2 dos ensaios da A3 (E005 em diante): dados com teste final reservado, contrato único de
modelo, métricas, bootstrap por trajetória e seleção congelada.

Convenção do E001 (veja experimentos/README.md): no instante t são conhecidos y[t] e a entrada aplicada
u[t]; u[t] gera y[t+1]. A divisão é feita por trajetória, antes de qualquer janela. Os valores exatos
(sementes, grades, número de reamostragens) estão em planejamento/pre-registro-a3.md e nos arquivos
config-*-v2.json.

Contrato de modelo
------------------
Toda família (persistência, ARX, NARX, GRU, Mamba-3) é avaliada por um objeto com:

    name: str          rótulo nas tabelas
    parameters: int    número de parâmetros ajustados
    simulate(y_past, u_past, u_future) -> y_hat

Cada linha b do lote é uma origem t, na ordem trajetória e depois origem (a mesma de `evaluate` do E001):

    y_past   (B, H)  y[t-H+1] ... y[t]       medições até a origem, inclusive (H = 256)
    u_past   (B, H)  u[t-H]   ... u[t-1]     u_past[:, i] é a entrada que produziu y_past[:, i]
    u_future (B, K)  u[t]     ... u[t+K-1]   entradas conhecidas; u_future[:, j] produz y_hat[:, j]
    y_hat    (B, K)  ŷ[t+1]   ... ŷ[t+K]     simulação livre, realimentando as próprias previsões (K = 50)

Regras do contrato:
- float64 e unidades físicas na entrada e na saída. A normalização é interna ao modelo e usa só o treino.
- Linhas independentes: o resultado não depende do tamanho nem da composição do lote.
- Nenhuma outra informação chega ao modelo. Os alvos y[t+1:] nunca são passados.
- Valores não finitos não são escondidos (sem recorte nem substituição). A divergência é detectada aqui.
- `time_aligned_pairs` devolve os pares (u[s], y[s]) do aquecimento, s = t-H+1..t, como a rede do E002 usa.

Uma família com sementes tem vários membros (um modelo por semente); as tradicionais têm um.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from experimento_duffing import duffing_step
from experimento_linear import ARX, Persistence, evaluation_origins, normalization_from_training, summarize_rollouts

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("treino", "validacao", "teste_final")
DEVELOPMENT_FILE = "dados-desenvolvimento.npz"
TEST_DIR = "teste-final"
TEST_FILE = f"{TEST_DIR}/dados-teste-final.npz"
USAGE_LOG = f"{TEST_DIR}/uso-teste-final.jsonl"
MANIFEST_FILE = "manifesto.json"
DATA_FORMAT = "tcc-ssm/dados/v2"
SELECTION_FORMAT = "tcc-ssm/selecao-congelada/v1"
DEFAULT_CODE_FILES = ("experimentos/protocolo_v2.py", "experimentos/avaliar_teste_final.py",
                      "experimentos/experimento_linear.py", "experimentos/experimento_duffing.py")


# ---------------------------------------------------------------------------------------------------
# Utilidades

def now_iso():
    """Horário de Brasília. Sem a base `tzdata` (Python no Windows), usa o fuso local do sistema."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat()
    except Exception:
        return datetime.now().astimezone().isoformat()


TEXT_SUFFIXES = {".py", ".json", ".jsonl", ".md", ".csv", ".txt"}


def sha256_text(data):
    """Hash de texto com fins de linha em LF: o mesmo no Windows (checkout com CRLF) e no macOS/Linux."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def sha256_file(path):
    data = Path(path).read_bytes()
    if Path(path).suffix.lower() in TEXT_SUFFIXES:
        return sha256_text(data)
    return hashlib.sha256(data).hexdigest()


def arrays_sha256(arrays):
    """Hash do conteúdo dos vetores. Ao contrário do hash do arquivo .npz, não depende da data gravada no zip."""
    digest = hashlib.sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(f"{key}|{value.dtype}|{value.shape}|".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def json_safe(value):
    """Converte tipos do NumPy; valores não finitos viram null (a divergência é marcada em campo próprio)."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative_path(path, start):
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(start).resolve())).as_posix()
    except ValueError:  # unidades diferentes no Windows
        return Path(path).resolve().as_posix()


# ---------------------------------------------------------------------------------------------------
# Contrato de modelo

def rollout_windows(u, y, origins, history, horizon):
    """Entradas do contrato para todas as trajetórias e origens. Os alvos ficam de fora, em `rollout_targets`."""
    origins = np.asarray(origins, dtype=int)
    if origins.min() < history:
        raise ValueError(f"A origem {origins.min()} não tem {history} medições e {history} entradas anteriores.")
    if origins.max() + horizon > u.shape[1]:
        raise ValueError("Origem/horizonte fora do intervalo de dados.")
    episode = np.repeat(np.arange(len(u)), len(origins))[:, None]
    t = np.tile(origins, len(u))[:, None]
    lags = np.arange(history)
    y_past = y[episode, t - history + 1 + lags]
    u_past = u[episode, t - history + lags]
    u_future = u[episode, t + np.arange(horizon)]
    return y_past, u_past, u_future


def rollout_targets(y, origins, horizon):
    """y[t+1] ... y[t+K] para cada origem, na mesma ordem das janelas. Nunca vai para o modelo."""
    origins = np.asarray(origins, dtype=int)
    episode = np.repeat(np.arange(len(y)), len(origins))[:, None]
    t = np.tile(origins, len(y))[:, None]
    return y[episode, t + 1 + np.arange(horizon)]


def time_aligned_pairs(y_past, u_past, u_future):
    """Pares (u[s], y[s]) para s = t-H+1..t, formato (B, H, 2): o aquecimento da rede do E002."""
    u_same_time = np.concatenate([u_past[:, 1:], u_future[:, :1]], axis=1)
    return np.stack([u_same_time, y_past], axis=-1)


class StepModelAdapter:
    """Contrato v2 para os modelos de um passo do E001/E003 (`predict_step`): persistência, ARX e NARX.

    Repete `experimento_linear.rollout`: os atrasos iniciais são as próprias medições, sem estimação do estado
    inicial. A estimação com a janela de 256 medições fica nos adaptadores dos tradicionais (Onda 2).
    """

    def __init__(self, model):
        self.model, self.name, self.parameters = model, model.name, model.parameters

    def simulate(self, y_past, u_past, u_future):
        batch, history = y_past.shape
        horizon = u_future.shape[1]
        out = np.empty((batch, horizon), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            for b in range(batch):
                u_history = np.concatenate([u_past[b], u_future[b]])   # u_history[i] = u[t-H+i]
                y_history = np.concatenate([y_past[b], np.empty(horizon)])  # y_history[i] = y[t-H+1+i]
                for j in range(horizon):
                    y_history[history + j] = self.model.predict_step(u_history[:history + j + 1],
                                                                     y_history[:history + j])
                out[b] = y_history[history:]
        return out


def simulate_checked(model, y_past, u_past, u_future, batch_size=None):
    """Chama `simulate` em lotes, com cópias das entradas, e confere o formato da saída."""
    total, horizon = len(y_past), u_future.shape[1]
    size = batch_size or total
    parts = []
    for start in range(0, total, size):
        part = slice(start, start + size)
        out = np.asarray(model.simulate(y_past[part].copy(), u_past[part].copy(), u_future[part].copy()),
                         dtype=np.float64)
        expected = (len(y_past[part]), horizon)
        if out.shape != expected:
            raise ValueError(f"{model.name}: saída com formato {out.shape}; o contrato exige {expected}.")
        parts.append(out)
    return np.concatenate(parts)


# ---------------------------------------------------------------------------------------------------
# Plantas e excitação: funções, para que outras plantas usem o mesmo gerador

@dataclass(frozen=True)
class SimulatedPlant:
    """Planta simulada. O protocolo não conhece as equações, só estas três funções."""
    name: str
    initial_state: Callable  # rng -> x0, formato (n_x,)
    simulate: Callable       # u (N, T), x0 (N, n_x) -> x (N, T+1, n_x); entradas sob segurador de ordem zero
    output: Callable         # x (N, T+1, n_x) -> saída sem ruído (N, T+1)


def duffing_plant(config):
    """Oscilador de Duffing do E003/E004 (`experimento_duffing.duffing_step`)."""
    plant, limits = config["plant"], config["initial_state"]

    def initial_state(rng):
        return np.array([rng.uniform(-limits["position_limit"], limits["position_limit"]),
                         rng.uniform(-limits["velocity_limit"], limits["velocity_limit"])])

    def simulate(u, x0):
        x = np.empty((len(u), u.shape[1] + 1, 2), dtype=np.float64)
        x[:, 0] = x0
        for t in range(u.shape[1]):  # todas as trajetórias do conjunto em paralelo
            x[:, t + 1] = duffing_step(x[:, t], u[:, t], plant)
        return x

    return SimulatedPlant("duffing", initial_state, simulate, lambda x: x[:, :, 0])


def aprbs(rng, steps, settings):
    """Patamares de duração e amplitude sorteadas, na mesma ordem de sorteio do E003."""
    u = np.empty(steps, dtype=np.float64)
    t = 0
    while t < steps:
        hold = int(rng.integers(settings["hold_min"], settings["hold_max"] + 1))
        u[t:min(t + hold, steps)] = rng.uniform(-settings["amplitude"], settings["amplitude"])
        t += hold
    return u


PLANTS = {"duffing": duffing_plant}
EXCITATIONS = {"aprbs": aprbs}


def validate_config(config):
    d, e, b = config["data"], config["evaluation"], config["bootstrap"]
    if min(d["train_episodes"], d["validation_episodes"], d["test_episodes"]) < 1:
        raise ValueError("Cada conjunto precisa de pelo menos uma trajetória.")
    if d["measurement_noise_std"] < 0:
        raise ValueError("O desvio do ruído deve ser não negativo.")
    horizons = sorted(set(e["horizons"]))
    if horizons[0] < 1 or e["origin_stride"] < 1 or e["history"] < 1:
        raise ValueError("Horizontes, passo e histórico devem ser positivos.")
    if e["warmup"] < e["history"]:
        raise ValueError("A primeira origem precisa de H medições e H entradas anteriores (warmup >= history).")
    if d["steps"] < e["warmup"] + horizons[-1]:
        raise ValueError("Trajetórias curtas demais para a avaliação.")
    if e["selection_horizon"] not in horizons:
        raise ValueError("O horizonte de escolha precisa estar entre os horizontes avaliados.")
    if e["divergence_factor"] <= 1:
        raise ValueError("O fator de divergência deve ser maior que 1.")
    if b["resamples"] < 1000 or not 0 < b["confidence"] < 1:
        raise ValueError("Use pelo menos 1000 reamostragens e confiança entre 0 e 1.")


def evaluation_settings(config):
    settings = dict(config["evaluation"])
    settings["horizons"] = sorted(set(settings["horizons"]))
    return settings


# ---------------------------------------------------------------------------------------------------
# Dados: geração, gravação e leitura do desenvolvimento

def generate_dataset(plant, excitation, config):
    """Trajetórias de treino, validação e teste final, cada uma com sua semente filha.

    Ordem de sorteio por trajetória: estado inicial, excitação, ruído. Com a planta de Duffing e a APRBS,
    reproduz exatamente `experimento_duffing.generate_dataset` para a mesma semente e os mesmos tamanhos.
    """
    d = config["data"]
    sizes = dict(zip(SPLITS, (d["train_episodes"], d["validation_episodes"], d["test_episodes"])))
    children = np.random.SeedSequence(config["seed"]).spawn(sum(sizes.values()))
    data, states, manifest, cursor = {}, {}, [], 0
    for split, count in sizes.items():
        u = np.empty((count, d["steps"]), dtype=np.float64)
        noise = np.empty((count, d["steps"] + 1), dtype=np.float64)
        x0, ids = [], []
        for episode in range(count):
            child = children[cursor]
            rng = np.random.default_rng(child)
            x0.append(plant.initial_state(rng))
            u[episode] = excitation(rng, d["steps"], config["excitation"])
            noise[episode] = rng.normal(0, d["measurement_noise_std"], d["steps"] + 1)
            ids.append(f"{split}-{episode:03d}")
            manifest.append({"id": ids[-1], "split": split, "spawn_key": list(child.spawn_key),
                             "seed_words": child.generate_state(4).tolist()})
            cursor += 1
        x = plant.simulate(u, np.array(x0))
        clean = plant.output(x)
        data[f"{split}_u"], data[f"{split}_y"], data[f"{split}_ids"] = u, clean + noise, np.array(ids)
        data[f"{split}_y_limpo"] = clean
        states[f"{split}_x"] = x
    return data, states, manifest


def save_dataset(output, data, states, manifest, config_bytes):
    """Grava o desenvolvimento e o teste final em arquivos separados; recusa pasta existente.

    O teste final (com a saída sem ruído e os estados dele) fica em `teste-final/` e só é lido por
    `load_final_test`. Estados e saídas sem ruído do desenvolvimento ficam em `diagnostico/`, fora dos modelos.
    """
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"A pasta {out} já existe. Escolha outra saída para preservar a execução.")
    (out / "diagnostico").mkdir(parents=True)
    (out / TEST_DIR).mkdir()
    noisy = any(not np.array_equal(data[f"{s}_y"], data[f"{s}_y_limpo"]) for s in SPLITS)
    development = {f"{s}_{k}": data[f"{s}_{k}"] for s in ("treino", "validacao") for k in ("u", "y", "ids")}
    test = {f"teste_final_{k}": data[f"teste_final_{k}"] for k in ("u", "y", "ids")}
    if noisy:
        test["teste_final_y_limpo"] = data["teste_final_y_limpo"]
        np.savez_compressed(out / "diagnostico" / "saida-sem-ruido-desenvolvimento.npz",
                            **{f"{s}_y_limpo": data[f"{s}_y_limpo"] for s in ("treino", "validacao")})
    np.savez_compressed(out / DEVELOPMENT_FILE, **development)
    np.savez_compressed(out / TEST_FILE, **test)
    if states:
        np.savez_compressed(out / "diagnostico" / "estados-desenvolvimento.npz",
                            **{k: v for k, v in states.items() if not k.startswith("teste_final")})
        np.savez_compressed(out / TEST_DIR / "estados-teste-final.npz",
                            **{k: v for k, v in states.items() if k.startswith("teste_final")})
    (out / "config.json").write_bytes(config_bytes)
    write_json(out / "normalizacao-treino.json", normalization_from_training(development))
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    write_json(out / MANIFEST_FILE, {
        "formato": DATA_FORMAT, "criado_em": now_iso(),
        "configuracao_sha256": sha256_text(config_bytes),
        "arquivos_sha256": {name: sha256_file(out / name) for name in files},
        "conteudo_sha256": {"desenvolvimento": arrays_sha256(development), "teste_final": arrays_sha256(test)},
        "ruido": noisy,
        "regra": "O teste final só é lido por protocolo_v2.load_final_test, com seleção congelada.",
        "trajetorias": manifest})
    return out


def generate_from_config(config_path, output):
    config_bytes = Path(config_path).read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    plant = PLANTS[config["plant"]["model"]](config)
    excitation = EXCITATIONS[config["excitation"]["type"]]
    data, states, manifest = generate_dataset(plant, excitation, config)
    return save_dataset(output, data, states, manifest, config_bytes)


def load_development(data_dir):
    """Treino e validação. Não abre nada em `teste-final/`."""
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / MANIFEST_FILE)
    if sha256_file(data_dir / DEVELOPMENT_FILE) != manifest["arquivos_sha256"][DEVELOPMENT_FILE]:
        raise ValueError("dados-desenvolvimento.npz não corresponde ao manifesto.")
    with np.load(data_dir / DEVELOPMENT_FILE) as f:
        development = {k: f[k] for k in f.files}
    return development, manifest


# ---------------------------------------------------------------------------------------------------
# Métricas e divergência

def metric_result(name, parameters, predictions, targets, settings, origins_per_episode):
    """`summarize_rollouts` do E001. Com valores não finitos, as linhas trazem RMSE infinito em vez de sumir."""
    with np.errstate(over="ignore", invalid="ignore"):
        result = summarize_rollouts(name, parameters, predictions, targets, settings, origins_per_episode)
    if not result["rows"]:
        std = float(targets.std())
        result["rows"] = [{"model": name, "horizon_steps": h, "rmse_endpoint_m": np.inf, "rmse_trajectory_m": np.inf,
                           "nrmse_endpoint": np.inf, "nrmse_trajectory": np.inf, "normalization_std_m": std,
                           "rollouts": len(predictions), "nonfinite_values": result["nonfinite_values"],
                           "parameters": parameters} for h in settings["horizons"]]
    return result


def trajectory_squared_errors(predictions, targets, n_trajectories):
    """Soma, sobre as origens de cada trajetória, do erro quadrático em cada passo: (trajetórias, K).

    Valor não finito vira infinito, para que a trajetória conte como divergente no bootstrap.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        squared = (predictions - targets) ** 2
    squared = np.where(np.isfinite(squared), squared, np.inf)
    return squared.reshape(n_trajectories, -1, squared.shape[1]).sum(axis=1)


def divergence_limit(development, factor):
    """Previsão acima de `factor` vezes a maior |y| medida no treino conta como divergência."""
    return factor * float(np.max(np.abs(development["treino_y"])))


def divergence_report(predictions, limit, n_trajectories):
    with np.errstate(invalid="ignore"):
        bad = ~np.isfinite(predictions) | (np.abs(predictions) > limit)
    per_rollout = bad.any(axis=1)
    per_trajectory = per_rollout.reshape(n_trajectories, -1).any(axis=1)
    return {"diverged": bool(per_rollout.any()), "divergent_rollouts": int(per_rollout.sum()),
            "affected_trajectories": [int(i) for i in np.flatnonzero(per_trajectory)],
            "nonfinite_values": int((~np.isfinite(predictions)).sum()), "limit_m": limit}


def evaluate_members(members, u, y, settings, y_clean=None, divergence_limit_m=None, batch_size=None):
    """Avalia cada membro (um por semente) nas mesmas origens e com as mesmas janelas.

    Devolve, por membro, as métricas do E001 contra a medição (e contra a saída sem ruído, se houver), as somas
    de erro quadrático por trajetória para o bootstrap e o relatório de divergência.
    """
    history, horizon = settings["history"], max(settings["horizons"])
    origins = evaluation_origins(u.shape[1], settings)
    windows = rollout_windows(u, y, origins, history, horizon)
    targets = rollout_targets(y, origins, horizon)
    clean_targets = rollout_targets(y_clean, origins, horizon) if y_clean is not None else None
    results = []
    for model in members:
        start = perf_counter()
        predictions = simulate_checked(model, *windows, batch_size=batch_size)
        entry = {"name": model.name, "parameters": model.parameters, "simulation_seconds": perf_counter() - start,
                 "measured": metric_result(model.name, model.parameters, predictions, targets, settings, len(origins)),
                 "squared_errors": trajectory_squared_errors(predictions, targets, len(u))}
        if clean_targets is not None:
            entry["clean"] = metric_result(model.name, model.parameters, predictions, clean_targets, settings,
                                           len(origins))
            entry["squared_errors_clean"] = trajectory_squared_errors(predictions, clean_targets, len(u))
        if divergence_limit_m is not None:
            entry["divergence"] = divergence_report(predictions, divergence_limit_m, len(u))
        results.append(entry)
    return {"origins": origins, "origins_per_episode": len(origins), "members": results}


# ---------------------------------------------------------------------------------------------------
# Validação: nota de cada candidato e escolha

def validation_score(members, development, settings, batch_size=None):
    """Nota de validação de um candidato: média, entre os membros, do RMSE da trajetória no horizonte de escolha.

    Se algum membro divergir, o candidato é descartado e o motivo fica registrado.
    """
    limit = divergence_limit(development, settings["divergence_factor"])
    evaluation = evaluate_members(members, development["validacao_u"], development["validacao_y"], settings,
                                  divergence_limit_m=limit, batch_size=batch_size)
    horizon = settings["selection_horizon"]
    per_member = []
    for m in evaluation["members"]:
        row = next(r for r in m["measured"]["rows"] if r["horizon_steps"] == horizon)
        per_member.append({"model": m["name"], "rmse_trajectory_m": row["rmse_trajectory_m"],
                           "divergence": m["divergence"]})
    diverged = [p["model"] for p in per_member if p["divergence"]["diverged"]]
    score = None if diverged else float(np.mean([p["rmse_trajectory_m"] for p in per_member]))
    return {"score": score, "discarded": bool(diverged),
            "reason": f"divergência na validação: {', '.join(diverged)}" if diverged else None,
            "parameters": members[0].parameters, "horizon_steps": horizon, "members": per_member}


def select_candidate(candidates, tie_tolerance):
    """Menor nota; empate numérico (dentro de `tie_tolerance`) vai para menos parâmetros e depois a ordem da grade."""
    valid = [i for i, c in enumerate(candidates)
             if not c.get("discarded") and c.get("score") is not None and np.isfinite(c["score"])]
    discarded = [{"index": i, "reason": c.get("reason")} for i, c in enumerate(candidates) if i not in valid]
    if not valid:
        raise RuntimeError("Nenhum candidato válido na validação.")
    best = min(candidates[i]["score"] for i in valid)
    tied = [i for i in valid if candidates[i]["score"] <= best + tie_tolerance]
    chosen = min(tied, key=lambda i: (candidates[i]["parameters"], i))
    return chosen, {"selected": chosen, "best_score": best, "tied": tied, "discarded": discarded}


# ---------------------------------------------------------------------------------------------------
# Incerteza: bootstrap por trajetória, combinado com as sementes

def percentile_interval(values, confidence):
    """Percentis empíricos (sem interpolação, aceita infinito). Com NaN (infinito menos infinito), não há intervalo."""
    if np.isnan(values).any():
        return None
    alpha = 1 - confidence
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2], method="inverted_cdf")
    return [float(low), float(high)]


def equivalence_verdict(interval):
    """Regra de H3 para a diferença a − b: só decide se o intervalo inteiro estiver de um lado do zero."""
    if interval is None:
        return "indeterminado"
    if interval[1] < 0:
        return "a_menor"
    if interval[0] > 0:
        return "b_menor"
    return "equivalentes"


def bootstrap_families(errors, origins_per_trajectory, horizon, kind="trajectory", resamples=10000, seed=0,
                       confidence=0.95, pairs=()):
    """Intervalos para o RMSE de cada família e para diferenças pareadas.

    errors: {família: (membros, trajetórias, K)}, somas de erro quadrático por trajetória e passo
    (`trajectory_squared_errors`), com as mesmas trajetórias de teste em todas as famílias.
    kind: "trajectory" (todos os passos até `horizon`) ou "endpoint" (só o passo `horizon`).

    Em cada reamostragem: sorteiam-se, com reposição, trajetórias inteiras, as mesmas para todas as famílias
    (por isso a diferença é pareada); sorteiam-se, com reposição e por família, os membros (sementes); o RMSE de
    cada membro é calculado nas trajetórias sorteadas e a nota da família é a média sobre os membros sorteados.
    Com um membro (tradicionais), resta o bootstrap por trajetória. A estimativa pontual é a média, entre os
    membros, do RMSE em todas as trajetórias.
    """
    names = sorted(errors)
    arrays = {name: np.asarray(errors[name], dtype=np.float64) for name in names}
    n_trajectories = {a.shape[1] for a in arrays.values()}
    if len(n_trajectories) != 1:
        raise ValueError("Todas as famílias precisam das mesmas trajetórias.")
    n_trajectories = n_trajectories.pop()
    rng = np.random.default_rng(seed)
    trajectory_index = rng.integers(0, n_trajectories, size=(resamples, n_trajectories))
    count = n_trajectories * origins_per_trajectory * (horizon if kind == "trajectory" else 1)
    replicates, families = {}, {}
    for name in names:
        a = arrays[name]
        values = a[:, :, :horizon].sum(axis=-1) if kind == "trajectory" else a[:, :, horizon - 1]
        with np.errstate(over="ignore", invalid="ignore"):
            member_point = np.sqrt(values.sum(axis=1) / count)
            member_replicates = np.sqrt(values[:, trajectory_index].sum(axis=-1) / count).T  # (B, membros)
        member_index = rng.integers(0, len(values), size=(resamples, len(values)))
        replicates[name] = np.take_along_axis(member_replicates, member_index, axis=1).mean(axis=1)
        families[name] = {"point": float(member_point.mean()), "members": member_point.tolist(),
                          "seed_std": float(member_point.std(ddof=1)) if len(values) > 1 else None,
                          "interval": percentile_interval(replicates[name], confidence)}
    comparisons = []
    for a, b in pairs:
        with np.errstate(invalid="ignore"):
            difference = replicates[a] - replicates[b]
            point = families[a]["point"] - families[b]["point"]
        interval = percentile_interval(difference, confidence)
        comparisons.append({"a": a, "b": b, "difference": point, "interval": interval,
                            "verdict": equivalence_verdict(interval)})
    return {"families": families, "comparisons": comparisons, "replicates": replicates,
            "settings": {"horizon_steps": horizon, "kind": kind, "resamples": resamples, "seed": seed,
                         "confidence": confidence, "trajectories": n_trajectories,
                         "origins_per_trajectory": origins_per_trajectory}}


def h3_verdict(comparisons, network, competitors):
    """H3 num cenário: a rede tem erro menor que cada competidor, com o intervalo da diferença abaixo de zero."""
    found = {(c["a"], c["b"]): c for c in comparisons}
    verdicts = {}
    for competitor in competitors:
        if (network, competitor) in found:
            verdicts[competitor] = found[(network, competitor)]["verdict"]
        else:
            flipped = {"a_menor": "b_menor", "b_menor": "a_menor"}
            verdicts[competitor] = flipped.get(found[(competitor, network)]["verdict"],
                                               found[(competitor, network)]["verdict"])
    return {"network": network, "verdicts": verdicts, "sustained": all(v == "a_menor" for v in verdicts.values())}


# ---------------------------------------------------------------------------------------------------
# Seleção congelada e acesso ao teste final

def freeze_selection(path, *, scenario, data_dir, families, best_traditional, comparisons=(), h3=None,
                     code_files=DEFAULT_CODE_FILES, code_root=ROOT, notes=None):
    """Grava o arquivo de seleção congelada. Recusa sobrescrever.

    families: {nome: {"adaptador": nome registrado ou "modulo:funcao",
                      "escolha": hiperparâmetros escolhidos,
                      "nota_validacao": {"rmse_trajetoria_m": ..., "horizonte_passos": 50, ...},
                      "membros": [{"semente": int ou None, "artefatos": {rótulo: caminho}}]}}
    Os caminhos dos artefatos são relativos à pasta do arquivo de seleção; o hash de cada um é gravado.
    """
    path, data_dir = Path(path), Path(data_dir)
    if path.exists():
        raise FileExistsError(f"{path} já existe. Uma seleção congelada não é sobrescrita; grave outra versão.")
    manifest = read_json(data_dir / MANIFEST_FILE)
    if best_traditional not in families:
        raise ValueError("`best_traditional` precisa ser uma das famílias.")
    frozen_families = {}
    for name, entry in families.items():
        missing = {"adaptador", "escolha", "nota_validacao", "membros"} - set(entry)
        if missing or not entry["membros"]:
            raise ValueError(f"Família {name}: faltam {sorted(missing) or ['membros']}.")
        members = []
        for member in entry["membros"]:
            artifacts = {label: {"caminho": Path(p).as_posix(), "sha256": sha256_file(path.parent / p)}
                         for label, p in member.get("artefatos", {}).items()}
            members.append({"semente": member.get("semente"), "artefatos": artifacts})
        frozen_families[name] = {**entry, "membros": members}
    selection = {
        "formato": SELECTION_FORMAT, "congelada": True, "cenario": scenario, "criada_em": now_iso(),
        "criterio": "RMSE da trajetória simulada no horizonte de escolha, na validação, contra a medição",
        "dados": {"diretorio": relative_path(data_dir, path.parent),
                  "desenvolvimento_sha256": manifest["arquivos_sha256"][DEVELOPMENT_FILE],
                  "teste_final_sha256": manifest["arquivos_sha256"][TEST_FILE],
                  "configuracao_sha256": manifest["configuracao_sha256"]},
        "codigo_sha256": {Path(f).as_posix(): sha256_file(Path(code_root) / f) for f in code_files},
        "familias": frozen_families, "melhor_tradicional": best_traditional,
        "comparacoes": [list(pair) for pair in comparisons], "h3": h3, "notas": notes}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, selection)
    return selection


def load_frozen_selection(path, code_root=ROOT, check_code=True):
    """Lê e confere a seleção: formato, campos, hashes do código e dos artefatos."""
    path = Path(path)
    selection = read_json(path)
    if selection.get("formato") != SELECTION_FORMAT or selection.get("congelada") is not True:
        raise ValueError(f"{path} não é uma seleção congelada no formato {SELECTION_FORMAT}.")
    for key in ("cenario", "dados", "codigo_sha256", "familias", "melhor_tradicional"):
        if key not in selection:
            raise ValueError(f"Seleção congelada sem o campo `{key}`.")
    if check_code:
        changed = [f for f, digest in selection["codigo_sha256"].items()
                   if not (Path(code_root) / f).is_file() or sha256_file(Path(code_root) / f) != digest]
        if changed:
            raise ValueError("Código alterado depois do congelamento: " + ", ".join(changed)
                             + ". Congele uma nova seleção (outro arquivo) antes de ler o teste final.")
    for name, entry in selection["familias"].items():
        for member in entry["membros"]:
            for label, artifact in member["artefatos"].items():
                if sha256_file(path.parent / artifact["caminho"]) != artifact["sha256"]:
                    raise ValueError(f"Artefato alterado depois do congelamento: {name}/{label}.")
    return selection


def selection_data_dir(selection, selection_path):
    return (Path(selection_path).parent / selection["dados"]["diretorio"]).resolve()


def load_final_test(data_dir, selection_path, caller=None, justification=None, code_root=ROOT):
    """Única porta de leitura do teste final.

    Exige a seleção congelada destes dados e acrescenta uma linha ao registro de uso antes de abrir o arquivo
    (data, chamador, hash da seleção). Uma segunda leitura só acontece com justificativa, que fica no registro.
    A proteção é de procedimento: o arquivo pode ser aberto por fora, mas todo uso legítimo fica registrado.
    """
    selection_path, data_dir = Path(selection_path), Path(data_dir)
    if not selection_path.is_file():
        raise FileNotFoundError(f"O teste final só é lido com a seleção congelada; não encontrei {selection_path}.")
    selection = load_frozen_selection(selection_path, code_root=code_root)
    manifest = read_json(data_dir / MANIFEST_FILE)
    if selection["dados"]["desenvolvimento_sha256"] != manifest["arquivos_sha256"][DEVELOPMENT_FILE]:
        raise ValueError("A seleção congelada foi feita com outros dados de desenvolvimento.")
    test_sha = sha256_file(data_dir / TEST_FILE)
    if test_sha != manifest["arquivos_sha256"][TEST_FILE] or test_sha != selection["dados"]["teste_final_sha256"]:
        raise ValueError("O arquivo do teste final não corresponde ao manifesto ou à seleção congelada.")
    log = data_dir / USAGE_LOG
    previous = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if log.exists() else []
    if previous and not justification:
        raise PermissionError(f"O teste final já foi lido {len(previous)} vez(es), a primeira em "
                              f"{previous[0]['data']}. Nova leitura só com justificativa, que fica registrada.")
    if caller is None:
        frame = inspect.stack()[1]
        caller = f"{Path(frame.filename).name}:{frame.function}"
    entry = {"data": now_iso(), "chamador": caller, "selecao": selection_path.name,
             "selecao_sha256": sha256_file(selection_path), "teste_final_sha256": test_sha,
             "uso": len(previous) + 1, "justificativa": justification}
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with np.load(data_dir / TEST_FILE) as f:
        return {k: f[k] for k in f.files}, entry


# ---------------------------------------------------------------------------------------------------
# Adaptadores registrados: constroem os membros de cada família a partir da seleção congelada

ADAPTERS: dict[str, Callable] = {}


def register_adapter(name):
    """Registra `loader(entry, member, context) -> modelo do contrato`.

    context = {"selection_dir": pasta da seleção, "development": treino e validação, "config": configuração}.
    """
    def decorate(loader):
        ADAPTERS[name] = loader
        return loader
    return decorate


def resolve_adapter(spec):
    """Nome registrado, ou "modulo:funcao" de um módulo de experimentos/ (cujo hash deve estar na seleção)."""
    if spec in ADAPTERS:
        return ADAPTERS[spec]
    module_name, separator, function = spec.partition(":")
    if not separator:
        raise KeyError(f"Adaptador desconhecido: {spec}")
    module = importlib.import_module(module_name)
    if Path(module.__file__).resolve().parent != Path(__file__).resolve().parent:
        raise ValueError(f"O adaptador {spec} precisa estar em experimentos/.")
    return getattr(module, function)


def artifact_path(member, label, context):
    return Path(context["selection_dir"]) / member["artefatos"][label]["caminho"]


def build_members(entry, context):
    loader = resolve_adapter(entry["adaptador"])
    return [loader(entry, member, context) for member in entry["membros"]]


@register_adapter("persistencia")
def load_persistence(entry, member, context):
    return StepModelAdapter(Persistence())


@register_adapter("arx_medido")
def load_arx(entry, member, context):
    """ARX do E001 gravado por `ARX.to_dict`, com os atrasos iniciais medidos (sem estimação do estado inicial)."""
    spec = read_json(artifact_path(member, "modelo", context))
    return StepModelAdapter(ARX(spec["output_order"], spec["input_order"], spec["coefficients"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("gerar", help="gera e grava os dados de um cenário (pasta nova)")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "gerar":
        print(f"Dados gravados em {generate_from_config(args.config, args.output).resolve()}")
