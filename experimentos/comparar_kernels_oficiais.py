"""E011: compara `mamba3_referencia.Mamba3SISO` com o módulo oficial `mamba_ssm.Mamba3` (SISO, kernels Triton).

Roda só numa GPU NVIDIA com `mamba_ssm` instalado do commit fixado (no TCC: WSL2 com Ubuntu, RTX 4060).
Os dois módulos recebem o mesmo `state_dict` e a mesma entrada; compara-se a saída do `forward` e os
gradientes em relação à entrada e aos parâmetros. O `step()` oficial não é usado.

Três versões do nosso bloco entram na comparação:
- float32, como nos ensaios do grupo;
- float64, como referência de alta precisão (a função dos autores ainda força ADT, DT e o ângulo
  acumulado para float32);
- float32 com Q, K, V, Trap, ângulos e Z arredondados para bfloat16 antes da função de referência,
  imitando as conversões de `mamba3_siso_combined`. Separa o erro do arredondamento das entradas
  do erro das contas internas do kernel.

Casos: pesos aleatórios (três sementes) e os pesos treinados de `resultados/mamba3-v1/` (configuração
selecionada, três sementes), estes com janelas dos dados de treino do E001.

Uso, dentro do WSL2, a partir da raiz do repositório:
    python experimentos/comparar_kernels_oficiais.py --out resultados/mamba3-kernels-oficiais-v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experimentos"))

from mamba3_referencia import Mamba3SISO, mamba3_siso_fwd_ref  # noqa: E402

UPSTREAM_COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"
BLOCK = {"d_model": 64, "d_state": 64, "expand": 2, "headdim": 64}
BATCH = 32
LENGTHS = (256, 512)
RANDOM_SEEDS = (0, 1, 2)
TRAINED_DIR = ROOT / "resultados" / "mamba3-v1"
DATA_DIR = ROOT / "resultados" / "linear-v1"


# ----------------------------------------------------------------------------- métricas
def p95_relative(ker: torch.Tensor, ref: torch.Tensor, eps=1e-6, ref_mag_mask=1e-2, p=0.95) -> float:
    """Mesma métrica de `relative_error` em tests/ops/triton/test_mamba3_siso.py (commit fixado):
    quantil 95% de |ker - ref| / (|ref| + eps), só onde |ref| >= 1e-2."""
    ker, ref = ker.detach().double(), ref.detach().double()
    mask = ref.abs() >= ref_mag_mask
    if not mask.any():
        return 0.0
    vals = ((ker - ref).abs()[mask] / (ref.abs()[mask] + eps)).flatten()
    k = max(1, min(vals.numel(), int(math.ceil(p * vals.numel()))))
    return vals.kthvalue(k).values.item()


def errors(a: torch.Tensor, ref: torch.Tensor) -> dict:
    a, ref = a.detach().double(), ref.detach().double()
    diff = a - ref
    ref_norm = ref.norm().item()
    return {
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "ref_max_abs": ref.abs().max().item(),
        "ref_rms": ref.pow(2).mean().sqrt().item(),
        "rel_max": diff.abs().max().item() / ref.abs().max().item(),
        "rel_l2": diff.norm().item() / ref_norm if ref_norm > 0 else 0.0,
        "p95_rel": p95_relative(a, ref),
        "finite": bool(torch.isfinite(a).all().item()),
    }


# ----------------------------------------------------------------------------- modelos
def official_block(device):
    from mamba_ssm import Mamba3  # importa só aqui: o módulo exige GPU NVIDIA e Triton
    return Mamba3(**BLOCK, device=device)  # is_mimo=False (SISO), rope_fraction=0.5, chunk_size=64


def check_state_dicts(official: nn.Module, ours: nn.Module) -> dict:
    a, b = official.state_dict(), ours.state_dict()
    shapes_a = {k: list(v.shape) for k, v in a.items()}
    shapes_b = {k: list(v.shape) for k, v in b.items()}
    dtypes_a = {k: str(v.dtype) for k, v in a.items()}
    return {
        "same_keys": sorted(a) == sorted(b),
        "only_official": sorted(set(a) - set(b)),
        "only_ours": sorted(set(b) - set(a)),
        "same_shapes": all(shapes_a[k] == shapes_b.get(k) for k in shapes_a),
        "parameters": {k: {"shape": shapes_a[k], "dtype": dtypes_a[k]} for k in sorted(a)},
    }


class Bf16InputsRef(nn.Module):
    """Nosso bloco em float32, com as conversões para bfloat16 que `mamba3_siso_combined` faz nas entradas."""

    def __init__(self, block: Mamba3SISO):
        super().__init__()
        self.block = block

    def forward(self, u):
        args, z = self.block._project(u)
        for name in ("Q", "K", "V", "Trap", "Angles", "Z"):
            if args[name] is not None:
                args[name] = args[name].to(torch.bfloat16).to(torch.float32)
        y, _ = mamba3_siso_fwd_ref(**args, chunk_size=self.block.chunk_size, dtype=torch.float32)
        return self.block._output(y, z, u.dtype)


class Regressor(nn.Module):
    """Mesmos atributos de `experimento_mamba3.Mamba3Regressor` (n_layers=1), para carregar os pesos
    salvos com `strict=True`. `forward` repete o caminho de sequência de `Mamba3Regressor._run`:
    encoder → h + bloco(RMSNorm(h)) → RMSNorm final → head. O bloco pode ser trocado pelo oficial."""

    def __init__(self, d_model=64, **block):
        super().__init__()
        self.encoder = nn.Linear(2, d_model)
        self.norms = nn.ModuleList([nn.RMSNorm(d_model)])
        self.blocks = nn.ModuleList([Mamba3SISO(d_model, **block)])
        self.final_norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def block_input(self, x):
        return self.norms[0](self.encoder(x))

    def forward(self, x, block=None):
        h = self.encoder(x)
        h = h + (block or self.blocks[0])(self.norms[0](h))
        return self.head(self.final_norm(h)).squeeze(-1)


# ----------------------------------------------------------------------------- comparação
def run(module, u, g):
    """Saída e gradientes de (saída · g).sum() em relação à entrada e aos parâmetros."""
    module.zero_grad(set_to_none=True)
    u = u.detach().clone().requires_grad_(True)
    y = module(u)
    (y * g.to(y.dtype)).sum().backward()
    torch.cuda.synchronize()
    grads = {k: p.grad.detach().clone() for k, p in module.named_parameters() if p.grad is not None}
    return y.detach(), u.grad.detach(), grads


def compare_block(official, ours32, ours64, ours_bf16in, u, seed):
    g = torch.randn(u.shape, generator=torch.Generator(device=u.device).manual_seed(10_000 + seed),
                    device=u.device)
    y_off, du_off, dp_off = run(official, u, g)
    y_32, du_32, dp_32 = run(ours32, u, g)
    y_64, du_64, dp_64 = run(ours64, u.double(), g.double())
    with torch.no_grad():
        y_off2 = official(u)  # segunda chamada: o kernel é determinístico nesta GPU?
        y_bf = ours_bf16in(u)
    param_names = sorted(dp_off)
    return {
        "output": {
            "official_vs_ours_fp32": errors(y_off, y_32),
            "official_vs_ours_fp64": errors(y_off, y_64),
            "ours_fp32_vs_ours_fp64": errors(y_32, y_64),
            "ours_bf16_inputs_vs_ours_fp64": errors(y_bf, y_64),
            "official_run_to_run_max_abs": (y_off - y_off2).abs().max().item(),
        },
        "grad_input": {
            "official_vs_ours_fp32": errors(du_off, du_32),
            "official_vs_ours_fp64": errors(du_off, du_64),
            "ours_fp32_vs_ours_fp64": errors(du_32, du_64),
        },
        "grad_parameters_rel_l2": {
            name: {"official_vs_ours_fp32": errors(dp_off[name], dp_32[name])["rel_l2"],
                   "official_vs_ours_fp64": errors(dp_off[name], dp_64[name])["rel_l2"],
                   "ours_fp32_vs_ours_fp64": errors(dp_32[name], dp_64[name])["rel_l2"]}
            for name in param_names
        },
        "dtypes": {"official_output": str(y_off.dtype), "ours_fp32_output": str(y_32.dtype),
                   "ours_fp64_output": str(y_64.dtype)},
    }


def build_trio(state_dict, device):
    official = official_block(device)
    official.load_state_dict(state_dict, strict=True)
    ours32 = Mamba3SISO(**BLOCK).to(device)
    ours32.load_state_dict(state_dict, strict=True)
    ours64 = Mamba3SISO(**BLOCK).to(device).double()
    ours64.load_state_dict(state_dict, strict=True)
    return official, ours32, ours64, Bf16InputsRef(ours32)


def random_cases(device):
    cases = []
    for seed in RANDOM_SEEDS:
        torch.manual_seed(seed)
        reference = official_block(device)  # inicialização do próprio módulo oficial
        state_dict = {k: v.detach().clone() for k, v in reference.state_dict().items()}
        check = check_state_dicts(reference, Mamba3SISO(**BLOCK).to(device))
        trio = build_trio(state_dict, device)
        for length in LENGTHS:
            u = torch.randn(BATCH, length, BLOCK["d_model"],
                            generator=torch.Generator(device=device).manual_seed(seed), device=device)
            t0 = perf_counter()
            result = compare_block(*trio, u, seed)
            result.update(case="pesos aleatórios", seed=seed, length=length, batch=BATCH,
                          input="normal padrão, (32, L, 64)", seconds=perf_counter() - t0)
            cases.append(result)
            print(f"aleatório semente {seed} L={length}: saída rel_l2 oficial×nosso fp32 = "
                  f"{result['output']['official_vs_ours_fp32']['rel_l2']:.2e}", flush=True)
    return cases, check


def training_windows(length, n_windows, seed):
    """Janelas [u[t], y[t]] normalizadas dos dados de TREINO do E001 (o teste não é lido)."""
    import numpy as np
    data = np.load(DATA_DIR / "dados.npz")
    stats = json.loads((DATA_DIR / "normalizacao-treino.json").read_text())
    u, y = data["treino_u"], data["treino_y"]  # y tem uma amostra a mais; como em experimento_mamba3.training_windows
    x = np.stack([(u - stats["u"]["mean"]) / stats["u"]["std"],
                  (y[:, :-1] - stats["y"]["mean"]) / stats["y"]["std"]], axis=-1)
    rng = np.random.default_rng(seed)
    episodes = rng.integers(0, x.shape[0], n_windows)
    starts = rng.integers(0, x.shape[1] - length + 1, n_windows)
    windows = np.stack([x[e, s:s + length] for e, s in zip(episodes, starts)])
    return torch.tensor(windows, dtype=torch.float32), stats["y"]["std"]


def trained_cases(device):
    selection = json.loads((TRAINED_DIR / "selecao-validacao.json").read_text())
    selected = selection["selected"]
    config = json.loads((TRAINED_DIR / "config.json").read_text())
    network = config["network"]
    assert network["n_layers"] == 1 and {k: network[k] for k in BLOCK} == BLOCK
    cases = []
    for path in sorted(TRAINED_DIR.glob(f"modelos/{selected}-semente*.pt")):
        seed = int(path.stem.rsplit("semente", 1)[1])
        net = Regressor(d_model=network["d_model"], d_state=network["d_state"], expand=network["expand"],
                        headdim=network["headdim"]).to(device)
        net.load_state_dict(torch.load(path, map_location=device), strict=True)
        block_sd = {k: v.detach().clone() for k, v in net.blocks[0].state_dict().items()}
        official, ours32, ours64, ours_bf = build_trio(block_sd, device)
        net64 = Regressor(**{k: network[k] for k in ("d_model", "d_state", "expand", "headdim")}).to(device)
        net64.load_state_dict(net.state_dict(), strict=True)
        net64 = net64.double()
        for length in LENGTHS:
            x, y_std = training_windows(length, BATCH, seed)
            x = x.to(device)
            with torch.no_grad():
                u_block = net.block_input(x)
            t0 = perf_counter()
            result = compare_block(official, ours32, ours64, ours_bf, u_block, seed)
            with torch.no_grad():  # rede inteira: mesma janela, bloco oficial no lugar do nosso
                y_net_off = net(x, block=official)
                y_net_32 = net(x)
                y_net_64 = net64(x.double())
            net_err = {}
            for key, (a, b) in {"official_vs_ours_fp32": (y_net_off, y_net_32),
                                "official_vs_ours_fp64": (y_net_off, y_net_64),
                                "ours_fp32_vs_ours_fp64": (y_net_32, y_net_64)}.items():
                net_err[key] = errors(a, b)  # em unidades normalizadas de y; abaixo, em metros
                net_err[key]["max_abs_m"] = net_err[key]["max_abs"] * y_std
                net_err[key]["rms_m"] = (a.double() - b.double()).pow(2).mean().sqrt().item() * y_std
            result.update(case="pesos treinados", model=str(path.relative_to(ROOT)), seed=seed, length=length,
                          batch=BATCH, input=f"janelas dos dados de treino de {DATA_DIR.relative_to(ROOT)}",
                          network_output=net_err, seconds=perf_counter() - t0)
            cases.append(result)
            print(f"treinado semente {seed} L={length}: bloco rel_l2 = "
                  f"{result['output']['official_vs_ours_fp32']['rel_l2']:.2e}; rede máx |Δy| = "
                  f"{net_err['official_vs_ours_fp32']['max_abs_m']:.2e} m", flush=True)
    return cases, selected


# ----------------------------------------------------------------------------- registro
def sha256(path: Path) -> dict:
    """Hash do arquivo como está no disco e com fins de linha LF (o checkout no Windows usa CRLF;
    o hash LF é o comparável com os registros feitos no Mac e com o conteúdo do commit)."""
    data = path.read_bytes()
    return {"disk": hashlib.sha256(data).hexdigest(),
            "lf": hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()}


def versions() -> dict:
    import einops
    import triton
    import mamba_ssm
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "einops": einops.__version__,
        "mamba_ssm": mamba_ssm.__version__,
        "mamba_ssm_path": str(Path(mamba_ssm.__file__).parent),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "CC": os.environ.get("CC"),
    }
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        info["nvidia_smi"] = f"falhou: {exc}"
    clone = os.environ.get("MAMBA_CLONE")
    if clone:
        info["mamba_clone_head"] = subprocess.run(["git", "-C", clone, "rev-parse", "HEAD"],
                                                  capture_output=True, text=True).stdout.strip()
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "resultados" / "mamba3-kernels-oficiais-v1")
    parser.add_argument("--sem-treinados", action="store_true", help="só pesos aleatórios")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Precisa de GPU NVIDIA com CUDA.")
    device = torch.device("cuda")
    # Sem TF32 nas multiplicações em float32 do nosso bloco (é o padrão do PyTorch; fixado aqui por clareza).
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    t0 = perf_counter()
    random_results, sd_check = random_cases(device)
    trained_results, selected = ([], None) if args.sem_treinados else trained_cases(device)
    record = {
        "status": "concluido",
        "created_at": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
        "scope": "E011: Mamba3SISO (PyTorch, funções de referência) contra mamba_ssm.Mamba3 SISO (Triton); "
                 "forward e gradientes; step() oficial não usado",
        "upstream": f"state-spaces/mamba@{UPSTREAM_COMMIT}",
        "block_config": BLOCK,
        "batch": BATCH,
        "lengths": list(LENGTHS),
        "precision": {
            "official": "parâmetros e entrada em float32; mamba3_siso_combined converte Q, K, V, Trap, ângulos "
                        "e Z para bfloat16; multiplicações tl.dot com operandos bfloat16 e acumulação float32; "
                        "saída do kernel em bfloat16, convertida para float32 antes de out_proj",
            "ours_fp32": "float32 em tudo, TF32 desligado",
            "ours_fp64": "float64, exceto ADT, DT e ângulo acumulado, que a função dos autores força para float32",
            "ours_bf16_inputs": "float32, com Q, K, V, Trap, ângulos e Z arredondados para bfloat16 antes",
        },
        "metrics": {
            "max_abs": "max |a - b|",
            "rel_max": "max |a - b| / max |b|",
            "rel_l2": "||a - b||_2 / ||b||_2",
            "p95_rel": "quantil 95% de |a - b| / (|b| + 1e-6) onde |b| >= 1e-2 (relative_error dos testes oficiais)",
            "gradientes": "de (saída · g).sum(), g normal padrão com semente fixa, igual nos dois lados",
        },
        "state_dict_check": sd_check,
        "trained_selected": selected,
        "versions": versions(),
        "source_sha256": {name: sha256(ROOT / "experimentos" / name)
                          for name in ("comparar_kernels_oficiais.py", "mamba3_referencia.py",
                                       "experimento_mamba3.py")},
        "random_weights": random_results,
        "trained_weights": trained_results,
        "total_seconds": perf_counter() - t0,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "execucao.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    print(f"Gravado {args.out / 'execucao.json'} em {record['total_seconds']:.1f} s")


if __name__ == "__main__":
    main()
