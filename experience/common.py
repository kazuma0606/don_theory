"""
common.py — 全実験で共有する共通コンポーネント

config.yaml を読み込み、以下を提供する：
  - cfg           : 設定辞書（全パラメータ）
  - device        : torch.device("cuda")
  - Dynamics      : 時間発展モジュール
  - build_interventions(d) : (E1, E2, a, M_base) を返す
  - build_observer(d, q)   : 観測モジュール C を返す
  - rollout(...)  : 時間発展 + 介入 + 観測
  - smooth_time(zs, kernel): 時間方向 1D 畳み込み
  - make_kernel(eps)        : ε 依存カーネル（Exp5 用）
  - build_target(...)       : θ* でのターゲット軌跡生成
  - loss_raw(theta, ...)    : 非平滑損失
  - loss_smooth(theta, ...) : 平滑損失
  - save_model(module, name): models/ に保存
  - load_model(module, name): models/ から読み込み
"""

from math import comb
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# ============================================================
# 設定読み込み
# ============================================================

_HERE = Path(__file__).parent
_CONFIG_PATH = _HERE / "config.yaml"

with open(_CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

RESULTS_DIR = _HERE / "results"
MODELS_DIR  = _HERE / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================
# デバイス（CUDA 前提）
# ============================================================

device = torch.device("cuda")

# ============================================================
# 0.2 Dynamics
# p_{t+1} = linear(p) + gamma * tanh(W(p))
# ============================================================

class Dynamics(nn.Module):
    """時間発展モジュール。
    p_{t+1} = A p_t + b + γ tanh(W p_t)
    linear: A p + b（bias 込み）、W: tanh 項（bias なし）
    """

    def __init__(self, d: int, gamma: float = None, seed: int = None):
        super().__init__()
        gamma = gamma if gamma is not None else cfg["gamma"]
        if seed is not None:
            torch.manual_seed(seed)
        self.linear = nn.Linear(d, d)
        self.W      = nn.Linear(d, d, bias=False)
        self.gamma  = gamma

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return self.linear(p) + self.gamma * torch.tanh(self.W(p))


# ============================================================
# 0.3 介入
# E1: p + θ₁ a  （加算型）
# E2: p + θ₂ (M_base @ p)  = (I + θ₂ M) p  （線形変換型）
# ============================================================

def build_interventions(d: int):
    """介入関数 E1, E2 と固定パラメータ a, M_base を構築して返す。

    Returns:
        E1      : (p, theta1) -> p + theta1 * a
        E2      : (p, theta2) -> p + theta2 * (M_base @ p)
        a       : 加算ベクトル, shape (d,), device=cuda
        M_base  : 固定ランダム行列, shape (d, d), device=cuda
    """
    a = torch.zeros(d, device=device)
    a[: d // 4] = 1.0  # 次元 0:d//4 のみ 1.0

    torch.manual_seed(cfg["M_base_seed"])
    M_base = (torch.randn(d, d) * cfg["M_base_scale"]).to(device)

    def E1(p: torch.Tensor, theta1: torch.Tensor) -> torch.Tensor:
        return p + theta1 * a

    def E2(p: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
        return p + theta2 * (p @ M_base.T)  # batch 対応: (batch,d) @ (d,d)^T

    return E1, E2, a, M_base


# ============================================================
# 0.4 観測
# ============================================================

def build_observer(d: int, q: int = None) -> nn.Linear:
    """観測モジュール C: ℝ^d → ℝ^q を構築して返す。"""
    q = q if q is not None else cfg["obs_dim"]
    return nn.Linear(d, q).to(device)


# ============================================================
# 0.5 rollout
# ============================================================

def rollout(
    p0: torch.Tensor,
    theta: torch.Tensor,
    dynamics: nn.Module,
    schedule: dict,
    T: int,
    observe,
) -> torch.Tensor:
    """時間発展 + 介入 + 観測を T ステップ実行する。

    Args:
        p0      : 初期状態, shape (batch, d)
        theta   : パラメータ (θ₁, θ₂), shape (2,)
        dynamics: Dynamics インスタンス
        schedule: {t: [(E_func, theta_idx), ...]}
                  theta_idx は 0 (θ₁) または 1 (θ₂)
        T       : タイムステップ数
        observe : 観測関数 p -> z, shape (batch, q)

    Returns:
        zs: shape (T+1, batch, q)
    """
    p = p0
    zs = []
    for t in range(T + 1):
        if t in schedule:
            for (E, idx) in schedule[t]:
                p = E(p, theta[idx])
        zs.append(observe(p))
        if t < T:
            p = dynamics(p)
    return torch.stack(zs, dim=0)


# ============================================================
# 0.6 smooth_time — 時間方向 1D 畳み込み
# ============================================================

def smooth_time(zs: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """zs を時間軸方向に kernel で畳み込む。

    Args:
        zs    : shape (T+1, batch, q)
        kernel: 1D カーネル, shape (K,), 正規化済み

    Returns:
        shape (T+1, batch, q)
    """
    # (T+1, batch, q) -> (batch*q, 1, T+1) for channel-wise conv
    T1, B, q = zs.shape
    zs_t = zs.permute(1, 2, 0).reshape(B * q, 1, T1)
    k = kernel.view(1, 1, -1).to(zs.device)
    pad = (kernel.numel() // 2,) * 2
    zs_padded = F.pad(zs_t, pad, mode="replicate")
    zs_smooth = F.conv1d(zs_padded, k)  # (B*q, 1, T+1)
    return zs_smooth.view(B, q, T1).permute(2, 0, 1)


def make_kernel(eps: float) -> torch.Tensor:
    """ε に比例した幅の二項係数カーネルを生成する（Exp5 用）。

    Args:
        eps: スケール係数（大きいほど幅広いカーネル）

    Returns:
        正規化済みカーネル Tensor, shape (K,), device=cpu
    """
    K = max(3, 2 * round(eps * 10) + 1)  # 奇数幅
    binom = torch.tensor(
        [float(comb(K - 1, k)) for k in range(K)], dtype=torch.float32
    )
    return binom / binom.sum()


# ============================================================
# 0.7 デフォルトカーネルとターゲット軌跡
# ============================================================

def build_default_kernel() -> torch.Tensor:
    """config.yaml の kernel_weights から正規化済みカーネルを返す。"""
    w = torch.tensor(cfg["kernel_weights"], dtype=torch.float32)
    return (w / w.sum()).to(device)


def build_target(dynamics: nn.Module, observe, C_obs: nn.Module = None):
    """θ* でのターゲット軌跡・固定初期状態・E1/E2 を構築して返す。

    Returns:
        p0_fixed  : shape (batch, d), device=cuda
        zs_target : shape (T+1, batch, q), detach 済み, device=cuda
        E1, E2    : 介入関数
        a, M_base : 介入の固定パラメータ
        schedule  : DEFAULT_SCHEDULE
        kernel    : デフォルトカーネル
    """
    d      = cfg["state_dim"]
    T      = cfg["T"]
    batch  = cfg["batch"]

    E1, E2, a, M_base = build_interventions(d)

    schedule = {
        cfg["E1_timestep"]: [(E1, 0)],
        cfg["E2_timestep"]: [(E2, 1)],
    }

    kernel = build_default_kernel()

    torch.manual_seed(cfg["p0_seed"])
    p0_fixed = torch.randn(batch, d, device=device)

    theta_star = torch.tensor(cfg["theta_star"], device=device)

    with torch.no_grad():
        zs_target = rollout(p0_fixed, theta_star, dynamics, schedule, T, observe)

    return p0_fixed, zs_target.detach(), E1, E2, a, M_base, schedule, kernel


# ============================================================
# 0.8 損失関数
# ============================================================

def loss_raw(
    theta: torch.Tensor,
    p0: torch.Tensor,
    zs_target: torch.Tensor,
    dynamics: nn.Module,
    schedule: dict,
    T: int,
    observe,
) -> torch.Tensor:
    """非平滑損失 J(θ) = mean‖z(θ) - z_target‖²"""
    zs = rollout(p0, theta, dynamics, schedule, T, observe)
    return ((zs - zs_target) ** 2).mean()


def loss_smooth(
    theta: torch.Tensor,
    p0: torch.Tensor,
    zs_target: torch.Tensor,
    dynamics: nn.Module,
    schedule: dict,
    T: int,
    observe,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """平滑損失 J_ε(θ) = mean‖smooth(z(θ)) - z_target‖²
    ターゲットは平滑化しない（論文 J_ε 定義に従う）。
    """
    zs   = rollout(p0, theta, dynamics, schedule, T, observe)
    zs_s = smooth_time(zs, kernel)
    return ((zs_s - zs_target) ** 2).mean()


# ============================================================
# モデル保存・読み込み
# ============================================================

def save_model(module: nn.Module, name: str) -> Path:
    """モデルの state_dict を models/<name>.pt に保存する。"""
    path = MODELS_DIR / f"{name}.pt"
    torch.save(module.state_dict(), path)
    return path


def load_model(module: nn.Module, name: str) -> nn.Module:
    """models/<name>.pt から state_dict を読み込む。"""
    path = MODELS_DIR / f"{name}.pt"
    module.load_state_dict(torch.load(path, map_location=device))
    return module


# ============================================================
# 0.9 非可換性サニティチェック
# ============================================================

def check_noncommutativity(d: int = None) -> float:
    """E1 ∘ E2 ≠ E2 ∘ E1 であることを確認する。

    Returns:
        diff (float): ‖E1(E2(p)) - E2(E1(p))‖
    """
    d = d if d is not None else cfg["state_dim"]
    E1, E2, _, _ = build_interventions(d)

    p_test = torch.randn(d, device=device)
    t1, t2 = 1.0, 1.0

    diff = (
        E1(E2(p_test, torch.tensor(t2, device=device)), torch.tensor(t1, device=device))
        - E2(E1(p_test, torch.tensor(t1, device=device)), torch.tensor(t2, device=device))
    ).norm().item()

    assert diff > 1e-3, f"非可換性が弱すぎます: {diff:.6f}"
    print(f"[sanity] ||E1*E2(p) - E2*E1(p)|| = {diff:.4f}  (expected ~= theta1*theta2*||Ma||)")
    return diff


if __name__ == "__main__":
    # --- 動作確認 ---
    print("=== common.py sanity check ===")
    print(f"device : {device}")
    print(f"config : d={cfg['state_dim']}, T={cfg['T']}, batch={cfg['batch']}")

    # Dynamics
    torch.manual_seed(cfg["dynamics_seed"])
    dyn = Dynamics(cfg["state_dim"]).to(device)
    save_model(dyn, "dynamics_seed0")
    print(f"Dynamics  : saved to models/dynamics_seed0.pt")

    # Observer
    obs_module = build_observer(cfg["state_dim"])
    save_model(obs_module, "observer_seed0")
    observe_fn = obs_module

    # 非可換性チェック
    check_noncommutativity()

    # ターゲット軌跡
    p0_fixed, zs_target, E1, E2, a, M_base, schedule, kernel = build_target(
        dyn, observe_fn
    )
    print(f"p0_fixed  : {p0_fixed.shape}")
    print(f"zs_target : {zs_target.shape}")

    # 損失確認
    theta_star = torch.tensor(cfg["theta_star"], device=device)
    L_raw    = loss_raw(theta_star, p0_fixed, zs_target, dyn, schedule, cfg["T"], observe_fn)
    L_smooth = loss_smooth(theta_star, p0_fixed, zs_target, dyn, schedule, cfg["T"], observe_fn, kernel)
    print(f"loss_raw(theta*)    = {L_raw.item():.6f}  (~= 0 expected)")
    print(f"loss_smooth(theta*) = {L_smooth.item():.6f}  (small value expected)")

    print("=== OK ===")
