"""
mollifier.py
============
Mollifier による離散 → C∞ 連続化の視覚化

- 離散的な介入パラメータ（ステップ関数）を
  Friedrichs Mollifier で平滑化する過程を示す
- ε → 0 での収束を視覚化
"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from pathlib import Path

IMG_DIR = Path(__file__).parent / "img"
IMG_DIR.mkdir(exist_ok=True)

# ── Mollifier の実装 ──────────────────────────────────────────────────────────

def mollifier_kernel(x: np.ndarray, eps: float) -> np.ndarray:
    """
    標準 Mollifier φ_ε(x)
      φ_ε(x) = C · exp(-1/(ε² - x²))  if |x| < ε
               0                         otherwise
    C は ∫φ_ε = 1 となる正規化定数
    """
    result = np.zeros_like(x, dtype=float)
    mask = np.abs(x) < eps
    xi = x[mask]
    denom = eps**2 - xi**2
    # 数値的安定性のためのクリップ
    denom = np.maximum(denom, 1e-15)
    result[mask] = np.exp(-1.0 / denom)
    # 正規化
    if result.sum() > 0:
        result /= result.sum() * (x[1] - x[0])
    return result


def mollify(f: np.ndarray, x: np.ndarray, eps: float) -> np.ndarray:
    """
    f_ε(x) = (f * φ_ε)(x)  （畳み込み）
    離散近似：数値積分
    """
    dx = x[1] - x[0]
    kernel = mollifier_kernel(x - x[len(x)//2], eps)
    # 畳み込み（numpy の convolve は "full" モード後に中心部を取る）
    conv = np.convolve(f, kernel * dx, mode="same")
    return conv


# ── 図1：Mollifier カーネル自体の形状 ────────────────────────────────────────

def plot_mollifier_kernel():
    x = np.linspace(-1.5, 1.5, 1000)
    epsilons = [1.0, 0.7, 0.4, 0.2]
    colors = ["#1565C0", "#2196F3", "#64B5F6", "#BBDEFB"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for eps, color in zip(epsilons, colors):
        kernel = mollifier_kernel(x, eps)
        # 正規化して高さを揃える（形の比較のため）
        if kernel.max() > 0:
            ax.plot(x, kernel / kernel.max(), color=color,
                    linewidth=2, label=f"ε = {eps}")

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("φ_ε(x)（正規化）", fontsize=12)
    ax.set_title(
        "Mollifier カーネル φ_ε(x) の形状\n"
        "ε → 0 でディラックのデルタ関数に収束（コンパクトサポート）",
        fontsize=12
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)

    plt.tight_layout()
    out = IMG_DIR / "04_mollifier_kernel.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"保存: {out}")


# ── 図2：離散介入パラメータの連続化 ──────────────────────────────────────────

def plot_smoothing():
    """
    離散的な介入パラメータ（ステップ関数）を
    Mollifier で平滑化していく過程。
    DL の embedding との違い：C∞ が証明されている。
    """
    N = 500
    x = np.linspace(0, 10, N)

    # 離散的な介入パラメータ（臨床での投薬量の段階的変化）
    f_discrete = np.zeros(N)
    f_discrete[(x >= 1) & (x < 3)]  = 0.3   # 低用量
    f_discrete[(x >= 3) & (x < 5)]  = 0.7   # 中用量
    f_discrete[(x >= 5) & (x < 6)]  = 1.0   # 高用量（ピーク）
    f_discrete[(x >= 6) & (x < 8)]  = 0.5   # 中用量（漸減）
    f_discrete[(x >= 8) & (x < 10)] = 0.1   # 低用量（維持）

    epsilons = [0.8, 0.4, 0.15, 0.05]
    colors   = ["#E53935", "#FF7043", "#FFA726", "#43A047"]
    labels   = [f"ε = {e}（粗い平滑化）" if e == max(epsilons)
                else f"ε = {e}（細かい平滑化）" if e == min(epsilons)
                else f"ε = {e}" for e in epsilons]

    fig, axes = plt.subplots(len(epsilons) + 1, 1, figsize=(10, 12), sharex=True)

    # 元の離散関数
    axes[0].step(x, f_discrete, where="post", color="black", linewidth=2)
    axes[0].set_ylabel("f(θ)（離散）", fontsize=10)
    axes[0].set_title(
        "Mollifier による離散介入パラメータの C∞ 連続化\n"
        "ε → 0 で元の関数に収束しながら、すべての段階で C∞ が保証される",
        fontsize=12
    )
    axes[0].fill_between(x, f_discrete, alpha=0.2, color="gray",
                         step="post", label="離散パラメータ（DL embedding の問題）")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.1, 1.2)

    # Mollifier 適用後
    for ax, eps, color, label in zip(axes[1:], epsilons, colors, labels):
        f_smooth = mollify(f_discrete, x, eps)
        ax.plot(x, f_smooth, color=color, linewidth=2, label=label)
        ax.step(x, f_discrete, where="post", color="gray",
                linewidth=0.8, alpha=0.4, linestyle="--")
        ax.set_ylabel("f_ε(θ)（C∞）", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.2)

        # C∞ であることを強調する注釈
        if eps == min(epsilons):
            ax.annotate(
                "C∞ 保証：無限回微分可能\n（勾配計算が数学的に正当）",
                xy=(7, 0.3), fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9", alpha=0.8)
            )

    axes[-1].set_xlabel("介入パラメータ θ（時間軸）", fontsize=11)

    plt.tight_layout()
    out = IMG_DIR / "05_mollifier_smoothing.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"保存: {out}")


# ── 図3：収束の定量的確認 ─────────────────────────────────────────────────────

def plot_convergence():
    """
    ε → 0 での収束を定量化。
    ‖f_ε - f‖_M0 = ‖f_ε - f‖∞ + ‖∇f_ε - ∇f‖∞ の変化。
    """
    N = 1000
    x = np.linspace(0, 10, N)
    dx = x[1] - x[0]

    f_discrete = np.zeros(N)
    f_discrete[(x >= 2) & (x < 5)] = 1.0
    f_discrete[(x >= 5) & (x < 8)] = 0.5

    epsilons = np.logspace(-1.5, 0, 40)
    errors_inf  = []  # ‖f_ε - f‖∞
    errors_grad = []  # ‖∇f_ε‖∞（滑らかさの指標）

    for eps in epsilons:
        f_smooth = mollify(f_discrete, x, eps)
        errors_inf.append(np.max(np.abs(f_smooth - f_discrete)))
        grad = np.gradient(f_smooth, dx)
        errors_grad.append(np.max(np.abs(grad)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(epsilons, errors_inf, "o-", color="#1565C0",
               linewidth=2, markersize=5, label=r"$\|f_\varepsilon - f\|_\infty$")
    ax1.set_xlabel("ε（Mollifier の幅）", fontsize=12)
    ax1.set_ylabel("一様ノルム誤差", fontsize=12)
    ax1.set_title(
        "収束性の確認\n"
        r"$\|f_\varepsilon - f\|_\infty \to 0$  as  $\varepsilon \to 0$",
        fontsize=12
    )
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.invert_xaxis()
    ax1.axvline(0.1, color="red", linestyle="--", alpha=0.5, label="実用的なε値")

    ax2.semilogx(epsilons, errors_grad, "s-", color="#E53935",
                 linewidth=2, markersize=5, label=r"$\|\nabla f_\varepsilon\|_\infty$（勾配の大きさ）")
    ax2.set_xlabel("ε（Mollifier の幅）", fontsize=12)
    ax2.set_ylabel("勾配ノルム（滑らかさの指標）", fontsize=12)
    ax2.set_title(
        "滑らかさの変化\nε が小さいほど急峻（元の不連続に近づく）",
        fontsize=12
    )
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.invert_xaxis()

    # ε の選択のトレードオフを注釈
    ax1.text(0.05, 0.35, "← 元の関数に近い\n   誤差大",
             transform=ax1.transAxes, fontsize=9, color="#E53935")
    ax1.text(0.6, 0.65, "← 平滑化過剰\n   誤差大",
             transform=ax1.transAxes, fontsize=9, color="#1565C0")

    plt.tight_layout()
    out = IMG_DIR / "06_mollifier_convergence.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"保存: {out}")


# ── メイン ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Mollifier の視覚化 ===")
    plot_mollifier_kernel()
    plot_smoothing()
    plot_convergence()
    print("完了。img/ ディレクトリを確認してください。")
