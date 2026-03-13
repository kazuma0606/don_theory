"""
noncommutativity.py
===================
MEDICUS 核心命題の視覚化：医療介入の非可換性

命題：E_a ∘ E_b(p) ≠ E_b ∘ E_a(p)

患者状態モデル：
  p = (腫瘍量, 免疫能, 組織完全性)  ∈ [0,1]³
  - 腫瘍量：低いほど良い
  - 免疫能：高いほど良い
  - 組織完全性：高いほど良い

非可換性の生物学的根拠：
  - 化療を先行すると免疫が抑制される
  - 免疫が抑制された状態での放射線治療は組織修復能力が落ちる
  → E_chemo ∘ E_radio ≠ E_radio ∘ E_chemo
"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from dataclasses import dataclass
from pathlib import Path

IMG_DIR = Path(__file__).parent / "img"
IMG_DIR.mkdir(exist_ok=True)


# ── 患者状態モデル ────────────────────────────────────────────────────────────

@dataclass
class PatientState:
    """患者状態ベクトル p = (腫瘍量, 免疫能, 組織完全性)"""
    tumor:  float  # [0,1]  低いほど良い
    immune: float  # [0,1]  高いほど良い
    tissue: float  # [0,1]  高いほど良い

    def clip(self) -> "PatientState":
        return PatientState(
            float(np.clip(self.tumor,  0, 1)),
            float(np.clip(self.immune, 0, 1)),
            float(np.clip(self.tissue, 0, 1)),
        )

    def to_array(self) -> np.ndarray:
        return np.array([self.tumor, self.immune, self.tissue])


def E_chemo(p: PatientState, intensity: float = 0.7) -> PatientState:
    """
    化学療法 E_c
    - 腫瘍縮小効果は免疫能に依存（免疫能が高いほど薬効が高い）
    - 強い免疫抑制作用（副作用）
    - 軽微な組織ダメージ
    """
    tumor_kill = intensity * 0.6 * (0.5 + 0.5 * p.immune)
    return PatientState(
        tumor  = p.tumor  * (1 - tumor_kill),
        immune = p.immune * (1 - intensity * 0.45),
        tissue = p.tissue * (1 - intensity * 0.05),
    ).clip()


def E_radio(p: PatientState, intensity: float = 0.7) -> PatientState:
    """
    放射線治療 E_r
    - 腫瘍縮小は直接的（免疫能に非依存）
    - 組織ダメージ：免疫能が低いほど大きい（修復能力の低下）
    - 軽微な免疫影響
    """
    radiation_damage = intensity * 0.30 * (1.5 - p.immune)
    return PatientState(
        tumor  = p.tumor  * (1 - intensity * 0.75),
        immune = p.immune * (1 - intensity * 0.08),
        tissue = p.tissue * (1 - radiation_damage),
    ).clip()


def J(p: PatientState) -> float:
    """
    アウトカム関数 J(p)（高いほど良い）
    J = -2·腫瘍量 + 1·免疫能 + 0.8·組織完全性
    """
    return -2.0 * p.tumor + 1.0 * p.immune + 0.8 * p.tissue


# ── 図1：状態軌跡の比較（単一患者） ──────────────────────────────────────────

def plot_state_trajectory():
    """
    1人の患者に対して
      順序A：化療 → 放射線
      順序B：放射線 → 化療
    の状態変化を並べて示す。
    """
    p0 = PatientState(tumor=0.8, immune=0.9, tissue=1.0)

    # 順序A：化療 → 放射線
    p_a1 = E_chemo(p0)
    p_a2 = E_radio(p_a1)

    # 順序B：放射線 → 化療
    p_b1 = E_radio(p0)
    p_b2 = E_chemo(p_b1)

    labels = ["初期状態", "1回目介入後", "2回目介入後"]
    states_A = [p0.to_array(), p_a1.to_array(), p_a2.to_array()]
    states_B = [p0.to_array(), p_b1.to_array(), p_b2.to_array()]

    dim_labels = ["腫瘍量\n(低いほど良)", "免疫能\n(高いほど良)", "組織完全性\n(高いほど良)"]
    colors_A = ["#2196F3", "#1976D2", "#0D47A1"]
    colors_B = ["#FF7043", "#E64A19", "#BF360C"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "医療介入の非可換性：患者状態の変化\n"
        r"$E_{\mathrm{chemo}} \circ E_{\mathrm{radio}}(p) \neq "
        r"E_{\mathrm{radio}} \circ E_{\mathrm{chemo}}(p)$",
        fontsize=13, y=1.02
    )

    for i, (ax, dim_label) in enumerate(zip(axes, dim_labels)):
        vals_A = [s[i] for s in states_A]
        vals_B = [s[i] for s in states_B]

        ax.plot(range(3), vals_A, "o-", color=colors_A[1],
                linewidth=2.5, markersize=8, label="順序A：化療→放射線")
        ax.plot(range(3), vals_B, "s--", color=colors_B[1],
                linewidth=2.5, markersize=8, label="順序B：放射線→化療")

        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel(dim_label, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 最終値の差を注釈
        diff = abs(vals_A[-1] - vals_B[-1])
        ax.annotate(
            f"差: {diff:.3f}",
            xy=(2, (vals_A[-1] + vals_B[-1]) / 2),
            xytext=(1.5, (vals_A[-1] + vals_B[-1]) / 2 + 0.15),
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9, color="gray"
        )

    plt.tight_layout()
    out = IMG_DIR / "01_state_trajectory.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"保存: {out}")


# ── 図2：アウトカム比較（初期腫瘍量を変化させて） ────────────────────────────

def plot_outcome_comparison():
    """
    初期腫瘍量を変化させたときの最終アウトカム J の比較。
    2本の線がずれていること = 非可換性の定量的証明。
    """
    tumor_range = np.linspace(0.1, 1.0, 80)
    J_A = []  # 化療 → 放射線
    J_B = []  # 放射線 → 化療
    diff = []

    for t in tumor_range:
        p0 = PatientState(tumor=t, immune=0.9, tissue=1.0)
        J_A.append(J(E_radio(E_chemo(p0))))
        J_B.append(J(E_chemo(E_radio(p0))))
        diff.append(J_A[-1] - J_B[-1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [2, 1]})

    # 上段：アウトカム比較
    ax1.plot(tumor_range, J_A, "-", color="#2196F3", linewidth=2.5,
             label=r"順序A：$E_{\mathrm{chemo}} \circ E_{\mathrm{radio}}(p)$（化療→放射線）")
    ax1.plot(tumor_range, J_B, "--", color="#FF7043", linewidth=2.5,
             label=r"順序B：$E_{\mathrm{radio}} \circ E_{\mathrm{chemo}}(p)$（放射線→化療）")
    ax1.fill_between(tumor_range, J_A, J_B,
                     where=[a > b for a, b in zip(J_A, J_B)],
                     alpha=0.15, color="#2196F3", label="順序Aが優位な領域")
    ax1.fill_between(tumor_range, J_A, J_B,
                     where=[a < b for a, b in zip(J_A, J_B)],
                     alpha=0.15, color="#FF7043", label="順序Bが優位な領域")

    ax1.set_ylabel("アウトカム J(p)（高いほど良い）", fontsize=11)
    ax1.set_title(
        "医療介入の非可換性：最終アウトカムの比較\n"
        r"$f(E_a, E_b) \neq f(E_b, E_a)$  $\Rightarrow$  介入順序が治療成績を決定する",
        fontsize=12
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="gray", linestyle=":", linewidth=1)

    # 下段：差（非可換性の強さ）
    ax2.bar(tumor_range, diff, width=0.012,
            color=["#2196F3" if d > 0 else "#FF7043" for d in diff],
            alpha=0.7)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xlabel("初期腫瘍量", fontsize=11)
    ax2.set_ylabel("J(順序A) − J(順序B)", fontsize=10)
    ax2.set_title("非可換性の強さ：アウトカム差分", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = IMG_DIR / "02_outcome_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"保存: {out}")


# ── 図3：非可換性ヒートマップ（2次元パラメータ空間） ─────────────────────────

def plot_noncommutativity_heatmap():
    """
    初期腫瘍量 × 介入強度の2次元空間で
    非可換性の強さ |J(A) - J(B)| をヒートマップ表示。
    MEDICUS空間の「景観」として可視化。
    """
    tumor_vals   = np.linspace(0.1, 1.0, 50)
    intensity_vals = np.linspace(0.2, 1.0, 50)
    Z = np.zeros((len(intensity_vals), len(tumor_vals)))

    for i, inten in enumerate(intensity_vals):
        for j, t in enumerate(tumor_vals):
            p0 = PatientState(tumor=t, immune=0.9, tissue=1.0)
            ja = J(E_radio(E_chemo(p0, inten), inten))
            jb = J(E_chemo(E_radio(p0, inten), inten))
            Z[i, j] = abs(ja - jb)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(tumor_vals, intensity_vals, Z, levels=20, cmap="RdYlBu_r")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$|J(E_c \circ E_r) - J(E_r \circ E_c)|$（非可換性の強さ）", fontsize=10)

    # 最大非可換点を強調
    max_idx = np.unravel_index(Z.argmax(), Z.shape)
    ax.plot(tumor_vals[max_idx[1]], intensity_vals[max_idx[0]],
            "w*", markersize=15, label=f"最大非可換点 ({Z.max():.3f})")

    ax.set_xlabel("初期腫瘍量", fontsize=12)
    ax.set_ylabel("介入強度", fontsize=12)
    ax.set_title(
        "非可換性の強さ：パラメータ空間ヒートマップ\n"
        "MEDICUS空間における介入順序依存性の景観",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.grid(False)

    plt.tight_layout()
    out = IMG_DIR / "03_noncommutativity_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"保存: {out}")


# ── メイン ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 非可換性の視覚化 ===")
    plot_state_trajectory()
    plot_outcome_comparison()
    plot_noncommutativity_heatmap()
    print("完了。img/ ディレクトリを確認してください。")
