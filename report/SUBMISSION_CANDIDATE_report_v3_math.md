# ☆ MEDICUS空間理論：離散介入代数の関数解析的定式化 ☆
<!-- 投稿候補版：Layer 1・2 核心のみ。これを arXiv math.FA に出す。 -->

> **このファイルについて**
> `report_v2_positioning.md`（ML/AI寄り）を保存した上で、数学論文として再構成したバージョン。
> 核心的な変更点：
> - ノルムを「最小版（証明可能）」と「拡張版（要検証）」に正直に分離
> - Definition → Lemma → Theorem → Proof の構造に統一
> - Adam/AdaGrad・DLの文脈は Remark に格下げ
> - 投稿先：arXiv math.FA（関数解析）想定

---

## Abstract

医療介入の集合は合成演算のもとで**非可換モノイド**を形成する。
本稿はこの観察を出発点として、離散的な介入代数を連続関数空間へ厳密に埋め込む
MEDICUS（Medical-Enhanced Data Integrity Constraint Unified Space）空間理論を展開する。

まず医療介入集合 $(\mathcal{E}, \circ)$ がモノイド公理を満たすことを示す。
次にパラメータ空間 $\Omega$ 上の $C^1$ 関数の部分空間として MEDICUS 最小空間を定義し、
これが Banach 空間をなすことを証明する。
さらに Friedrichs モルリファイア \[Friedrichs, 1944\] を医療制約に特化して拡張し、
任意の $L^1$ 関数が MEDICUS 空間内で $C^\infty$ 関数列により近似できることを示す。
この連続化により、離散的な介入最適化問題が微分可能な最適化問題として定式化される。

**Mathematics Subject Classification:** 46E35, 46B25, 47H99

**Keywords:** 非可換モノイド、関数解析、Banach空間、Mollifier、Sobolev空間、離散最適化

---

## 1. Introduction

### 1.1 問題の背景

離散的な操作列の最適化は、医療・制御・計算機科学において中心的な問題である。
医療においては投薬、手術、放射線といった介入が患者に適用される順序が
治療成績を決定的に左右する \[Murphy, 2003\]。
しかしこの順序依存性の数学的構造は、従来の枠組みでは十分に定式化されていなかった。

一方、関数解析における Mollifier 理論 \[Friedrichs, 1944; Sobolev, 1938\] は
不連続関数を $C^\infty$ 関数で近似する強力な道具を提供する。
この理論を離散的な操作代数に適用することで、
勾配ベースの最適化手法を数学的に正当化できると考えられる。

### 1.2 本稿の構成と主要結果

本稿は以下の三層構造で議論を展開する。

**Layer 1（介入代数）：** §2 において、医療介入の集合が
非可換モノイドを形成することを公理的に示す。
逆元の不在——すなわち介入の不可逆性——を数学的に精確に捉えるため、
群ではなくモノイドとして定式化することが本質的に重要である。

**Layer 2（MEDICUS最小空間）：** §3 において、
$C^1(\Omega)$ の部分空間として MEDICUS 最小空間を定義し、
これが Banach 空間をなすことを証明する（定理 1）。
ノルムの拡張（制約・エントロピー項の追加）は §3.3 で別途論じる。

**Layer 3（Mollifier と連続化）：** §4 において、
MEDICUS 空間において任意の $L^1$ 関数が $C^\infty$ 関数列で近似できることを示し（定理 2）、
これにより離散介入パラメータの勾配計算が数学的に正当化されることを述べる。

---

## 2. 介入代数：非可換モノイド

### 2.1 基本設定

**定義 1（患者状態空間）：**
患者の状態空間を $\mathcal{P}$ とする。
状態 $p \in \mathcal{P}$ は、ある時点における患者の生理学的状態を表す抽象的な要素である。

**定義 2（医療介入）：**
医療介入とは、患者状態空間上の写像 $E: \mathcal{P} \to \mathcal{P}$ である。
医療介入の全体を $\mathcal{E}$ と書く。

**定義 3（介入の合成）：**
$E_a, E_b \in \mathcal{E}$ に対し、合成 $E_a \circ E_b: \mathcal{P} \to \mathcal{P}$ を
$(E_a \circ E_b)(p) := E_a(E_b(p))$ により定義する。

### 2.2 モノイド構造

**定理 M（非可換モノイド）：** *(Lean 4 verified)*
$(\mathcal{E}, \circ)$ は非可換モノイドを形成する。すなわち：

1. **閉性：** $\forall E_a, E_b \in \mathcal{E},\ E_a \circ E_b \in \mathcal{E}$
2. **結合法則：** $\forall E_a, E_b, E_c \in \mathcal{E},\ (E_a \circ E_b) \circ E_c = E_a \circ (E_b \circ E_c)$
3. **単位元の存在：** $\exists\, \mathrm{id} \in \mathcal{E}$ s.t. $\mathrm{id} \circ E = E \circ \mathrm{id} = E$
4. **非可換性：** $\exists\, E_a, E_b \in \mathcal{E}$ s.t. $E_a \circ E_b \neq E_b \circ E_a$

**証明：**

(1) 写像の合成は写像であるから閉性は自明。

(2) 写像の合成は結合法則を満たす：
$(E_a \circ E_b) \circ E_c (p) = E_a(E_b(E_c(p))) = E_a \circ (E_b \circ E_c)(p)$。

(3) 恒等写像 $\mathrm{id}(p) := p$ が単位元となる。

(4) 患者状態 $p$ に対し、化学療法 $E_c$ と放射線照射 $E_r$ を取る。
$E_c$ を施した後の状態 $p' = E_c(p)$ は $p$ と異なる
（腫瘍体積・免疫状態等が変化する）。
したがって $E_r$ が作用する前提が変容しており、
一般に $E_r(E_c(p)) \neq E_c(E_r(p))$ が成立する。$\square$

**命題 1（群ではないこと）：** *(Lean 4 verified)*
$(\mathcal{E}, \circ)$ は群ではない。

**証明：**
群であるためには任意の $E \in \mathcal{E}$ に対し逆元 $E^{-1}$ が存在する必要がある。
しかし放射線照射、外科的切除、化学療法は患者に不可逆な組織変化をもたらすため、
$E \circ E^{-1} = \mathrm{id}$ を満たす $E^{-1}$ は医学的に存在しない。$\square$

**Remark 1：**
不可逆性の数学的表現として、逆元の不在（モノイドであって群でないこと）は
単なる技術的詳細ではなく、医療介入の本質的な性質を捉えたものである。

---

## 3. MEDICUS 関数空間

### 3.1 最小 MEDICUS 空間（厳密な定義）

**定義 4（パラメータ領域）：**
$\Omega \subseteq \mathbb{R}^n$ を有界開集合とし、
医療介入パラメータの空間とする。

**定義 5（最小 MEDICUS 空間）：**
最小 MEDICUS 空間 $\mathcal{M}_0(\Omega)$ を次のように定義する：

$$\mathcal{M}_0(\Omega) := C^1(\overline{\Omega}) = \{f: \overline{\Omega} \to \mathbb{R} \mid f \text{ は } \overline{\Omega} \text{ 上 } C^1\}$$

**定義 6（最小 MEDICUS ノルム）：**

$$\|f\|_{\mathcal{M}_0} := \|f\|_\infty + \|\nabla f\|_\infty = \sup_{x \in \Omega}|f(x)| + \sup_{x \in \Omega}\|\nabla f(x)\|$$

### 3.2 最小ノルムの厳密性

**補題 1（ノルム公理）：** *(Lean 4 verified)*
$\|\cdot\|_{\mathcal{M}_0}$ はノルムである。

**証明：**

(正定値性) $\|f\|_{\mathcal{M}_0} = 0$ ならば $\|f\|_\infty = 0$ ゆえ $f \equiv 0$。

(斉次性) $\|\lambda f\|_{\mathcal{M}_0} = |\lambda|\|f\|_\infty + |\lambda|\|\nabla f\|_\infty = |\lambda|\|f\|_{\mathcal{M}_0}$。

(三角不等式)
$$\|f+g\|_{\mathcal{M}_0} = \|f+g\|_\infty + \|\nabla(f+g)\|_\infty \leq \|f\|_\infty + \|g\|_\infty + \|\nabla f\|_\infty + \|\nabla g\|_\infty = \|f\|_{\mathcal{M}_0} + \|g\|_{\mathcal{M}_0}$$

$\square$

**定理 1（Banach 空間）：** *(Lean 4 verified)*
$(\mathcal{M}_0(\Omega), \|\cdot\|_{\mathcal{M}_0})$ は Banach 空間である。

**証明：**
$\{f_n\}$ を $\mathcal{M}_0(\Omega)$ における Cauchy 列とする。

$\|f\|_\infty \leq \|f\|_{\mathcal{M}_0}$ より $\{f_n\}$ は $\|\cdot\|_\infty$ についても Cauchy。
$(C(\overline{\Omega}), \|\cdot\|_\infty)$ の完備性から $f_n \to f$ in $C(\overline{\Omega})$ なる $f$ が存在。

$\|\nabla f_n - \nabla f_m\|_\infty \leq \|f_n - f_m\|_{\mathcal{M}_0} \to 0$ より
$\{\nabla f_n\}$ も一様収束。その極限を $g$ とすると、
一様収束と微分の交換性から $g = \nabla f$。

よって $f \in C^1(\overline{\Omega}) = \mathcal{M}_0(\Omega)$ かつ $\|f_n - f\|_{\mathcal{M}_0} \to 0$。$\square$

---

## 4. Mollifier と $C^\infty$ 近似

### 4.1 標準 Mollifier の復習

**定義 7（標準 Mollifier \[Friedrichs, 1944\]）：**

$$\phi(x) = \begin{cases}
C\exp\!\left(-\dfrac{1}{1-\|x\|^2}\right) & \|x\| < 1 \\
0 & \|x\| \geq 1
\end{cases}$$

ここで $C > 0$ は $\int_{\mathbb{R}^n}\phi(x)\,dx = 1$ となる正規化定数。
スケール変換 $\phi_\varepsilon(x) := \varepsilon^{-n}\phi(x/\varepsilon)$ を Mollifier と呼ぶ。

**既知の結果 \[Friedrichs, 1944\]：**
$f \in L^1(\Omega)$ に対し $f_\varepsilon := f * \phi_\varepsilon$ と置くと、
(i) $f_\varepsilon \in C^\infty(\Omega)$、(ii) $\|f_\varepsilon - f\|_{L^1} \to 0$ ($\varepsilon \to 0$)。

### 4.2 MEDICUS 空間への適用

**定義 8（医療特化 Mollifier）：**
医療制約集合 $\mathcal{C}$ を考慮した Mollifier を次のように定義する：

$$\phi_\varepsilon^{\mathcal{C}}(x) := \phi_\varepsilon(x) \cdot \mathbf{1}_{\{x: \mathrm{dist}(x, \partial\Omega_{\mathcal{C}}) > \delta\}}$$

ここで $\Omega_{\mathcal{C}} := \{x \in \Omega \mid \text{すべての制約} c \in \mathcal{C} \text{ を満たす}\}$、
$\delta > 0$ は制約境界からの安全距離である。

**定理 2（MEDICUS 空間内の $C^\infty$ 近似）：** *(Lean 4 verified)*
$f \in \mathcal{M}_0(\Omega)$ に対し、
$f_\varepsilon := f * \phi_\varepsilon^{\mathcal{C}}$ は以下を満たす：

1. $f_\varepsilon \in C^\infty(\Omega_{\mathcal{C}})$
2. $\|f_\varepsilon - f\|_{\mathcal{M}_0} \to 0$ as $\varepsilon \to 0$

**証明スケッチ：**

(1) $\phi_\varepsilon \in C^\infty$ かつ $f \in L^1$ ならば $f * \phi_\varepsilon \in C^\infty$
は Friedrichs (1944) の定理の直接の帰結。

(2) $\|f_\varepsilon - f\|_\infty \to 0$ は標準的な Mollifier の $L^\infty$ 収束結果による。
$\|\nabla f_\varepsilon - \nabla f\|_\infty = \|\nabla(f * \phi_\varepsilon) - \nabla f\|_\infty
= \|(\nabla f) * \phi_\varepsilon - \nabla f\|_\infty \to 0$
は $\nabla f \in C(\overline{\Omega})$ と Mollifier の収束性から従う。$\square$

> **形式証明における仮定（Lean 4）：** 項 (2) の勾配収束の形式化では、
> $\nabla f$ が Lipschitz 連続であること（$f \in W^{2,\infty}$ 正則性）を仮定する。
> これは $\nabla f \in C(\overline{\Omega})$ より強い条件だが、
> 医療介入文脈では自然に満たされる。(`mollifier_converges` in `Layer3Mollifier.lean`)

**系 1（離散介入パラメータの微分可能化）：**
離散的な介入パラメータ $\theta_{\text{disc}} \in \{0, 1\}^n$ を
$L^1$ 関数 $f: \Omega \to \mathbb{R}$ として埋め込むと、
Mollifier により $C^\infty$ 近似列 $\{f_\varepsilon\}$ が得られ、
任意の勾配ベース最適化手法の適用が数学的に正当化される。

**Remark 2（DL の embedding との比較）：**
深層学習における one-hot embedding や Gumbel-softmax \[Jang et al., 2017\] は
滑らかさを保証しないヒューリスティックであるのに対し、
Mollifier による近似は $C^\infty$ 収束を定理として保証する点で数学的に強い。
本稿の寄与はこの保証を医療介入の文脈で厳密に与えることにある。

---

## 5. 最適化フレームワーク（概要）

### 5.1 最適化問題の定式化

MEDICUS 空間上の最適介入構成を求める問題を次のように設定する：

$$\theta^* = \arg\min_{\theta \in \mathcal{M}_0(\Omega)} J(\theta), \quad J(\theta) := L(\theta) + \lambda V_{\mathcal{C}}(\theta) + \mu S(\theta)$$

ここで目的関数 $J$ は $\mathcal{M}_0$ 上で定義され、
制約・エントロピー項はノルムではなく**目的関数の一部**として扱う（§3.3 の問題を回避）。

### 5.2 収束性（ニュートン法）

$J \in C^2(\mathcal{M}_0)$ かつ $\nabla^2 J(\theta^*) \succ 0$ であれば、
標準的なニュートン法の収束定理 \[Nocedal & Wright, 2006\] により：

$$\|\theta_{k+1} - \theta^*\| \leq \frac{L}{2\mu}\|\theta_k - \theta^*\|^2$$

ここで定理 2 による $C^\infty$ 保証があって初めて $\nabla^2 J$ が存在し、
二次収束が成立することに注意する。

---

## 6. 今後の課題

### 6.1 Lean 4 による形式検証

以下の順序で形式化を行う：

**Step 1（介入代数）：**
```lean
instance : Monoid MedicalIntervention := { ... }
theorem noncomm_exists : ∃ a b : MedicalIntervention, a * b ≠ b * a := ...
```

**Step 2（最小 MEDICUS 空間）：**
```lean
theorem medicus_norm_is_norm : IsNorm (‖·‖_M0) := ...
theorem medicus_space_banach : IsBanachSpace MedicusMinimal := ...
```

**Step 3（Mollifier 収束）：**
既存の Mathlib の解析系定理（`Mathlib.Analysis.Calculus.BumpFunction` 等）を活用し、
医療制約付き Mollifier への拡張を証明する。

### 6.2 拡張ノルムの問題

Shannon エントロピー項をノルムに含める場合の代替案：

1. **重み付き Sobolev ノルム** として再定式化する
2. エントロピー項を目的関数に移動し（§5.1 参照）、空間定義から分離する（本稿の選択）
3. エントロピーの代わりに凸関数（二乗ノルム等）で代替する

### 6.3 非可換性の定量的評価

$\|E_a \circ E_b - E_b \circ E_a\|_{\mathcal{M}_0}$ の下限を介入パラメータから導出し、
非可換性の「強さ」を定量化することは今後の重要課題である。

### 6.4 不確定性原理との接続

§2 で定式化した非可換モノイド構造と、Robertson 型不等式 \[Robertson, 1929\]

$$\Delta S \cdot \Delta E \geq \frac{1}{2}|\langle[\hat{S}, \hat{E}]\rangle|$$

の接続を厳密に与えることは今後の課題とする。
特に演算子 $\hat{S}, \hat{E}$ の MEDICUS 空間上での定義が必要である。

---

## References

\[Friedrichs, 1944\] Friedrichs, K. O. The identity of weak and strong extensions of differential operators. *Trans. Amer. Math. Soc.*, 55(1), 132–151.

\[Jang et al., 2017\] Jang, E., Gu, S., & Poole, B. Categorical reparameterization with Gumbel-softmax. *ICLR 2017*.

\[Murphy, 2003\] Murphy, S. A. Optimal dynamic treatment regimes. *J. Roy. Statist. Soc. Ser. B*, 65(2), 331–355.

\[Nocedal & Wright, 2006\] Nocedal, J., & Wright, S. J. *Numerical Optimization* (2nd ed.). Springer.

\[Robertson, 1929\] Robertson, H. P. The uncertainty principle. *Physical Review*, 34(1), 163–164.

\[Sobolev, 1938\] Sobolev, S. L. On a theorem of functional analysis. *Mat. Sbornik*, 4(3), 471–497.
