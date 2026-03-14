"""
build_pdf.py — manuscript_v_20260314.md → PDF 変換スクリプト

数式は matplotlib の mathtext で PNG レンダリング後に埋め込む。
図は figures.md の採番に従って本文中の参照直後に挿入する。
"""

import re
import io
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    PageBreak, Table, TableStyle, KeepTogether, HRFlowable,
    ListFlowable, ListItem,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ============================================================
# Unicode フォント登録（Arial / Windows）
# ============================================================

FONT_DIR = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Arial",       str(FONT_DIR / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Bold",  str(FONT_DIR / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Arial-Italic",str(FONT_DIR / "ariali.ttf")))
pdfmetrics.registerFont(TTFont("Arial-BoldItalic", str(FONT_DIR / "arialbi.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily(
    "Arial",
    normal="Arial",
    bold="Arial-Bold",
    italic="Arial-Italic",
    boldItalic="Arial-BoldItalic",
)

# ReportLab の <b> <i> タグが Arial ファミリーを参照するよう登録
FONT_NORMAL     = "Arial"
FONT_BOLD       = "Arial-Bold"
FONT_ITALIC     = "Arial-Italic"
FONT_BOLDITALIC = "Arial-BoldItalic"
FONT_MONO       = "Courier"  # Courier は組み込みで ASCII のみだが数式以外に使用

# ============================================================
# パス設定
# ============================================================

BASE    = Path(__file__).parent
ROOT    = BASE.parent
REPORT  = BASE
EXP_RES = ROOT / "experience" / "results"
VIS_IMG = ROOT / "visualization" / "img"
OUT_PDF = REPORT / "manuscript_v_20260314.pdf"
MATH_CACHE = BASE / "_math_cache"
MATH_CACHE.mkdir(exist_ok=True)

# ============================================================
# 図ファイルマッピング（figures.md 採番と一致）
# ============================================================

FIGURES = {
    "Fig. 01": VIS_IMG / "01_state_trajectory.png",
    "Fig. 02": VIS_IMG / "02_outcome_comparison.png",
    "Fig. 03": VIS_IMG / "03_noncommutativity_heatmap.png",
    "Fig. 04": VIS_IMG / "04_mollifier_kernel.png",
    "Fig. 05": VIS_IMG / "05_mollifier_smoothing.png",
    "Fig. 06": VIS_IMG / "06_mollifier_convergence.png",
    "Fig. 07": EXP_RES / "exp04" / "exp4_heatmap.png",
    "Fig. 08": EXP_RES / "exp04" / "exp4_surface.png",
    "Fig. 09": EXP_RES / "exp01_modify" / "exp1m_grad_smoothness.png",
    "Fig. 10": EXP_RES / "exp05" / "exp5_scaling.png",
    "Fig. A1": EXP_RES / "exp01_modify" / "exp1m_loss_curves.png",
    "Fig. A2": EXP_RES / "exp01_modify" / "exp1m_theta_dist.png",
    "Fig. A3": EXP_RES / "exp01_modify" / "exp1m_grad_norms.png",
    "Fig. A4": EXP_RES / "exp02" / "exp2_conv_rate.png",
    "Fig. A5": EXP_RES / "exp02" / "exp2_theta_dist.png",
    "Fig. A6": EXP_RES / "exp03" / "exp3_conv_rate.png",
    "Fig. A7": EXP_RES / "exp03" / "exp3_theta_dist.png",
    "Fig. A8": EXP_RES / "exp_extra_01" / "extra1_convergence_speed.png",
    "Fig. A9": EXP_RES / "exp_extra_01" / "extra1_loss_curves.png",
    "Fig. A10": EXP_RES / "exp_extra_01" / "extra1_grad_norms.png",
}

# 図番号 → キャプション
FIGURE_CAPTIONS = {
    "Fig. 01": "State trajectory visualization under non-commutative interventions.",
    "Fig. 02": "Outcome comparison: E_a∘E_b versus E_b∘E_a.",
    "Fig. 03": "Heatmap of ‖E_a∘E_b(p) − E_b∘E_a(p)‖ quantifying non-commutativity.",
    "Fig. 04": "Mollifier kernel φ_ε for multiple values of ε.",
    "Fig. 05": "Mollifier smoothing of a step function: C∞ approximation.",
    "Fig. 06": "‖f_ε − f‖_∞ convergence as ε → 0 (continuous mollifier, Theorem 1).",
    "Fig. 07": "Heatmap of J(θ) (left) and J_ε(θ) (right) over θ ∈ [−3,3]² (log scale). "
               "Red dot: θ* = (1,1). Smoothing compresses the value range and displaces the minimum.",
    "Fig. 08": "3D surface of J(θ) (left) and J_ε(θ) (right). "
               "J exhibits high-frequency undulations; J_ε presents a smooth bowl geometry.",
    "Fig. 09": "Step-to-step gradient norm variation |Δ‖∇J‖| for raw J (red) and "
               "J_ε,sym (blue). The reduced variation under smoothing is consistent with "
               "the Lipschitz gradient property established in Layer 3.",
    "Fig. 10": "Upper: MAE vs ε (Theorem 1 proxy). Lower: ‖θ_ε* − θ*‖ vs ε (Corollary 2). "
               "Both decrease monotonically as ε → 0.",
    "Fig. A1": "Loss curves: raw J vs J_ε,sym (20 initializations, Adam).",
    "Fig. A2": "Distance ‖θ − θ*‖ over steps: raw J vs J_ε,sym.",
    "Fig. A3": "Gradient norm trajectories: raw J vs J_ε,sym.",
    "Fig. A4": "Convergence rate across 10 dynamics seeds (Exp2).",
    "Fig. A5": "Final ‖θ_ε* − θ*‖ across 10 dynamics seeds (Exp2).",
    "Fig. A6": "Convergence rate at d = 32/64/128 (Exp3).",
    "Fig. A7": "Final ‖θ_ε* − θ*‖ at d = 32/64/128 (Exp3).",
    "Fig. A8": "Convergence step distribution (boxplot): 4 conditions, Adam vs L-BFGS.",
    "Fig. A9": "Loss curves: 4 conditions (raw/smooth_sym × Adam/L-BFGS).",
    "Fig. A10": "Gradient norm trajectories: 4 conditions.",
}

# 本文中のどのキーワードの後に挿入するか
FIGURE_TRIGGERS = {
    "Fig. 01": "Fig. 01",
    "Fig. 02": "Fig. 02",
    "Fig. 03": "Fig. 03",
    "Fig. 04": "Fig. 04",
    "Fig. 05": "Fig. 05",
    "Fig. 06": "Fig. 06",
    "Fig. 07": "Fig. 07",
    "Fig. 08": "Fig. 08",
    "Fig. 09": "Fig. 09",
    "Fig. 10": "Fig. 10",
}

# ============================================================
# 数式レンダリング
# ============================================================

def _fix_for_mathtext(expr: str) -> str:
    """matplotlib mathtext が受け付けない記号を修正。"""
    fixes = [
        (r'\\le\b',  r'\\leq'),
        (r'\\ge\b',  r'\\geq'),
        (r'\\lVert', r'\\|'),
        (r'\\rVert', r'\\|'),
        (r'\\norm\{([^}]+)\}', r'\\|\1\\|'),
        (r'\\text\{([^}]+)\}', r'\\mathrm{\1}'),
        (r'\\mathrm\{([^}]*)\}', r'\\rm \1'),
        (r'\\operatorname\{([^}]+)\}', r'\\mathrm{\1}'),
        (r'\\quad', r'\ \ '),
        (r'\\qquad', r'\ \ \ \ '),
        (r'\\,', r'\ '),
        (r'\\;', r'\ '),
        (r'\\!', r''),
        (r'\\bigl', r'\\left'),
        (r'\\bigr', r'\\right'),
        (r'\\Bigl', r'\\left'),
        (r'\\Bigr', r'\\right'),
        (r'\\big',  r''),
        (r'\\square', r'\\Box'),
        (r'\\url\{[^}]+\}', r''),
        (r'\\texttt\{([^}]+)\}', r'\\mathrm{\1}'),
        (r'\\begin\{[^}]+\}', r''),
        (r'\\end\{[^}]+\}', r''),
        (r'\\\\', r'\ '),
        (r'&', r'\ '),
        (r'\\label\{[^}]+\}', r''),
        (r'\\tag\{[^}]+\}', r''),
    ]
    result = expr
    for pat, rep in fixes:
        result = re.sub(pat, rep, result)
    return result


def render_math(expr: str, fontsize: int = 11, display: bool = False) -> str:
    """数式を PNG にレンダリングし、ファイルパスを返す。キャッシュあり。失敗時は None。"""
    fixed = _fix_for_mathtext(expr)
    key = hashlib.md5((fixed + str(fontsize) + str(display)).encode()).hexdigest()
    path = MATH_CACHE / f"{key}.png"
    if path.exists():
        return str(path)

    try:
        fig, ax = plt.subplots(figsize=(0.01, 0.01))
        ax.axis("off")
        text = ax.text(
            0.5, 0.5, f"${fixed}$",
            fontsize=fontsize,
            ha="center", va="center",
            transform=ax.transAxes,
        )
        fig.canvas.draw()
        bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())
        dpi = 150
        w = max(bbox.width / dpi + 0.1, 0.5)
        h = max(bbox.height / dpi + 0.1, 0.3)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(w, h))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.axis("off")
        ax.text(
            0.5, 0.5, f"${fixed}$",
            fontsize=fontsize,
            ha="center", va="center",
            transform=ax.transAxes,
        )
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                    facecolor="white", pad_inches=0.05)
        plt.close(fig)
        return str(path)
    except Exception as e:
        plt.close("all")
        return None  # 失敗時は None を返す


def render_display_math(expr: str) -> str:
    """ディスプレイ数式（$$...$$）をレンダリング。"""
    return render_math(expr, fontsize=13, display=True)


def render_inline_math(expr: str) -> str:
    """
    インライン数式専用レンダラー。
    pad_inches を最小化し、自然サイズで使うことで本文テキストと揃える。
    """
    fixed = _fix_for_mathtext(expr)
    key = hashlib.md5((fixed + "inline10").encode()).hexdigest()
    path = MATH_CACHE / f"{key}_il.png"
    if path.exists():
        return str(path)

    fontsize = 10
    dpi = 150
    try:
        # Step 1: bbox 計測
        fig, ax = plt.subplots(figsize=(0.01, 0.01))
        ax.axis("off")
        t = ax.text(0.5, 0.5, f"${fixed}$",
                    fontsize=fontsize, ha="center", va="center",
                    transform=ax.transAxes)
        fig.canvas.draw()
        bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
        plt.close(fig)

        # Step 2: content bbox ぴったりのサイズで描画、余白は最小
        w_in = max(bbox.width  / dpi + 0.02, 0.1)
        h_in = max(bbox.height / dpi + 0.02, 0.15)
        fig, ax = plt.subplots(figsize=(w_in, h_in))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.axis("off")
        ax.text(0.5, 0.5, f"${fixed}$",
                fontsize=fontsize, ha="center", va="center",
                transform=ax.transAxes)
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight",
                    facecolor="white", pad_inches=0.01)
        plt.close(fig)
        return str(path)
    except Exception:
        plt.close("all")
        return None


# ============================================================
# スタイル定義
# ============================================================

PAGE_W, PAGE_H = A4
MARGIN_L = 25 * mm
MARGIN_R = 25 * mm
MARGIN_T = 25 * mm
MARGIN_B = 20 * mm
TEXT_W = PAGE_W - MARGIN_L - MARGIN_R

ACCENT  = HexColor("#1a3a5c")
GRAY    = HexColor("#555555")
LGRAY   = HexColor("#888888")
BGGRAY  = HexColor("#f5f5f5")

def make_styles():
    styles = getSampleStyleSheet()
    base = dict(fontName="Arial", leading=16)

    s = {}

    s["title"] = ParagraphStyle("title",
        fontName="Arial-Bold", fontSize=16, leading=22,
        textColor=ACCENT, spaceAfter=6, spaceBefore=0,
        alignment=TA_CENTER)

    s["author"] = ParagraphStyle("author",
        fontName="Arial", fontSize=12, leading=16,
        textColor=GRAY, spaceAfter=4, alignment=TA_CENTER)

    s["abstract_head"] = ParagraphStyle("abstract_head",
        fontName="Arial-Bold", fontSize=10, leading=14,
        textColor=ACCENT, spaceAfter=2, spaceBefore=8, alignment=TA_CENTER)

    s["abstract"] = ParagraphStyle("abstract",
        fontName="Arial", fontSize=9.5, leading=14,
        leftIndent=15*mm, rightIndent=15*mm,
        spaceAfter=12, alignment=TA_JUSTIFY)

    s["h1"] = ParagraphStyle("h1",
        fontName="Arial-Bold", fontSize=13, leading=18,
        textColor=ACCENT, spaceBefore=14, spaceAfter=4,
        borderPadding=(0, 0, 2, 0))

    s["h2"] = ParagraphStyle("h2",
        fontName="Arial-Bold", fontSize=11, leading=15,
        textColor=ACCENT, spaceBefore=10, spaceAfter=3)

    s["h3"] = ParagraphStyle("h3",
        fontName="Arial-BoldItalic", fontSize=10, leading=14,
        textColor=GRAY, spaceBefore=7, spaceAfter=2)

    s["body"] = ParagraphStyle("body",
        fontName="Arial", fontSize=10, leading=15,
        spaceAfter=6, alignment=TA_JUSTIFY)

    s["body_indent"] = ParagraphStyle("body_indent",
        fontName="Arial", fontSize=10, leading=15,
        leftIndent=10*mm, spaceAfter=4, alignment=TA_JUSTIFY)

    s["theorem"] = ParagraphStyle("theorem",
        fontName="Arial-Italic", fontSize=10, leading=15,
        leftIndent=8*mm, rightIndent=8*mm, spaceAfter=4,
        backColor=BGGRAY, borderPadding=4)

    s["caption"] = ParagraphStyle("caption",
        fontName="Arial-Italic", fontSize=8.5, leading=12,
        textColor=GRAY, spaceAfter=8, alignment=TA_CENTER)

    s["ref"] = ParagraphStyle("ref",
        fontName="Arial", fontSize=8.5, leading=13,
        leftIndent=8*mm, firstLineIndent=-8*mm,
        spaceAfter=3)

    s["code"] = ParagraphStyle("code",
        fontName="Courier", fontSize=8, leading=12,
        leftIndent=8*mm, spaceAfter=4, backColor=BGGRAY)

    s["bullet"] = ParagraphStyle("bullet",
        fontName="Arial", fontSize=10, leading=14,
        leftIndent=8*mm, firstLineIndent=-4*mm, spaceAfter=2)

    return s

# ============================================================
# テキスト前処理ユーティリティ
# ============================================================

def escape_xml(text: str) -> str:
    """ReportLab XML 特殊文字をエスケープ。"""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def process_inline(text: str) -> str:
    """
    インライン要素（太字・斜体・インライン数式）を ReportLab タグに変換。
    インライン数式 $...$ は matplotlib mathtext で PNG レンダリングし、
    ReportLab の <img> タグとして段落中に埋め込む。
    フォールバック: レンダリング失敗時は Unicode 近似テキスト。
    """
    # Arial に含まれない特殊 Unicode 文字を ASCII に正規化
    text = text.replace("\u2011", "-")   # U+2011 non-breaking hyphen → -
    # U+2013 en dash, U+2014 em dash は Arial に含まれるのでそのまま残す
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')

    # インライン数式を一時プレースホルダーに退避
    math_store = {}   # key → (tag_string, is_img)

    def stash_math(m):
        key = f"\x00MATH{len(math_store)}\x00"
        expr = m.group(1)

        # インライン専用レンダラーで PNG 生成（余白最小）
        img_path = render_inline_math(expr)
        if img_path:
            try:
                with PILImage.open(img_path) as pil_img:
                    w_px, h_px = pil_img.size
                # 自然サイズをそのまま使用（DPI=150 → points）
                # fontsize=10, pad=0.01 なので h ≈ 11-14pt、本文と釣り合う
                DPI = 150
                w_pt = w_px * 72.0 / DPI
                h_pt = h_px * 72.0 / DPI
                tag = (
                    f'<img src="{img_path}" '
                    f'width="{w_pt:.1f}" height="{h_pt:.1f}" '
                    f'valign="middle"/>'
                )
                math_store[key] = (tag, True)
            except Exception:
                math_store[key] = (math_to_unicode(expr), False)
        else:
            math_store[key] = (math_to_unicode(expr), False)

        return key

    text = re.sub(r'\$([^$\n]+?)\$', stash_math, text)

    # XML エスケープ（数式プレースホルダーは \x00 を含むので影響なし）
    text = escape_xml(text)

    # 太字 **...**
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # 斜体 *...*
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # インラインコード `...`
    text = re.sub(r'`([^`]+)`', r'<font name="Courier" size="9">\1</font>', text)

    # 数式を復元
    for key, (val, is_img) in math_store.items():
        placeholder = escape_xml(key)  # \x00 chars pass through escape_xml unchanged
        if is_img:
            text = text.replace(placeholder, val)           # <img> タグをそのまま挿入
        else:
            text = text.replace(placeholder, f'<i>{escape_xml(val)}</i>')  # Unicode fallback

    return text


def math_to_unicode(expr: str) -> str:
    """
    LaTeX 数式を近似的な Unicode テキストに変換。
    完全な変換は行わず、読みやすい近似を返す。
    """
    replacements = [
        # Greek letters
        (r'\\varepsilon', 'ε'), (r'\\epsilon', 'ε'),
        (r'\\theta', 'θ'), (r'\\Theta', 'Θ'),
        (r'\\phi', 'φ'), (r'\\varphi', 'φ'),
        (r'\\rho', 'ρ'), (r'\\sigma', 'σ'),
        (r'\\alpha', 'α'), (r'\\beta', 'β'),
        (r'\\gamma', 'γ'), (r'\\delta', 'δ'),
        (r'\\lambda', 'λ'), (r'\\mu', 'μ'),
        (r'\\tau', 'τ'), (r'\\omega', 'ω'), (r'\\Omega', 'Ω'),
        (r'\\nabla', '∇'), (r'\\Delta', 'Δ'), (r'\\partial', '∂'),
        # Operators and relations
        (r'\\infty', '∞'), (r'\\to', '->'), (r'\\mapsto', '|->'),
        (r'\\leq', '≤'), (r'\\geq', '≥'),
        (r'\\le\b', '≤'), (r'\\ge\b', '≥'),   # \le / \ge without q
        (r'\\neq', '≠'), (r'\\approx', '≈'), (r'\\sim', '~'),
        (r'\\circ', '∘'), (r'\\cdot', '·'),
        (r'\\star', '⋆'), (r'\\times', '×'), (r'\\ast', '*'),
        (r'\\sum', 'Σ'), (r'\\prod', 'Π'), (r'\\int', '∫'),
        (r'\\mathbb\{R\}', 'ℝ'), (r'\\mathbb\{N\}', 'ℕ'), (r'\\mathbb\{Z\}', 'ℤ'),
        (r'\\in\b', '∈'), (r'\\subset', '⊂'),
        (r'\\cup', '∪'), (r'\\cap', '∩'),
        (r'\\forall', '∀'), (r'\\exists', '∃'),
        (r'\\langle', '('), (r'\\rangle', ')'),   # ⟨⟩ は Arial 未収録 → ()
        # Norm delimiters — most specific first, then plain \|
        # CRITICAL: r'\\\|' matches the two-char sequence backslash+pipe.
        # Do NOT use r'\\|' here — in regex that means "backslash OR empty string".
        (r'\\left\s*\\\|', '||'),   # \left\|  → ||  (‖ U+2016 は Arial 未収録)
        (r'\\right\s*\\\|', '||'),  # \right\|
        (r'\\\|', '||'),             # plain \|
        (r'\\left', ''), (r'\\right', ''),
        # Braced font/text commands
        (r'\\mathrm\{([^}]+)\}', r'\1'),
        (r'\\mathbf\{([^}]+)\}', r'\1'),
        (r'\\mathcal\{([^}]+)\}', r'\1'),
        (r'\\text\{([^}]+)\}', r'\1'),
        (r'\\operatorname\{([^}]+)\}', r'\1'),
        (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1/\2)'),
        # Super/subscripts (must come after all command expansions)
        (r'\^(\{[^}]+\}|\S)', lambda m: _superscript(m.group(1).strip('{}'))),
        (r'_(\{[^}]+\}|\S)',  lambda m: _subscript(m.group(1).strip('{}'))),
        # Remove remaining braces
        (r'\{([^}]*)\}', r'\1'),
    ]
    result = expr
    for pat, rep in replacements:
        if callable(rep):
            result = re.sub(pat, rep, result)
        else:
            result = re.sub(pat, rep, result)
    return result.strip()


def _superscript(s: str) -> str:
    # Arial は基本的な上付き数字 ⁰-⁹ は含むが ⁿ ⁱ ᵈ 等のラテン文字上付きは
    # 含まない場合があり □ になる。数字のみ変換し、それ以外は ^s 表記を使う。
    sup_map = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    try:
        result = s.translate(sup_map)
        if result == s:
            return '^' + s   # 非数字: ^* ^n 等、そのまま ASCII で表示
        return result
    except Exception:
        return f'^{s}'


def _subscript(s: str) -> str:
    sub_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    try:
        result = s.translate(sub_map)
        if result == s:
            # No digit substitution occurred — keep underscore as visual separator
            return '_' + s
        return result
    except Exception:
        return f'_{s}'


# ============================================================
# 図フローアブル生成
# ============================================================

def make_figure_flowable(fig_key: str, styles: dict, max_w=None) -> list:
    """図と キャプション を Flowable リストで返す。"""
    path = FIGURES.get(fig_key)
    if path is None or not path.exists():
        return []

    if max_w is None:
        max_w = TEXT_W - 10 * mm

    try:
        img = PILImage.open(path)
        iw, ih = img.size
        scale = min(max_w / iw, (120 * mm) / ih)
        w, h = iw * scale, ih * scale
        rl_img = RLImage(str(path), width=w, height=h)
    except Exception:
        return []

    caption_text = FIGURE_CAPTIONS.get(fig_key, fig_key)
    caption = Paragraph(
        f"<b>{fig_key}.</b> {escape_xml(caption_text)}",
        styles["caption"]
    )
    return [
        Spacer(1, 4 * mm),
        KeepTogether([rl_img, Spacer(1, 2 * mm), caption]),
        Spacer(1, 4 * mm),
    ]


# ============================================================
# Markdown パーサー → Flowable リスト
# ============================================================

def parse_markdown(md_text: str, styles: dict) -> list:
    """Markdown テキストを ReportLab Flowable リストに変換。"""
    flowables = []
    inserted_figs = set()

    def add_figure_if_referenced(text_block: str):
        """テキストブロックに fig 参照があれば図を挿入。"""
        for fig_key in FIGURES:
            if fig_key in text_block and fig_key not in inserted_figs:
                # Fig. 07–08 のような範囲参照を処理
                figs_to_insert = [fig_key]
                # 範囲指定（例: Fig. 07–08）
                m = re.search(r'Fig\.\s+0?(\d+)[\u2013\u2014-]0?(\d+)', text_block)
                if m:
                    start, end = int(m.group(1)), int(m.group(2))
                    for n in range(start, end + 1):
                        k = f"Fig. {n:02d}" if n < 10 else f"Fig. {n}"
                        if k != fig_key:
                            figs_to_insert.append(k)

                for fk in figs_to_insert:
                    if fk not in inserted_figs and fk in FIGURES:
                        flowables.extend(make_figure_flowable(fk, styles))
                        inserted_figs.add(fk)

    lines = md_text.split("\n")
    i = 0
    in_abstract = False
    in_references = False
    in_appendix = False
    skip_next = False
    bullet_buffer = []

    def flush_bullets():
        nonlocal bullet_buffer
        if not bullet_buffer:
            return
        items = []
        for item_text in bullet_buffer:
            items.append(ListItem(
                Paragraph(process_inline(item_text), styles["bullet"]),
                leftIndent=12*mm, bulletIndent=4*mm, value="bullet",
            ))
        flowables.append(ListFlowable(items, bulletType="bullet",
                                      bulletFontSize=8, leftIndent=8*mm))
        flowables.append(Spacer(1, 3*mm))
        bullet_buffer = []

    def flush_para(para_lines, style_key="body"):
        nonlocal in_abstract, in_references
        if not para_lines:
            return
        text = " ".join(l.strip() for l in para_lines if l.strip())
        if not text:
            return

        # 図参照をチェックして図を後挿入するためにテキストを記録
        processed = process_inline(text)
        flowables.append(Paragraph(processed, styles[style_key]))
        add_figure_if_referenced(text)

    para_buf = []
    current_style = "body"

    while i < len(lines):
        line = lines[i]
        raw = line.rstrip()

        # ---- HR / section separator ----
        if re.match(r'^---+$', raw):
            flush_bullets()
            flush_para(para_buf, current_style)
            para_buf = []
            flowables.append(HRFlowable(width="100%", thickness=0.5,
                                        color=HexColor("#cccccc"), spaceAfter=4))
            i += 1
            continue

        # ---- Header ----
        m = re.match(r'^(#{1,3})\s+\*?\*?(.+?)\*?\*?$', raw)
        if m:
            flush_bullets()
            flush_para(para_buf, current_style)
            para_buf = []
            level = len(m.group(1))
            heading = m.group(2).strip().strip('*')

            # Abstract 特別処理
            if heading.lower() in ("abstract",):
                in_abstract = True
                flowables.append(Paragraph("Abstract", styles["abstract_head"]))
                i += 1
                continue

            in_abstract = False
            if "References" in heading:
                in_references = True
                flowables.append(PageBreak())
            if "Appendix" in heading:
                in_appendix = True

            style_key = {1: "h1", 2: "h2", 3: "h3"}.get(level, "h2")
            flowables.append(Spacer(1, 3*mm))
            # process_inline を使うことで $...$ のインライン数式レンダリングと
            # U+2011 ノンブレークハイフンの正規化も行う
            flowables.append(Paragraph(process_inline(heading), styles[style_key]))
            i += 1
            continue

        # ---- 著者行（太字のない短い行、h1 直後）----
        if re.match(r'^\*\*(.+?)\*\*$', raw) and len(raw) < 80:
            flush_para(para_buf, current_style)
            para_buf = []
            name = raw.strip("*")
            flowables.append(Paragraph(escape_xml(name), styles["author"]))
            i += 1
            continue

        # ---- ディスプレイ数式 $$...$$ ----
        if raw.strip().startswith("$$"):
            flush_bullets()
            flush_para(para_buf, current_style)
            para_buf = []
            math_lines = []
            if raw.strip() == "$$":
                i += 1
                while i < len(lines) and lines[i].strip() != "$$":
                    math_lines.append(lines[i])
                    i += 1
                i += 1  # closing $$
            else:
                content = raw.strip()[2:]
                if content.endswith("$$"):
                    content = content[:-2]
                math_lines = [content]
                i += 1

            expr = " ".join(l.strip() for l in math_lines).strip()
            if expr:
                img_path = render_display_math(expr)
                rendered = False
                if img_path:
                    try:
                        img = PILImage.open(img_path)
                        iw, ih = img.size
                        dpi = 150
                        scale = min((TEXT_W * 0.7) / (iw / dpi * 72), 1.0)
                        w = (iw / dpi * 72) * scale
                        h = (ih / dpi * 72) * scale
                        rl_img = RLImage(img_path, width=w, height=h)
                        table = Table([[rl_img]], colWidths=[TEXT_W])
                        table.setStyle(TableStyle([
                            ("ALIGN", (0,0), (-1,-1), "CENTER"),
                            ("TOPPADDING", (0,0), (-1,-1), 4),
                            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
                        ]))
                        flowables.append(table)
                        rendered = True
                    except Exception:
                        pass
                if not rendered:
                    flowables.append(Paragraph(
                        f'<i>{escape_xml(math_to_unicode(expr))}</i>',
                        styles["body_indent"]
                    ))
            continue

        # ---- コードブロック ----
        if raw.startswith("```"):
            flush_bullets()
            flush_para(para_buf, current_style)
            para_buf = []
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            if code_lines:
                code_text = "<br/>".join(
                    escape_xml(l) for l in code_lines
                )
                flowables.append(Paragraph(
                    f'<font name="Courier" size="8">{code_text}</font>',
                    styles["code"]
                ))
            continue

        # ---- テーブル ----
        if raw.startswith("|") and "|" in raw[1:]:
            flush_bullets()
            flush_para(para_buf, current_style)
            para_buf = []
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            # セパレータ行を除去
            data_lines = [l for l in table_lines
                          if not re.match(r'^\|[-| :]+\|$', l)]
            if data_lines:
                table_data = []
                for tl in data_lines:
                    # \| inside $...$ must not be treated as column separator
                    _tl = re.sub(r'\$[^$]*\$', lambda m: m.group().replace("|", "\x00PIPE\x00"), tl)
                    cells = [c.strip().replace("\x00PIPE\x00", "|") for c in _tl.strip("|").split("|")]
                    row = [Paragraph(process_inline(c), ParagraphStyle(
                        "tc", fontName="Arial", fontSize=8.5, leading=12))
                        for c in cells]
                    table_data.append(row)
                ncols = max(len(r) for r in table_data)
                col_w = TEXT_W / ncols
                t = Table(table_data, colWidths=[col_w] * ncols)
                t.setStyle(TableStyle([
                    ("FONTNAME",    (0,0), (-1,0),  "Arial-Bold"),
                    ("FONTSIZE",    (0,0), (-1,-1), 8.5),
                    ("BACKGROUND",  (0,0), (-1,0),  HexColor("#dde4ee")),
                    ("ROWBACKGROUNDS", (0,1), (-1,-1),
                     [white, HexColor("#f7f9fc")]),
                    ("GRID",        (0,0), (-1,-1), 0.4, HexColor("#bbbbbb")),
                    ("VALIGN",      (0,0), (-1,-1), "TOP"),
                    ("TOPPADDING",  (0,0), (-1,-1), 3),
                    ("BOTTOMPADDING",(0,0), (-1,-1), 3),
                    ("LEFTPADDING", (0,0), (-1,-1), 4),
                ]))
                flowables.append(Spacer(1, 2*mm))
                flowables.append(t)
                flowables.append(Spacer(1, 3*mm))
            continue

        # ---- 箇条書き ----
        m_bullet = re.match(r'^[-*+]\s+(.+)', raw)
        if m_bullet:
            flush_para(para_buf, current_style)
            para_buf = []
            bullet_buffer.append(m_bullet.group(1))
            i += 1
            continue

        # ---- 番号付きリスト ----
        m_num = re.match(r'^\d+\.\s+(.+)', raw)
        if m_num:
            flush_para(para_buf, current_style)
            para_buf = []
            flush_bullets()
            flowables.append(Paragraph(
                f"• {process_inline(m_num.group(1))}",
                styles["bullet"]
            ))
            i += 1
            continue

        # ---- 引用ブロック > ----
        if raw.startswith("> "):
            flush_bullets()
            flush_para(para_buf, current_style)
            para_buf = []
            content = raw[2:].strip().strip("*")
            flowables.append(Paragraph(
                f"<i>{process_inline(content)}</i>",
                styles["body_indent"]
            ))
            i += 1
            continue

        # ---- 参考文献行 [N] ----
        if in_references and re.match(r'^\[(\d+)\]', raw):
            flush_bullets()
            flush_para(para_buf, current_style)
            para_buf = []
            flowables.append(Paragraph(process_inline(raw), styles["ref"]))
            i += 1
            continue

        # ---- 空行 ----
        if not raw.strip():
            flush_bullets()
            if para_buf:
                flush_para(para_buf,
                           "abstract" if in_abstract else current_style)
                para_buf = []
            else:
                flowables.append(Spacer(1, 2*mm))
            i += 1
            continue

        # ---- 通常テキスト行 ----
        # 区切り線のみの行をスキップ
        if re.match(r'^-{3,}$', raw):
            i += 1
            continue

        para_buf.append(raw)
        i += 1

    # 残りをフラッシュ
    flush_bullets()
    flush_para(para_buf, current_style)

    # 未挿入の Appendix 図を末尾に追加
    for fig_key in FIGURES:
        if fig_key.startswith("Fig. A") and fig_key not in inserted_figs:
            flowables.extend(make_figure_flowable(fig_key, styles, max_w=TEXT_W))
            inserted_figs.add(fig_key)

    return flowables


# ============================================================
# ページ番号フッター
# ============================================================

def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 8)
    canvas.setFillColor(LGRAY)
    page_num = canvas.getPageNumber()
    canvas.drawCentredString(PAGE_W / 2, 12 * mm, str(page_num))
    canvas.restoreState()


# ============================================================
# メイン処理
# ============================================================

def build_pdf():
    md_path = REPORT / "manuscript_v_20260314.md"
    md_text = md_path.read_text(encoding="utf-8")

    # --- タイトルブロックを特別処理 ---
    title_match = re.search(
        r'#\s+\*\*(.+?)\*\*\s*\n+\*\*(.+?)\*\*',
        md_text, re.DOTALL
    )

    styles = make_styles()

    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=MARGIN_L,
        rightMargin=MARGIN_R,
        topMargin=MARGIN_T,
        bottomMargin=MARGIN_B,
        title="Differentiable Optimization of Non-Commutative Intervention Sequences via Banach-Space Smoothing",
        author="Hisanori Yoshimura",
    )

    flowables = []

    # タイトル
    flowables.append(Spacer(1, 8 * mm))
    flowables.append(Paragraph(
        "Differentiable Optimization of Non-Commutative Intervention Sequences<br/>"
        "via Banach-Space Smoothing",
        styles["title"]
    ))
    flowables.append(Spacer(1, 3 * mm))
    flowables.append(Paragraph("Hisanori Yoshimura", styles["author"]))
    flowables.append(HRFlowable(width="60%", thickness=1,
                                color=ACCENT, spaceAfter=6))
    flowables.append(Spacer(1, 4 * mm))

    # タイトル行とタイトルブロックをスキップして本文パース
    # 最初の --- 以降から開始
    body_start = md_text.find("\n---\n", md_text.find("Hisanori Yoshimura"))
    if body_start == -1:
        body_start = 0
    body_text = md_text[body_start + 5:]  # "---\n" をスキップ

    flowables.extend(parse_markdown(body_text, styles))

    print(f"Building PDF: {OUT_PDF}")
    doc.build(flowables, onFirstPage=on_page, onLaterPages=on_page)
    print(f"Done: {OUT_PDF}")


if __name__ == "__main__":
    build_pdf()
