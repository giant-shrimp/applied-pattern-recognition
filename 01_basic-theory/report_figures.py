"""基礎理論 演習レポート：答えの検算と解答図の描画．

問題1  2クラスの正規分布によるベイズ識別の決定境界
問題2  体積不変の正規化とKL展開（主成分分析）

実行すると，検算結果を標準出力に表示し，解答用の図を output/ に保存する．
    python report_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Ellipse

OUT_DIR = Path(__file__).resolve().parent / "output"

BLUE = "#1f4fd8"
RED = "#e03030"
GREEN = "#1a8f3c"


def set_japanese_font():
    """環境にある日本語フォントを選ぶ（図の凡例・ラベル用）．"""
    candidates = ["IPAGothic", "Yu Gothic", "Meiryo", "Hiragino Sans",
                  "Noto Sans CJK JP", "MS Gothic"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["mathtext.fontset"] = "cm"


# ---------------------------------------------------------------- 問題1
def problem1():
    r3 = np.sqrt(3.0)
    x1 = np.array([[-5, 1], [-1, 1], [-3, 0], [-3, 2]], dtype=float)
    x2 = np.array([[1, 1], [5, 1], [3, 0], [3, 2],
                   [3 - 2 * r3, 1], [3 + 2 * r3, 1],
                   [3, 1 - r3], [3, 1 + r3]])

    # 事前確率（クラスの出現頻度）
    n1, n2 = len(x1), len(x2)
    p1, p2 = n1 / (n1 + n2), n2 / (n1 + n2)

    # 平均ベクトルと共分散行列の最尤推定値（分母は各クラスのパターン数）
    mu1, mu2 = x1.mean(axis=0), x2.mean(axis=0)
    sigma1 = (x1 - mu1).T @ (x1 - mu1) / n1
    sigma2 = (x2 - mu2).T @ (x2 - mu2) / n2

    def g(x, mu, sigma, prior):
        """正規分布に対するベイズ識別関数（d = 2）．"""
        d = x - mu
        return (-0.5 * d @ np.linalg.inv(sigma) @ d - np.log(2 * np.pi)
                - 0.5 * np.log(np.linalg.det(sigma)) + np.log(prior))

    print("=== 問題1 ===")
    print(f"P(w1) = {p1:.4f}, P(w2) = {p2:.4f}")
    print("mu1 =", mu1, " mu2 =", mu2)
    print("Sigma1 =\n", sigma1)
    print("Sigma2 =\n", sigma2)

    # 決定境界 (x1 + 9)^2 / 72 + (x2 - 1)^2 / 18 = 1 の検算
    center, a, b = np.array([-9.0, 1.0]), 6 * np.sqrt(2), 3 * np.sqrt(2)
    t = np.linspace(0, 2 * np.pi, 361)
    boundary = center + np.column_stack([a * np.cos(t), b * np.sin(t)])
    diff = [g(p, mu1, sigma1, p1) - g(p, mu2, sigma2, p2) for p in boundary]
    print(f"楕円上での |g1 - g2| の最大値: {np.max(np.abs(diff)):.2e}")
    ok1 = all(g(p, mu1, sigma1, p1) > g(p, mu2, sigma2, p2) for p in x1)
    ok2 = all(g(p, mu1, sigma1, p1) < g(p, mu2, sigma2, p2) for p in x2)
    print("w1 のパターンをすべて w1 と識別:", ok1)
    print("w2 のパターンをすべて w2 と識別:", ok2)

    # 図：決定境界と与えられたパターン
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.set_facecolor("#fdeaea")
    ax.add_patch(Ellipse(center, 2 * a, 2 * b, facecolor="#e3ecfd",
                         edgecolor=GREEN, linewidth=2.2, zorder=1))
    ax.grid(color="0.75", linewidth=0.6, zorder=1.5)
    ax.scatter(x1[:, 0], x1[:, 1], s=55, c=BLUE, marker="o",
               label=r"$\omega_1$ のパターン", zorder=4)
    ax.scatter(x2[:, 0], x2[:, 1], s=55, c=RED, marker="x", linewidths=2,
               label=r"$\omega_2$ のパターン", zorder=4)
    ax.scatter(*mu1, s=90, c=BLUE, marker="+", linewidths=2, zorder=4)
    ax.scatter(*mu2, s=90, c=RED, marker="+", linewidths=2, zorder=4)
    ax.scatter([], [], s=90, c="0.3", marker="+", linewidths=2,
               label="平均ベクトル")
    ax.plot([], [], color=GREEN, linewidth=2.2, label="決定境界")
    ax.text(-14.5, 2.8, r"$\omega_1$ と識別される領域", color=BLUE,
            fontsize=11, ha="center")
    ax.text(3.2, 5.6, r"$\omega_2$ と識別される領域", color=RED,
            fontsize=11, ha="center")
    ax.set_xlim(-19, 8)
    ax.set_ylim(-5, 7)
    ax.set_aspect("equal")
    ax.set_xticks(range(-18, 9, 2))
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(loc="lower right", fontsize=9, framealpha=1)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "problem1_decision_boundary.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- 問題2
def problem2():
    x = np.array([[2, 8], [2, 4], [6, 4], [6, 0]], dtype=float)
    n = len(x)

    # 体積不変の正規化（標本分散の分母は N - 1）
    x_bar = x.mean(axis=0)
    s = np.sqrt(((x - x_bar) ** 2).sum(axis=0) / (n - 1))
    a = np.prod(s) ** (1 / len(s)) / s      # a_j = (prod s_k)^(1/d) / s_j
    z = (x - x_bar) * a

    # KL展開（経験共分散行列の分母は N）
    sigma = z.T @ z / n                     # z の平均は 0
    eigvals, eigvecs = np.linalg.eigh(sigma)
    u1 = eigvecs[:, np.argmax(eigvals)]
    if u1[0] < 0:                           # 符号を (1, -1)/sqrt(2) の向きにそろえる
        u1 = -u1
    y = z @ u1

    print("=== 問題2 ===")
    print("標本平均 =", x_bar)
    print("標本分散 =", s ** 2)
    print("a =", a, " det A =", np.prod(a))
    print("正規化後のパターン z =\n", z)
    print("正規化後の標本分散 =", z.var(axis=0, ddof=1))
    print("経験共分散行列 =\n", sigma)
    print("固有値 =", np.sort(eigvals)[::-1])
    print("変換ベクトル u1 =", u1)
    print("変換後のパターン y =", y)

    # 図：変換後のパターン（数直線）
    fig, ax = plt.subplots(figsize=(7, 1.7))
    ax.scatter(y, np.zeros(n), s=70, c=RED, zorder=3)
    for k, v in enumerate(y, start=1):
        ax.text(v, 0.3, rf"$y_{k}\approx{v:.3f}$", ha="center",
                fontsize=11, color=RED)
    ax.set_ylim(-0.5, 0.9)
    ax.set_xlim(-5.5, 5.5)
    ax.set_yticks([])
    ax.set_xticks(range(-5, 6))
    ax.set_xlabel(r"$y$")
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "problem2_number_line.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    set_japanese_font()
    problem1()
    problem2()
    print(f"図を {OUT_DIR} に保存しました．")
