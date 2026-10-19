# -*- coding: utf-8 -*-
"""_patch_listacp_cn.py — make the Chinese LISTA-CP remark parallel to the English one and
remove the inaccurate "spectral-norm ball" phrasing (review R3 m3.10 / section 6).

The earlier Chinese sentence asserted that the learnable correction is constrained inside a
spectral-norm ball around T^phys.  The actual device is an L-infinity amplitude budget plus a
per-kernel spectral clamp used as a DESIGN target; that is a neighbourhood condition, and the
English version does not (and should not) claim more.
"""
import io, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))

OLD = ("LISTA-CP\\cite{listacp}以 $W_1^kA+W_2^k=I$ 把每层两个可学习矩阵耦合到已知测量矩阵 $A$。"
       "本文的锚定是其非线性类似物：可学习修正被约束在\\emph{相对于}已知物理算子 "
       "$T^{\\mathrm{phys}}$ 的谱范数球内，且被保证的对象不是稀疏码的恢复，"
       "而是物理迭代的不动点与有界性。")

NEW = ("LISTA-CP\\cite{listacp}以\\emph{等式} $W_1^kA+W_2^k=I$ 把每层两个可学习矩阵耦合到已知测量矩阵 $A$。"
       "本文的构造与之是\\emph{同族}关系而非类比，我们不做更强的主张：二者都把可学习部分系于一个已知算子，"
       "但 LISTA-CP 用的是权重上的等式约束，本文施加的是 Lipschitz（幅度）预算与逐核谱钳位——"
       "这是邻域条件而非等式；且被保证的对象不是稀疏码的恢复（本文不涉及），"
       "而是物理迭代的不动点与有界性。")


def main():
    p = os.path.join(ROOT, "paper_cn", "main.tex")
    s = io.open(p, encoding="utf-8").read()
    assert s.count(OLD) == 1, "anchor count = %d" % s.count(OLD)
    io.open(p, "w", encoding="utf-8").write(s.replace(OLD, NEW, 1))
    print("CN LISTA remark rewritten")


if __name__ == "__main__":
    main()
