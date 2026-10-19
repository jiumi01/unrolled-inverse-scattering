# -*- coding: utf-8 -*-
"""_sync_cn_round4.py -- all Chinese-manuscript synchronisations for revision round R3/R4.

Consolidated on purpose: R2 m3.10 noted that the release carries a large number of one-off
`_*.py` scripts, so this round keeps ONE re-runnable CN synchroniser instead of a new script per
edit.  It is idempotent -- an entry already applied is reported as such rather than only failing
when the source string is gone -- so it can be re-run after further edits.

Each entry mirrors an English edit:
  (1) sec:avail        -- R2 m3.10: sealed drivers vs single-use scripts distinguishable in the
                          release (naming convention + snapshot provenance note; files are not
                          moved, because frozen md5 manifests record paths).
  (2) conclusion       -- R2 m3.2: interval notation must not describe a discrete sweep.
  (3) V-A setup        -- R2 m3.5: disclose the depth-curve schedule and the uniform-40 re-run,
                          with the comparability control stated at the MEASURED re-run floor
                          instead of the retired 5e-5 guess.
  (4) V-B saturation   -- R2 m3.5 / Q3.4, executed on the artifact: under the uniform schedule the
                          archived K=16 -> K=24 uptick is NOT reproduced, so the claim drops from
                          strict monotonicity to resolvability at the measured re-run floor; the
                          replacement names BOTH arms (the unconstrained one does rise).
"""
import io, os, sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
P = os.path.join(ROOT, "paper_cn", "main.tex")

OLD_AVAIL_TAIL = "合成散射数据由随稿生成器以固定种子产出，无需外部下载；配置即第\\ref{sec:setup}节所述。"

AVAIL_ADD = ("发布件中，封存的驱动脚本与一次性修补/探针脚本以文件名区分——驱动不带前导下划线"
             "（\\texttt{isp\\_core.py}、\\texttt{spec\\_tools.py}、\\texttt{gen\\_numbers.py}、"
             "\\texttt{make\\_figs.py}、\\texttt{run\\_*.py}），一次性脚本带（\\texttt{\\_*.py}）"
             "——快照的溯源说明同时列出两类；我们未把它们移入不同目录，因为冻结清单记录的是路径，"
             "移动会使清单中每一条校验和失效。合成散射数据由随稿生成器以固定种子产出，无需外部下载；"
             "配置即第\\ref{sec:setup}节所述。")

V_A_OLD = ("为此我们在完全相同的优化器、批序与种子下，对每个展开深度（$K\\in\\{2,4,8,16,24\\}$）"
           "统一以 $40$ 轮重训并并列报告；该重放由阳性对照校验——在任何新数字产生之前，"
           "它必须把归档的 $40$ 轮点（$K\\le8$）复现到优于 $5\\times10^{-5}$。")
V_A_NEW = ("为此我们在完全相同的优化器、批序与种子下，对每个展开深度（$K\\in\\{2,4,8,16,24\\}$）"
           "统一以 $40$ 轮重训，并以 $K\\le8$ 各点——其日程\\emph{未}改变——作为必须复现归档值的"
           "阳性对照。该对照不能是等值检验：驱动脚本在各深度臂之间不重设种子，故模型的初始化"
           "取决于此前扫描消耗了多少全局随机数；本次重放在计数上忠实，但无法证明在种类上忠实。"
           "因此判据改为流水线自身在同一协议下实测的复跑散布，即测试 RRE 上的 "
           "$\\numv{\\NumDepUniFloor}$（第\\ref{sec:lim}节），而实测的 $K\\le8$ 漂移为 "
           "$\\numv{\\NumDepUniDrift}$，可见本次重放是可比的。此外每个臂都显式设种子，"
           "故跨 $K$ 变化的只有深度。")
# Final state of the V-A paragraph: V_A_NEW plus the quantification of the schedule effect on the
# two depths whose schedule actually changed (so "the replay is comparable" is not the only number).
V_A_FINAL = ("为此我们在完全相同的优化器、批序与种子下，对每个展开深度（$K\\in\\{2,4,8,16,24\\}$）"
             "统一以 $40$ 轮重训，并以 $K\\le8$ 各点——其日程\\emph{未}改变——作为必须复现归档值的"
             "阳性对照。该对照不能是等值检验：驱动脚本在各深度臂之间不重设种子，故模型的初始化"
             "取决于此前扫描消耗了多少全局随机数；本次重放在计数上忠实，但无法证明在种类上忠实。"
             "因此判据改为流水线自身在同一协议下实测的复跑散布，即测试 RRE 上的 "
             "$\\numv{\\NumDepUniFloor}$（第\\ref{sec:lim}节），而实测的 $K\\le8$ 漂移为 "
             "$\\numv{\\NumDepUniDrift}$，可见本次重放是可比的。约束臂上日程\\emph{确实}改变的两个"
             "深度分别移动 $\\numv{\\NumDepUniShiftSixteen}$（$K=16$）与 "
             "$\\numv{\\NumDepUniShiftTwentyFour}$（$K=24$），均低于该地板，故日程效应本身亦未被"
             "分辨。此外每个臂都显式设种子，故跨 $K$ 变化的只有深度。")

V_B_OLD = ("约束曲线基本单调（除 $K=16\\to24$ 处有 $0.2\\%$ 的轻微回升外，在每个实测深度上都下降）"
           "并饱和，与深度分解（几何深度项 + 修正地板）一致；")
# Intermediate and final states of the V-B sentence.  The FINAL state quotes the archived
# percentage from the artifact instead of from the text (that was the S12 defect reappearing
# inside its own fix) and gives the uniform-40 endpoints of the contested interval, so the claim
# is checkable from the manuscript alone.
V_B_NEW = ("约束曲线在每个实测深度上都下降并在 $K=16$ 处饱和，与深度分解（几何深度项 + 修正地板）一致。"
           "归档曲线所显示的 $K=16\\to24$ 处 $0.2\\%$ 回升，在把各深度统一日程重训后\\emph{未被复现}"
           "（第\\ref{sec:exp}节）：同一区间给出 $\\numv{\\NumDepDeltaUniPct}\\%$，"
           "故我们不再把该曲线描述为非单调。该区间在任何情形下均为\\emph{未分辨}——"
           "\\emph{两个}臂的变化都低于实测复跑地板（$\\numv{\\NumDepUniFloor}$；"
           "无约束臂在该处实际上升 $\\numv{\\NumDepDeltaUncUniPct}\\%$）——"
           "故本文所称“饱和”是关于\\emph{可分辨性}的陈述"
           "（$K=16$ 之后没有超出仪器噪声的变化被分辨出来），而非严格单调性主张；")
V_B_FINAL = ("约束曲线在每个实测深度上都下降并在 $K=16$ 处饱和，与深度分解（几何深度项 + 修正地板）一致。"
             "归档曲线所显示的 $K=16\\to24$ 处 $\\numv{\\NumDepDeltaArchPct}\\%$ 回升，"
             "在把各深度统一日程重训后\\emph{未被复现}——同一区间读出 "
             "$\\numv{\\NumDepUniConsSixteen}\\!\\to\\!\\numv{\\NumDepUniConsTwentyFour}$，"
             "即 $\\numv{\\NumDepDeltaUniPct}\\%$（第\\ref{sec:exp}节）——"
             "故我们不再把该曲线描述为非单调。该区间在任何情形下均为\\emph{未分辨}——"
             "\\emph{两个}臂的变化都低于实测复跑地板（$\\numv{\\NumDepUniFloor}$；"
             "无约束臂在该处实际上升 $\\numv{\\NumDepDeltaUncUniPct}\\%$）——"
             "故本文所称“饱和”是关于\\emph{可分辨性}的陈述"
             "（$K=16$ 之后没有超出仪器噪声的变化被分辨出来），而非严格单调性主张；")

# Each entry is (alternative_anchors, replacement).  The anchor set is a TUPLE because a chain of
# edits to one region breaks the naive single-anchor test: after the follow-up edit, the earlier
# entry's replacement no longer occurs verbatim, so that entry MISSES and the whole script aborts
# (this happened for real -- three consecutive runs of the previous version reported two spurious
# misses).  Accepting several pre-states makes the script order-tolerant across intermediate
# revisions, and testing the replacement first makes it idempotent.
REPS = [
    # (0) abstract -- R3 n3.4: the English abstract was shortened until BOTH word-count
    #     conventions fit under 250 (see _swap_abstract.py); the CN abstract mirrors the two
    #     content removals.  Entry kept here so the consolidation is complete.
    (("因此在每个实测步长（其跨度覆盖 $\\alpha\\in[0.05,1]$）上全局欧氏压缩都不可达，稳定性保证必须分层给出。",),
     "因此全局欧氏压缩在实测步长上不可达，稳定性必须分层给出。"),
    (("合成散射数据由随稿生成器以固定种子产出，无需外部下载；配置即第\\ref{sec:setup}节所述。",),
     AVAIL_ADD),
    (("在\\emph{每个实测步长}（跨度覆盖 $\\alpha\\in[0.05,1]$）上\\emph{不是}全局压缩的",),
     "在\\emph{每个实测步长}上\\emph{不是}全局压缩的"),
    # (1) V-A -- R2 m3.5: disclose the schedule + the uniform-40 re-run, with the comparability
    #     control at the MEASURED re-run floor rather than the retired 5e-5 guess.
    ((V_A_OLD, V_A_NEW), V_A_FINAL),
    # (2) V-B -- R2 m3.5 / Q3.4 resolved on the artifact: under the uniform schedule the archived
    #     K=16 -> K=24 rise is NOT reproduced, so the claim drops from strict monotonicity to
    #     resolvability at the measured re-run floor.  BOTH arms are named on purpose, so the
    #     paragraph cannot be read as selective reporting (the unconstrained arm does rise).
    ((V_B_OLD, V_B_NEW), V_B_FINAL),
    # (3) fig:depth caption -- the figure now carries BOTH schedules (solid = archived, dashed =
    #     uniform 40-epoch), so the caption must say which is which, and must state that the
    #     solid/dashed separation at K<=8 is at most the measured re-run floor (the replay control
    #     asserts exactly that).
    (("两种展开变体都在单层代价匹配的前提下、以约 $12$ 倍的更少外层迭代达到经典迭代收敛后的精度"
      "（第\\ref{sec:exp}节）。}",),
     "两种展开变体都在单层代价匹配的前提下、以约 $12$ 倍的更少外层迭代达到经典迭代收敛后的精度"
     "（第\\ref{sec:exp}节）。实线为归档扫描曲线；虚线为把\\emph{每个}深度都改用同一 $40$ 轮日程"
     "重训后的同样两臂（第\\ref{sec:exp}节）——即实线仅在 $K\\le8$ 处采用的日程。由于两个日程在 "
     "$K\\le8$ 处一致，该处实线与虚线之差\\emph{至多}为实测复跑地板"
     "（$\\numv{\\NumDepUniFloor}$），这正是复放阳性对照所断言的；虚线扫描按臂设种子，"
     "故跨 $K$ 变化的只有深度。}"),
]


def main():
    s = io.open(P, encoding="utf-8").read()

    # --- normalisation, and TWO bugs of this very script -----------------------------
    # The availability entry's replacement text CONTAINS its anchor (the new paragraph ends
    # with the sentence it was inserted before).  Consequence 1: the first version of this
    # file tested `old in s` first and re-inserted the paragraph on every run, so three runs
    # meant three copies.  Consequence 2, worse: the de-duplication used
    #     re.sub(r"(?:prefix)+", prefix, s)
    # and in a regex REPLACEMENT string "\t" is a tab, so the 6 occurrences of `\texttt` in
    # the paragraph came back as TAB+"exttt" -- the copy stopped matching the prefix, so it
    # survived the collapse as silently broken LaTeX while the count check reported "x1".
    # Fixes: (i) repair both damaged and duplicated copies here using str.replace only (which
    # does NOT interpret escapes), (ii) below, test the *replacement* for presence before the
    # anchor, (iii) assert afterwards that the paragraph appears exactly once AND that no tab
    # character survives in the availability section.
    avail_prefix = AVAIL_ADD[: -len(OLD_AVAIL_TAIL)]
    mangled = avail_prefix.replace("\\t", "\t")
    n_bad = s.count(avail_prefix) + s.count(mangled)
    removed = 0
    if n_bad > 1:                      # repair only when there is something to repair
        for bad in (avail_prefix, mangled):
            while bad and bad in s:
                s = s.replace(bad, "", 1)
                removed += 1
    if removed:
        print("[fix  ] removed %d copy/copies of the availability paragraph" % removed)
    # Re-insert exactly one clean copy immediately before the sentence it precedes.
    if avail_prefix not in s:
        if OLD_AVAIL_TAIL not in s:
            print("[abort] anchor sentence for the availability paragraph not found")
            return 4
        s = s.replace(OLD_AVAIL_TAIL, AVAIL_ADD, 1)

    applied = missed = 0
    for olds, new in REPS:
        label = olds[0][:44]
        if new in s:
            print("[have ] %s" % label)
            continue
        for old in olds:
            if old in s and old != new:
                s = s.replace(old, new, 1)
                applied += 1
                print("[apply] %s" % label)
                break
        else:
            print("[MISS ] %s" % label)
            missed += 1
    if missed:
        print("[abort] %d entry/entries not found; nothing written" % missed)
        return 2
    # Instrument self-checks on the one entry whose replacement contains its anchor.
    # NOTE on the mangling signature: `\texttt` itself contains the substring "exttt", so a
    # naive `"exttt" in region` flags all 42 legitimate occurrences.  The actual signature of
    # the escape corruption is a TAB immediately followed by "exttt" (the "\t" of "\texttt"
    # consumed as a tab escape); the tab count already covers it, and it is asserted separately
    # so the diagnostic is unambiguous.
    n_final = s.count(avail_prefix)
    i = s.find("\\label{sec:avail}")
    j = s.find("\\section", i + 1) if i >= 0 else -1
    region = s[i:j] if (i >= 0 and j > i) else ""
    tabs = region.count("\t")
    mangled_mark = region.count("\t" + "exttt")
    if n_final != 1 or tabs or mangled_mark:
        print("[abort] availability paragraph x%d, %d tab char(s), %d mangled macro marker(s)"
              " -> not written" % (n_final, tabs, mangled_mark))
        return 3
    io.open(P, "w", encoding="utf-8", newline="\n").write(s)
    print("[done] CN synchronised (%d applied; availability paragraph x1, 0 tab, no mangling)"
          % applied)
    return 0


if __name__ == "__main__":
    sys.exit(main())
