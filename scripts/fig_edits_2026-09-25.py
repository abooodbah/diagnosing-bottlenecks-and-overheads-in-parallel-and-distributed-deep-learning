# -*- coding: utf-8 -*-
"""2026-09-25 figure-label edits from the companion clarity review:
panel (a) counts marked as this configuration's; panel (c) timer-coverage
note; panel (d) title, axis label, and transport-explicit legend; routing
panels labeled as the selected worker's. Edits companion_figures.py."""
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "companion_figures.py")
t = io.open(P, encoding="utf-8").read()

subs = [
# fig1 panel (a) footer
('ax.text(50, -1.5, "(a) state, observed collectives, and timer boundaries", ha="center", fontsize=7.8)',
 'ax.text(50, -1.5, "(a) state, collectives observed in this configuration, and timer boundaries", ha="center", fontsize=7.8)'),
# fig2 panel (c): timer-coverage note under the legend
('''    c.legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, frameon=False, columnspacing=0.9)
    c.set_title("(c) four-node composition", fontsize=8.5)''',
 '''    c.legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=2, frameon=False, columnspacing=0.9)
    c.text(0.5, -0.62, "H100 ZeRO-3 bars use DeepSpeed-layer intervals; all others use torch-layer timers",
           transform=c.transAxes, ha="center", va="top", fontsize=5.4, color="#444444")
    c.set_title("(c) four-node composition", fontsize=8.5)'''),
# fig2 panel (d): explicit transport suffixes
('''            lab = "%s %d node%s%s" % ("H100" if plat == "h100" else "Intel", n, "" if n == 1 else "s",
                                      ", TCP" if tcp else (", host-staged" if host else ""))''',
 '''            sfx = ", TCP" if tcp else (", IB staged" if host else (", IB" if (plat == "h100" and n > 1) else ""))
            lab = "%s %d node%s%s" % ("H100" if plat == "h100" else "Intel", n, "" if n == 1 else "s", sfx)'''),
# fig2 panel (d): axis label and title
('d.set_xlabel("training step (warm-up + measured)", fontsize=8)',
 'd.set_xlabel("iteration index (warm-up + measured)", fontsize=8)'),
('d.set_title("(d) ZeRO-3 step time per logged iteration", fontsize=8.5)',
 'd.set_title("(d) ZeRO-3 step per iteration (logger enabled)", fontsize=8.5)'),
# fig3: mark the bar panels as the selected worker's
('for ax, p, title in ((a, "pvc", "Intel (16 ranks)"), (b, "h100", "H100 (8 ranks)")):',
 'for ax, p, title in ((a, "pvc", "Intel (16 ranks), selected worker"), (b, "h100", "H100 (8 ranks), selected worker")):'),
]
for a, b in subs:
    assert t.count(a) == 1, a[:70]
    t = t.replace(a, b)
io.open(P, "w", encoding="utf-8", newline="\n").write(t)
print("figure script edited")
