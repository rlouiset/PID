"""
Regenerate the two appendix figures from rho.json and capacity.json.
Produces clip_supgk_source.pdf and clip_supgk_mech.pdf.

Uses R_source / H(Y) as the y-axis (H(Y)=log 4 = 1.386 nats), so numbers match the tables.
All three seeds are kept: the one Stage-1 failure affects only I_1/R_imin, not R_source.

    python make_figures.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt

HY = np.log(4)


def load(path):
    with open(path) as f:
        return json.load(f)


def agg(rows, key_filter, xkey, field):
    """mean/std of `field`/HY*100 grouped by xkey, over seeds."""
    xs = sorted({r[xkey] for r in rows if key_filter(r)})
    means, stds = [], []
    for x in xs:
        vals = [100 * r[field] / HY for r in rows if key_filter(r) and r[xkey] == x]
        means.append(np.mean(vals)); stds.append(np.std(vals))
    return np.array(xs), np.array(means), np.array(stds)


def agg_raw(rows, key_filter, xkey, field):
    xs = sorted({r[xkey] for r in rows if key_filter(r)})
    return np.array(xs), np.array([np.mean([r[field] for r in rows
                     if key_filter(r) and r[xkey] == x]) for x in xs])


# ---------------- Figure 1: source redundancy ----------------
rho = load("rho.json")
cap = load("capacity.json")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)

x, mc, sc = agg(rho, lambda r: True, "rho", "R_src_clip")
_, mg, sg = agg(rho, lambda r: True, "rho", "R_src_supgk")
axL.errorbar(x, mg, sg, marker="o", capsize=3, label="SupGK")
axL.errorbar(x, mc, sc, marker="s", capsize=3, label="CLIP")
axL.axhline(86, ls="--", c="grey", lw=1.2, label="ceiling")
axL.set_xlabel(r"nuisance shared, $\rho$"); axL.set_ylabel(r"source redundancy (\% of $H(Y)$)")
axL.set_title(r"sharing sweep ($d_{emb}=512$)", fontsize=10)
axL.set_ylim(0, 100); axL.legend(frameon=False, fontsize=9, loc="lower left")

x, mc, sc = agg(cap, lambda r: True, "d_emb", "R_src_clip")
_, mg, sg = agg(cap, lambda r: True, "d_emb", "R_src_supgk")
axR.errorbar(x, mg, sg, marker="o", capsize=3, label="SupGK")
axR.errorbar(x, mc, sc, marker="s", capsize=3, label="CLIP")
axR.axhline(86, ls="--", c="grey", lw=1.2)
axR.set_xscale("log", base=2); axR.set_xticks([32, 64, 128, 256, 512])
axR.set_xticklabels([32, 64, 128, 256, 512])
axR.set_xlabel(r"embedding size $d_{emb}$")
axR.set_title(r"capacity sweep ($\rho=1$)", fontsize=10)

fig.tight_layout(); fig.savefig("clip_supgk_source.pdf", bbox_inches="tight")
print("wrote clip_supgk_source.pdf")

# ---------------- Figure 2: mechanism (R2 of z_c, z_n from CLIP) ----------------
fig2, ax = plt.subplots(figsize=(5.6, 3.4))
x, zc = agg_raw(rho, lambda r: True, "rho", "r2_zc_clip")
_, zn = agg_raw(rho, lambda r: True, "rho", "r2_zn_clip")
ax.plot(x, zc, marker="o", label=r"signal $z_c$")
ax.plot(x, zn, marker="^", label=r"nuisance $z_n$")
ax.set_xlabel(r"nuisance shared, $\rho$"); ax.set_ylabel(r"linear recoverability $R^2$ (CLIP)")
ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=9, loc="upper left")
fig2.tight_layout(); fig2.savefig("clip_supgk_mech.pdf", bbox_inches="tight")
print("wrote clip_supgk_mech.pdf")