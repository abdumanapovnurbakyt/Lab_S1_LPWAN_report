import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SFC = {7: CAT[0], 9: CAT[1], 12: CAT[2]}
MARK = {7: "o", 9: "s", 12: "^"}
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NB = " "

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.labelcolor": INK2, "axes.labelsize": 11, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.edgecolor": BASE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 9,
    "legend.frameon": False, "legend.fontsize": 10.5, "lines.linewidth": 1.7,
    "xtick.major.size": 3, "ytick.major.size": 3,
})


def ru(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def arr(x):
    return np.array([np.nan if v is None else v for v in x], dtype=float)


def n_fmt(v, _=None):
    return f"{int(round(v)):,}".replace(",", NB)


def n_short(v, _=None):
    return f"{v / 1000:g}k" if v >= 1000 else f"{int(round(v))}"


def t_label(t):
    if t < 60:
        return f"{t} с"
    if t < 3600:
        return f"{t // 60} мин"
    return f"{t // 3600} ч" if t < 86400 else "1 сут"


def lum(hexc):
    r, g, b = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(hexc)]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def text_on(hexc):
    l = lum(hexc)
    return INK if (l + 0.05) / 0.05 >= 1.05 / (l + 0.05) else "#ffffff"


def spread(ys, gap):
    ys = np.array(ys, float)
    order = np.argsort(ys)
    out = ys.copy()
    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        if out[j] - out[i] < gap:
            out[j] = out[i] + gap
    return out


def end_labels(ax, items, gap, x=1.012, transform_y=lambda v: v):
 
    xl, yl = ax.get_xlim(), ax.get_ylim()
    ys = spread([transform_y(i[0]) for i in items], gap)
    logy = transform_y(1.0) != 1.0
    tr = ax.get_yaxis_transform()
    for (y0, txt, col, mk), y in zip(items, ys):
        yy = 10 ** y if logy else y
        ax.plot([x + 0.006], [yy], marker=mk, ms=6.5, color=col, mec=SURFACE, mew=0.8, ls="none", transform=tr,
                clip_on=False, zorder=6)
        ax.annotate(txt, xy=(x + 0.024, yy), xycoords=("axes fraction", "data"), color=INK, fontsize=11,
                    fontweight="bold", va="center", ha="left", annotation_clip=False)
    ax.set_xlim(xl)
    ax.set_ylim(yl)


def tag(ax, x, y, text, color, marker="o"):
    ax.plot([x], [y], marker=marker, ms=6, color=color, mec=SURFACE, mew=0.8, ls="none", zorder=6)
    ax.text(x, y, "  " + text, color=INK, fontsize=11, fontweight="bold", va="center", ha="left", zorder=6)


def clean_axes(ax, grid_x=False):
    ax.spines["left"].set_color(BASE)
    ax.spines["bottom"].set_color(BASE)
    if grid_x:
        ax.grid(axis="x", color=GRID, lw=0.7)


def save(fig, out, name):
    d = os.path.join(out, "figures")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def log_ticks(ax, lo, hi):
    from matplotlib.ticker import NullLocator
    t = [m * 10 ** e for e in range(-1, 7) for m in (1, 2, 5)]
    t = [v for v in t if lo <= v <= hi]
    ax.set_xticks(t)
    ax.xaxis.set_major_formatter(FuncFormatter(n_fmt))
    ax.xaxis.set_minor_locator(NullLocator())


def set_n_axis(ax, ticks=(10, 100, 500, 1000, 5000, 50000)):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(n_short))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def fig_aloha(R, out):
    a = R["aloha"]
    G = np.array(a["G"])
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 4.0))
    th_p, th_s = np.array(a["theory_pdr"]), np.array(a["theory_s"])
    nc_p, nc_s = np.array(a["nocap"]["pdr"]), np.array(a["nocap"]["s"])
    c_p, c_s = np.array(a["cap"]["pdr"]), np.array(a["cap"]["s"])

    ax = axs[0]
    ax.plot(G, th_p * 100, color=INK2, ls="--", lw=1.2)
    ax.plot(G, nc_p * 100, color=CAT[0], marker="o", ms=4.5, markevery=3, lw=1.5)
    ax.plot(G, c_p * 100, color=CAT[1], marker="s", ms=4.5, markevery=3, lw=1.5)
    ax.set_xlabel("Нагрузка G (пакетов за время одного пакета)")
    ax.set_ylabel("PDR, %")
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 102)
    ax.set_title(r"PDR; теория $e^{-2G}$")
    clean_axes(ax)

    ax = axs[1]
    ax.plot(G, th_s * 100, color=INK2, ls="--", lw=1.2)
    ax.plot(G, nc_s * 100, color=CAT[0], marker="o", ms=4.5, markevery=3, lw=1.5)
    ax.plot(G, c_s * 100, color=CAT[1], marker="s", ms=4.5, markevery=3, lw=1.5)
    pk, pc = a["peak_nocap"], a["peak_cap"]
    ax.axhline(100 / (2 * np.e), color=MUTED, lw=0.8, ls=":")
    s_half = a.get("s_at_half", pk["S"])
    ax.annotate(f"Чистая ALOHA:\n{ru(s_half * 100)} % при G = 0,5",
                xy=(0.5, s_half * 100), xytext=(1.3, 13.2), color=INK, fontsize=11,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkA=0, shrinkB=4))
    ax.annotate(f"Capture: макс. {ru(pc['S'] * 100)} %\nпри G = {ru(pc['G'], 2)}",
                xy=(pc["G"], pc["S"] * 100), xytext=(1.35, max(c_s) * 100 - 0.5), color=INK, fontsize=11,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkA=0, shrinkB=4))
    ax.set_xlabel("Нагрузка G")
    ax.set_ylabel("Пропускная способность S, %")
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, max(c_s) * 100 + 6)
    ax.set_title(r"S = G·PDR; теория $G\,e^{-2G}$")
    clean_axes(ax)
    h = [Line2D([], [], color=INK2, ls="--", lw=1.2, label="Теория (чистая ALOHA)"),
         Line2D([], [], color=CAT[0], marker="o", ms=5, lw=1.5, label="Симуляция, без capture"),
         Line2D([], [], color=CAT[1], marker="s", ms=5, lw=1.5, label="Симуляция, capture 6 дБ")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.92, bottom=0.25, wspace=0.27)
    return save(fig, out, "fig1_aloha.png")


def fig_pdr_vs_n(R, out):
    cfg = R["meta"]["config"]
    ts, sfs, req = cfg["grid"]["interval_s"], cfg["grid"]["sf"], cfg["grid"]["n_devices"]
    fig, axs = plt.subplots(2, 2, figsize=(9.1, 6.7), sharex=True, sharey=True)
    for ax, t in zip(axs.flat, ts):
        ends = []
        for n in req:
            ax.axvline(n, color=GRID, lw=0.7, zorder=0)
        for sf in sfs:
            sw = R["sweep"][f"{sf}_{t}"]
            N, m, ci = np.array(sw["N"]), arr(sw["pdr_mean"]) * 100, arr(sw["pdr_ci"]) * 100
            nm = R["nmax"][f"{sf}_{t}"]
            ls = "-" if nm["legal"] else (0, (4, 2))
            ax.fill_between(N, np.maximum(m - ci, 0), np.minimum(m + ci, 100), color=SFC[sf], alpha=0.16, lw=0)
            ax.plot(N, m, color=SFC[sf], lw=1.6, ls=ls)
            idx = [sw["N"].index(n) for n in req]
            ax.plot(N[idx], m[idx], ls="none", marker=MARK[sf], ms=6.5, mfc=SFC[sf], mec=SURFACE, mew=1.1)
            if nm["flag"] == "":
                ax.plot([nm["n_max"]], [95], ls="none", marker="D", ms=6.5, mfc=SURFACE, mec=SFC[sf], mew=1.6, zorder=5)
            ends.append((m[-1], f"SF{sf}", SFC[sf], MARK[sf]))
        ax.axhline(95, color=MUTED, lw=0.9, ls=":")
        ax.text(46000, 96.2, "95 %", color=INK2, fontsize=10, va="bottom", ha="right")
        set_n_axis(ax, (3, 10, 100, 1000, 5000, 50000))
        ax.set_xlim(3, 50000)
        ax.set_ylim(0, 102)
        ax.set_title(f"Один пакет раз в {t_label(t)} ({t} с)")
        end_labels(ax, ends, 9)
        clean_axes(ax)
    for ax in axs[1]:
        ax.set_xlabel("Число устройств N")
    for ax in axs[:, 0]:
        ax.set_ylabel("PDR, %")
    h = [Line2D([], [], color=SFC[s], marker=MARK[s], ms=6, lw=1.6, label=f"SF{s}") for s in sfs]
    h += [Line2D([], [], color=INK2, marker="D", ms=6, mfc=SURFACE, mew=1.5, ls="none", label="N_max (PDR = 95 %)"),
          Line2D([], [], color=INK2, ls=(0, (4, 2)), lw=1.4, label="режим превышает duty cycle 1 %")]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.085, right=0.925, top=0.95, bottom=0.12, wspace=0.2, hspace=0.24)
    return save(fig, out, "fig2_pdr_vs_n.png")


def fig_heatmap(R, out):
    cfg = R["meta"]["config"]
    ts, sfs, req = cfg["grid"]["interval_s"], cfg["grid"]["sf"], cfg["grid"]["n_devices"]
    cmap = LinearSegmentedColormap.from_list("seq", RAMP)
    fig, axs = plt.subplots(1, len(sfs), figsize=(9.9, 3.3), sharey=True)
    im = None
    for ax, sf in zip(np.atleast_1d(axs), sfs):
        M = np.array([[R["sweep"][f"{sf}_{t}"]["pdr_mean"][R["sweep"][f"{sf}_{t}"]["N"].index(n)] for t in ts] for n in req])
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(req)):
            for j, t in enumerate(ts):
                col = cmap(M[i, j])
                hexc = matplotlib.colors.to_hex(col)
                mark = "" if R["nmax"][f"{sf}_{t}"]["legal"] else "*"
                ax.text(j, i, f"{ru(M[i, j] * 100)}{mark}", ha="center", va="center", color=text_on(hexc),
                        fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(ts)))
        ax.set_xticklabels([t_label(t) for t in ts])
        ax.set_yticks(range(len(req)))
        ax.set_yticklabels([f"N = {n_fmt(n)}" for n in req])
        ax.set_xticks(np.arange(-0.5, len(ts), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(req), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=2.5)
        ax.grid(which="major", visible=False)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"SF{sf}")
        ax.set_xlabel("Интервал между пакетами")
    cax = fig.add_axes([0.945, 0.22, 0.011, 0.66])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("PDR, %")
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cb.set_ticklabels(["0", "25", "50", "75", "100"])
    cb.outline.set_visible(False)
    fig.text(0.01, 0.01, "* режим превышает ограничение duty cycle 1 % (юридически недопустим в EU868); число – PDR, %",
             color=INK2, fontsize=10)
    fig.subplots_adjust(left=0.1, right=0.93, top=0.88, bottom=0.22, wspace=0.08)
    return save(fig, out, "fig3_heatmap.png")


def fig_nmax(R, out):
    cfg = R["meta"]["config"]
    ts, sfs = cfg["grid"]["interval_s"], cfg["grid"]["sf"]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ends = []
    for sf in sfs:
        v = [R["nmax"][f"{sf}_{t}"] for t in ts]
        n = np.array([x["n_max"] for x in v])
        lo = n - np.array([x["seed_min"] for x in v])
        hi = np.array([x["seed_max"] for x in v]) - n
        ax.errorbar(ts, n, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], color=SFC[sf], lw=1.7, marker=MARK[sf], ms=6.5,
                    capsize=2.5, elinewidth=1.0, mfc=SFC[sf], mec=SURFACE)
        for x, t in zip(v, ts):
            if not x["legal"]:
                ax.plot([t], [x["n_max"]], marker=MARK[sf], ms=7, mfc=SURFACE, mec=SFC[sf], mew=1.6, ls="none", zorder=5)
        ax.plot(ts, [x["analytic"] for x in v], color=SFC[sf], lw=1.1, ls=(0, (4, 2)), alpha=0.9)
        ends.append((n[-1], f"SF{sf}", SFC[sf], MARK[sf]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ts)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: t_label(int(round(v)))))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.yaxis.set_major_formatter(FuncFormatter(n_fmt))
    ax.set_xlim(ts[0] * 0.9, ts[-1] * 1.1)
    ax.grid(axis="both", color=GRID, lw=0.7)
    ax.set_xlabel("Интервал между пакетами одного устройства")
    ax.set_ylabel("N_max: устройств на шлюз при PDR ≥ 95 %")
    ax.set_title("Ёмкость одного шлюза (EU868, 3 канала)")
    end_labels(ax, ends, 0.12, x=1.015, transform_y=np.log10)
    h = [Line2D([], [], color=INK2, lw=1.7, marker="o", ms=6, label="симуляция (с capture), разброс по сериям"),
         Line2D([], [], color=INK2, lw=1.1, ls=(0, (4, 2)), label="аналитика ALOHA без capture"),
         Line2D([], [], color=INK2, marker="o", mfc=SURFACE, mew=1.5, ls="none", label="режим превышает duty cycle 1 %")]
    ax.legend(handles=h, loc="upper left")
    clean_axes(ax, grid_x=True)
    fig.subplots_adjust(left=0.14, right=0.88, top=0.92, bottom=0.13)
    return save(fig, out, "fig4_nmax.png")


def fig_sensitivity(R, out):
    s = R["sensitivity"]
    cfg = R["meta"]["config"]
    sfs = cfg["grid"]["sf"]
    names = [("base", "Базовая модель"),
             ("no_capture", "Без capture"),
             ("capture_1db", "Порог capture 1 дБ"),
             ("channels_8", "8 каналов\nвместо 3"),
             ("imperfect_sf", "Неидеальная\nортогональность SF"),
             ("demod_16", "16 демодуляторов\nвместо 8")]
    fig, axs = plt.subplots(1, len(sfs), figsize=(9.8, 3.6), sharey=True)
    for ax, sf in zip(np.atleast_1d(axs), sfs):
        vals = [s["n_max"][k][str(sf)] for k, _ in names]
        flags = [s["flags"][k][str(sf)] for k, _ in names]
        y = np.arange(len(names))[::-1]
        ax.hlines(y, 1, vals, color=GRID, lw=1.2, zorder=1)
        ax.axvline(vals[0], color=SFC[sf], lw=0.9, ls=(0, (3, 2)), alpha=0.8)
        ax.scatter(vals, y, s=[70] + [46] * (len(names) - 1), color=SFC[sf], edgecolor=SURFACE, linewidth=1.2, zorder=3,
                   marker=MARK[sf])
        for v, yy, f in zip(vals, y, flags):
            ax.text(v * 1.18, yy, f"{'>' if f == '>' else ('<' if f == '<' else '')}{n_fmt(v)}", va="center", fontsize=10.5,
                    color=INK)
        ax.set_xscale("log")
        lo = max(min(vals) * 0.5, 1)
        hi = max(vals) * 3.5
        ax.set_xlim(lo, hi)
        log_ticks(ax, lo, hi)
        ax.set_yticks(y)
        ax.set_yticklabels([n for _, n in names])
        ax.set_title(f"SF{sf}")
        ax.set_xlabel("N_max при PDR ≥ 95 %")
        ax.grid(axis="x", color=GRID, lw=0.7)
        ax.grid(axis="y", visible=False)
        clean_axes(ax)
    fig.subplots_adjust(left=0.19, right=0.985, top=0.88, bottom=0.19, wspace=0.12)
    return save(fig, out, "fig5_sensitivity.png")


def fig_adr(R, out):
    cfg = R["meta"]["config"]
    mix, adr = R["mix"], R["adr"]
    fig, axs = plt.subplots(2, 2, figsize=(9.3, 7.0))

    ax = axs[0, 0]
    fr = np.array(mix["frac12"]) * 100
    for key, ls, lab in (("ideal", "-", "идеальная ортогональность SF"), ("imperfect", (0, (4, 2)), "неидеальная (Croce)")):
        net = arr([x[0] for x in mix["net"][key]]) * 100
        ax.plot(fr, net, color=INK, lw=1.8, ls=ls, marker="D" if key == "ideal" else None, ms=4.5, markevery=1)
        ax.plot(fr, arr(mix["sf7"][key]) * 100, color=SFC[7], lw=1.5, ls=ls)
        ax.plot(fr, arr(mix["sf12"][key]) * 100, color=SFC[12], lw=1.5, ls=ls)
    ax.set_xlabel("Доля устройств на SF12, % (остальные на SF7)")
    ax.set_ylabel("PDR, %")
    ax.set_ylim(0, 102)
    ax.set_xlim(0, 100)
    ax.set_title(f"A. Доля SF12-устройств (N = {mix['N']})")
    tag(ax, 62, 72, "вся сеть", INK, "D")
    tag(ax, 36, 93, "SF7-устройства", SFC[7], MARK[7])
    tag(ax, 50, 24, "SF12-устройства", SFC[12], MARK[12])
    h = [Line2D([], [], color=INK2, lw=1.7, label="идеальная ортогональность SF"),
         Line2D([], [], color=INK2, lw=1.7, ls=(0, (4, 2)), label="неидеальная (Croce et al.)")]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    clean_axes(ax)

    ax = axs[0, 1]
    radii = np.array(adr["radii_km"])
    shades = {500: "#86b6ef", 1000: "#2a78d6", 5000: "#0d366b"}
    ends = []
    for n in adr["N"]:
        m = arr([x[0] for x in adr["pdr"][str(n)]]) * 100
        c = arr([x[1] for x in adr["pdr"][str(n)]]) * 100
        col = shades[n]
        ax.fill_between(radii, m - c, m + c, color=col, alpha=0.18, lw=0)
        ax.plot(radii, m, color=col, marker="o", ms=5, lw=1.6, mec=SURFACE)
        ends.append((m[-1], f"N = {n_fmt(n)}", col, "o"))
    m_imp = arr([x[0] for x in adr["pdr_imperfect_1000"]]) * 100
    ax.plot(radii, m_imp, color=shades[1000], lw=1.2, ls=(0, (4, 2)))
    ax.set_ylim(0, 102)
    ax.set_xlim(0.5, 10.3)
    ax.set_xlabel("Радиус соты, км")
    ax.set_ylabel("PDR, %")
    ax.set_title("B. ADR: рост радиуса соты")
    end_labels(ax, ends, 7)
    ax.text(0.7, 5, "пунктир: N = 1000,\nнеидеальная ортогональность", color=INK2, fontsize=10)
    clean_axes(ax, grid_x=True)

    ax = axs[1, 0]
    sfs_all = [7, 8, 9, 10, 11, 12]
    cmap = LinearSegmentedColormap.from_list("seq", RAMP)
    M = np.array([[adr["sf_share"][str(r)][i] for r in adr["radii_km"]] for i in range(6)])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(6):
        for j in range(len(radii)):
            ax.text(j, i, f"{M[i, j] * 100:.0f}", ha="center", va="center", fontsize=10.5,
                    color=text_on(matplotlib.colors.to_hex(cmap(M[i, j]))), fontweight="bold")
    ax.set_xticks(range(len(radii)))
    ax.set_xticklabels([f"{r:g}" for r in radii])
    ax.set_yticks(range(6))
    ax.set_yticklabels([f"SF{s}" for s in sfs_all])
    ax.set_xticks(np.arange(-0.5, len(radii), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Радиус соты, км")
    ax.set_title("C. Доли SF после ADR, % (N = 1000)")

    ax = axs[1, 1]
    names = [("ниже чувствительности", 1, CAT[7], "v"), ("коллизия", 3, CAT[6], "o"), ("нет свободного демодулятора", 2, CAT[4], "s")]
    for nm, k, col, mk in names:
        y = np.array([adr["causes"][str(r)][k] for r in adr["radii_km"]]) * 100
        ax.plot(radii, y, color=col, marker=mk, ms=5, lw=1.6, mec=SURFACE)
    ax.set_xlim(0.5, 10.3)
    ax.set_xlabel("Радиус соты, км")
    ax.set_ylabel("Потери, % от переданных")
    ax.set_title("D. Причины потерь, ADR (N = 1000)")
    ax.legend(handles=[Line2D([], [], color=c, marker=m, ms=5, lw=1.6, label=n) for n, _, c, m in names], loc="upper left")
    clean_axes(ax, grid_x=True)
    fig.subplots_adjust(left=0.085, right=0.89, top=0.95, bottom=0.07, wspace=0.34, hspace=0.3)
    return save(fig, out, "fig6_adr_sf12.png")


def fig_battery(R, out):
    cfg = R["meta"]["config"]
    en = R["energy"]
    sfs, req = cfg["grid"]["sf"], cfg["grid"]["n_devices"]
    ts_all = en["intervals_s"]
    fig, axs = plt.subplots(1, 2, figsize=(9.3, 4.1), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    ends = []
    for sf in sfs:
        rows = [en["unconfirmed"][str(sf)][str(t)] for t in ts_all]
        y = [r["lifetime_years"] for r in rows]
        ax.plot(ts_all, y, color=SFC[sf], lw=1.7, marker=MARK[sf], ms=6, mec=SURFACE)
        for r, t in zip(rows, ts_all):
            if not r["legal"]:
                ax.plot([t], [r["lifetime_years"]], marker=MARK[sf], ms=7, mfc=SURFACE, mec=SFC[sf], mew=1.6, ls="none", zorder=5)
        ends.append((y[-1], f"SF{sf}", SFC[sf], MARK[sf]))
    ax.axhline(10, color=MUTED, lw=0.9, ls=":")
    ax.text(ts_all[0] * 0.95, 10.6, "10 лет", color=INK2, fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([60, 600, 3600, 21600, 86400])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: t_label(int(round(v)))))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlim(ts_all[0] * 0.85, ts_all[-1] * 1.15)
    ax.set_xlabel("Интервал между пакетами")
    ax.set_ylabel("Время работы, лет (лог. шкала)")
    ax.set_title("A. Unconfirmed")
    ax.grid(axis="both", color=GRID, lw=0.7)
    end_labels(ax, ends, 0.12, x=1.015, transform_y=np.log10)
    ax.legend(handles=[Line2D([], [], color=INK2, marker="o", mfc=SURFACE, mew=1.5, ls="none", label="превышает duty cycle 1 %")],
              loc="lower right")
    clean_axes(ax, grid_x=True)

    ax = axs[1]
    T = cfg["grid"]["base_interval_s"]
    x = np.arange(len(req))
    ends = []
    for sf in sfs:
        conf = [en["confirmed"][str(sf)][str(T)][str(n)]["lifetime_years"] for n in req]
        base = en["unconfirmed"][str(sf)][str(T)]["lifetime_years"]
        ax.plot(x, conf, color=SFC[sf], lw=1.7, marker=MARK[sf], ms=6, mec=SURFACE)
        ax.plot([x[0], x[-1]], [base, base], color=SFC[sf], lw=1.0, ls=(0, (4, 2)), alpha=0.9)
        ends.append((conf[-1], f"SF{sf}", SFC[sf], MARK[sf]))
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n_fmt(n) for n in req])
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlabel("Число устройств N в соте")
    ax.set_ylabel("Время работы, лет")
    ax.set_title(f"B. Confirmed, 1 пакет / {t_label(T)}")
    ax.set_xlim(-0.2, len(req) - 0.8)
    ax.grid(axis="both", color=GRID, lw=0.7)
    ax.legend(handles=[Line2D([], [], color=INK2, lw=1.7, marker="o", ms=5, label="confirmed (повторы при потере)"),
                       Line2D([], [], color=INK2, lw=1.0, ls=(0, (4, 2)), label="unconfirmed (для сравнения)")],
              loc="lower left")
    end_labels(ax, ends, 0.1, x=1.02, transform_y=np.log10)
    clean_axes(ax, grid_x=True)
    fig.subplots_adjust(left=0.09, right=0.915, top=0.9, bottom=0.15, wspace=0.34)
    return save(fig, out, "fig7_battery.png")


def fig_energy_share(R, out):
    from energy import COMPONENT_NAMES, COMPONENTS
    cfg = R["meta"]["config"]
    en = R["energy"]
    sfs = cfg["grid"]["sf"]
    T = cfg["grid"]["base_interval_s"]
    fig, axs = plt.subplots(1, 2, figsize=(9.3, 3.3), sharey=True)
    for ax, (key, ttl) in zip(axs, (("share", f"A. Весь цикл ({t_label(T)})"),
                                    ("share_active", "B. Активная часть (без сна)"))):
        y = np.arange(len(sfs))[::-1]
        left = np.zeros(len(sfs))
        for ci, comp in enumerate(COMPONENTS):
            vals = np.array([en["unconfirmed"][str(sf)][str(T)][key].get(comp, 0.0) for sf in sfs]) * 100
            ax.barh(y, vals, left=left, height=0.58, color=CAT[ci], edgecolor=SURFACE, linewidth=1.5)
            for yy, v, l in zip(y, vals, left):
                if v >= 5:
                    ax.text(l + v / 2, yy, f"{v:.0f}".replace(".", ","), ha="center", va="center", fontsize=10.5,
                            color=text_on(CAT[ci]), fontweight="bold")
            left += vals
        for yy, sf in zip(y, sfs):
            u = en["unconfirmed"][str(sf)][str(T)]
            rx = (u["share"]["rx1"] + u["share"]["rx2"]) if key == "share" else (u["share_active"]["rx1"] + u["share_active"]["rx2"])
            ax.text(101.5, yy, f"RX {rx * 100:.0f} %".replace(".", ","), va="center", ha="left", fontsize=11,
                    fontweight="bold", color=INK, clip_on=False)
        ax.set_yticks(y)
        ax.set_yticklabels([f"SF{s}" for s in sfs])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Доля, %")
        ax.set_title(ttl)
        ax.grid(axis="x", color=GRID, lw=0.7)
        ax.grid(axis="y", visible=False)
        clean_axes(ax)
    h = [matplotlib.patches.Patch(color=CAT[i], label=COMPONENT_NAMES[c]) for i, c in enumerate(COMPONENTS)]
    fig.legend(handles=h, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.06, right=0.9, top=0.88, bottom=0.27, wspace=0.3)
    return save(fig, out, "fig8_energy_share.png")


FIGS = [fig_aloha, fig_pdr_vs_n, fig_heatmap, fig_nmax, fig_sensitivity, fig_adr, fig_battery, fig_energy_share]


def make_all(R, out):
    for f in FIGS:
        p = f(R, out)
        print(f"    {os.path.relpath(p, out)}")


if __name__ == "__main__":
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    R = json.load(open(os.path.join(here, "results.json"), encoding="utf-8"))
    make_all(R, here)
