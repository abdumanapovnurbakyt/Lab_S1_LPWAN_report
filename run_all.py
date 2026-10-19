import argparse
import copy
import csv
import datetime as dt
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import energy
from lora import SFS
from network import aloha_point, build_params, simulate_once

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
TARGET = 0.95

_CFG = None


def _init(cfg):
    global _CFG
    _CFG = cfg


def _work(kw):
    return simulate_once(_CFG, **kw)


def run_many(cfg, kws, jobs):
    if jobs <= 1 or len(kws) < 8:
        return [simulate_once(cfg, **kw) for kw in kws]
    with ProcessPoolExecutor(jobs, initializer=_init, initargs=(cfg,)) as ex:
        return list(ex.map(_work, kws, chunksize=max(1, len(kws) // (jobs * 8))))


def clean(o):
   
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def mean_ci(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    sd = float(np.std(xs, ddof=1))
    return m, T975.get(n - 1, 1.96) * sd / math.sqrt(n)


def nmax_from_curve(ns, pdr, target=TARGET):
 
    ns = np.asarray(ns, dtype=float)
    pdr = np.asarray(pdr, dtype=float)
    ok = np.flatnonzero(pdr >= target)
    if ok.size == 0:
        return float(ns[0]), "<"
    i = int(ok[-1])
    if i == len(ns) - 1:
        return float(ns[-1]), ">"
    x0, x1 = math.log10(ns[i]), math.log10(ns[i + 1])
    y0, y1 = pdr[i], pdr[i + 1]
    return 10 ** (x0 + (y0 - target) / (y0 - y1) * (x1 - x0)), ""


def analytic_nmax(cfg, sf, period_s, channels=None):
   
    p = build_params(cfg, channels)
    return -math.log(TARGET) / 2 * p.channels * period_s / p.toa[sf - 7]


def sweep_ns(cfg, required, points=None, lo=None, hi=None):
    g = cfg["grid"]
    pts = np.geomspace(lo or g["n_sweep_min"], hi or g["n_sweep_max"], points or g["n_sweep_points"])
    return sorted(set(int(round(x)) for x in pts) | set(required))


def main(quick=False, jobs=None):
    t_start = time.time()
    cfg = json.load(open(os.path.join(HERE, "config.json"), encoding="utf-8"))
    if quick:
        cfg["sim"]["seeds"] = 2
        cfg["sim"]["target_packets"] = 15000
        cfg["grid"]["n_sweep_points"] = 10
    seeds = cfg["sim"]["seeds"]
    jobs = jobs or os.cpu_count() or 1
    os.makedirs(OUT, exist_ok=True)
    grid = cfg["grid"]
    req_n, sfs, ts = grid["n_devices"], grid["sf"], grid["interval_s"]
    base_t = grid["base_interval_s"]
    ns_all = sweep_ns(cfg, req_n)
    p0 = build_params(cfg)
    R = {"meta": {"created": dt.datetime.now().isoformat(timespec="seconds"), "quick": quick, "seeds": seeds,
                  "jobs": jobs, "config": cfg}}
    print(f"Серий: {seeds}, процессов: {jobs}, точек N: {len(ns_all)}")

    R["airtime_ms"] = {str(s): float(p0.toa[s - 7] * 1000) for s in SFS}
    R["duty"] = {str(s): {"toa_s": float(p0.toa[s - 7]), "min_interval_s": energy.min_interval_s(cfg, s)} for s in SFS}

    print("1/7 проверка на пределе ALOHA ...")
    g_grid = [round(x, 2) for x in np.arange(0.05, 2.51, 0.05)]
    m_pk, n_al = (40_000, 2) if quick else (250_000, 3)
    al = {"G": g_grid, "theory_pdr": [math.exp(-2 * g) for g in g_grid], "theory_s": [g * math.exp(-2 * g) for g in g_grid]}
    for name, cap in (("nocap", False), ("cap", True)):
        pdr_l, s_l = [], []
        for g in g_grid:
            vals = [aloha_point(cfg, g, m_pk, i, capture=cap) for i in range(n_al)]
            pdr_l.append(float(np.mean([v[0] for v in vals])))
            s_l.append(float(np.mean([v[1] for v in vals])))
        al[name] = {"pdr": pdr_l, "s": s_l}
    i_pk = int(np.argmax(al["nocap"]["s"]))
    al["peak_nocap"] = {"G": g_grid[i_pk], "S": al["nocap"]["s"][i_pk]}
    i_pc = int(np.argmax(al["cap"]["s"]))
    al["peak_cap"] = {"G": g_grid[i_pc], "S": al["cap"]["s"][i_pc]}
    al["peak_theory"] = {"G": 0.5, "S": 1 / (2 * math.e)}
    al["s_at_half"] = al["nocap"]["s"][g_grid.index(0.5)]
    al["max_abs_dev_pdr"] = float(max(abs(a - b) for a, b in zip(al["nocap"]["pdr"], al["theory_pdr"])))
    R["aloha"] = al
    print(f"    S_max (без capture) = {al['peak_nocap']['S'] * 100:.2f} % при G = {al['peak_nocap']['G']}; "
          f"с capture = {al['peak_cap']['S'] * 100:.1f} % при G = {al['peak_cap']['G']}")

    print("2/7 сетка PDR(N, SF, интервал) ...")
    tags, kws = [], []
    for sf in sfs:
        for t in ts:
            for n in ns_all:
                for sd in range(seeds):
                    tags.append(("sweep", sf, t, n))
                    kws.append(dict(seed_idx=sd, n_dev=n, period_s=t, mode="single", sf=sf))
    for sf in sfs:
        for n in req_n:
            for sd in range(seeds):
                tags.append(("pure", sf, base_t, n))
                kws.append(dict(seed_idx=sd, n_dev=n, period_s=base_t, mode="single", sf=sf, capture="off", demod="off"))
    res = run_many(cfg, kws, jobs)
    agg = defaultdict(list)
    for tag, r in zip(tags, res):
        agg[tag].append(r)

    sweep = {}
    for sf in sfs:
        for t in ts:
            ps = np.array([[r["pdr"] for r in agg[("sweep", sf, t, n)]] for n in ns_all])        # [N, seeds]
            m = ps.mean(axis=1)
            ci = [mean_ci(list(row))[1] for row in ps]
            causes = [np.sum([r["causes"] for r in agg[("sweep", sf, t, n)]], axis=0).tolist() for n in ns_all]
            sweep[f"{sf}_{t}"] = {"sf": sf, "interval_s": t, "N": ns_all, "pdr_mean": m.tolist(), "pdr_ci": ci,
                                  "per_seed": ps.T.tolist(), "causes": causes}
    R["sweep"] = sweep

    nmax = {}
    for sf in sfs:
        for t in ts:
            sw = sweep[f"{sf}_{t}"]
            est, flag = nmax_from_curve(ns_all, sw["pdr_mean"])
            per_seed = [nmax_from_curve(ns_all, ps)[0] for ps in sw["per_seed"]]
            nmax[f"{sf}_{t}"] = {"sf": sf, "interval_s": t, "n_max": est, "flag": flag, "seed_min": min(per_seed),
                                 "seed_max": max(per_seed), "analytic": analytic_nmax(cfg, sf, t),
                                 "legal": energy.summary_unconfirmed(cfg, sf, t)["legal"]}
    R["nmax"] = nmax

    pure = {}
    for sf in sfs:
        rows = []
        for n in req_n:
            g = n * p0.toa[sf - 7] / (base_t * p0.channels)
            m, ci = mean_ci([r["pdr"] for r in agg[("pure", sf, base_t, n)]])
            rows.append({"N": n, "G_per_channel": g, "sim": m, "ci": ci, "theory": math.exp(-2 * g)})
        pure[str(sf)] = rows
    R["pure_check"] = pure

    print("3/7 чувствительность модели (capture, каналы) ...")
    variants = {"no_capture": {"capture": "off"}, "capture_1db": {"capture": 1.0},
                "channels_8": {"channels": 8}, "imperfect_sf": {"imperfect": True}, "demod_16": {"demod": 16}}
    ns_s = [n for n in ns_all if 10 <= n <= 20000]
    tags, kws = [], []
    for vn, vkw in variants.items():
        for sf in sfs:
            for n in ns_s:
                for sd in range(seeds):
                    tags.append((vn, sf, n))
                    kws.append(dict(seed_idx=sd, n_dev=n, period_s=base_t, mode="single", sf=sf, **vkw))
    res = run_many(cfg, kws, jobs)
    agg2 = defaultdict(list)
    for tag, r in zip(tags, res):
        agg2[tag].append(r["pdr"])
    sens, sens_flags = {"base": {}}, {"base": {}}
    for sf in sfs:
        sw = sweep[f"{sf}_{base_t}"]
        curve = [sw["pdr_mean"][sw["N"].index(n)] for n in ns_s]
        sens["base"][str(sf)], sens_flags["base"][str(sf)] = nmax_from_curve(ns_s, curve)
    for vn in variants:
        sens[vn], sens_flags[vn] = {}, {}
        for sf in sfs:
            curve = [float(np.mean(agg2[(vn, sf, n)])) for n in ns_s]
            sens[vn][str(sf)], sens_flags[vn][str(sf)] = nmax_from_curve(ns_s, curve)
    R["sensitivity"] = {"interval_s": base_t, "n_max": sens, "flags": sens_flags}

    print("4/7 сценарии SF12 / ADR ...")
    fr_grid = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]
    n_mix = 1000
    tags, kws = [], []
    for fr in fr_grid:
        for imp in (False, True):
            for sd in range(seeds):
                tags.append(("mix", fr, imp))
                kws.append(dict(seed_idx=sd, n_dev=n_mix, period_s=base_t, mode="mix", frac12=fr, imperfect=imp))
    radii = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    n_adr = [500, 1000, 5000]
    for rk in radii:
        for n in n_adr:
            for sd in range(seeds):
                tags.append(("adr", rk, n, False))
                kws.append(dict(seed_idx=sd, n_dev=n, period_s=base_t, mode="adr", radius_km=rk))
        for sd in range(seeds):
            tags.append(("adr", rk, 1000, True))
            kws.append(dict(seed_idx=sd, n_dev=1000, period_s=base_t, mode="adr", radius_km=rk, imperfect=True))
    ns_adr = sweep_ns(cfg, [], points=12 if not quick else 8, lo=20, hi=5000)
    radii_n = [1.0, 2.0, 3.0, 5.0]
    for rk in radii_n:
        for n in ns_adr:
            for sd in range(min(3, seeds)):
                tags.append(("adrn", rk, n))
                kws.append(dict(seed_idx=sd, n_dev=n, period_s=base_t, mode="adr", radius_km=rk))
    res = run_many(cfg, kws, jobs)
    agg3 = defaultdict(list)
    for tag, r in zip(tags, res):
        agg3[tag].append(r)

    def per_sf_pdr(rs, sf):
        i = sf - 7
        sent = sum(r["per_sf_sent"][i] for r in rs)
        recv = sum(r["per_sf_recv"][i] for r in rs)
        return recv / sent if sent else float("nan")

    mix = {"N": n_mix, "interval_s": base_t, "frac12": fr_grid, "net": {}, "sf12": {}, "sf7": {}}
    for imp, key in ((False, "ideal"), (True, "imperfect")):
        mix["net"][key] = [mean_ci([r["pdr"] for r in agg3[("mix", fr, imp)]]) for fr in fr_grid]
        mix["sf12"][key] = [per_sf_pdr(agg3[("mix", fr, imp)], 12) for fr in fr_grid]
        mix["sf7"][key] = [per_sf_pdr(agg3[("mix", fr, imp)], 7) for fr in fr_grid]
    R["mix"] = mix

    adr = {"radii_km": radii, "interval_s": base_t, "N": n_adr, "pdr": {}, "sf_share": {}, "pdr_by_sf": {},
           "causes": {}, "pdr_imperfect_1000": []}
    for n in n_adr:
        adr["pdr"][str(n)] = [mean_ci([r["pdr"] for r in agg3[("adr", rk, n, False)]]) for rk in radii]
    for rk in radii:
        rs = agg3[("adr", rk, 1000, False)]
        dev = np.mean([r["devices_per_sf"] for r in rs], axis=0)
        adr["sf_share"][str(rk)] = (dev / dev.sum()).tolist()
        adr["pdr_by_sf"][str(rk)] = [per_sf_pdr(rs, s) for s in SFS]
        adr["causes"][str(rk)] = (np.sum([r["causes"] for r in rs], axis=0) / np.sum([r["sent"] for r in rs])).tolist()
        adr["pdr_imperfect_1000"].append(mean_ci([r["pdr"] for r in agg3[("adr", rk, 1000, True)]]))
    R["adr"] = adr
    adr_nmax = {}
    for rk in radii_n:
        curve = [float(np.mean([r["pdr"] for r in agg3[("adrn", rk, n)]])) for n in ns_adr]
        adr_nmax[str(rk)] = list(nmax_from_curve(ns_adr, curve))
    R["adr_nmax"] = {"interval_s": base_t, "N": ns_adr, "n_max": adr_nmax}

    print("5/7 энергомодель ...")
    en = {"intervals_s": cfg["energy"]["intervals_s"], "unconfirmed": {}, "confirmed": {}, "shares_t600": {},
          "rx2_sf9_t600": {}, "ack_capacity_per_s": {}}
    for sf in SFS:
        en["unconfirmed"][str(sf)] = {str(t): energy.summary_unconfirmed(cfg, sf, t) for t in cfg["energy"]["intervals_s"]}
    cfg9 = copy.deepcopy(cfg)
    cfg9["energy"]["rx2_sf"] = 9
    for sf in SFS:
        en["rx2_sf9_t600"][str(sf)] = energy.summary_unconfirmed(cfg9, sf, base_t)
    for sf in sfs:
        en["confirmed"][str(sf)] = {}
        for t in ts:
            sw = sweep[f"{sf}_{t}"]
            en["confirmed"][str(sf)][str(t)] = {
                str(n): energy.summary_confirmed(cfg, sf, t, sw["pdr_mean"][sw["N"].index(n)]) for n in req_n}
    for sf in SFS:
        en["ack_capacity_per_s"][str(sf)] = energy.ack_capacity_per_s(cfg, sf)
    R["energy"] = en

    print("6/7 CSV ...")
    with open(os.path.join(OUT, "pdr_grid.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["sf", "interval_s", "n_devices", "G_per_channel", "pdr_mean_pct", "ci95_pct", "legal_duty_cycle"])
        for sf in sfs:
            for t in ts:
                sw = sweep[f"{sf}_{t}"]
                for n in req_n:
                    i = sw["N"].index(n)
                    g = n * p0.toa[sf - 7] / (t * p0.channels)
                    w.writerow([sf, t, n, f"{g:.4f}".replace(".", ","), f"{sw['pdr_mean'][i] * 100:.2f}".replace(".", ","),
                                f"{sw['pdr_ci'][i] * 100:.2f}".replace(".", ","), "да" if nmax[f"{sf}_{t}"]["legal"] else "нет"])
    with open(os.path.join(OUT, "n_max_95.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["sf", "interval_s", "n_max_sim", "flag", "seed_min", "seed_max", "n_max_analytic_noCapture"])
        for k, v in nmax.items():
            w.writerow([v["sf"], v["interval_s"], f"{v['n_max']:.0f}", v["flag"], f"{v['seed_min']:.0f}",
                        f"{v['seed_max']:.0f}", f"{v['analytic']:.0f}"])
    with open(os.path.join(OUT, "battery_lifetime.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["sf", "interval_s", "toa_ms", "duty_cycle_pct", "legal", "cycle_mJ", "avg_uA", "lifetime_years",
                    "tx_share_pct", "rx_share_pct"])
        for sf in SFS:
            for t in cfg["energy"]["intervals_s"]:
                u = en["unconfirmed"][str(sf)][str(t)]
                w.writerow([sf, t, f"{u['toa_s'] * 1000:.1f}".replace(".", ","), f"{u['duty_cycle'] * 100:.3f}".replace(".", ","),
                            "да" if u["legal"] else "нет", f"{u['cycle_mj']:.2f}".replace(".", ","),
                            f"{u['avg_ua']:.2f}".replace(".", ","), f"{u['lifetime_years']:.2f}".replace(".", ","),
                            f"{u['tx_share'] * 100:.1f}".replace(".", ","), f"{u['rx_total_share'] * 100:.1f}".replace(".", ",")])

    with open(os.path.join(OUT, "adr_n_max_95.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["cell_radius_km", "interval_s", "n_max_adr", "flag"])
        for rk, (v, fl) in adr_nmax.items():
            w.writerow([rk.replace(".", ","), base_t, f"{v:.0f}", fl])

    R["meta"]["elapsed_s"] = time.time() - t_start
    R = clean(R)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1, allow_nan=False)

    print("7/7 графики ...")
    import plots
    plots.make_all(R, OUT)
    summary(R, cfg)
    print(f"\nГотово за {time.time() - t_start:.0f} с. Результаты: {OUT}")


def summary(R, cfg):
    sfs, req_n = cfg["grid"]["sf"], cfg["grid"]["n_devices"]
    base_t = cfg["grid"]["base_interval_s"]
    print(f"\nPDR, % (интервал {base_t} с, {R['meta']['seeds']} серий):")
    print("  SF   " + "".join(f"N={n:<8}" for n in req_n))
    for sf in sfs:
        sw = R["sweep"][f"{sf}_{base_t}"]
        print(f"  SF{sf:<3} " + "".join(f"{sw['pdr_mean'][sw['N'].index(n)] * 100:<10.1f}" for n in req_n))
    print(f"\nМаксимум устройств на шлюз при PDR >= 95 % (интервал {base_t} с):")
    for sf in sfs:
        v = R["nmax"][f"{sf}_{base_t}"]
        print(f"  SF{sf}: {v['n_max']:.0f}{v['flag']}  (по сериям {v['seed_min']:.0f}–{v['seed_max']:.0f}; "
              f"аналитика без capture {v['analytic']:.0f})")
    print("\nВремя работы от батареи, лет (без подтверждений):")
    for sf in sfs:
        print(f"  SF{sf}: " + ", ".join(f"{t} с – {R['energy']['unconfirmed'][str(sf)][str(t)]['lifetime_years']:.1f}"
                                         for t in (60, 600, 3600)))
    print(f"\nN_max при ADR (интервал {base_t} с): " + ", ".join(
        f"R = {rk} км – {v:.0f}{fl}" for rk, (v, fl) in R["adr_nmax"]["n_max"].items()))
    a = R["aloha"]
    print(f"\nALOHA: S_max = {a['peak_nocap']['S'] * 100:.2f} % при G = {a['peak_nocap']['G']} (теория 18,39 %); "
          f"с capture {a['peak_cap']['S'] * 100:.1f} %")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="быстрая проверка (2 серии, меньше пакетов)")
    ap.add_argument("--jobs", type=int, help="число процессов (по умолчанию – число ядер)")
    a = ap.parse_args()
    main(a.quick, a.jobs)
