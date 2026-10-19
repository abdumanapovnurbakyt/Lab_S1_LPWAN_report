import heapq
import math
from dataclasses import dataclass

import numpy as np

from lora import SFS, airtime_s, uplink_phy_bytes


CROCE = np.array([
    [1, -8, -9, -9, -9, -9],
    [-11, 1, -11, -12, -13, -13],
    [-15, -13, 1, -13, -14, -15],
    [-19, -18, -17, 1, -17, -18],
    [-22, -22, -21, -20, 1, -20],
    [-25, -25, -25, -24, -23, 1],
], dtype=float)

CAUSE_OK, CAUSE_LINK, CAUSE_PATH, CAUSE_COLLISION = 0, 1, 2, 3
CAUSE_NAMES = ("доставлено", "ниже чувствительности", "нет свободного демодулятора", "коллизия")


@dataclass
class Params:
    channels: int
    toa: np.ndarray            # время в эфире по SF7..SF12, с
    sens: np.ndarray           # чувствительность по SF7..SF12, дБм
    capture_db: object         # float или None (None: любое перекрытие = потеря)
    imperfect: bool            # неидеальная ортогональность SF
    demod: object              # число демодуляторов или None
    tx_dbm: float
    pl_d0_db: float
    d0_km: float
    exponent: float
    sigma_db: float
    adr_margin_db: float


def build_params(cfg, channels=None, capture="cfg", imperfect=None, demod="cfg") -> Params:
    r, n = cfg["radio"], cfg["network"]
    toa = np.array([airtime_s(s, uplink_phy_bytes(r["payload_app_bytes"], r["mac_overhead_bytes"]),
                              r["bw_hz"], r["cr"], r["preamble_syms"], r["explicit_header"], r["crc"])
                    for s in SFS])
    sens = np.array([r["sensitivity_dbm"][str(s)] for s in SFS], dtype=float)
    if capture == "cfg":
        capture = n["capture_threshold_db"]
    elif capture == "off":
        capture = None
    if imperfect is None:
        imperfect = not n["orthogonal_sf"]
    if demod == "cfg":
        demod = n["gateway_demodulators"]
    elif demod == "off":
        demod = None
    pl = n["pathloss"]
    return Params(channels or n["channels"], toa, sens, capture, imperfect, demod, r["tx_power_dbm"],
                  pl["pl_d0_db"], pl["d0_km"], pl["exponent"], pl["shadowing_sigma_db"], n["adr_margin_db"])


def make_devices(rng, n, radius_km, p: Params):
    """Расстояния (км) и RSSI на шлюзе (дБм) для n устройств, равномерно в круге."""
    d = np.maximum(radius_km * np.sqrt(rng.random(n)), 0.01)
    pl = p.pl_d0_db + 10 * p.exponent * np.log10(d / p.d0_km)
    return d, p.tx_dbm - pl + rng.normal(0.0, p.sigma_db, n)


def adr_assign(rssi, p: Params):
    """ADR (упрощённо): наименьший SF, для которого RSSI >= чувствительность + запас; иначе SF12."""
    sf = np.full(len(rssi), 12)
    for i in range(len(SFS) - 1, -1, -1):           
        sf[rssi >= p.sens[i] + p.adr_margin_db] = SFS[i]
    return sf


def periodic_packets(rng, n_dev, period_s, k_periods, channels):
    dev = np.tile(np.arange(n_dev), k_periods)
    k = np.repeat(np.arange(k_periods), n_dev)
    start = k * period_s + rng.random(k_periods * n_dev) * period_s
    ch = rng.integers(0, channels, k_periods * n_dev)
    return start, dev, ch


def _collisions(start, ch, sf, pw, p: Params):
    dead = np.zeros(len(start), dtype=bool)
    toa = p.toa
    sfs_present = np.unique(sf)
    for c in range(p.channels):
        idx_c = np.flatnonzero(ch == c)
        if idx_c.size < 2:
            continue
        groups = {}
        for s in sfs_present:
            g = idx_c[sf[idx_c] == s]
            if g.size == 0:
                continue
            g = g[np.argsort(start[g], kind="stable")]
            groups[int(s)] = (g, start[g], np.concatenate(([0.0], np.cumsum(pw[g]))))
        for si, (gi, sti, _) in groups.items():
            end_i = sti + toa[si - 7]
            pi = pw[gi]
            for sj, (gj, stj, cumj) in groups.items():
                if si != sj and not p.imperfect:
                    continue                              
                lo = np.searchsorted(stj, sti - toa[sj - 7], side="right")
                hi = np.searchsorted(stj, end_i, side="left")
                cnt = hi - lo
                interf = cumj[hi] - cumj[lo]
                if si == sj:
                    cnt = cnt - 1                         
                    interf = interf - pi
                has = cnt > 0
                if not has.any():
                    continue
                if si == sj:
                    if p.capture_db is None:
                        bad = has
                    else:
                        bad = has & (pi < interf * 10 ** (p.capture_db / 10))
                else:
                    bad = has & (pi < interf * 10 ** (CROCE[si - 7, sj - 7] / 10))
                dead[gi[bad]] = True
    return dead


def _no_path(start, end, cand, demod):
    out = np.zeros(len(start), dtype=bool)
    idx = np.flatnonzero(cand)
    if demod is None or idx.size <= demod:
        return out
    order = np.argsort(start[idx], kind="stable")
    st = start[idx][order]
    en = end[idx][order]
    conc = np.arange(1, len(st) + 1) - np.searchsorted(np.sort(en), st, side="right")
    if conc.max() <= demod:
        return out
    busy = []
    drop = []
    st_l, en_l = st.tolist(), en.tolist()
    for pos, (s, e) in enumerate(zip(st_l, en_l)):
        while busy and busy[0] <= s:
            heapq.heappop(busy)
        if len(busy) >= demod:
            drop.append(pos)
        else:
            heapq.heappush(busy, e)
    out[idx[order[drop]]] = True
    return out


def resolve(start, ch, sf, rssi, p: Params):
    pw = 10 ** (rssi / 10.0)                           
    end = start + p.toa[sf - 7]
    link_fail = rssi < p.sens[sf - 7]
    collided = _collisions(start, ch, sf, pw, p)
    no_path = _no_path(start, end, ~link_fail, p.demod)
    cause = np.zeros(len(start), dtype=np.int8)
    cause[collided] = CAUSE_COLLISION
    cause[no_path] = CAUSE_PATH
    cause[link_fail] = CAUSE_LINK
    return cause == CAUSE_OK, cause


def rng_for(*key):
    return np.random.default_rng(np.random.SeedSequence([int(abs(k)) for k in key]))


def simulate_once(cfg, seed_idx, n_dev, period_s, mode="single", sf=7, frac12=0.0, radius_km=None,
                  channels=None, capture="cfg", imperfect=None, demod="cfg", target_packets=None):
  
    p = build_params(cfg, channels, capture, imperfect, demod)
    s = cfg["sim"]
    mode_id = {"single": 0, "mix": 1, "adr": 2}[mode]
   
    rng = rng_for(s["base_seed"], seed_idx, n_dev, int(period_s), mode_id, sf, int(round(frac12 * 1000)),
                  int(round((radius_km or 0) * 100)), p.channels)
    if radius_km is None:
        radius_km = cfg["network"]["adr_radius_km"] if mode == "adr" else cfg["network"]["disk_radius_km"]
    _, rssi_dev = make_devices(rng, n_dev, radius_km, p)
    if mode == "single":
        sf_dev = np.full(n_dev, sf)
    elif mode == "mix":
        sf_dev = np.full(n_dev, 7)
        sf_dev[rng.permutation(n_dev)[:int(round(frac12 * n_dev))]] = 12
    else:
        sf_dev = adr_assign(rssi_dev, p)
    k_periods = max(s["min_periods"], math.ceil((target_packets or s["target_packets"]) / n_dev))
    start, dev, ch = periodic_packets(rng, n_dev, period_s, k_periods, p.channels)
    sf_pk, rssi_pk = sf_dev[dev], rssi_dev[dev]
    received, cause = resolve(start, ch, sf_pk, rssi_pk, p)
    guard = 2 * p.toa.max()
    valid = (start >= guard) & (start <= k_periods * period_s - guard)
    per_sf_sent = np.array([np.sum(valid & (sf_pk == x)) for x in SFS])
    per_sf_recv = np.array([np.sum(valid & (sf_pk == x) & received) for x in SFS])
    causes = np.array([np.sum(valid & (cause == c)) for c in range(4)])
    sent = int(valid.sum())
    return {
        "pdr": float(received[valid].mean()) if sent else float("nan"),
        "sent": sent,
        "per_sf_sent": per_sf_sent.tolist(),
        "per_sf_recv": per_sf_recv.tolist(),
        "causes": causes.tolist(),
        "devices_per_sf": [int(np.sum(sf_dev == x)) for x in SFS],
        "periods": k_periods,
    }


def aloha_point(cfg, g_load, m_packets, seed_idx, capture=False, sf=7):
   
    p = build_params(cfg, channels=1, capture="cfg" if capture else "off", imperfect=False, demod="off")
    rng = rng_for(cfg["sim"]["base_seed"], 777, seed_idx, int(round(g_load * 1000)), int(capture))
    toa = p.toa[sf - 7]
    horizon = m_packets * toa / g_load
    start = rng.random(m_packets) * horizon
    ch = np.zeros(m_packets, dtype=int)
    sfv = np.full(m_packets, sf)
    if capture:
        _, rssi = make_devices(rng, m_packets, cfg["network"]["disk_radius_km"], p)
        rssi = np.maximum(rssi, p.sens[sf - 7] + 1.0)  
    else:
        rssi = np.full(m_packets, -100.0)
    received, _ = resolve(start, ch, sfv, rssi, p)
    guard = 2 * toa
    valid = (start >= guard) & (start <= horizon - guard)
    pdr = float(received[valid].mean())
    return pdr, g_load * pdr
