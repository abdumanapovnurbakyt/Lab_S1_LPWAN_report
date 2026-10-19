from lora import airtime_s, symbol_time_s, uplink_phy_bytes

COMPONENTS = ("tx", "rx1", "rx2", "mcu", "sleep")
COMPONENT_NAMES = {"tx": "Передача (TX)", "rx1": "Окно RX1", "rx2": "Окно RX2",
                   "mcu": "МК и датчик", "sleep": "Сон"}


def rx_window_s(cfg, sf):
    e, r = cfg["energy"], cfg["radio"]
    return e["rx_symbols"] * symbol_time_s(sf, r["bw_hz"]) + e["rx_margin_s"]


def toa_uplink(cfg, sf):
    r = cfg["radio"]
    return airtime_s(sf, uplink_phy_bytes(r["payload_app_bytes"], r["mac_overhead_bytes"]), r["bw_hz"],
                     r["cr"], r["preamble_syms"], r["explicit_header"], r["crc"])


def duty_cycle(cfg, sf, interval_s):
    return toa_uplink(cfg, sf) / interval_s


def min_interval_s(cfg, sf):
    return toa_uplink(cfg, sf) / cfg["network"]["duty_cycle"]


def cycle_unconfirmed(cfg, sf, interval_s):
    e = cfg["energy"]
    toa = toa_uplink(cfg, sf)
    t_rx1, t_rx2 = rx_window_s(cfg, sf), rx_window_s(cfg, e["rx2_sf"])
    t_sleep = interval_s - e["mcu_active_s"] - toa - t_rx1 - t_rx2
    q = {"tx": e["tx_ma"] * toa, "rx1": e["rx_ma"] * t_rx1, "rx2": e["rx_ma"] * t_rx2,
         "mcu": e["mcu_active_ma"] * e["mcu_active_s"], "sleep": e["sleep_ua"] / 1000.0 * max(t_sleep, 0.0)}
    return q, {"toa": toa, "rx1": t_rx1, "rx2": t_rx2}


def self_discharge_ma(cfg):
    e = cfg["energy"]
    return e["battery_mah"] * e["self_discharge_pct_per_year"] / 100.0 / 8760.0


def lifetime_years(cfg, avg_ma):
    e = cfg["energy"]
    total = avg_ma + self_discharge_ma(cfg)
    return e["battery_mah"] * e["usable_fraction"] / total / 8760.0


def summary_unconfirmed(cfg, sf, interval_s):
    q, t = cycle_unconfirmed(cfg, sf, interval_s)
    total = sum(q.values())
    avg_ma = total / interval_s
    e = cfg["energy"]
    return {
        "sf": sf, "interval_s": interval_s, "toa_s": t["toa"], "duty_cycle": t["toa"] / interval_s,
        "legal": t["toa"] / interval_s <= cfg["network"]["duty_cycle"],
        "charge_mc": q, "energy_mj": {k: v * e["supply_v"] for k, v in q.items()},
        "cycle_mc": total, "cycle_mj": total * e["supply_v"],
        "share": {k: v / total for k, v in q.items()},
        "share_active": {k: v / (total - q["sleep"]) for k, v in q.items() if k != "sleep"},
        "avg_ua": avg_ma * 1000.0,
        "lifetime_years": lifetime_years(cfg, avg_ma),
        "rx_total_share": (q["rx1"] + q["rx2"]) / total,
        "tx_share": q["tx"] / total,
    }


def summary_confirmed(cfg, sf, interval_s, pdr):
 
    e = cfg["energy"]
    q = 1.0 - pdr
    retries = e["confirmed_max_retries"]
    toa = toa_uplink(cfg, sf)
    t_rx1, t_rx2 = rx_window_s(cfg, sf), rx_window_s(cfg, e["rx2_sf"])
    t_ack = airtime_s(sf, e["ack_phy_bytes"], cfg["radio"]["bw_hz"], cfg["radio"]["cr"],
                      cfg["radio"]["preamble_syms"], cfg["radio"]["explicit_header"], cfg["radio"]["crc"])
    q_fail = e["tx_ma"] * toa + e["rx_ma"] * (t_rx1 + t_rx2)
    q_succ = e["tx_ma"] * toa + e["rx_ma"] * (t_rx1 + t_ack)
    p_deliver = 1.0 - q ** (retries + 1)
    exp_fail = sum(q ** n for n in range(1, retries + 2))
    q_msg = q_fail * exp_fail + q_succ * p_deliver
    attempts = exp_fail + p_deliver
    t_active = e["mcu_active_s"] * attempts
    q_mcu = e["mcu_active_ma"] * t_active
    t_sleep = interval_s - t_active - (toa + t_rx1 + t_rx2) * attempts
    q_sleep = e["sleep_ua"] / 1000.0 * max(t_sleep, 0.0)
    total = q_msg + q_mcu + q_sleep
    avg_ma = total / interval_s
    return {"sf": sf, "interval_s": interval_s, "pdr_attempt": pdr, "p_deliver": p_deliver, "attempts": attempts,
            "cycle_mc": total, "avg_ua": avg_ma * 1000.0, "lifetime_years": lifetime_years(cfg, avg_ma),
            "t_ack_s": t_ack}


def ack_capacity_per_s(cfg, sf):
    r, e = cfg["radio"], cfg["energy"]
    t_ack = airtime_s(sf, e["ack_phy_bytes"], r["bw_hz"], r["cr"], r["preamble_syms"], r["explicit_header"], r["crc"])
    return cfg["network"]["duty_cycle"] / t_ack
