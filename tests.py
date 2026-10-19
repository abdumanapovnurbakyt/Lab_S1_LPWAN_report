import json
import math
import os

import numpy as np

import energy
from lora import airtime_s
from network import (CROCE, aloha_point, build_params, periodic_packets, resolve, rng_for, simulate_once)

CFG = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"), encoding="utf-8"))
OK = []


def check(name, cond, info=""):
    assert cond, f"НЕ ПРОЙДЕНО: {name} {info}"
    OK.append(name)
    print(f"  ок  {name} {info}")


def pk(p, rows):
    """rows: (start, channel, sf, rssi) -> массивы для resolve"""
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1].astype(int), a[:, 2].astype(int), a[:, 3]


def test_airtime():
    check("ToA SF7, 20 байт = 56,6 мс", abs(airtime_s(7, 20) - 0.056576) < 1e-6, f"({airtime_s(7, 20) * 1000:.1f} мс)")
    check("ToA SF12, 10 байт = 991,2 мс", abs(airtime_s(12, 10) - 0.991232) < 1e-6, f"({airtime_s(12, 10) * 1000:.1f} мс)")
    check("ToA SF12, 20 байт = 1318,9 мс", abs(airtime_s(12, 20) - 1.318912) < 1e-6)
    check("ToA SF7, 23 байта = 61,7 мс", abs(airtime_s(7, 23) - 0.061696) < 1e-6)
    toas = [airtime_s(s, 33) for s in range(7, 13)]
    check("ToA растёт с SF", all(a < b for a, b in zip(toas, toas[1:])))


def test_collisions():
    p = build_params(CFG, channels=2, demod="off")
    t = p.toa[0]                                   # SF7
    r, _ = resolve(*pk(p, [(0, 0, 7, -80), (t / 2, 0, 7, -95)]), p)
    check("capture: сильный пакет (+15 дБ) выживает, слабый теряется", list(r) == [True, False])
    r, _ = resolve(*pk(p, [(0, 0, 7, -80), (t / 2, 0, 7, -83)]), p)
    check("capture: разница 3 дБ -> оба потеряны", list(r) == [False, False])
    r, _ = resolve(*pk(p, [(0, 0, 7, -80), (t / 2, 0, 7, -86.01)]), p)
    check("capture: разница 6 дБ -> сильный выживает", list(r) == [True, False])
    r, _ = resolve(*pk(p, [(0, 0, 7, -80), (t * 0.3, 0, 7, -87), (t * 0.6, 0, 7, -87)]), p)
    check("capture: помехи суммируются по мощности", not r[0])
    r, _ = resolve(*pk(p, [(0, 0, 7, -80), (t / 2, 0, 8, -80)]), p)
    check("разные SF не мешают друг другу", list(r) == [True, True])
    r, _ = resolve(*pk(p, [(0, 0, 7, -80), (t / 2, 1, 7, -80)]), p)
    check("разные каналы не мешают друг другу", list(r) == [True, True])
    r, _ = resolve(*pk(p, [(0, 0, 7, -80), (t + 0.001, 0, 7, -80)]), p)
    check("без перекрытия коллизии нет", list(r) == [True, True])
    pn = build_params(CFG, channels=2, capture="off", demod="off")
    r, _ = resolve(*pk(pn, [(0, 0, 7, -80), (t / 2, 0, 7, -100)]), pn)
    check("без capture любое перекрытие = потеря обоих", list(r) == [False, False])
    r, c = resolve(*pk(p, [(0, 0, 7, -124), (5, 0, 12, -124)]), p)
    check("ниже чувствительности SF7 теряется, SF12 принимается", list(r) == [False, True])
    pi = build_params(CFG, channels=2, imperfect=True, demod="off")
    r, _ = resolve(*pk(pi, [(0, 0, 7, -100), (0.01, 0, 12, -95)]), pi)
    check("неидеальная ортогональность: помеха SF12 на +5 дБ терпима для SF7 (порог −9 дБ)", r[0])
    r, _ = resolve(*pk(pi, [(0, 0, 7, -100), (0.01, 0, 12, -85)]), pi)
    check("неидеальная ортогональность: помеха SF12 на +15 дБ убивает SF7", not r[0])
    check("матрица Croce: диагональ положительная, остальное отрицательное",
          all(CROCE[i, i] > 0 for i in range(6)) and (CROCE[~np.eye(6, dtype=bool)] < 0).all())


def test_demodulators():
    p = build_params(CFG, channels=16, demod=8)
    rows = [(0.001 * i, i, 7, -80) for i in range(10)]  
    r, c = resolve(*pk(p, rows), p)
    check("шлюз: из 10 одновременных принято 8", int(r.sum()) == 8, f"(принято {int(r.sum())})")
    p0 = build_params(CFG, channels=16, demod="off")
    r, _ = resolve(*pk(p0, rows), p0)
    check("без ограничения демодуляторов приняты все 10", int(r.sum()) == 10)


def test_generator():
    rng = rng_for(1, 2, 3)
    start, dev, ch = periodic_packets(rng, 50, 600.0, 10, 3)
    cnt = np.bincount(dev, minlength=50)
    check("каждое устройство передаёт по 1 разу за период", (cnt == 10).all())
    per = (start // 600).astype(int)
    ok = all(len(set(per[dev == d])) == 10 for d in range(50))
    check("пакеты равномерно по периодам, момент случайный", ok and start.min() >= 0 and start.max() < 6000)
    check("каналы случайны и в допустимом диапазоне", set(np.unique(ch)) == {0, 1, 2})


def test_aloha_limit():
    """Главная проверка: предел чистой ALOHA 1/(2e) ≈ 18,4 % при G = 0,5."""
    peak = 1 / (2 * math.e)
    best_s, best_g = 0.0, None
    print("  G      PDR(симул.)  PDR(теория e^-2G)  S(симул.)  S(теория)")
    worst = 0.0
    for g in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]:
        pdr, s = aloha_point(CFG, g, 300_000, 1, capture=False)
        th = math.exp(-2 * g)
        worst = max(worst, abs(pdr - th))
        print(f"  {g:<5} {pdr:.4f}       {th:.4f}             {s:.4f}     {g * th:.4f}")
        if s > best_s:
            best_s, best_g = s, g
    check("ALOHA: PDR(G) совпадает с e^(-2G)", worst < 0.005, f"(макс. отклонение {worst:.4f})")
    check("ALOHA: максимум пропускной способности при G = 0,5", best_g == 0.5, f"(G* = {best_g})")
    check("ALOHA: S_max ≈ 18,4 %", abs(best_s - peak) < 0.004, f"(S_max = {best_s * 100:.2f} %, теория {peak * 100:.2f} %)")
    _, s_cap = aloha_point(CFG, 0.5, 300_000, 1, capture=True)
    check("capture повышает пропускную способность выше предела 18,4 %", s_cap > best_s + 0.02,
          f"(S = {s_cap * 100:.1f} % при G = 0,5)")


def test_energy():
    e = CFG["energy"]
    s7 = energy.summary_unconfirmed(CFG, 7, 600)
    toa = energy.toa_uplink(CFG, 7)
    check("энергия TX = U·I·ToA", abs(s7["energy_mj"]["tx"] - e["supply_v"] * e["tx_ma"] * toa) < 1e-9)
    check("доли энергии в сумме = 100 %", abs(sum(s7["share"].values()) - 1) < 1e-12)
    lt = [energy.summary_unconfirmed(CFG, sf, 600)["lifetime_years"] for sf in (7, 9, 12)]
    check("время работы падает с ростом SF", lt[0] > lt[1] > lt[2], f"({lt[0]:.1f} > {lt[1]:.1f} > {lt[2]:.1f} лет)")
    lt = [energy.summary_unconfirmed(CFG, 9, T)["lifetime_years"] for T in (60, 600, 3600)]
    check("время работы растёт с интервалом", lt[0] < lt[1] < lt[2])
    c1 = energy.summary_confirmed(CFG, 7, 600, 1.0)
    u = energy.summary_unconfirmed(CFG, 7, 600)
    check("confirmed при PDR=100 %: ровно одна попытка", abs(c1["attempts"] - 1) < 1e-12)
    c2 = energy.summary_confirmed(CFG, 7, 600, 0.3)
    check("confirmed: чем ниже PDR, тем больше попыток и меньше срок", c2["attempts"] > 2 and c2["lifetime_years"] < c1["lifetime_years"])
    check("duty cycle: SF12 при 60 с нарушает 1 %, SF7 при 60 с нет",
          not energy.summary_unconfirmed(CFG, 12, 60)["legal"] and energy.summary_unconfirmed(CFG, 7, 60)["legal"])


def test_simulation():
    pdr = [simulate_once(CFG, 0, n, 600, "single", sf=9)["pdr"] for n in (100, 1000, 5000)]
    check("PDR убывает с ростом N", pdr[0] > pdr[1] > pdr[2], f"({pdr[0]:.3f} > {pdr[1]:.3f} > {pdr[2]:.3f})")
    pdr = [simulate_once(CFG, 0, 1000, 600, "single", sf=s)["pdr"] for s in (7, 9, 12)]
    check("PDR убывает с ростом SF (длиннее пакет)", pdr[0] > pdr[1] > pdr[2], f"({pdr[0]:.3f} > {pdr[1]:.3f} > {pdr[2]:.3f})")
    r = simulate_once(CFG, 0, 1000, 600, "single", sf=7)
    check("учёт пакетов: sent = сумма по SF = сумма причин", r["sent"] == sum(r["per_sf_sent"]) == sum(r["causes"]))
    a = simulate_once(CFG, 3, 500, 600, "adr")
    b = simulate_once(CFG, 3, 500, 600, "adr")
    check("воспроизводимость: тот же seed -> тот же результат", a == b)
    check("ADR: устройства распределены по нескольким SF", sum(1 for x in a["devices_per_sf"] if x > 0) >= 4)
    f0 = simulate_once(CFG, 0, 1000, 600, "mix", frac12=0.0)["pdr"]
    f5 = simulate_once(CFG, 0, 1000, 600, "mix", frac12=0.5)["pdr"]
    check("доля SF12 снижает PDR сети", f5 < f0, f"({f0:.3f} -> {f5:.3f})")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"[{name}]")
            fn()
    print(f"\nПроверок: {len(OK)}")
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ")
