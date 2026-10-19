import math

SFS = (7, 8, 9, 10, 11, 12)


def symbol_time_s(sf: int, bw_hz: int = 125000) -> float:
    return (2 ** sf) / bw_hz


def airtime_s(sf: int, phy_payload_bytes: int, bw_hz: int = 125000, cr: int = 1,
              preamble_syms: int = 8, explicit_header: bool = True, crc: bool = True) -> float:
  
    tsym = symbol_time_s(sf, bw_hz)
    de = 1 if tsym > 0.016 else 0
    ih = 0 if explicit_header else 1
    num = 8 * phy_payload_bytes - 4 * sf + 28 + 16 * (1 if crc else 0) - 20 * ih
    den = 4 * (sf - 2 * de)
    n_payload = 8 + max(math.ceil(num / den) * (cr + 4), 0)
    t_preamble = (preamble_syms + 4.25) * tsym
    return t_preamble + n_payload * tsym


def uplink_phy_bytes(app_bytes: int, mac_overhead_bytes: int = 13) -> int:
    return app_bytes + mac_overhead_bytes
