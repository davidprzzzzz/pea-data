#!/usr/bin/env python3
"""Compact digest of data/latest.json for the weekly Claude macro review.
usage: python tools/brief.py <repo_dir>"""
import json
import os
import sys

d = json.load(open(os.path.join(sys.argv[1], "data", "latest.json")))
m = d["meta"]
print(f"AS OF {m['as_of']} ({m['week']}) | priced {m['priced_n']}/{m['universe_n']} | pairs tested {m['pairs_tested']}, "
      f"cointegrated {m['pairs_cointegrated']} | signals {m['n_signals']} + watch {m['n_watch']}")
if m.get("warnings"):
    print("WARNINGS:", " | ".join(m["warnings"]))
print("\nMACRO REGIME:", d["macro"]["regime_text"])
for f in d["macro"]["factors"]:
    print(f"  {f['label']:<20} last {f['last']}  1w {f['chg_1w']}{f['chg_unit']}  4w {f['chg_4w']}{f['chg_unit']}  z4 {f['z4']}  1y pct {f['pctile_1y']}")
print("\nSECTORS (macro tailwind, 1m ret, #signals):")
for s in sorted(d["sectors"], key=lambda s: -(s["tailwind"] or 0)):
    print(f"  {s['sector']:<32} tail {s['tailwind']:+.2f}  1m {(s['ret_1m'] or 0) * 100:+.1f}%  sig {s['n_signals']}")
print("\nIDEAS:")
for i in d["ideas"]:
    L, lv, st, v, mc = i["long"], i["levels"], i["stats"], i.get("value") or {}, i.get("macro") or {}
    ref = i["ref"]["ticker"] or "sector basket"
    drv = ", ".join(f"{x['f']} {x['contrib']:+.2f}" for x in mc.get("drivers", []))
    print(f"[{i['id']}] #{i['rank']} {i['tier']} {i['type']} LONG {L['ticker']} {L['name']} ({L['country']}) vs {ref} | "
          f"score {i['score']} | z {st['z']} hl {st.get('half_life')}d | exp {lv['exp_pct'] * 100:+.1f}% risk "
          f"{lv['risk_pct'] * 100:.1f}% hz {lv['horizon_days']}d | fPE {v.get('fpe')} (sect {v.get('sec_fpe')}) "
          f"EV/EBITDA {v.get('ev_ebitda')} P/B {v.get('pb')} value_z {v.get('value_z')} | macro pct {mc.get('pctile')} "
          f"[{drv}] | bt {i['backtest']} | earnings {L.get('earnings_date')} | flags {i['flags']}")
t = d["track"]["summary"]
print(f"\nTRACK RECORD: {t}")
