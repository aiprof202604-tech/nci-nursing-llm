#!/usr/bin/env python3
"""Consistently wrong items (30/30 identical incorrect answers in a cell) by model: at any temperature and excluding
T = 0, on all 90 items and on the 68 information-complete items, with the temperatures at which each item was
unanimously wrong. Requires outputs/cells.pkl from revision_analyses.py. No API calls."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import revision_analyses as ra
import sensitivity_subsets as ss

def main():
    cells = pd.read_pickle(ra.OUT / "cells.pkl")
    complete = set(cells["item"].unique()) - set(ss.CONTENT) - set(ss.CONTEXT)
    assert len(complete) == 68, len(complete)
    out = {}
    uw_all = cells[(cells.phat >= 0.999) & (cells.modal_correct == 0)]
    for m in ra.MODELS:
        uw = uw_all[uw_all.model == m]
        all90 = sorted(uw["item"].unique()); noT0 = sorted(uw[uw.temp > 0]["item"].unique())
        out[m] = dict(any_temperature=len(all90), excluding_T0=len(noT0),
                      any_temperature_complete68=sum(i in complete for i in all90),
                      excluding_T0_complete68=sum(i in complete for i in noT0),
                      temperatures_by_item={i: sorted(float(t) for t in uw[uw["item"] == i].temp) for i in all90})
        print(ra.LAB[m], {k: v for k, v in out[m].items() if k != "temperatures_by_item"})
        for i, ts in out[m]["temperatures_by_item"].items(): print(f"    {i}: unanimous wrong at T = {ts}")
    out["distinct_items"] = dict(any_temperature=int(uw_all["item"].nunique()),
                                 excluding_T0=int(uw_all[uw_all.temp > 0]["item"].nunique()))
    print("distinct items:", out["distinct_items"])
    json.dump(out, open(ra.OUT / "consistently_wrong_by_temperature.json", "w"), indent=1)
    print("written", ra.OUT / "consistently_wrong_by_temperature.json")

if __name__ == "__main__":
    main()
