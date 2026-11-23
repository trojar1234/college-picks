"""Model 2.0 walk-forward backtest: every model's prediction for every game since 2017, each made only
from information available before that game, then the ensemble and confidence model on top."""
import numpy as np
import pandas as pd

import v2_data as VD
import v2_ensemble as VE
import v2_eval as EV
import v2_models as VM
import v2_ratings as VR


def build(SEASONS, pfeat=None, passing=None, v1_wf=None, params=None):
    G = VD.games_table(SEASONS)
    TG = VD.team_games(G)
    P = VD.personnel(sorted(G.season.unique()), pfeat)
    QB = VD.qb_context(G, passing)
    pre, fins, filt = VR.run_all(TG, G, P, params)
    S = VM.side_rows(G, pre, P, QB)
    S = VM.walk(S)
    Gm = VM.to_games(S, VM.MODELS)
    base = G[["game_id", "season", "wk", "start", "home", "away", "neutral", "home_cls", "away_cls",
              "home_pts", "away_pts", "completed"]].copy()
    base["margin"] = base.home_pts - base.away_pts
    base["total"] = base.home_pts + base.away_pts
    base["mismatch"] = ((base.home_cls != "fbs") | (base.away_cls != "fbs")).astype(int)
    add_context(base, G)
    Gm = base.merge(Gm, on="game_id", how="inner")
    Gm = Gm.merge(pre[["game_id", "pts_vm", "pts_vt", "drives_h", "drives_a"]], on="game_id", how="left")
    Gm = Gm.merge(VM.massey(G), on="game_id", how="left").merge(VM.elo(G), on="game_id", how="left")
    if v1_wf is not None and len(v1_wf):
        v = v1_wf[["game_id", "pred_h", "pred_a"]].drop_duplicates("game_id")
        Gm = Gm.merge(pd.DataFrame(dict(game_id=v.game_id, m_v1=v.pred_h - v.pred_a, t_v1=v.pred_h + v.pred_a)),
                      on="game_id", how="left")
    L = VD.lines_table(SEASONS)
    L = pd.DataFrame(dict(game_id=L.game_id, line_m_open=-L.cfbd_spread_open, line_m_close=-L.cfbd_spread,
                          line_t_open=L.cfbd_total_open, line_t_close=L.cfbd_total))
    Gm = Gm.merge(L, on="game_id", how="left")
    # the closing and opening lines as forecasts, for comparison only
    Gm["m_close"], Gm["t_close"] = Gm.line_m_close, Gm.line_t_close
    Gm["m_open"], Gm["t_open"] = Gm.line_m_open, Gm.line_t_open
    Gm, weights = VE.walk_ensemble(Gm)
    Gm = VE.walk_conf(Gm, with_disagree=True)
    Gm = Gm.rename(columns={"sd_m": "sd_m_dis", "sd_t": "sd_t_dis"})
    Gm = VE.walk_conf(Gm, with_disagree=False)
    Gm["sd_m"], Gm["sd_t"] = Gm.sd_m_nodis, Gm.sd_t_nodis  # the confidence model in use (no disagreement term)
    Gm["agree"] = VE.agreement(Gm)
    Gm["sd_m_ens"], Gm["sd_t_ens"] = Gm.sd_m, Gm.sd_t
    for c in VM.MODELS + ["massey", "elo", "v1", "close", "open"]:
        if f"m_{c}" in Gm:
            Gm[f"sd_m_{c}"] = EV.hist_sd(Gm, f"m_{c}", "margin")
            Gm[f"sd_t_{c}"] = EV.hist_sd(Gm, f"t_{c}", "total")
    return dict(G=G, TG=TG, P=P, QB=QB, pre=pre, fins=fins, filt=filt, S=S, Gm=Gm, weights=weights)


def add_context(base, G):
    """Game context the ensemble may use: both teams Power 4, late season (Nov+), neutral site."""
    import cfb_data as Dt
    g = G.set_index("game_id")
    th = [Dt.tier(c, k, s) for c, k, s in zip(g.home_conf.reindex(base.game_id), g.home_cls.reindex(base.game_id),
                                               base.season)]
    ta = [Dt.tier(c, k, s) for c, k, s in zip(g.away_conf.reindex(base.game_id), g.away_cls.reindex(base.game_id),
                                               base.season)]
    base["x_p4p4"] = [float(a == "P4" and b == "P4") for a, b in zip(th, ta)]
    base["x_late"] = (base.wk >= 10).astype(float)
    base["x_neutral"] = base.neutral.astype(float)
    return base


ALL = VM.MODELS + ["ens", "massey", "elo", "v1", "open", "close"]


def report(Gm):
    rep = {}
    for split in EV.SPLITS:
        r = dict(accuracy=EV.accuracy(Gm, ALL, split))
        r["vs_v1"] = [EV.paired(Gm, "ens", "v1", split, k) for k in ("margin", "total")]
        r["vs_best_component"] = [EV.paired(Gm, "ens", m, split, "margin") for m in VM.MODELS]
        r["market_info"] = EV.market_info(Gm, "ens", split)
        r["market_info_v1"] = EV.market_info(Gm, "v1", split)
        r["picks"] = EV.picks_record(Gm, "ens", split)
        r["confidence"] = EV.confidence_test(Gm, split)
        r["calibration"] = EV.calibration(Gm, split)
        rep[split] = r
    return rep
