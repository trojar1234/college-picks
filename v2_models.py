"""Model 2.0 component models and baselines.

Every model projects each team's points from information available before kickoff, and margin and
total follow from those. Models that learn coefficients (2-5) are refit each season on earlier
seasons only, so every backtest prediction is out of sample.

  1 Bayesian ratings      points-scored filter (v2_ratings)                 no fitting beyond the filter
  2 Possession model      expected drives x points per drive, calibrated   2 coefficients
  3 Matchup model         unit-vs-unit efficiency ratings and interactions  ridge regression
  4 Personnel model       model 1 + roster, recruiting, transfers, QB       ridge on model 1's misses
  5 Machine learning      gradient-boosted trees on every rating            sklearn HistGradientBoosting
  Baselines: Massey (ridge on points, refit weekly) and margin Elo.
"""
import numpy as np
import pandas as pd

import v2_ratings as VR

EFF = [m for m in VR.METRICS if m not in ("pts",)]
INTER = ["rppa", "pppa", "rsr", "psr", "xpl", "stuff", "line", "fin", "opp_rate", "fpos"]
PERS = ["talent_z", "rec_z", "ret_all", "ret_pass", "ret_rush", "oret", "dret", "qbl_z", "qbd", "exp", "tin_z", "tout_z"]
MODELS = ["bayes", "poss", "matchup", "personnel", "ml"]
NAMES = {"bayes": "Bayesian ratings", "poss": "Possession model", "matchup": "Matchup model",
         "personnel": "Personnel model", "ml": "Machine learning", "massey": "Massey (baseline)",
         "elo": "Elo (baseline)", "v1": "Current model (v1)"}


def _z(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def side_rows(G, pre, pers, qb):
    """Two rows per game: the team's offense against the opponent's defense, every pregame number."""
    g = G[["game_id", "season", "wk", "start", "home", "away", "neutral", "home_cls", "away_cls",
           "home_pts", "away_pts", "completed"]].merge(pre, on="game_id", how="inner")
    if qb is not None:
        g = g.merge(qb, on="game_id", how="left")
    P = []
    for s, df in pers.items():
        d = df.copy()
        for c, z in (("talent", "talent_z"), ("rec", "rec_z"), ("qbl", "qbl_z"), ("tin", "tin_z"), ("tout", "tout_z")):
            d[z] = _z(d[c])
        d["season"] = s
        P.append(d)
    P = pd.concat(P, ignore_index=True) if P else pd.DataFrame(columns=["team", "season"] + PERS)
    for c in PERS:
        if c not in P:
            P[c] = np.nan
    P = P[["team", "season"] + PERS].drop_duplicates(["team", "season"])
    out = []
    for me, op, sfx, osfx, sgn in (("home", "away", "h", "a", 1), ("away", "home", "a", "h", -1)):
        r = pd.DataFrame(dict(game_id=g.game_id, season=g.season, wk=g.wk, side=sfx, team=g[me], opp=g[op],
                              h=np.where(g.neutral, 0, sgn), cls=g[f"{me}_cls"], opp_cls=g[f"{op}_cls"],
                              pts=g[f"{me}_pts"], completed=g.completed))
        r["bayes"] = g[f"pts_{sfx}"].to_numpy()
        r["pts_o"], r["pts_d"] = g[f"pts_o{sfx}"].to_numpy(), g[f"pts_d{osfx}"].to_numpy()
        r["pts_mu"], r["pts_hfa"] = g["pts_mu"].to_numpy(), g["pts_hfa"].to_numpy()
        r["pts_sd_o"] = g[f"pts_sdo_{sfx}"].to_numpy()
        for m in EFF:
            if f"{m}_{sfx}" in g:
                r[f"{m}_pred"] = g[f"{m}_{sfx}"].to_numpy()
                r[f"{m}_o"] = g[f"{m}_o{sfx}"].to_numpy()
                r[f"{m}_d"] = g[f"{m}_d{osfx}"].to_numpy()
        r["qbchg"] = g.get(f"qbchg_{sfx}", pd.Series(0.0, index=g.index)).fillna(0.0).to_numpy()
        r["qbchg_opp"] = g.get(f"qbchg_{osfx}", pd.Series(0.0, index=g.index)).fillna(0.0).to_numpy()
        out.append(r)
    S = pd.concat(out, ignore_index=True)
    S = S.merge(P.add_suffix("_t").rename(columns={"team_t": "team", "season_t": "season"}), on=["team", "season"], how="left")
    S = S.merge(P.add_suffix("_x").rename(columns={"team_x": "opp", "season_x": "season"}), on=["opp", "season"], how="left")
    S["fbs"] = (S.cls == "fbs").astype(float)
    S["opp_fbs"] = (S.opp_cls == "fbs").astype(float)
    S["early"] = 1.0 / (S.wk.clip(upper=20) + 1.0)
    return S


# ---------------- Model 2: possession ----------------

def poss_X(S):
    """Points = drives x what each drive is worth: points per drive, how often a drive reaches the
    opponent's 40 times points per trip there, field position, and passing/rushing efficiency per drive."""
    D = S["drives_pred"]
    cols = [np.ones(len(S)), D * S["ppd_pred"], D * S["opp_rate_pred"] * S["fin_pred"], D * (80 - S["fpos_pred"]) / 80,
            D * S["pppa_pred"], D * S["rppa_pred"], S.h.astype(float)]
    X = np.column_stack([np.asarray(c, float) for c in cols])
    return np.where(np.isfinite(X), X, 0.0)


def fit_poss(train):
    t = train[train.pts.notna()]
    return np.linalg.lstsq(poss_X(t), t.pts.to_numpy(float), rcond=None)[0]


def pred_poss(S, b):
    return np.clip(poss_X(S) @ b, 0.5, None)


# ---------------- Model 3: matchup (ridge) ----------------

def matchup_X(S):
    cols = {}
    for m in EFF:
        if f"{m}_o" in S:
            cols[f"{m}_o"] = S[f"{m}_o"]
            cols[f"{m}_d"] = S[f"{m}_d"]
    for m in INTER:
        if f"{m}_o" in S:
            cols[f"{m}_x"] = S[f"{m}_o"] * S[f"{m}_d"]
    if "rshare_o" in S and "rppa_d" in S:
        cols["run_vs_weak_run_d"] = S["rshare_o"] * (S["rppa_d"] - S["pppa_d"])
        cols["pass_vs_weak_pass_d"] = (1 - S["rshare_o"]) * (S["pppa_d"] - S["rppa_d"])
    cols["h"] = S.h.astype(float)
    cols["fbs"], cols["opp_fbs"] = S.fbs, S.opp_fbs
    return pd.DataFrame(cols).fillna(0.0)


class Ridge:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.m, self.s = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.m) / self.s
        self.b0 = float(np.mean(y))
        self.b = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(Z.shape[1]), Z.T @ (np.asarray(y, float) - self.b0))
        return self

    def predict(self, X):
        return self.b0 + ((np.asarray(X, float) - self.m) / self.s) @ self.b


# ---------------- Model 4: personnel ----------------

def personnel_X(S):
    cols = {}
    for c in PERS:
        cols[f"{c}_t"] = S.get(f"{c}_t")
        cols[f"{c}_x"] = S.get(f"{c}_x")
        # preseason information matters most early; let each feature fade as games accumulate
        cols[f"{c}_t_early"] = S.get(f"{c}_t") * S.early
        cols[f"{c}_x_early"] = S.get(f"{c}_x") * S.early
    cols["qbchg"], cols["qbchg_opp"] = S.qbchg, S.qbchg_opp
    cols["sd_o"] = S.pts_sd_o
    cols["early"] = S.early
    return pd.DataFrame(cols).astype(float).fillna(0.0)


# ---------------- Model 5: gradient boosting ----------------

def ml_X(S):
    keep = ["pts_o", "pts_d", "pts_mu", "h", "fbs", "opp_fbs", "early", "qbchg", "qbchg_opp", "pts_sd_o"]
    keep += [c for c in S.columns if c.endswith("_o") or c.endswith("_d")]
    keep += [f"{c}_t" for c in PERS] + [f"{c}_x" for c in PERS]
    keep = list(dict.fromkeys(k for k in keep if k in S))
    return S[keep].astype(float)


def fit_ml(train, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    t = train[train.pts.notna()]
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=60,
                                      l2_regularization=1.0, random_state=seed)
    X = ml_X(t)
    cols = [c for c in X.columns if X[c].notna().mean() > 0.2 and X[c].nunique() > 1]
    m.fit(X[cols], t.pts.to_numpy(float))
    m.cols_ = cols
    return m


# ---------------- walk-forward over seasons ----------------

def walk(S, first=2017, alphas=(300.0, 300.0), seasons=None):
    """Out-of-sample team-point predictions for every model, each season fitted on earlier seasons."""
    S = S.copy()
    for m in MODELS[1:]:
        S[m] = np.nan
    for s in (seasons or sorted(S.season.unique())):
        if s < first:
            continue
        tr = S[(S.season < s) & S.pts.notna() & (S.season >= 2016)]
        cur = S.season == s
        if len(tr) < 1000:
            continue
        b = fit_poss(tr)
        S.loc[cur, "poss"] = pred_poss(S[cur], b)
        r = Ridge(alphas[0]).fit(matchup_X(tr), tr.pts)
        S.loc[cur, "matchup"] = np.clip(r.predict(matchup_X(S[cur])), 0.5, None)
        r4 = Ridge(alphas[1]).fit(personnel_X(tr), tr.pts - tr.bayes)
        S.loc[cur, "personnel"] = np.clip(S.loc[cur, "bayes"] + r4.predict(personnel_X(S[cur])), 0.5, None)
        ml = fit_ml(tr)
        S.loc[cur, "ml"] = np.clip(ml.predict(ml_X(S[cur])[ml.cols_]), 0.5, None)
    S["bayes"] = S["bayes"].clip(lower=0.5)
    return S


def to_games(S, cols):
    """Team rows -> one row per game: margin (home - away) and total for each model."""
    h = S[S.side == "h"].set_index("game_id")
    a = S[S.side == "a"].set_index("game_id")
    out = pd.DataFrame(index=h.index)
    for c in cols:
        out[f"m_{c}"] = h[c] - a[c]
        out[f"t_{c}"] = h[c] + a[c]
    return out.reset_index()


# ---------------- Baselines ----------------

def massey(G, lam=4.0, carry=0.5, hfa_games=60.0):
    """Points Massey: each team's offense and defense from a ridge fit on this season's earlier games,
    shrunk toward last season's ratings. Refit every week."""
    rows = []
    prev = {}
    for s in sorted(G.season.unique()):
        Gs = G[G.season == s]
        teams = sorted(set(Gs.home) | set(Gs.away))
        ix = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        cls = dict(zip(Gs.home, Gs.home_cls))
        cls.update(dict(zip(Gs.away, Gs.away_cls)))
        new = {"fbs": (0.0, 0.0), "fcs": (-9.0, 9.0), "lower": (-16.0, 16.0)}  # newcomers start at their level
        po = np.array([carry * prev[t][0] if t in prev else new.get(cls.get(t), (0.0, 0.0))[0] for t in teams])
        pdf = np.array([carry * prev[t][1] if t in prev else new.get(cls.get(t), (0.0, 0.0))[1] for t in teams])
        mu_prev = prev.get("_mu", 27.5)
        done = Gs.iloc[0:0]
        for wk, grp in Gs.groupby("wk", sort=True):
            # fit on games before this week
            d = done
            k = len(d)
            if k:
                A = np.zeros((2 * k, 2 * n + 2))
                y = np.zeros(2 * k)
                for j, r in enumerate(d.itertuples()):
                    hh = 0 if r.neutral else 1
                    for jj, (t, o, pts, hv) in enumerate(((r.home, r.away, r.home_pts, hh), (r.away, r.home, r.away_pts, -hh))):
                        A[2 * j + jj, ix[t]] = 1
                        A[2 * j + jj, n + ix[o]] = 1
                        A[2 * j + jj, 2 * n] = 1
                        A[2 * j + jj, 2 * n + 1] = hv
                        y[2 * j + jj] = pts
                pen = np.r_[np.full(2 * n, lam), 0.0, 0.0]
                prior = np.r_[po, pdf, 0.0, 0.0]
                M = A.T @ A + np.diag(pen)
                M[2 * n, 2 * n] += 1e-3
                M[2 * n + 1, 2 * n + 1] += hfa_games
                rhs = A.T @ y + pen * prior + np.r_[np.zeros(2 * n), 1e-3 * mu_prev, hfa_games * 1.3]
                x = np.linalg.solve(M, rhs)
            else:
                x = np.r_[po, pdf, mu_prev, 1.3]
            for r in grp.itertuples():
                hh = 0 if r.neutral else 1
                ph = x[2 * n] + hh * x[2 * n + 1] + x[ix[r.home]] + x[n + ix[r.away]]
                pa = x[2 * n] - hh * x[2 * n + 1] + x[ix[r.away]] + x[n + ix[r.home]]
                rows.append(dict(game_id=r.game_id, m_massey=ph - pa, t_massey=ph + pa))
            done = pd.concat([done, grp[grp.completed & grp.home_pts.notna()]])
        prev = {t: (x[ix[t]], x[n + ix[t]]) for t in teams}
        prev["_mu"] = x[2 * n]
    return pd.DataFrame(rows)


def elo(G, k=20.0, hfa=55.0, regress=0.33, scale=25.0):
    """Margin-of-victory Elo (FiveThirtyEight-style). Margin only; totals use the league's average so far."""
    R = {}
    rows = []
    for s in sorted(G.season.unique()):
        Gs = G[G.season == s]
        for t in set(Gs.home) | set(Gs.away):
            base = 1500.0 if t != "_lower division" else 1150.0
            if t in R:
                R[t] = base + (1 - regress) * (R[t] - base)
        tot_sum, tot_n = 55.0 * 20, 20
        for wk, grp in Gs.groupby("wk", sort=True):
            for r in grp.itertuples():
                for t in (r.home, r.away):
                    if t not in R:
                        R[t] = 1500.0 if r.home_cls == "fbs" or r.away_cls == "fbs" else 1350.0
                hh = 0 if r.neutral else 1
                dr = R[r.home] - R[r.away] + hh * hfa
                rows.append(dict(game_id=r.game_id, m_elo=dr / scale, t_elo=tot_sum / tot_n))
            for r in grp[grp.completed & grp.home_pts.notna()].itertuples():
                hh = 0 if r.neutral else 1
                dr = R[r.home] - R[r.away] + hh * hfa
                exp = 1 / (1 + 10 ** (-dr / 400))
                mov = r.home_pts - r.away_pts
                res = 1.0 if mov > 0 else 0.0 if mov < 0 else 0.5
                mult = np.log(abs(mov) + 1) * 2.2 / ((dr if mov > 0 else -dr) * 0.001 + 2.2)
                delta = k * mult * (res - exp)
                R[r.home] += delta
                R[r.away] -= delta
                tot_sum += r.home_pts + r.away_pts
                tot_n += 1
    return pd.DataFrame(rows)
