"""Model 2.0 rating engine: Bayesian (Kalman-filter) team ratings, one filter per statistic.

For a statistic y (points scored, points per drive, rushing efficiency ...), every team-game is
    y(team on offense) = mu + hfa * h + off[team] + def[opponent] + noise
where h is +1 at home, -1 away, 0 neutral. Each filter keeps a mean and a full covariance for every
team's offense and defense, so every rating carries its own uncertainty, and a forecast's uncertainty
is computed from it (not assumed equal for all games).

Chronology is the guard against look-ahead: games are processed week by week, every game's
before-game ratings are recorded first, and only then are that week's results absorbed. Ratings drift
a little each week (process noise) so recent games count more, as evidence warrants.

Between seasons each rating restarts from a preseason prior: a regression, fitted only on earlier
seasons, of how teams' final ratings carried over given last season's rating and the personnel tables
(talent, recruiting, returning production, quarterback, transfers, roster experience). Its variance is
team-specific: teams returning less production start with wider uncertainty.
"""
import numpy as np
import pandas as pd

try:  # in-place rank-1 update (about twice as fast as numpy's outer product on big covariances)
    from scipy.linalg.blas import dger as _dger
except Exception:  # pragma: no cover
    _dger = None

# statistic -> (column in team_games, weight column or None, weight scale)
METRICS = {
    "pts": ("pts", None, 1.0),
    "ppd": ("ppd", "ng_drives", 11.0),
    "drives": ("drives", None, 1.0),
    "rppa": ("rppa", "rplays", 30.0),
    "pppa": ("pppa", "pplays", 30.0),
    "rsr": ("rsr", "rplays", 30.0),
    "psr": ("psr", "pplays", 30.0),
    "xpl": ("xpl", "plays", 60.0),
    "stuff": ("stuff", "rplays", 30.0),
    "line": ("line", "rplays", 30.0),
    "rshare": ("rshare", "plays", 60.0),
    "opp_rate": ("opp_rate", "ng_drives", 11.0),
    "fin": ("fin", "opps", 5.0),
    "fpos": ("fpos", "ng_drives", 11.0),
}

# Filter settings (observation noise R and weekly drift q, as shares of the statistic's game-to-game
# variance), chosen on one-step-ahead errors in 2017-2022: points 0.5/0.005; points per drive, rushing and
# passing efficiency all did best near 0.5-0.75 / 0.006, and drives (pace) at 0.75 / 0.006.
PARAMS = {m: (0.6, 0.006) for m in METRICS}
PARAMS.update(pts=(0.5, 0.005), drives=(0.75, 0.006))

PRIOR_FEATS = ["talent_z", "rec_z", "ret_all", "oret", "dret", "qbl_z", "qbd", "exp", "tin_z", "tout_z", "lower"]


def _z(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    return ((s - s.mean()) / sd) if sd and sd > 0 else s * 0


class PairFilter:
    def __init__(self, teams, R, q, mu0, sd0, hfa0=0.0):
        self.teams = list(teams)
        self.ix = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        self.n = n
        self.x = np.zeros(2 + 2 * n)
        self.x[0], self.x[1] = mu0, hfa0
        self.P = np.asfortranarray(np.diag(np.r_[sd0 ** 2 * 0.25, (sd0 * 0.15) ** 2, np.full(2 * n, sd0 ** 2)]))
        self.R, self.q = R, q
        self.qmu = 0.0

    def add_teams(self, teams, sd):
        new = [t for t in teams if t not in self.ix]
        if not new:
            return
        n0, k = self.n, len(new)
        x, P = self.x, self.P
        # layout: [mu, hfa, off(n), def(n)] -> grow both blocks
        xo, xd = x[2:2 + n0], x[2 + n0:]
        x2 = np.r_[x[:2], xo, np.zeros(k), xd, np.zeros(k)]
        idx_old = np.r_[0, 1, np.arange(2, 2 + n0), np.arange(2 + n0 + k, 2 + 2 * n0 + k)]
        P2 = np.zeros((len(x2), len(x2)))
        P2[np.ix_(idx_old, idx_old)] = P
        for j in range(k):
            P2[2 + n0 + j, 2 + n0 + j] = sd ** 2
            P2[2 + 2 * n0 + k + j, 2 + 2 * n0 + k + j] = sd ** 2
        self.teams += new
        for t in new:
            self.ix[t] = len(self.ix)
        self.n = n0 + k
        self.x, self.P = x2, np.asfortranarray(P2)

    def o(self, t):
        return 2 + self.ix[t]

    def d(self, t):
        return 2 + self.n + self.ix[t]

    def vec(self, team, opp, h):
        return [0, 1, self.o(team), self.d(opp)], np.array([1.0, float(h), 1.0, 1.0])

    def predict(self, team, opp, h):
        i, a = self.vec(team, opp, h)
        return float(a @ self.x[i]), float(a @ self.P[np.ix_(i, i)] @ a)

    def diff_var(self, home, away, h):
        """Variance of (home points - away points) and of (home + away) predictions."""
        i1, a1 = self.vec(home, away, h)
        i2, a2 = self.vec(away, home, -h)
        idx = sorted(set(i1) | set(i2))
        pos = {k: j for j, k in enumerate(idx)}
        v1, v2 = np.zeros(len(idx)), np.zeros(len(idx))
        for k, a in zip(i1, a1):
            v1[pos[k]] += a
        for k, a in zip(i2, a2):
            v2[pos[k]] += a
        S = self.P[np.ix_(idx, idx)]
        dm, dt = v1 - v2, v1 + v2
        return float(dm @ S @ dm), float(dt @ S @ dt)

    def update(self, team, opp, h, y, w=1.0):
        i, a = self.vec(team, opp, h)
        Pa = self.P[:, i] @ a
        s = float(a @ Pa[i]) + self.R / max(w, 0.05)
        e = y - float(a @ self.x[i])
        K = Pa / s
        self.x += K * e
        if _dger is not None and self.P.flags.f_contiguous:
            self.P = _dger(-1.0, K, Pa, a=self.P, overwrite_a=1)
        else:
            self.P -= np.outer(K, Pa)
        return e, s

    def drift(self, q=None):
        q = self.q if q is None else q
        k = np.arange(2, 2 + 2 * self.n)
        self.P[k, k] += q
        self.P[0, 0] += self.qmu  # the league's scoring level drifts within a season too

    def ratings(self):
        n = self.n
        return pd.DataFrame(dict(team=self.teams, off=self.x[2:2 + n], deff=self.x[2 + n:],
                                 off_sd=np.sqrt(np.diag(self.P)[2:2 + n]), def_sd=np.sqrt(np.diag(self.P)[2 + n:])))


def _prior_frame(teams, prev, pers, tiers):
    """Features for the preseason prior of each team (one row per team)."""
    df = pd.DataFrame(dict(team=list(teams)))
    if prev is not None:
        df = df.merge(prev[["team", "off", "deff"]].rename(columns={"off": "prev_off", "deff": "prev_def"}),
                      on="team", how="left")
    else:
        df["prev_off"] = df["prev_def"] = np.nan
    p = pers.copy() if pers is not None else pd.DataFrame(columns=["team"])
    for c, z in (("talent", "talent_z"), ("rec", "rec_z"), ("qbl", "qbl_z"), ("tin", "tin_z"), ("tout", "tout_z")):
        p[z] = _z(p[c]) if c in p else np.nan
    keep = ["team"] + [c for c in dict.fromkeys(PRIOR_FEATS + ["ret_all"]) if c in p and c != "lower"]
    df = df.merge(p[keep], on="team", how="left")
    for c in PRIOR_FEATS + ["ret_all"]:
        if c not in df:
            df[c] = np.nan
    df["lower"] = df.team.map(lambda t: float(tiers.get(t) != "fbs"))
    df["has_prev"] = df.prev_off.notna().astype(float)
    return df


class PriorModel:
    """Ridge regression: this season's final rating ~ last season's final rating + personnel.
    Fitted on (season-1 -> season) pairs from earlier seasons only."""

    def __init__(self, alpha=5.0):
        self.alpha = alpha
        self.coef = {}
        self.var = {}

    @staticmethod
    def X(df, side):
        prev = df[f"prev_{side}"].fillna(0.0)
        cols = [np.ones(len(df)), prev, df.has_prev]
        for c in PRIOR_FEATS:
            cols.append(df[c].fillna(0.0) if c in df else np.zeros(len(df)))
        return np.column_stack(cols).astype(float)

    def fit(self, pairs):
        """pairs: DataFrame of prior features plus final_off, final_def."""
        for side in ("off", "def"):
            y = pairs[f"final_{side}"].to_numpy(float)
            X = self.X(pairs, side)
            pen = np.diag([0.0, 0.1, 0.1] + [self.alpha] * (X.shape[1] - 3))
            b = np.linalg.solve(X.T @ X + pen, X.T @ y)
            r = y - X @ b
            # team-specific variance: residual^2 against returning production and whether a rating existed
            ret = pairs.ret_all.fillna(pairs.ret_all.median() if pairs.ret_all.notna().any() else 0.5).to_numpy(float)
            V = np.column_stack([np.ones(len(r)), 1 - ret, 1 - pairs.has_prev.to_numpy(float)])
            v = np.linalg.lstsq(V, r ** 2, rcond=None)[0]
            self.coef[side] = b
            self.var[side] = (v, float(np.mean(r ** 2)))

    def predict(self, df, side):
        if side not in self.coef:
            return None, None
        m = self.X(df, side) @ self.coef[side]
        v, base = self.var[side]
        ret = df.ret_all.fillna(0.4).to_numpy(float)
        V = np.column_stack([np.ones(len(df)), 1 - ret, 1 - df.has_prev.to_numpy(float)])
        var = np.clip(V @ v, 0.35 * base, 4.0 * base)
        return m, var


def run_filter(metric, TG, G, pers, tiers, R, q, carry=0.6, sd0=None, prior_alpha=5.0, record=True, qmu=0.0, mu0=None):
    """Run one statistic's filter over every season. Returns (pregame table, final ratings by season).
    pregame: one row per game with home/away offense and defense ratings before the game, and for
    points the variance of the projected margin and total."""
    col, wcol, wscale = METRICS[metric]
    tg = TG[TG[col].notna()]
    # starting values from the first (burn-in) season only, so no later season informs an earlier forecast
    y0 = tg[tg.season == tg.season.min()][col].to_numpy(float)
    sd0 = float(np.nanstd(y0)) * 0.5 if sd0 is None else sd0
    mu0 = float(np.nanmean(y0)) if mu0 is None else mu0
    seasons = sorted(G.season.unique())
    F = None
    finals, pairs_hist = {}, []
    pre_rows = []
    pm = PriorModel(prior_alpha)
    for s in seasons:
        Gs = G[G.season == s]
        teams = sorted(set(Gs.home) | set(Gs.away))
        prev = finals.get(s - 1)
        pf = _prior_frame(teams, prev, pers.get(s), tiers.get(s, {}))
        if F is None:
            F = PairFilter(teams, R, q, mu0, sd0)
            F.qmu = qmu
        else:
            F.add_teams(teams, sd0)
            # restart every team from its preseason prior
            pm_ok = len(pairs_hist) >= 1
            if pm_ok:
                pm.fit(pd.concat(pairs_hist, ignore_index=True))
            ix = {t: i for i, t in enumerate(pf.team)}
            for side in ("off", "def"):
                if pm_ok:
                    m, v = pm.predict(pf, side)
                else:
                    pv = pf[f"prev_{side}"].to_numpy(float)
                    m = np.where(np.isnan(pv), 0.0, carry * pv)
                    v = np.full(len(pf), sd0 ** 2)
                for t, mm, vv in zip(pf.team, m, v):
                    k = F.o(t) if side == "off" else F.d(t)
                    F.x[k] = mm
                    F.P[k, :] = 0.0
                    F.P[:, k] = 0.0
                    F.P[k, k] = vv
            F.P[0, 0] += (0.1 * sd0) ** 2
            F.P[1, 1] += (0.05 * sd0) ** 2
        # center: league-average rating 0 for FBS teams keeps mu identifiable
        for (wk,), grp in Gs.groupby(["wk"], sort=True):
            if record:
                for r in grp.itertuples():
                    h = 0 if r.neutral else 1
                    ph, vh = F.predict(r.home, r.away, h)
                    pa, va = F.predict(r.away, r.home, -h)
                    row = dict(game_id=r.game_id, **{f"{metric}_h": ph, f"{metric}_a": pa,
                                                     f"{metric}_oh": F.x[F.o(r.home)], f"{metric}_dh": F.x[F.d(r.home)],
                                                     f"{metric}_oa": F.x[F.o(r.away)], f"{metric}_da": F.x[F.d(r.away)],
                                                     f"{metric}_mu": F.x[0], f"{metric}_hfa": F.x[1]})
                    if metric == "pts":
                        vm, vt = F.diff_var(r.home, r.away, h)
                        row.update(pts_vm=vm, pts_vt=vt, pts_vh=vh, pts_va=va,
                                   pts_sdo_h=np.sqrt(F.P[F.o(r.home), F.o(r.home)]),
                                   pts_sdo_a=np.sqrt(F.P[F.o(r.away), F.o(r.away)]))
                    pre_rows.append(row)
            obs = tg[tg.game_id.isin(grp.game_id)]
            if len(obs):
                F.drift()
                for r in obs.itertuples():
                    w = (getattr(r, wcol) / wscale) if wcol else 1.0
                    if not np.isfinite(w) or w <= 0:
                        continue
                    F.update(r.team, r.opp, r.h, getattr(r, col), w)
        fin = F.ratings()
        fin = fin[fin.team.isin(teams)]
        finals[s] = fin
        pf2 = pf.merge(fin[["team", "off", "deff"]].rename(columns={"off": "final_off", "deff": "final_def"}),
                       on="team", how="left")
        pairs_hist.append(pf2[pf2.final_off.notna()])
    return pd.DataFrame(pre_rows), finals, F


def tiers_by_season(G):
    out = {}
    for s, g in G.groupby("season"):
        t = dict(zip(g.home, g.home_cls))
        t.update(dict(zip(g.away, g.away_cls)))
        out[s] = t
    return out


def run_all(TG, G, pers, params=None, metrics=None):
    """All filters. params: {metric: (R, q)}; defaults are estimated from the data."""
    tiers = tiers_by_season(G)
    pre, fins, filt = None, {}, {}
    for m in (metrics or METRICS):
        col = METRICS[m][0]
        v = float(np.nanvar(TG[TG.season == TG[TG[col].notna()].season.min()][col]))  # burn-in season only
        r, qq = (params or PARAMS).get(m, (0.6, 0.006))
        R, q = r * v, qq * v
        p, f, F = run_filter(m, TG, G, pers, tiers, R, q)
        pre = p if pre is None else pre.merge(p, on="game_id", how="outer")
        fins[m] = f
        filt[m] = F
    return pre, fins, filt
