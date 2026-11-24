"""Model 2.0 score simulator: a possession engine that turns projected points into a full joint
distribution of final scores.

Each team gets a number of drives (shared game pace, so both teams' drive counts move together).
Each drive ends in a touchdown, a field goal or nothing, with probabilities set so the expected points
match the ensemble's projection for that team. Touchdowns are worth 6 plus a try (usually 1, sometimes
0 or 2). Tied games go to overtime. Scores are therefore always whole, never negative, and land on 3s
and 7s as real scores do, so margins of 3, 7 and 10 come out more often than a smooth curve says.

On top of drive-by-drive luck, each simulation shifts both teams' expected points by a game-specific
amount (sd_m for the margin, sd_t for the total, from v2_ensemble's confidence model), so the spread of
outcomes matches the size of the model's real misses on earlier seasons.
"""
import numpy as np


def _drive_probs(e):
    """Per-drive touchdown and field-goal chances for expected points per drive e."""
    e = np.clip(e, 0.02, 5.5)
    pfg = np.clip(0.11 + 0.04 * (e - 2.0), 0.03, 0.20)
    ptd = (e - 3.0 * pfg) / 6.97
    low = ptd < 0.005
    ptd = np.where(low, 0.005, ptd)
    pfg = np.where(low, np.clip((e - 6.97 * 0.005) / 3.0, 0.0, 0.3), pfg)
    tot = ptd + pfg
    over = tot > 0.92
    ptd = np.where(over, ptd * 0.92 / tot, ptd)
    pfg = np.where(over, pfg * 0.92 / tot, pfg)
    return ptd, pfg


def _team_points(rng, D, mu):
    e = mu / D
    ptd, pfg = _drive_probs(e)
    td = rng.binomial(D, ptd)
    fg = rng.binomial(D - td, np.clip(pfg / np.clip(1 - ptd, 1e-6, None), 0, 1))
    extra = rng.binomial(td, 0.95) + 2 * rng.binomial(td, 0.01)
    return 6 * td + extra + 3 * fg


def simulate(mu_h, mu_a, drives, sd_m, sd_t, n=20000, seed=0, pilot=True):
    """Returns (home_scores, away_scores) integer arrays of length n."""
    rng = np.random.default_rng(seed)
    drives = float(np.clip(drives if np.isfinite(drives) else 11.5, 7.0, 18.0))

    def draw(sm, st, rng):
        base = np.clip(np.rint(rng.normal(drives, 1.2, n)), 6, 22).astype(int)
        dh = base + rng.integers(0, 2, n) - (rng.random(n) < 0.5)
        da = base + rng.integers(0, 2, n) - (rng.random(n) < 0.5)
        dh, da = np.clip(dh, 5, None), np.clip(da, 5, None)
        zm, zt = rng.normal(0, 1, n) * sm, rng.normal(0, 1, n) * st
        mh = np.clip(mu_h + (zt + zm) / 2, 0.3, None)
        ma = np.clip(mu_a + (zt - zm) / 2, 0.3, None)
        h = _team_points(rng, dh, mh)
        a = _team_points(rng, da, ma)
        # overtime: alternate possessions until someone leads
        tie = h == a
        if tie.any():
            ph = np.clip(0.5 + (mu_h - mu_a) / 60.0, 0.1, 0.9)
            win = rng.random(tie.sum()) < ph
            pts = np.where(rng.random(tie.sum()) < 0.7, 7, 3)
            h[tie] += np.where(win, pts, 0)
            a[tie] += np.where(win, 0, pts)
        return h, a

    sm, st = sd_m, sd_t
    target_h, target_a = mu_h, mu_a
    if pilot:  # remove the drive-to-drive luck already in the engine from the extra shift
        h0, a0 = draw(0.0, 0.0, np.random.default_rng(seed + 1))
        vm, vt = np.var(h0 - a0), np.var(h0 + a0)
        sm = float(np.sqrt(max(sd_m ** 2 - vm, 0.0)))
        st = float(np.sqrt(max(sd_t ** 2 - vt, 0.0)))
        # the zero-point floor and overtime nudge the averages; shift the inputs once so the simulated
        # average scores match the published projection
        h1, a1 = draw(sm, st, np.random.default_rng(seed + 2))
        mu_h = max(mu_h + (target_h - h1.mean()), 0.3)
        mu_a = max(mu_a + (target_a - a1.mean()), 0.3)
    return draw(sm, st, rng)


def summarize(h, a, lines=None):
    """Win chance, 80% ranges, medians and cover/over chances at given lines (home spread, total)."""
    m, t = h - a, h + a
    out = dict(p_home=float((m > 0).mean()), home_med=int(np.median(h)), away_med=int(np.median(a)),
               m_lo=float(np.quantile(m, 0.1)), m_hi=float(np.quantile(m, 0.9)),
               t_lo=float(np.quantile(t, 0.1)), t_hi=float(np.quantile(t, 0.9)))
    if lines:
        sp, tot = lines
        if sp is not None and np.isfinite(sp):
            win, push = (m + sp > 0).mean(), (m + sp == 0).mean()
            out.update(p_home_cover=float(win / max(1 - push, 1e-9)), p_push_spread=float(push))
        if tot is not None and np.isfinite(tot):
            ov, push = (t > tot).mean(), (t == tot).mean()
            out.update(p_over=float(ov / max(1 - push, 1e-9)), p_push_total=float(push))
    # most likely exact scores
    pairs, cnt = np.unique(np.stack([h, a], 1), axis=0, return_counts=True)
    top = np.argsort(-cnt)[:3]
    out["likely"] = [[int(pairs[i][0]), int(pairs[i][1]), round(float(cnt[i] / len(h)), 3)] for i in top]
    return out
