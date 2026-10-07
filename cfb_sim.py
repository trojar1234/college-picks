"""Monte Carlo game simulator.

Each team gets a number of possessions. On each possession it scores a
touchdown (7), a field goal (3), or nothing, with probabilities set so the
average matches the model's expected points. A shared "game script" shock
links the two teams' scoring, and a team-level shock adds the extra
variance real games have. Ties go to overtime. This produces realistic,
lumpy scores (24-17, 31-28) instead of 26.4-19.8.
"""
import numpy as np

FG_PER_TD = 0.5  # field goals per touchdown on an average drive
RNG = np.random.default_rng(7)


def simulate(e_home, e_away, drives, sd, rho, n=10000, rng=RNG):
    D = np.clip(np.rint(drives + rng.normal(0, 1.0, n)), 6, 22).astype(int)
    eg = rng.normal(size=n)
    out = []
    for e in (e_home, e_away):
        z = np.sqrt(rho) * eg + np.sqrt(1 - rho) * rng.normal(size=n)
        v = (max(e, 0.5) / drives) * np.exp(sd * z - sd * sd / 2)
        p7 = np.clip(v / (7 + 3 * FG_PER_TD), 0.005, 0.8)
        p3 = np.clip(FG_PER_TD * p7, 0, 0.95 - p7)
        n7 = rng.binomial(D, p7)
        n3 = rng.binomial(D - n7, p3 / (1 - p7))
        out.append((7 * n7 + 3 * n3, p7, p3))
    (h, ph7, ph3), (a, pa7, pa3) = out
    h, a = h.copy(), a.copy()
    tied = np.where(h == a)[0]
    for _ in range(8):  # overtime periods
        if len(tied) == 0:
            break
        for arr, p7, p3 in ((h, ph7, ph3), (a, pa7, pa3)):
            u = rng.random(len(tied))
            q7 = np.clip(p7[tied] * 1.6, 0, 0.7)
            q3 = np.clip(p3[tied] * 1.6, 0, 0.95 - q7)
            arr[tied] += np.where(u < q7, 7, np.where(u < q7 + q3, 3, 0))
        tied = tied[h[tied] == a[tied]]
    if len(tied):
        h[tied] += np.where(rng.random(len(tied)) < 0.5, 2, 0)
        a[tied] += np.where(h[tied] == a[tied], 2, 0)
    return h, a


def summarize(h, a, spread=None, total=None):
    m = h - a
    pairs, counts = np.unique(np.stack([h, a], 1), axis=0, return_counts=True)
    top = np.argsort(-counts)[:5]
    s = dict(
        home_med=int(np.median(h)),
        away_med=int(np.median(a)),
        home_mean=float(h.mean()),
        away_mean=float(a.mean()),
        margin=float(m.mean()),
        total_mean=float((h + a).mean()),
        p_home_win=float((m > 0).mean()),
        likely=[[int(pairs[i][0]), int(pairs[i][1]), round(float(counts[i] / len(h)), 4)] for i in top],
        hist=np.histogram(np.clip(m, -45, 45), bins=np.arange(-45, 48, 3))[0].tolist(),
    )
    # Medians can contradict the mean margin; nudge so the projected score agrees with it.
    if (s["home_med"] - s["away_med"]) * s["margin"] < 0 or (s["home_med"] == s["away_med"]):
        s["home_med"] = int(round(s["home_mean"]))
        s["away_med"] = int(round(s["away_mean"]))
        if s["home_med"] == s["away_med"]:
            s["home_med" if s["margin"] >= 0 else "away_med"] += 1
    if spread is not None and spread == spread:
        cover = m + spread  # home covers if margin + spread > 0
        s["p_home_cover"] = float((cover > 0).mean())
        s["p_push_spread"] = float((cover == 0).mean())
    if total is not None and total == total:
        t = h + a
        s["p_over"] = float((t > total).mean())
        s["p_push_total"] = float((t == total).mean())
    return s


def calibrate(target_var, rho, e=27.0, drives=12.0):
    """Pick the team-level shock size so simulated score variance matches real errors."""
    rng = np.random.default_rng(1)
    best, best_err = 0.3, 1e9
    for sd in np.linspace(0.0, 0.9, 19):
        h, _ = simulate(e, e, drives, sd, rho, n=20000, rng=rng)
        err = abs(h.var() - target_var)
        if err < best_err:
            best, best_err = float(sd), err
    return best
