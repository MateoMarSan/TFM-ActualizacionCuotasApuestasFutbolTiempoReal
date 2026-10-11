"""Evaluación en el conjunto de test (el último 15 % de la temporada).

Por un lado las métricas del paper: log-probabilidad y error de calibración
por acción, para jugadores y equipos. Por otro, la comparación con las casas
de football-data (solo hay cuotas de antes del partido): 1X2 y Más/Menos 2,5
goles, y el 1X2 del modelo a lo largo del partido.

  python evaluate.py                   # contra Pinnacle al cierre
  python evaluate.py --bookmaker B365
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

from data_pipeline import FootballDataset, split_datasets, ACTIONS, A
from odds_generator import OddsGenerator, prob_over, calibrate
from historial import load_fd, find_fd_match     # cuotas y emparejamiento con football-data
from train import load_model, predict_match, column_at_minute

# (competition_id, season_id) StatsBomb -> (división, temporada) football-data
FD_FILES = {(11, 27): ('SP1', '1516'), (2, 27): ('E0', '1516'), (12, 27): ('I1', '1516'),
            (7, 27): ('F1', '1516'), (9, 27): ('D1', '1516')}
LEAGUE_NAMES = {11: 'La Liga', 2: 'Premier League', 12: 'Serie A', 7: 'Ligue 1',
                9: 'Bundesliga'}

# Columnas 1X2 por casa (alternativas según la temporada del fichero)
BOOKMAKERS = {
    'PSC':  [('PSCH', 'PSCD', 'PSCA')],                                  # Pinnacle cierre
    'PS':   [('PSH', 'PSD', 'PSA')],                                     # Pinnacle apertura
    'B365': [('B365H', 'B365D', 'B365A')],
    'Avg':  [('AvgH', 'AvgD', 'AvgA'), ('BbAvH', 'BbAvD', 'BbAvA')],     # media mercado
}
OU_COLS = [('PC>2.5', 'PC<2.5'), ('Avg>2.5', 'Avg<2.5'), ('BbAv>2.5', 'BbAv<2.5')]
EPS = 1e-12


# Métricas
def poisson_logpmf(y: np.ndarray, lam: np.ndarray) -> np.ndarray:
    return y * np.log(np.maximum(lam, EPS)) - lam - gammaln(y + 1)


def calibration(p: np.ndarray, hit: np.ndarray, n_bins: int = 20) -> tuple[float, pd.DataFrame]:
    """Error de calibración (media ponderada de |p − frecuencia| por intervalo)
    y la curva de calibración."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    df = pd.DataFrame({'bin': bins, 'p': p, 'hit': hit})
    curve = df.groupby('bin').agg(p=('p', 'mean'), freq=('hit', 'mean'), n=('p', 'size'))
    ece = float((curve.n * (curve.p - curve.freq).abs()).sum() / curve.n.sum())
    return ece, curve


def modal_calibration_data(y: np.ndarray, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidad del valor modal de la Poisson (⌊λ⌋) y si acertó."""
    mode = np.floor(lam)
    return np.exp(poisson_logpmf(mode, lam)), (y == mode).astype(np.float32)


def brier_score(probs: np.ndarray, outcome: np.ndarray) -> float:
    """Brier multiclase: media de Σ_k (p_k − y_k)². Rango [0, 2]."""
    y = np.eye(probs.shape[1])[outcome]
    return float(((probs - y) ** 2).sum(axis=1).mean())


def log_loss(probs: np.ndarray, outcome: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(outcome)), outcome], EPS, 1)).mean())


def correlation(a: np.ndarray, b: np.ndarray) -> dict:
    pr, pp = stats.pearsonr(a, b)
    sr, sp = stats.spearmanr(a, b)
    return {'pearson': pr, 'pearson_p': pp, 'spearman': sr, 'spearman_p': sp}


def demargin(odds: np.ndarray) -> np.ndarray:
    inv = 1.0 / odds
    return inv / inv.sum(axis=-1, keepdims=True)


def pick_cols(row: pd.Series, alternatives: list[tuple]) -> tuple | None:
    for cols in alternatives:
        if all(c in row.index and pd.notna(row[c]) for c in cols):
            return cols
    return None


# Evaluación
def evaluate(args):
    model, a = load_model(args.checkpoint, args.device)
    dataset = FootballDataset(a['preset'], a['max_matches'], a['data_dir'],
                              convocatoria=a.get('convocatoria', False))
    splits = dict(zip(['train', 'val', 'test'], split_datasets(dataset)))
    indices = splits[args.split].indices
    T = a['temperature']
    gen = OddsGenerator(alpha=args.alpha, margin=args.margin, temperature=T)
    print(f"Evaluando {len(indices)} partidos ({args.split}) con {args.checkpoint} "
          f"(1X2 calibrado con T = {T:.3f})")

    # Acumuladores de las métricas del paper
    logp = {('team', a_): [] for a_ in ACTIONS} | {('player', a_): [] for a_ in ACTIONS}
    calib = {k: ([], []) for k in logp}
    logp_outcome, calib_outcome = [], ([], [])

    odds_files, bet_rows, inplay, unmatched = {}, [], [], []
    for n, i in enumerate(indices):
        meta = dataset.meta[i]
        pred = predict_match(model, dataset, i, args.device)

        # Métricas del paper (todas las columnas del partido)
        for level, lam, y in [('team', pred['lam_team'], pred['y_team']),
                              ('player', pred['lam_player'], pred['y_player'])]:
            for a_, name in enumerate(ACTIONS):
                lam_a, y_a = lam[..., a_].ravel(), y[..., a_].ravel()
                logp[(level, name)].append(poisson_logpmf(y_a, lam_a).astype(np.float32))
                p_mode, hit = modal_calibration_data(y_a, lam_a)
                calib[(level, name)][0].append(p_mode.astype(np.float32))
                calib[(level, name)][1].append(hit)
        po = calibrate(pred['p_outcome'], T)         # 1X2 calibrado, como en producción
        logp_outcome.append(np.log(np.clip(po[:, meta_outcome(meta)], EPS, 1)))
        calib_outcome[0].append(po.max(1))
        calib_outcome[1].append((po.argmax(1) == meta_outcome(meta)).astype(np.float32))

        # En juego: 1X2 del modelo a distintos minutos
        for minute in args.minutes:
            c = column_at_minute(pred['clock'], minute)
            o_t = gen.generate(pred['p_outcome'][c], pred['lam_team'][:, c],
                               pred['current_team'][:, c])
            inplay.append({'match_id': meta['match_id'], 'minute': minute,
                           'outcome': meta_outcome(meta),
                           **{f'p{k}': v for k, v in zip('HDA', po[c])},
                           # ¿Señalan las cuotas de goles un favorito distinto al del 1X2?
                           'incoherente_modelo': favorito(po[c]) != favorito(o_t['_probs']['1X2_poisson_modelo']),
                           'incoherente_ajustado': favorito(po[c]) != favorito(o_t['_probs']['1X2_poisson'])})

        # Pre-partido vs casa de apuestas
        key = (meta['competition_id'], meta['season_id'])
        if key not in FD_FILES:
            continue
        if key not in odds_files:
            odds_files[key] = load_fd(*FD_FILES[key])
        real = find_fd_match(odds_files[key], meta)
        cols = pick_cols(real, BOOKMAKERS[args.bookmaker]) if real is not None else None
        if cols is None:
            unmatched.append(f"{meta['home_team']} vs {meta['away_team']} ({meta['match_date']})")
            continue
        o = gen.generate(pred['p_outcome'][0], pred['lam_team'][:, 0], pred['current_team'][:, 0])
        odds_real = real[list(cols)].to_numpy(dtype=float)
        ou_cols = pick_cols(real, OU_COLS)
        lam_goals = pred['lam_team'][:, 0, A['goals']].sum()
        bet_rows.append({
            'match_id': meta['match_id'], 'date': meta['match_date'],
            'league': LEAGUE_NAMES.get(meta['competition_id']),
            'home': meta['home_team'], 'away': meta['away_team'],
            'score': f"{meta['home_score']}-{meta['away_score']}",
            'outcome': meta_outcome(meta),
            'over25': int(meta['home_score'] + meta['away_score'] > 2.5),
            **{f'p_model_{k}': v for k, v in zip('HDA', po[0])},
            **{f'p_poisson_{k}': v for k, v in zip('HDA', o['_probs']['1X2_poisson_modelo'])},
            **{f'p_coherente_{k}': v for k, v in zip('HDA', o['_probs']['1X2_poisson'])},
            **{f'p_book_{k}': v for k, v in zip('HDA', demargin(odds_real))},
            **{f'odds_model_{k}': o[f'1X2_{s}'] for k, s in zip('HDA', ['home', 'draw', 'away'])},
            **{f'odds_book_{k}': v for k, v in zip('HDA', odds_real)},
            'p_model_over25': prob_over(0, lam_goals, 2.5),
            'p_book_over25': demargin(real[list(ou_cols)].to_numpy(dtype=float))[0]
                             if ou_cols else np.nan,
            'expected_goals_model': lam_goals,
        })
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(indices)} partidos")

    if unmatched:
        print(f"Aviso: {len(unmatched)} partidos sin cuotas reales, p. ej.: {unmatched[:3]}")

    os.makedirs(args.out_dir, exist_ok=True)
    paper_table = report_paper_metrics(logp, calib, logp_outcome, calib_outcome, args)
    report_betting(pd.DataFrame(bet_rows), pd.DataFrame(inplay), args)
    paper_table.to_csv(os.path.join(args.out_dir, 'metricas_paper.csv'), index=False)
    print(f"\nResultados guardados en {args.out_dir}/")


def favorito(p) -> int:
    """Favorito entre local (0) y visitante (1), ignorando el empate."""
    return int(p[2] > p[0])


def meta_outcome(meta: dict) -> int:
    h, a = meta['home_score'], meta['away_score']
    return 0 if h > a else (1 if h == a else 2)


# Informes
def report_paper_metrics(logp, calib, logp_outcome, calib_outcome, args) -> pd.DataFrame:
    rows, curves = [], {}
    for name in ACTIONS:
        row = {'accion': name}
        for level in ('team', 'player'):
            row[f'logp_{level}'] = float(np.concatenate(logp[(level, name)]).mean())
            p, hit = (np.concatenate(x) for x in calib[(level, name)])
            row[f'calib_{level}'], curve = calibration(p, hit)
            if level == 'team':
                curves[name] = curve
        rows.append(row)
    ece_o, curve_o = calibration(np.concatenate(calib_outcome[0]),
                                 np.concatenate(calib_outcome[1]))
    rows.append({'accion': 'resultado_1X2',
                 'logp_team': float(np.concatenate(logp_outcome).mean()),
                 'calib_team': ece_o})
    table = pd.DataFrame(rows)

    print(f"\n{'=' * 72}\nMÉTRICAS DEL PAPER (Tabla 1) — test, todos los eventos\n{'=' * 72}")
    print(f"{'Acción':<18}{'log-prob eq.':>13}{'calib. eq.':>11}"
          f"{'log-prob jug.':>15}{'calib. jug.':>12}")
    for r in table.itertuples():
        lp_p = f"{r.logp_player:>15.3f}" if pd.notna(r.logp_player) else f"{'':>15}"
        c_p = f"{r.calib_player:>12.3f}" if pd.notna(r.calib_player) else f"{'':>12}"
        print(f"{r.accion:<18}{r.logp_team:>13.3f}{r.calib_team:>11.3f}{lp_p}{c_p}")

    plot_calibration(curves | {'resultado_1X2': curve_o}, args)
    return table


def report_betting(df: pd.DataFrame, inplay: pd.DataFrame, args):
    if df.empty:
        print("No hay partidos con cuotas reales emparejadas.")
        return
    y = df.outcome.to_numpy()
    P = {name: df[[f'p_{src}_{k}' for k in 'HDA']].to_numpy()
         for name, src in [('model', 'model'), ('poisson', 'poisson'), ('book', 'book')]}
    freq = np.bincount(y, minlength=3) / len(y)

    print(f"\n{'=' * 72}\nPRE-PARTIDO 1X2 — {len(df)} partidos de test  |  casa: {args.bookmaker}"
          f"  ({df.date.min()} → {df.date.max()})\n{'=' * 72}")
    print(f"{'Fuente':<34}{'Brier':>9}{'Log-loss':>10}{'Acierto':>9}")
    table = []
    for name, p in [('Modelo (cabeza 1X2)', P['model']),
                    ('Modelo (Poisson de goles)', P['poisson']),
                    (f'Casa de apuestas ({args.bookmaker})', P['book']),
                    ('Uniforme (1/3)', np.full_like(P['model'], 1 / 3)),
                    ('Frecuencias de la muestra*', np.tile(freq, (len(y), 1)))]:
        row = {'fuente': name, 'brier': brier_score(p, y), 'log_loss': log_loss(p, y),
               'acierto': float((p.argmax(1) == y).mean())}
        table.append(row)
        print(f"{name:<34}{row['brier']:>9.4f}{row['log_loss']:>10.4f}{row['acierto']:>9.1%}")
    print("  * usa el resultado observado del propio test: cota optimista")

    # Coherencia entre la cabeza 1X2 y los mercados de goles
    P_coh = df[[f'p_coherente_{k}' for k in 'HDA']].to_numpy()
    print("\nCOHERENCIA 1X2 ↔ mercados de goles (favorito distinto entre local y visitante)")
    filas_coh = []
    for nombre, Q in [('Goles del modelo (sin ajustar)', P['poisson']),
                      ('Goles con reparto coherente', P_coh)]:
        dis = np.mean([favorito(a) != favorito(b) for a, b in zip(P['model'], Q)])
        dif = float(np.abs(P['model'] - Q).mean())
        dis_j = inplay['incoherente_modelo' if 'sin' in nombre else 'incoherente_ajustado'].mean()
        filas_coh.append({'mercados_goles': nombre, 'favorito_distinto_prepartido': dis,
                          'favorito_distinto_en_juego': dis_j, 'dif_media_prob_prepartido': dif})
        print(f"  {nombre:<32} antes del partido {dis:6.1%} | en juego {dis_j:6.1%} | "
              f"diferencia media de probabilidad {dif:.3f}")
    pd.DataFrame(filas_coh).to_csv(os.path.join(args.out_dir, 'coherencia.csv'), index=False)

    om = df[[f'odds_model_{k}' for k in 'HDA']].to_numpy().ravel()
    ob = df[[f'odds_book_{k}' for k in 'HDA']].to_numpy().ravel()
    corr = {'Cuotas decimales': correlation(om, ob),
            'Prob. implícitas': correlation(P['model'].ravel(), P['book'].ravel()),
            'P(victoria local)': correlation(P['model'][:, 0], P['book'][:, 0])}
    print(f"\nCorrelación modelo vs casa (n={len(om)} cuotas):")
    for name, c in corr.items():
        print(f"  {name:<20} Pearson r={c['pearson']:+.3f} (p={c['pearson_p']:.2g})"
              f"   Spearman ρ={c['spearman']:+.3f} (p={c['spearman_p']:.2g})")
    print(f"  Diferencia media entre probabilidades: {np.abs(P['model'] - P['book']).mean():.4f}")

    print("\nBrier 1X2 por liga:")
    for lg, g in df.groupby('league'):
        yy = g.outcome.to_numpy()
        print(f"  {lg:<16} n={len(g):>3}  modelo {brier_score(P['model'][g.index], yy):.4f}"
              f"  casa {brier_score(P['book'][g.index], yy):.4f}")

    ou = df.dropna(subset=['p_book_over25'])
    yo = ou.over25.to_numpy()
    print(f"\nMÁS/MENOS 2,5 GOLES — {len(ou)} partidos, Más en el {yo.mean():.0%}")
    for name, p in [('Modelo', ou.p_model_over25.to_numpy()),
                    ('Casa (media mercado)', ou.p_book_over25.to_numpy())]:
        P2 = np.stack([1 - p, p], axis=1)
        print(f"  {name:<22} Brier {brier_score(P2, yo) / 2:.4f}  Log-loss {log_loss(P2, yo):.4f}")
    c = correlation(ou.p_model_over25.to_numpy(), ou.p_book_over25.to_numpy())
    print(f"  Correlación P(Más): Pearson r={c['pearson']:+.3f} (p={c['pearson_p']:.2g})")

    print(f"\nEN JUEGO 1X2 — modelo vs resultado final")
    curve = []
    for t, g in inplay.groupby('minute'):
        p, yy = g[['pH', 'pD', 'pA']].to_numpy(), g.outcome.to_numpy()
        curve.append({'minute': t, 'brier': brier_score(p, yy), 'log_loss': log_loss(p, yy)})
        print(f"  minuto {t:>3}: Brier {curve[-1]['brier']:.4f}  Log-loss {curve[-1]['log_loss']:.4f}")
    curve = pd.DataFrame(curve)

    df.to_csv(os.path.join(args.out_dir, 'prepartido_por_partido.csv'), index=False)
    pd.DataFrame(table).to_csv(os.path.join(args.out_dir, 'prepartido_metricas.csv'), index=False)
    pd.DataFrame([{'medida': k, **v} for k, v in corr.items()]).to_csv(
        os.path.join(args.out_dir, 'correlaciones.csv'), index=False)
    curve.to_csv(os.path.join(args.out_dir, 'en_juego_metricas.csv'), index=False)
    plot_betting(P['model'], P['book'], curve, table[2]['brier'], args)


# Figuras
def plot_calibration(curves: dict, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 5, figsize=(15, 9), sharex=True, sharey=True)
    for ax, (name, c) in zip(axes.ravel(), curves.items()):
        ax.bar(c.p, c.n / c.n.max(), width=0.05, color='#d0d0d0')   # densidad
        ax.plot(c.p, c.freq, color='#2a6fdb', lw=1.5)
        ax.plot([0, 1], [0, 1], ':', color='#d9534f')
        ax.set_title(name, fontsize=10)
    for ax in axes.ravel()[len(curves):]:
        ax.axis('off')
    fig.supxlabel('Probabilidad predicha del valor modal')
    fig.supylabel('Frecuencia real')
    fig.suptitle('Calibración por acción (equipos, todos los eventos del test)')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'calibracion.png'), dpi=150)


def plot_betting(P_model, P_book, curve, brier_book, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for k, (lbl, c) in enumerate(zip(['Local', 'Empate', 'Visitante'],
                                     ['#2a6fdb', '#8a8a8a', '#d9534f'])):
        ax[0].scatter(P_book[:, k], P_model[:, k], label=lbl, color=c, alpha=0.6, s=14)
    ax[0].plot([0, 1], [0, 1], 'k--', lw=1)
    ax[0].set(xlabel=f'Probabilidad casa ({args.bookmaker}, sin margen)',
              ylabel='Probabilidad modelo', title='1X2 pre-partido', xlim=(0, 1), ylim=(0, 1))
    ax[0].legend()
    ax[1].plot(curve.minute, curve.brier, 'o-', color='#2a6fdb', label='Modelo en juego')
    ax[1].axhline(brier_book, color='#d9534f', ls='--', label=f'Casa pre-partido ({args.bookmaker})')
    ax[1].set(xlabel='Minuto', ylabel='Brier 1X2 (menor es mejor)',
              title='Brier del modelo a lo largo del partido')
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'evaluacion_cuotas.png'), dpi=150)


def parse_args():
    p = argparse.ArgumentParser(description='Evalúa el modelo (métricas del paper y cuotas)')
    p.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--bookmaker', default='PSC', choices=list(BOOKMAKERS))
    p.add_argument('--alpha', type=float, default=0.94)
    p.add_argument('--margin', type=float, default=0.05)
    p.add_argument('--minutes', type=int, nargs='+', default=[0, 15, 30, 45, 60, 75, 85])
    p.add_argument('--device', default='cpu')
    p.add_argument('--out_dir', default='results')
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())
