"""Compara el modelo sin convocatoria (v2) con el de la convocatoria completa
(v3) en los 232 partidos de test.

En los jugadores solo se cuentan los que juegan: acertar que un suplente que
no sale no hace nada es muy fácil y le daría ventaja al v3.

Antes hay que pasar evaluate.py con cada versión.
Salida: results/comparacion_v2_v3.csv
"""
import numpy as np
import pandas as pd

from data_pipeline import FootballDataset, split_datasets, ACTIONS
from evaluate import poisson_logpmf
from train import load_model, predict_match

MODELOS = {'v2': 'antiguo/v2_sin_convocatoria/checkpoints/best_model.pt',
           'v3': 'checkpoints/best_model.pt'}
RESULTADOS = {'v2': 'antiguo/v2_sin_convocatoria/results', 'v3': 'results'}


def logp_acciones(nombre):
    model, a = load_model(MODELOS[nombre])
    ds = FootballDataset(a['preset'], a['max_matches'], a['data_dir'],
                         convocatoria=a.get('convocatoria', False))
    _, _, test = split_datasets(ds)
    jug, eq = {k: [] for k in ACTIONS}, {k: [] for k in ACTIONS}
    n_jug = []
    for i in test.indices:
        pred = predict_match(model, ds, i)
        juega = ds.records[i]['minutes'][:, -1] > 0           # jugadores que juegan
        n_jug.append(len(juega))
        for k, name in enumerate(ACTIONS):
            lp = poisson_logpmf(pred['y_player'][juega, :, k], pred['lam_player'][juega, :, k])
            jug[name].append(lp.ravel())
            lt = poisson_logpmf(pred['y_team'][..., k], pred['lam_team'][..., k])
            eq[name].append(lt.ravel())
    print(f"{nombre}: {np.mean(n_jug):.1f} jugadores por partido")
    return ({k: float(np.concatenate(v).mean()) for k, v in jug.items()},
            {k: float(np.concatenate(v).mean()) for k, v in eq.items()})


def cuotas_de_mas(df):
    om = df[[f'odds_model_{k}' for k in 'HDA']].to_numpy()
    pb = df[[f'p_book_{k}' for k in 'HDA']].to_numpy()
    y = np.eye(3)[df.outcome.to_numpy()]
    valor = om > 1 / pb
    ganancia = (y * om - 1)[valor]
    return valor.mean(), valor.sum(), float(np.median((pb * om - 1)[valor])), ganancia.mean()


def main():
    filas = []
    lp = {v: logp_acciones(v) for v in MODELOS}
    for name in ACTIONS:
        filas.append({'medida': f'logp_jugador_{name}', 'v2': lp['v2'][0][name], 'v3': lp['v3'][0][name]})
    for name in ACTIONS:
        filas.append({'medida': f'logp_equipo_{name}', 'v2': lp['v2'][1][name], 'v3': lp['v3'][1][name]})

    pre = {v: pd.read_csv(f'{RESULTADOS[v]}/prepartido_por_partido.csv') for v in MODELOS}
    assert (pre['v2'].match_id.values == pre['v3'].match_id.values).all()
    y = pre['v2'].outcome.to_numpy()
    ll = {v: -np.log(pre[v][[f'p_model_{k}' for k in 'HDA']].to_numpy()[np.arange(len(y)), y])
          for v in MODELOS}
    d = ll['v3'] - ll['v2']
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"1X2 pre-partido, log-loss v3 − v2 = {d.mean():+.4f}  (IC 95 %: {lo:+.4f} a {hi:+.4f})")
    filas.append({'medida': 'logloss_1x2_pre_dif_v3_menos_v2', 'v2': lo, 'v3': hi})
    for v in MODELOS:
        p = pre[v][[f'p_model_{k}' for k in 'HDA']].to_numpy()
        print(f"{v}: P media (1, X, 2) = {np.round(p.mean(0), 3)}")
        frac, n, ev_med, roi = cuotas_de_mas(pre[v])
        print(f"{v}: cuotas por encima de la justa de Pinnacle {frac:.1%} ({n}), "
              f"ventaja mediana {ev_med:.1%}, beneficio del apostante {roi:+.1%}")
        filas += [{'medida': 'pct_cuotas_de_mas', v: frac}, {'medida': 'beneficio_apostante', v: roi},
                  {'medida': 'ventaja_mediana', v: ev_med}]

    df = pd.DataFrame(filas).groupby('medida', sort=False).first().reset_index()
    df.to_csv('results/comparacion_v2_v3.csv', index=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
