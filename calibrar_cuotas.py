"""Ajuste de alfa y del margen M con las cuotas reales de football-data
(cinco grandes ligas, 2012/13 a 2015/16).

python calibrar_cuotas.py
Con Bet365 sale M = 5,3 % y alfa = 0,94, que es lo que usa OddsGenerator.
"""
import glob, os
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

D = r'C:\Users\usuario\OneDrive\Documentos\Python\TFM\cache\historico'
fr = []
for f in glob.glob(os.path.join(D, '*.csv')):
    if os.path.basename(f).split('_')[0] not in ('E0', 'SP1', 'I1', 'D1', 'F1'):
        continue
    d = pd.read_csv(f, encoding='latin-1')
    d['temp'] = os.path.basename(f)[-8:-4]
    fr.append(d)
df = pd.concat(fr, ignore_index=True)
df = df[df.FTR.isin(['H', 'D', 'A'])]
y = df.FTR.map({'H': 0, 'D': 1, 'A': 2}).values
print('partidos', len(df), df.temp.value_counts().to_dict())


def transf(p, a):
    w = np.power(p, a); return w / w.sum(1, keepdims=True)


def odds(c):
    cols = [c + 'H', c + 'D', c + 'A']
    if not all(x in df for x in cols): return None
    return df[cols].apply(pd.to_numeric, errors='coerce').values


# referencia "verdadera": Pinnacle de cierre sin margen (proporcional)
ref = odds('PSC')
for casa in ['B365', 'BW', 'IW', 'WH', 'VC', 'PS', 'PSC']:
    o = odds(casa)
    ok = np.isfinite(o).all(1) & (o > 1).all(1)
    if casa != 'PSC': okr = ok & np.isfinite(ref).all(1) & (ref > 1).all(1)
    else: okr = ok
    inv = 1 / o[ok]
    M = inv.sum(1).mean() - 1
    q = inv / inv.sum(1, keepdims=True)
    yy = y[ok]
    # alfa por resultados: p = q^(1/a) normalizado, máx. verosimilitud
    nll = lambda a: -np.log(transf(q, 1 / a)[np.arange(len(yy)), yy]).mean()
    a_res = minimize_scalar(nll, bounds=(0.5, 1.5), method='bounded').x
    # alfa frente a Pinnacle cierre: q_casa ≈ transf(p_PSC, a)
    if casa != 'PSC':
        r = 1 / ref[okr]; p = r / r.sum(1, keepdims=True)
        qi = 1 / o[okr]; qi = qi / qi.sum(1, keepdims=True)
        se = lambda a: ((transf(p, a) - qi) ** 2).sum(1).mean()
        a_ps = minimize_scalar(se, bounds=(0.5, 1.5), method='bounded').x
    else:
        a_ps = float('nan')
    print(f'{casa:5s} n={ok.sum():5d}  M={M*100:5.2f}%  alfa(resultados)={a_res:.3f}  alfa(vs PSC)={a_ps:.3f}  logloss={nll(1.0):.4f}')

# margen del O/U 2.5 (media de casas)
for c in [('BbAv>2.5', 'BbAv<2.5'), ('B365>2.5', 'B365<2.5'), ('P>2.5', 'P<2.5')]:
    if c[0] in df:
        o = df[list(c)].apply(pd.to_numeric, errors='coerce').values
        o = o[np.isfinite(o).all(1)]
        print('O/U', c, len(o), f'M={((1/o).sum(1).mean()-1)*100:.2f}%')
