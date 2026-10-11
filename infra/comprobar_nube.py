"""Comprueba que el endpoint de SageMaker da lo mismo que el modelo en local.

Coge varios partidos de test, los corta en distintos momentos, monta la
rejilla igual que la Lambda procesadora y compara las dos predicciones.

python infra/comprobar_nube.py
"""

import io
import json
import os
import sys

import boto3
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from data_pipeline import FootballDataset, load_events, split_datasets   # noqa: E402
from procesado import build_inputs, key_event_type, process_events       # noqa: E402
from train import load_model, predict_match                              # noqa: E402

N_PARTIDOS, CORTES = 5, (0.25, 0.5, 1.0)


def main():
    model, a = load_model(os.path.join(RAIZ, 'checkpoints', 'best_model.pt'))
    ds = FootballDataset(a['preset'], a['max_matches'], a['data_dir'],
                         convocatoria=a.get('convocatoria', False))
    _, _, test = split_datasets(ds)
    sm = boto3.client('sagemaker-runtime')
    dif_max = 0.0
    for i in test.indices[:N_PARTIDOS]:
        rec, meta = ds.records[i], ds.meta[i]
        meta_directo = {k: v for k, v in meta.items() if k not in ('home_score', 'away_score')}
        squad = list(zip(rec['player_ids'].tolist(), rec['side'].tolist(), rec['pos'].tolist()))
        p_str = dict(zip(rec['player_ids'].tolist(), ds.p_str[i]))
        local = predict_match(model, ds, i)
        eventos = load_events(meta['match_id'])
        claves = [k for k, e in enumerate(eventos) if key_event_type(e) is not None]
        for corte in CORTES:
            ultimo = claves[int(corte * (len(claves) - 1))]
            r = process_events(eventos[:ultimo + 1], meta_directo, squad)
            x = build_inputs(r, np.stack([p_str[p] for p in r['player_ids']]),
                             ds.t_str[i], ds.h_team[i], ds.h_game[i])
            buf = io.BytesIO()
            np.savez_compressed(buf, **{k: v for k, v in x.items() if k.startswith('x_')})
            resp = sm.invoke_endpoint(EndpointName='tfm-axial-endpoint', ContentType='application/x-npz',
                                      Accept='application/json', Body=buf.getvalue())
            nube = json.loads(resp['Body'].read())
            c = nube['columnas'] - 1                  # misma columna en la predicción local
            d = max(np.abs(np.array(nube['p_outcome']) - local['p_outcome'][c]).max(),
                    np.abs(np.array(nube['lam_team']) - local['lam_team'][:, c]).max())
            dif_max = max(dif_max, d)
            print(f"{meta['home_team']} - {meta['away_team']}: columna {c:>3}, diferencia máx. {d:.1e}")
    print(f'\nDiferencia máxima entre nube y local: {dif_max:.1e}')


if __name__ == '__main__':
    main()
