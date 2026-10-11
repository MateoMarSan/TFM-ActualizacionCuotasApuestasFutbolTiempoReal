"""Sube a DynamoDB (tfm-nivel-previo) la información previa de los partidos de
test: convocatoria, nivel previo, Elo e historia. El resultado no se sube.

También deja la lista de partidos en infra/dist/partidos.json para el
simulador.

python infra/preparar_partidos.py
"""

import io
import json
import os
import sys

import boto3
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
from data_pipeline import FootballDataset, split_datasets   # noqa: E402

DIST = os.path.join(RAIZ, 'infra', 'dist')


def main():
    ds = FootballDataset('top5_1516')
    _, _, test = split_datasets(ds)
    tabla = boto3.resource('dynamodb').Table('tfm-nivel-previo')
    partidos = []
    with tabla.batch_writer() as escritor:
        for i in test.indices:
            meta = {k: v for k, v in ds.meta[i].items() if k not in ('home_score', 'away_score')}
            buffer = io.BytesIO()
            rec = ds.records[i]
            # Jugadores del partido (con su lado y posición): el procesador los
            # necesita desde el principio, igual que en el entrenamiento
            np.savez_compressed(buffer, player_ids=rec['player_ids'],
                                side=rec['side'], pos=rec['pos'],
                                p_str=ds.p_str[i], t_str=ds.t_str[i],
                                h_team=ds.h_team[i], h_game=ds.h_game[i])
            escritor.put_item(Item={'match_id': int(meta['match_id']),
                                    'meta': json.dumps(meta, default=str),
                                    'datos': buffer.getvalue()})
            partidos.append({'match_id': int(meta['match_id']), 'fecha': meta['match_date'],
                             'local': meta['home_team'], 'visitante': meta['away_team']})
    os.makedirs(DIST, exist_ok=True)
    with open(os.path.join(DIST, 'partidos.json'), 'w', encoding='utf-8') as f:
        json.dump(partidos, f, ensure_ascii=False, indent=1)
    print(f'{len(partidos)} partidos de test preparados en DynamoDB (tfm-nivel-previo)')


if __name__ == '__main__':
    main()
