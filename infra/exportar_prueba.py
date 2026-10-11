"""Descarga de DynamoDB las cuotas publicadas en una prueba y las guarda en
results/aws/prueba_<match_id>.json y .csv, con un resumen de la latencia.

python infra/exportar_prueba.py --match_id 3754206
"""

import argparse
import json
import os

import boto3
import numpy as np
import pandas as pd
from boto3.dynamodb.conditions import Key

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--match_id', type=int, required=True)
    args = p.parse_args()

    tabla = boto3.resource('dynamodb').Table('tfm-cuotas')
    items, kwargs = [], {'KeyConditionExpression': Key('match_id').eq(args.match_id)}
    while True:
        resp = tabla.query(**kwargs)
        items += resp['Items']
        if 'LastEvaluatedKey' not in resp:
            break
        kwargs['ExclusiveStartKey'] = resp['LastEvaluatedKey']
    mensajes = sorted((json.loads(it['datos']) for it in items), key=lambda m: m['seq'])
    if not mensajes:
        raise SystemExit('No hay cuotas publicadas para ese partido')

    salida = os.path.join(RAIZ, 'results', 'aws')
    os.makedirs(salida, exist_ok=True)
    base = os.path.join(salida, f'prueba_{args.match_id}')
    with open(base + '.json', 'w', encoding='utf-8') as f:
        json.dump(mensajes, f, ensure_ascii=False, indent=1)

    filas = [{
        'seq': m['seq'], 'minuto': m['minuto'], 'segundo': m['segundo'], 'evento': m['evento'],
        'goles_local': m['marcador'][0], 'goles_visitante': m['marcador'][1],
        'cuota_1': m['cuotas']['1X2_home'], 'cuota_X': m['cuotas']['1X2_draw'],
        'cuota_2': m['cuotas']['1X2_away'],
        'p_1': m['probabilidades']['1X2'][0], 'p_X': m['probabilidades']['1X2'][1],
        'p_2': m['probabilidades']['1X2'][2],
        'mas_2.5': m['cuotas']['goals_ou_2.5_over'],
        'goles_esperados_local': m['goles_esperados_restantes'][0],
        'goles_esperados_visitante': m['goles_esperados_restantes'][1],
        'latencia_modelo_ms': m['latencia_modelo_ms'],
    } for m in mensajes]
    df = pd.DataFrame(filas)
    df.to_csv(base + '.csv', index=False)

    lat = df.latencia_modelo_ms.to_numpy()
    print(f'{len(df)} actualizaciones (seq {df.seq.min()}–{df.seq.max()}) guardadas en {base}.json/.csv')
    print(f'Latencia del modelo: mediana {np.median(lat):.0f} ms, p95 {np.percentile(lat, 95):.0f} ms, '
          f'mín {lat.min()} ms, máx {lat.max()} ms')


if __name__ == '__main__':
    main()
