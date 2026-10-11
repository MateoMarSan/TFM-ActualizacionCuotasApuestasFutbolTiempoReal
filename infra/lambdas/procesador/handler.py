"""Lambda procesadora. Por cada evento que llega de Kinesis:
  1. lo guarda en tfm-historial;
  2. si es un evento clave, rehace el partido hasta ahí con procesado.py,
     añade el nivel previo (tfm-nivel-previo), llama al endpoint y pasa el
     resultado a la Lambda de cuotas.
"""

import base64
import io
import json
import os
import time

import boto3
import numpy as np
from boto3.dynamodb.conditions import Key

from procesado import build_inputs, key_event_type, process_events

dynamodb = boto3.resource('dynamodb')
historial = dynamodb.Table(os.environ['TABLA_HISTORIAL'])
nivel_previo = dynamodb.Table(os.environ['TABLA_NIVEL'])
sagemaker = boto3.client('sagemaker-runtime')
lambda_client = boto3.client('lambda')

DOS_DIAS = 2 * 24 * 3600
_cache_nivel = {}          # se reutiliza mientras la Lambda siga "caliente"


def cargar_nivel_previo(match_id: int) -> dict | None:
    """Nivel previo del partido (lo sube infra/preparar_partidos.py)."""
    if match_id not in _cache_nivel:
        item = nivel_previo.get_item(Key={'match_id': match_id}).get('Item')
        if item is None:
            return None
        datos = np.load(io.BytesIO(bytes(item['datos'])))
        _cache_nivel[match_id] = {
            'meta': json.loads(item['meta']),
            'squad': list(zip(datos['player_ids'].tolist(), datos['side'].tolist(),
                              datos['pos'].tolist())),
            'p_str': dict(zip(datos['player_ids'].tolist(), datos['p_str'])),
            'p_str_media': datos['p_str'].mean(axis=0),
            't_str': datos['t_str'], 'h_team': datos['h_team'], 'h_game': datos['h_game'],
        }
    return _cache_nivel[match_id]


def eventos_del_partido(match_id: int) -> list[dict]:
    """Todos los eventos guardados del partido, en orden."""
    eventos, kwargs = [], {'KeyConditionExpression': Key('match_id').eq(match_id)}
    while True:
        resp = historial.query(**kwargs)
        eventos += [json.loads(it['evento']) for it in resp['Items']]
        if 'LastEvaluatedKey' not in resp:
            return eventos
        kwargs['ExclusiveStartKey'] = resp['LastEvaluatedKey']


def predecir(match_id: int, nivel: dict) -> None:
    """Monta la rejilla del partido hasta el último evento y llama al modelo."""
    eventos = eventos_del_partido(match_id)
    r = process_events(eventos, nivel['meta'], nivel['squad'])
    if r is None:
        return
    # Nivel previo de cada jugador, en el orden en que process_events los coloca
    p_str = np.stack([nivel['p_str'].get(pid, nivel['p_str_media']) for pid in r['player_ids']])
    x = build_inputs(r, p_str, nivel['t_str'], nivel['h_team'], nivel['h_game'])

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{k: v for k, v in x.items() if k.startswith('x_')})
    t0 = time.time()
    resp = sagemaker.invoke_endpoint(EndpointName=os.environ['ENDPOINT'],
                                     ContentType='application/x-npz',
                                     Accept='application/json', Body=buffer.getvalue())
    pred = json.loads(resp['Body'].read())
    latencia_ms = round((time.time() - t0) * 1000)

    ultimo = eventos[-1]
    mensaje = {
        'match_id': match_id,
        'seq': len(r['clock']),                          # nº de eventos clave
        'minuto': ultimo['minute'], 'segundo': ultimo['second'],
        'evento': key_event_type(ultimo),
        'local': nivel['meta']['home_team'], 'visitante': nivel['meta']['away_team'],
        'acumulado': r['counts_t'].sum(axis=1).tolist(),  # acciones ya hechas (2 x 12)
        'p_outcome': pred['p_outcome'], 'lam_team': pred['lam_team'],
        'latencia_modelo_ms': latencia_ms,
    }
    lambda_client.invoke(FunctionName=os.environ['FUNCION_CUOTAS'],
                         InvocationType='Event', Payload=json.dumps(mensaje))
    print(json.dumps({'match_id': match_id, 'seq': mensaje['seq'],
                      'minuto': mensaje['minuto'], 'latencia_modelo_ms': latencia_ms}))


def handler(event, context):
    expira = int(time.time()) + DOS_DIAS
    for record in event['Records']:
        msg = json.loads(base64.b64decode(record['kinesis']['data']))
        match_id, evento = int(msg['match_id']), msg['evento']
        historial.put_item(Item={'match_id': match_id, 'idx': int(evento['index']),
                                 'evento': json.dumps(evento), 'expira': expira})
        if key_event_type(evento) is None:
            continue
        nivel = cargar_nivel_previo(match_id)
        if nivel is None:
            print(f'Sin nivel previo para el partido {match_id}: se ignora')
            continue
        predecir(match_id, nivel)
    return {'procesados': len(event['Records'])}
