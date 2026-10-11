"""Lambda de la web.

  GET /                        la página
  GET /api/partidos            partidos con cuotas
  GET /api/cuotas?match_id=N   cuotas de un partido
"""

import json
import os
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

tabla = boto3.resource('dynamodb').Table(os.environ['TABLA_CUOTAS'])
HTML = open(os.path.join(os.path.dirname(__file__), 'index.html'), encoding='utf-8').read()


def respuesta(cuerpo, tipo='application/json', estado=200):
    if tipo == 'application/json':
        cuerpo = json.dumps(cuerpo, default=lambda o: int(o) if isinstance(o, Decimal) else str(o))
    return {'statusCode': estado, 'headers': {'Content-Type': tipo, 'Cache-Control': 'no-store'},
            'body': cuerpo}


def partidos():
    """Último estado de cada partido con cuotas."""
    ultimos, kwargs = {}, {}
    while True:
        resp = tabla.scan(**kwargs)
        for it in resp['Items']:
            mid = int(it['match_id'])
            if mid not in ultimos or int(it['seq']) > int(ultimos[mid]['seq']):
                ultimos[mid] = it
        if 'LastEvaluatedKey' not in resp:
            break
        kwargs['ExclusiveStartKey'] = resp['LastEvaluatedKey']
    lista = []
    for mid, it in ultimos.items():
        d = json.loads(it['datos'])
        lista.append({'match_id': mid, 'local': d['local'], 'visitante': d['visitante'],
                      'marcador': d['marcador'], 'minuto': d['minuto'],
                      'publicado': d['publicado']})
    return sorted(lista, key=lambda p: -p['publicado'])


def cuotas(match_id: int):
    items, kwargs = [], {'KeyConditionExpression': Key('match_id').eq(match_id)}
    while True:
        resp = tabla.query(**kwargs)
        items += [json.loads(it['datos']) for it in resp['Items']]
        if 'LastEvaluatedKey' not in resp:
            return items
        kwargs['ExclusiveStartKey'] = resp['LastEvaluatedKey']


def handler(event, context):
    ruta = event.get('rawPath', '/')
    params = event.get('queryStringParameters') or {}
    if ruta == '/api/partidos':
        return respuesta(partidos())
    if ruta == '/api/cuotas' and 'match_id' in params:
        return respuesta(cuotas(int(params['match_id'])))
    if ruta in ('/', '/index.html'):
        return respuesta(HTML, tipo='text/html; charset=utf-8')
    return respuesta({'error': 'no encontrado'}, estado=404)
