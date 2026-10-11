"""Lambda de la plataforma: lee la cola SQS y guarda cada actualización de
cuotas en DynamoDB (tfm-cuotas) para la web.
"""

import json
import os
import time

import boto3

tabla = boto3.resource('dynamodb').Table(os.environ['TABLA_CUOTAS'])
DOS_DIAS = 2 * 24 * 3600


def handler(event, context):
    for record in event['Records']:
        msg = json.loads(record['body'])
        tabla.put_item(Item={
            'match_id': int(msg['match_id']),
            'seq': int(msg['seq']),
            'datos': json.dumps(msg),
            'expira': int(time.time()) + DOS_DIAS,
        })
    return {'guardadas': len(event['Records'])}
