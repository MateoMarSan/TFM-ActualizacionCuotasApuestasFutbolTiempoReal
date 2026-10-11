"""Lambda de cuotas: recibe las predicciones del modelo, calcula las cuotas con
odds_generator.py y las publica en SNS.
"""

import json
import os
import time

import boto3
import numpy as np

from odds_generator import OddsGenerator

sns = boto3.client('sns')
generador = OddsGenerator(alpha=float(os.environ['ALFA']),
                          margin=float(os.environ['MARGEN']),
                          temperature=float(os.environ['TEMPERATURA']))


def handler(event, context):
    odds = generador.generate(np.array(event['p_outcome']),
                              np.array(event['lam_team']),
                              np.array(event['acumulado']))
    probs = odds.pop('_probs')
    marcador = odds.pop('_score')
    lam_goles = odds.pop('_lam_goles')      # reparto coherente con el 1X2
    mensaje = {
        'match_id': event['match_id'], 'seq': event['seq'],
        'minuto': event['minuto'], 'segundo': event['segundo'], 'evento': event['evento'],
        'local': event['local'], 'visitante': event['visitante'],
        'marcador': list(marcador),
        'cuotas': odds, 'probabilidades': probs,
        'goles_esperados_restantes': [round(lam_goles[0], 3), round(lam_goles[1], 3)],
        'latencia_modelo_ms': event['latencia_modelo_ms'],
        'publicado': int(time.time() * 1000),
    }
    sns.publish(TopicArn=os.environ['TEMA_CUOTAS'], Message=json.dumps(mensaje))
    return {'ok': True}
