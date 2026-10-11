"""Simula el directo: manda a Kinesis los eventos de un partido de test, en
orden y con sus tiempos reales divididos por la velocidad. El descanso se
salta.

  python infra/simulador.py                     # partido al azar, x30
  python infra/simulador.py --velocidad 60
  python infra/simulador.py --match_id 3825739
  python infra/simulador.py --listar
"""

import argparse
import json
import os
import random
import sys
import time

import boto3

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('STATSBOMB_DATA', r'D:\04 Master\BigData\TFM\open-data')
PARTIDOS = os.path.join(RAIZ, 'infra', 'dist', 'partidos.json')
FLUJO = 'tfm-eventos'


def segundos_de_juego(e: dict) -> float:
    """Instante del evento en segundos de juego (la 2ª parte empieza en el 45:00)."""
    return e['minute'] * 60 + e['second'] + float(e['timestamp'][-3:]) / 1000


def main():
    p = argparse.ArgumentParser(description='Simula un partido en directo enviándolo a Kinesis')
    p.add_argument('--match_id', type=int)
    p.add_argument('--velocidad', type=float, default=30.0,
                   help='Factor de aceleración (30 = un partido en ~3 minutos)')
    p.add_argument('--listar', action='store_true')
    args = p.parse_args()

    if not os.path.exists(PARTIDOS):
        sys.exit('Primero hay que desplegar y preparar los partidos (infra/desplegar.py)')
    partidos = json.load(open(PARTIDOS, encoding='utf-8'))
    if args.listar:
        for m in partidos:
            print(f"{m['match_id']}  {m['fecha']}  {m['local']} - {m['visitante']}")
        return
    partido = (next(m for m in partidos if m['match_id'] == args.match_id)
               if args.match_id else random.choice(partidos))
    mid = partido['match_id']
    with open(os.path.join(DATA_DIR, 'data', 'events', f'{mid}.json'), encoding='utf-8') as f:
        eventos = json.load(f)

    print(f"Simulando {partido['local']} - {partido['visitante']} ({partido['fecha']}), "
          f"{len(eventos)} eventos, velocidad x{args.velocidad:g}")
    kinesis = boto3.client('kinesis')
    inicio_real = time.time()
    t0 = segundos_de_juego(eventos[0])
    goles = {}
    lote = []
    for k, e in enumerate(eventos):
        # Esperar hasta el instante (acelerado) en que ocurrió el evento
        objetivo = inicio_real + (segundos_de_juego(e) - t0) / args.velocidad
        espera = objetivo - time.time()
        if espera > 0.05 and lote:
            kinesis.put_records(StreamName=FLUJO, Records=lote)
            lote = []
        if espera > 0:
            time.sleep(espera)
        # La clave de partición es el partido: sus eventos llegan siempre en orden
        lote.append({'Data': json.dumps({'match_id': mid, 'evento': e}).encode(),
                     'PartitionKey': str(mid)})
        if len(lote) == 500:
            kinesis.put_records(StreamName=FLUJO, Records=lote)
            lote = []
        if e['type']['name'] == 'Shot' and e['shot']['outcome']['name'] == 'Goal':
            goles[e['team']['name']] = goles.get(e['team']['name'], 0) + 1
            print(f"  {e['minute']}' ¡Gol de {e['team']['name']}!  {goles}")
        if e['type']['name'] in ('Half Start', 'Half End') and k % 2 == 0:
            print(f"  {e['minute']}' {e['type']['name']} (periodo {e['period']})")
        # Saltar el descanso: al empezar la 2.ª parte se reinicia el reloj
        if e['type']['name'] == 'Half Start' and e['period'] == 2 and k + 1 < len(eventos):
            inicio_real, t0 = time.time(), segundos_de_juego(e)
    if lote:
        kinesis.put_records(StreamName=FLUJO, Records=lote)
    print(f'Partido enviado en {time.time() - inicio_real:.0f} s (segunda parte). Goles: {goles}')


if __name__ == '__main__':
    main()
