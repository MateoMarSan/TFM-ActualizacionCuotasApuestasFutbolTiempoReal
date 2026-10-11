"""Diagrama de la arquitectura de AWS para la memoria.

python figura_arquitectura.py [salida.png]
"""

import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FONDO, BORDE, TEXTO, GRIS = '#eef2f6', '#c9d3de', '#1a202c', '#4a5568'
OFF, RT, FIN = '#b7791f', '#2b6cb0', '#2f855a'
FLECHA = '#9fb3c8'


def caja(ax, x, y, w, h, etiqueta, titulo, sub, color, sub2=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2',
                                fc='white', ec=BORDE, lw=0.9))
    ax.add_patch(FancyBboxPatch((x, y + 0.6), 0.55, h - 1.2, boxstyle='square,pad=0',
                                fc=color, ec='none'))
    ax.add_patch(FancyBboxPatch((x + 2, y + h / 2 - 1.6), 7.2, 3.2,
                                boxstyle='round,pad=0,rounding_size=0.6',
                                fc='#edf2f7', ec=BORDE, lw=0.7))
    ax.text(x + 5.6, y + h / 2, etiqueta, ha='center', va='center', fontsize=7.5,
            family='monospace', color=color, weight='bold')
    ty = y + h / 2 + (1.6 if sub else 0)
    ax.text(x + 11, ty, titulo, ha='left', va='center', fontsize=10.5, weight='bold', color=TEXTO)
    if sub:
        ax.text(x + 11, y + h / 2 - 1.7, sub, ha='left', va='center', fontsize=8.3, color=GRIS)


def flecha(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=FLECHA, lw=1.3, mutation_scale=11))


def cabecera(ax, x, w, y, texto, color, fondo):
    ax.add_patch(FancyBboxPatch((x, y), w, 4, boxstyle='round,pad=0,rounding_size=0.8',
                                fc=fondo, ec=BORDE, lw=0.8))
    ax.text(x + w / 2, y + 2, texto, ha='center', va='center', fontsize=9.5,
            weight='bold', color=color)


def main(salida):
    fig = plt.figure(figsize=(10.5, 12.2), dpi=200)
    fig.patch.set_facecolor(FONDO)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 122)
    ax.axis('off')

    H, PASO = 6.6, 9.2                   # alto de caja y separación vertical
    xl, xr, w = 2.5, 54, 48.5            # columnas izquierda y derecha

    # Columna izquierda: entrenamiento y preparación (offline)
    cabecera(ax, xl, w, 116.5, 'ENTRENAMIENTO Y PREPARACIÓN — OFFLINE', OFF, '#f5ecdc')
    izq = [
        ('Datos', 'StatsBomb + football-data.co.uk', '1.551 partidos 2015/16 · resultados 2012–2016'),
        ('PC', 'Entrenamiento local', 'Axial Transformer en PyTorch · CPU · unas 9 h'),
        ('S3', 'Artefacto del modelo', 'Pesos (3,3 MB) + código de inferencia'),
        ('SM-R', 'SageMaker Model Registry', 'Registra y aprueba la versión del modelo'),
    ]
    ys_izq = [106 - i * PASO for i in range(len(izq))]
    for (et, t, s), y in zip(izq, ys_izq):
        caja(ax, xl, y, w, H, et, t, s, OFF)
    for a, b in zip(ys_izq, ys_izq[1:]):
        flecha(ax, xl + w / 2, a, xl + w / 2, b + H)

    # Preparación de los partidos -> tabla de nivel previo
    y_prep = ys_izq[-1] - PASO - 2.5
    caja(ax, xl, y_prep, w, H, 'PC', 'preparar_partidos.py', 'Nivel previo, Elo y convocatoria, con los mismos datos', OFF)

    # Columna derecha: tiempo real
    cabecera(ax, xr, w, 116.5, 'TIEMPO REAL — AWS', RT, '#e1ecf7')
    der = [
        ('Sim', 'Simulador de directo', 'Eventos de StatsBomb en orden · hasta ×60'),
        ('KDS', 'Kinesis Data Streams', 'tfm-eventos · en orden por partido'),
        ('λ', 'Lambda procesadora', 'Reconstruye la rejilla con procesado.py'),
    ]
    ys_der = [106 - i * PASO for i in range(len(der))]
    for (et, t, s), y in zip(der, ys_der):
        caja(ax, xr, y, w, H, et, t, s, RT)
    for a, b in zip(ys_der, ys_der[1:]):
        flecha(ax, xr + w / 2, a, xr + w / 2, b + H)

    # Dos tablas de DynamoDB bajo el procesador
    y_ddb = ys_der[-1] - PASO - 1.5
    wd = (w - 2) / 2
    hd = H + 1.6
    for i, (t, s1, s2) in enumerate([('tfm-historial', 'Eventos del partido', 'en curso'),
                                     ('tfm-nivel-previo', 'Nivel previo y', 'convocatoria')]):
        x = xr + i * (wd + 2)
        ax.add_patch(FancyBboxPatch((x, y_ddb - 1.6), wd, hd, boxstyle='round,pad=0,rounding_size=1.2',
                                    fc='white', ec=BORDE, lw=0.9))
        ax.add_patch(FancyBboxPatch((x, y_ddb - 1.0), 0.55, hd - 1.2, boxstyle='square,pad=0', fc=RT, ec='none'))
        ax.add_patch(FancyBboxPatch((x + 2, y_ddb + 3.4), 6, 2.6, boxstyle='round,pad=0,rounding_size=0.6',
                                    fc='#edf2f7', ec=BORDE, lw=0.7))
        ax.text(x + 5, y_ddb + 4.7, 'DDB', ha='center', va='center', fontsize=7.5,
                family='monospace', color=RT, weight='bold')
        ax.text(x + 9.5, y_ddb + 4.7, 'DynamoDB', ha='left', va='center', fontsize=10, weight='bold', color=TEXTO)
        ax.text(x + 2, y_ddb + 1.6, t, ha='left', va='center', fontsize=8.3, family='monospace', color=TEXTO)
        ax.text(x + 2, y_ddb - 0.3, f'{s1} {s2}', ha='left', va='center', fontsize=8, color=GRIS)
    # procesador <-> tablas
    yb = ys_der[-1]
    flecha(ax, xr + wd / 2, yb, xr + wd / 2, y_ddb - 1.6 + hd)                    # escribe historial
    flecha(ax, xr + wd + 2 + wd / 2, y_ddb - 1.6 + hd, xr + wd + 2 + wd / 2, yb)  # lee nivel previo
    ax.text(xr + wd / 2 + 1, (yb + y_ddb + hd - 1.6) / 2, 'guarda', fontsize=7.5, color=GRIS, va='center')
    ax.text(xr + wd + 2 + wd / 2 + 1, (yb + y_ddb + hd - 1.6) / 2, 'lee', fontsize=7.5, color=GRIS, va='center')

    # preparar_partidos.py -> tfm-nivel-previo (de offline a tiempo real)
    x_np = xr + wd + 2 + wd / 2
    ax.plot([xl + w, x_np, x_np], [y_prep + H / 2, y_prep + H / 2, y_ddb - 1.6],
            color=FLECHA, lw=1.3)
    flecha(ax, x_np, y_prep + H / 2 + 0.01, x_np, y_ddb - 1.6)
    ax.text((xl + w + x_np) / 2, y_prep + H / 2 + 1, 'antes de cada partido', fontsize=7.5,
            color=GRIS, ha='center')

    # Convergencia
    y_sep = y_prep - 3.5
    ax.plot([2.5, 102.5], [y_sep, y_sep], color=BORDE, lw=0.9)
    ax.text(52.5, y_sep - 2.2, 'AMBOS FLUJOS SE UNEN EN EL MODELO', ha='center', va='center',
            fontsize=8.5, color=GRIS)

    xc, wc = 22, 61
    fin = [
        ('SM-E', 'SageMaker Endpoint serverless', 'Ejecuta el Axial Transformer · mediana 135 ms'),
        ('λ', 'Lambda de cuotas', 'odds_generator.py: temperatura, Poisson, α y margen'),
        ('SNS/SQS', 'SNS tfm-cuotas → SQS tfm-plataforma', 'Reparte cada actualización de cuotas'),
        ('λ', 'Lambda plataforma → DynamoDB', 'Guarda las cuotas en tfm-cuotas'),
        ('API', 'API Gateway + Lambda API', 'Web de la plataforma de apuestas'),
    ]
    y0 = y_sep - 11.5
    ys_fin = [y0 - i * PASO for i in range(len(fin))]
    for (et, t, s), y in zip(fin, ys_fin):
        caja(ax, xc, y, wc, H, et, t, s, FIN)
    for a, b in zip(ys_fin, ys_fin[1:]):
        flecha(ax, xc + wc / 2, a, xc + wc / 2, b + H)

    # Registro del modelo y procesador -> endpoint
    ytop = ys_fin[0] + H
    ax.plot([xl + w / 2, xl + w / 2], [ys_izq[-1], ys_izq[-1] - 1.5], color=FLECHA, lw=1.3)
    ax.plot([xl + w / 2, 1.2, 1.2, xc - 4, xc - 4],
            [ys_izq[-1] - 1.5, ys_izq[-1] - 1.5, ytop + 2.5, ytop + 2.5, ytop + 2.5],
            color=FLECHA, lw=1.3)
    flecha(ax, xc - 4, ytop + 2.5, xc + 14, ytop + 0.05)
    ax.text(3, ytop + 4, 'pesos del modelo', fontsize=7.5, color=GRIS)
    x_rt = xr + w + 1.3
    ax.plot([xr + w, x_rt, x_rt, xc + wc + 4], [yb + H / 2, yb + H / 2, ytop + 2.5, ytop + 2.5],
            color=FLECHA, lw=1.3)
    flecha(ax, xc + wc + 4, ytop + 2.5, xc + wc - 14, ytop + 0.05)
    ax.text(xr + w - 0.5, ytop + 4, 'rejilla del partido (.npz)', fontsize=7.5, color=GRIS, ha='right')

    # Pie: piezas transversales
    ax.text(52.5, ys_fin[-1] - 3.2,
            'Todo se crea con CloudFormation · un rol IAM por pieza · aviso de gasto con AWS Budgets',
            ha='center', va='center', fontsize=8.3, color=GRIS, style='italic')

    ax.set_ylim(ys_fin[-1] - 5, 122)
    fig.savefig(salida, facecolor=FONDO, bbox_inches='tight', pad_inches=0.15)
    print(f'Figura guardada en {salida}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'results/arquitectura_aws.png')
