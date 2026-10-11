"""Figuras de la memoria que no saca evaluate.py: la evolución del 1X2 en la
prueba de AWS y las cuotas del modelo frente a las de Pinnacle.

python figuras_memoria.py   (después de evaluate.py)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AZUL, GRIS, ROJO = '#2a6fdb', '#8a8a8a', '#d9534f'
TINTA, SUAVE = '#333333', '#bdbdbd'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': SUAVE, 'axes.labelcolor': TINTA,
                     'xtick.color': TINTA, 'ytick.color': TINTA})


def evolucion_partido():
    d = pd.read_csv('results/aws/prueba_3754206.csv')
    t = d.minuto + d.segundo / 60
    fig, ax = plt.subplots(figsize=(9, 4))
    for col, color, nombre in [('p_1', AZUL, 'Newcastle'), ('p_X', GRIS, 'Empate'),
                               ('p_2', ROJO, 'Manchester City')]:
        ax.step(t, d[col], where='post', color=color, lw=2)
        ax.text(t.iloc[-1] + 1, d[col].iloc[-1], nombre, color=TINTA, va='center', fontsize=9)
    # Goles: líneas verticales con el nuevo marcador
    marcador = d.goles_local.astype(str) + '–' + d.goles_visitante.astype(str)
    cambio = marcador != marcador.shift()
    for i in d.index[cambio][1:]:
        ax.axvline(t[i], color=SUAVE, ls='--', lw=1)
        ax.text(t[i] + 0.5, 0.97, f'Gol: {marcador[i]}', fontsize=9, color=TINTA, va='top')
    ax.set_xlim(0, t.iloc[-1] + 14)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Minuto')
    ax.set_ylabel('Probabilidad')
    ax.grid(axis='y', color='#eeeeee')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig('results/evolucion_partido_aws.png', dpi=150)


def cuotas_frente_pinnacle():
    df = pd.read_csv('results/prepartido_por_partido.csv')
    om = df[[f'odds_model_{k}' for k in 'HDA']].to_numpy()
    justa = 1 / df[[f'p_book_{k}' for k in 'HDA']].to_numpy()
    valor = om > justa
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    ax.scatter(justa[~valor], om[~valor], s=16, color=GRIS, alpha=0.6,
               label=f'El modelo paga menos que la cuota justa ({(~valor).sum()})')
    ax.scatter(justa[valor], om[valor], s=16, color=ROJO, alpha=0.7,
               label=f'El modelo paga más: apuesta a favor del apostante ({valor.sum()})')
    lim = [1.05, 100]
    ax.plot(lim, lim, color=TINTA, lw=1)
    # Ejemplo del texto: victoria del Valencia en el Barcelona–Valencia (1-2)
    r = df.index[(df.home == 'Barcelona') & (df.away == 'Valencia')][0]
    ax.annotate('Barcelona–Valencia (1–2):\nvictoria visitante', (justa[r, 2], om[r, 2]),
                xytext=(1.3, 45), fontsize=9, color=TINTA,
                arrowprops=dict(arrowstyle='-', color=TINTA, lw=0.8))
    ax.set_xscale('log'); ax.set_yscale('log')
    ticks = [1.5, 2, 3, 5, 10, 20, 40, 80]
    ax.set_xticks(ticks); ax.set_xticklabels([str(x).replace('.', ',') for x in ticks])
    ax.set_yticks(ticks); ax.set_yticklabels([str(x).replace('.', ',') for x in ticks])
    ax.minorticks_off()
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Cuota justa de Pinnacle (sin margen)')
    ax.set_ylabel('Cuota ofrecida por el modelo')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fontsize=8.5, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig('results/cuotas_frente_pinnacle.png', dpi=150)


if __name__ == '__main__':
    evolucion_partido()
    cuotas_frente_pinnacle()
    print('Figuras guardadas en results/')
