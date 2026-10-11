"""Paso de las predicciones del modelo a cuotas.

El modelo da el 1X2 y, para cada equipo, la lambda de Poisson de lo que le
queda en cada acción. Sumando lo que ya ha pasado salen los mercados: 1X2,
próximo gol y Más/Menos de goles, tiros, córners y amarillas.

El 1X2 y los goles vienen de salidas distintas del modelo y a veces no cuadran
(por ejemplo, dan favoritos distintos). Para evitarlo se mantienen el 1X2 y el
total de goles esperados, y esos goles se reparten entre los dos equipos de
forma que Poisson dé el mismo 1X2.

Luego se aplica la temperatura del 1X2 (train.py --calibrar), el sesgo
favorito-outsider (p^alfa normalizado) y el margen: cuota = 1 / (p (1 + M)).
"""

import math
import numpy as np

from procesado import A

# Líneas Más/Menos del partido completo (valores habituales en las casas)
DEFAULT_LINES = {'goals': 2.5, 'shots': 24.5, 'corners': 9.5, 'yellow_cards': 4.5}


# Probabilidades
def poisson_pmf(lam: float, max_k: int) -> np.ndarray:
    """P(k) = e^{-λ} λ^k / k!  para k = 0..max_k (en espacio logarítmico)."""
    k = np.arange(max_k + 1)
    log_p = -lam + k * math.log(max(lam, 1e-12)) - np.array([math.lgamma(i + 1) for i in k])
    return np.exp(log_p)


def prob_over(current: float, lam_remaining: float, line: float) -> float:
    """P(total final > línea) = P(current + Poisson(λ_restante) > línea)."""
    need = math.floor(line) + 1 - int(current)          # eventos que aún faltan
    if need <= 0:
        return 1.0
    max_k = int(lam_remaining + 10 * math.sqrt(lam_remaining + 1) + need)
    p = 1.0 - poisson_pmf(lam_remaining, max_k)[:need].sum()
    return float(min(max(p, 0.0), 1.0))          # evita -1e-12 por redondeo


def next_goal_probs(lam_home: float, lam_away: float) -> np.ndarray:
    """[P(marca local), P(marca visitante), P(no hay más goles)] con dos
    procesos de Poisson independientes: el primero en ocurrir se reparte en
    proporción a sus tasas."""
    lam = lam_home + lam_away
    p_none = math.exp(-lam)
    if lam <= 0:
        return np.array([0.0, 0.0, 1.0])
    return np.array([lam_home / lam * (1 - p_none), lam_away / lam * (1 - p_none), p_none])


def calibrate(p: np.ndarray, temperature: float) -> np.ndarray:
    """Calibración por temperatura: equivale a dividir los logits entre T."""
    z = np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)) / temperature
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def poisson_1x2(home_goals: int, away_goals: int, lam_home: float,
                lam_away: float, max_add: int = 10) -> np.ndarray:
    """1X2 alternativo a partir de los goles que quedan (Poisson independientes).
    Sirve para comparar con la cabeza 1X2 del modelo."""
    ph, pa = poisson_pmf(lam_home, max_add), poisson_pmf(lam_away, max_add)
    joint = np.outer(ph, pa)
    diff = (home_goals + np.arange(max_add + 1))[:, None] - \
           (away_goals + np.arange(max_add + 1))[None, :]
    p = np.array([joint[diff > 0].sum(), joint[diff == 0].sum(), joint[diff < 0].sum()])
    return p / p.sum()


def coherent_goal_split(p_1x2: np.ndarray, total: float, home_goals: int,
                        away_goals: int) -> tuple[float, float]:
    """
    Reparte los goles que quedan (`total`) entre local y visitante, λ_h = s·total
    y λ_a = (1 − s)·total, eligiendo la proporción s cuyo 1X2 por Poisson (con
    el marcador actual) más se parece a `p_1x2`. Búsqueda en rejilla, en dos
    pasadas (gruesa y fina), solo con NumPy.
    """
    if total < 0.02:                         # casi no quedan goles: el reparto da igual
        return total / 2, total / 2
    objetivo = np.asarray(p_1x2, dtype=np.float64)
    error = lambda s: float(((poisson_1x2(home_goals, away_goals, s * total,
                                          (1 - s) * total) - objetivo) ** 2).sum())
    rejilla = np.linspace(0.01, 0.99, 99)
    s0 = rejilla[int(np.argmin([error(s) for s in rejilla]))]
    fina = np.linspace(max(0.001, s0 - 0.01), min(0.999, s0 + 0.01), 41)
    s = fina[int(np.argmin([error(x) for x in fina]))]
    return float(s * total), float((1 - s) * total)


# De probabilidad a cuota
def apply_asymmetric_overround(probs: np.ndarray, alpha: float = 0.94) -> np.ndarray:
    """Sesgo favorito-outsider: p_k^α / Σ_j p_j^α  (Albertsen 2017)."""
    assert 0 < alpha <= 1.0, "alpha debe estar en (0, 1]"
    powered = np.power(np.asarray(probs, dtype=np.float64) + 1e-12, alpha)
    return powered / powered.sum()


def prob_to_decimal_odds(p: float, margin: float = 0.0) -> float:
    """Cuota decimal = 1 / (p · (1 + margen))."""
    return round(1.0 / max(p * (1.0 + margin), 1e-6), 2)


# Generador
class OddsGenerator:
    """
    Convierte las predicciones del modelo en un instante del partido en cuotas.

    Uso:
        gen = OddsGenerator(alpha=0.94, margin=0.05)
        odds = gen.generate(p_outcome, lam_team, current_team)
    """

    def __init__(self, alpha: float = 0.94, margin: float = 0.05, lines: dict = None,
                 temperature: float = 1.0, coherent: bool = True):
        self.alpha = alpha
        self.margin = margin
        self.temperature = temperature
        self.coherent = coherent          # mercados de goles coherentes con el 1X2
        self.lines = {**DEFAULT_LINES, **(lines or {})}

    def _odds(self, probs: np.ndarray) -> list[float]:
        adj = apply_asymmetric_overround(probs, self.alpha)
        return [prob_to_decimal_odds(p, self.margin) for p in adj]

    def generate(self, p_outcome: np.ndarray, lam_team: np.ndarray,
                 current_team: np.ndarray) -> dict:
        """
        p_outcome:    (3,)     P(local, empate, visitante) de la cabeza 1X2 (sin calibrar)
        lam_team:     (2, 12)  λ de las acciones que quedan por equipo
        current_team: (2, 12)  acciones ya realizadas por equipo
        """
        g = A['goals']
        hg, ag = int(current_team[0, g]), int(current_team[1, g])
        lam_h_modelo, lam_a_modelo = float(lam_team[0, g]), float(lam_team[1, g])
        p_1x2 = calibrate(p_outcome, self.temperature)
        if self.coherent:
            lam_h, lam_a = coherent_goal_split(p_1x2, lam_h_modelo + lam_a_modelo, hg, ag)
        else:
            lam_h, lam_a = lam_h_modelo, lam_a_modelo

        probs = {
            '1X2': p_1x2,
            'next_goal': next_goal_probs(lam_h, lam_a),
        }
        for action, line in self.lines.items():
            a = A[action]
            p_over = prob_over(current_team[:, a].sum(), lam_team[:, a].sum(), line)
            probs[f'{action}_ou_{line}'] = np.array([p_over, 1.0 - p_over])

        labels = {'1X2': ['home', 'draw', 'away'],
                  'next_goal': ['home', 'away', 'none']}
        odds = {}
        for market, p in probs.items():
            names = labels.get(market, ['over', 'under'])
            for name, o in zip(names, self._odds(p)):
                odds[f'{market}_{name}'] = o

        odds['_probs'] = {k: v.round(4).tolist() for k, v in probs.items()}
        # 1X2 por Poisson con el reparto usado (coherente) y con el del modelo
        odds['_probs']['1X2_poisson'] = poisson_1x2(hg, ag, lam_h, lam_a).round(4).tolist()
        odds['_probs']['1X2_poisson_modelo'] = poisson_1x2(
            hg, ag, lam_h_modelo, lam_a_modelo).round(4).tolist()
        odds['_lam_goles'] = [round(lam_h, 4), round(lam_a, 4)]
        odds['_score'] = (hg, ag)
        return odds


# Ejemplo
if __name__ == '__main__':
    gen = OddsGenerator(alpha=0.94, margin=0.05)

    # Minuto 60, local gana 1-0. El modelo predice que quedan 0,6 goles del
    # local y 0,4 del visitante, 9 tiros, 3 córners y 1,5 amarillas.
    lam = np.zeros((2, 12))
    cur = np.zeros((2, 12))
    lam[:, A['goals']] = [0.6, 0.4]
    lam[:, A['shots']] = [5.0, 4.0]
    lam[:, A['corners']] = [1.5, 1.5]
    lam[:, A['yellow_cards']] = [0.7, 0.8]
    cur[:, A['goals']] = [1, 0]
    cur[:, A['shots']] = [10, 6]
    cur[:, A['corners']] = [4, 3]
    cur[:, A['yellow_cards']] = [1, 2]

    odds = gen.generate(np.array([0.72, 0.20, 0.08]), lam, cur)
    for k, v in odds.items():
        if not k.startswith('_'):
            print(f"  {k:<26}{v}")
    print("  Probabilidades justas:", odds['_probs'])
    overround = sum(1 / odds[f'1X2_{k}'] for k in ('home', 'draw', 'away'))
    print(f"  Overround 1X2: {overround:.3f}")
