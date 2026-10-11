"""Historia de los equipos con los resultados de football-data.co.uk.

StatsBomb solo tiene completa la 2015/16, así que en las primeras jornadas no
hay forma de saber qué equipos son buenos. Con los resultados desde 2012/13
(primera y segunda) se calcula el Elo de cada equipo y sus números de la
temporada anterior, siempre con partidos de fechas anteriores.

Las cuotas no entran en el modelo, solo se usan para compararlo.
Aquí está también el emparejamiento de partidos entre StatsBomb y
football-data.
"""

import os
import difflib
import unicodedata
import numpy as np
import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'historico')
SEASONS = ['1213', '1314', '1415', '1516']
# Primera y segunda división de cada liga
DIVISIONS = {'E0': 'E1', 'SP1': 'SP2', 'I1': 'I2', 'F1': 'F2', 'D1': 'D2'}
# competition_id de StatsBomb -> división de football-data
FD_DIVISION = {11: 'SP1', 2: 'E0', 12: 'I1', 7: 'F1', 9: 'D1'}
SEASON_CODE = {27: '1516', 90: '2021'}

# Parámetros del Elo (inspirados en el World Football Elo)
ELO_TOP, ELO_SECOND = 1500.0, 1300.0      # valor inicial en primera / segunda
ELO_K = 20.0                              # velocidad de actualización
ELO_HOME = 60.0                           # ventaja de jugar en casa (puntos Elo)
ELO_REGRESSION = 0.8                      # al empezar temporada: 80 % propio, 20 % media

from procesado import D_HIST_TEAM, D_HIST_GAME   # dimensiones de estas features


# Lectura de ficheros
def load_fd(div: str, season: str) -> pd.DataFrame:
    """Fichero de football-data (lo descarga la primera vez)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'{div}_{season}.csv')
    if not os.path.exists(path):
        r = requests.get(f'https://www.football-data.co.uk/mmz4281/{season}/{div}.csv',
                         timeout=30)
        r.raise_for_status()
        with open(path, 'wb') as f:
            f.write(r.content)
    df = pd.read_csv(path, encoding='latin-1').dropna(subset=['HomeTeam', 'Date', 'FTHG'])
    df['date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed').dt.date
    df['div'], df['season'] = div, season
    return df


def load_all_results() -> pd.DataFrame:
    frames = [load_fd(d, s)[['date', 'div', 'season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
              for s in SEASONS for top, second in DIVISIONS.items() for d in (top, second)]
    return pd.concat(frames, ignore_index=True).sort_values('date', kind='stable')


# Emparejamiento StatsBomb <-> football-data
def _norm(name: str) -> str:
    """Minúsculas, sin tildes ni siglas de club, para comparar nombres."""
    s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()
    drop = {'fc', 'cf', 'ac', 'as', 'afc', 'ogc', 'ss', 'us', 'sc', 'rc', 'cd', 'ud',
            'sd', 'rcd', 'club', 'de', 'calcio', 'vfl', 'vfb', 'tsg', '1.'}
    return ' '.join(w for w in s.replace('-', ' ').split() if w not in drop)


def _sim(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    return max(difflib.SequenceMatcher(None, a, b).ratio(), 1.0 if (a in b or b in a) else 0.0)


def find_fd_match(fd: pd.DataFrame, meta: dict) -> pd.Series | None:
    """Fila de football-data del partido de StatsBomb: misma fecha (±1 día),
    mismo marcador y nombres de equipos parecidos."""
    date = pd.to_datetime(meta['match_date']).date()
    one_day = pd.Timedelta(days=1)
    cand = fd[(fd.date >= date - one_day) & (fd.date <= date + one_day) &
              (fd.FTHG == meta['home_score']) & (fd.FTAG == meta['away_score'])]
    if cand.empty:
        return None
    score = cand.apply(lambda r: _sim(meta['home_team'], r.HomeTeam) +
                                 _sim(meta['away_team'], r.AwayTeam), axis=1)
    return cand.loc[score.idxmax()] if score.max() >= 0.8 else None


# Elo
def expected_home(elo_home: float, elo_away: float) -> float:
    """Resultado esperado del local (1 = gana, 0,5 = empata) según el Elo."""
    return 1.0 / (1.0 + 10 ** (-(elo_home + ELO_HOME - elo_away) / 400.0))


def compute_elo(results: pd.DataFrame) -> dict:
    """
    Recorre todos los partidos por fecha y devuelve, para cada partido
    (fecha, local, visitante), el Elo de ambos equipos antes de jugarlo.
    Los partidos de una misma fecha no se ven entre sí.
    """
    elo, pre = {}, {}
    team_div = {}
    for season, g_season in results.groupby('season', sort=True):
        # Regresión a la media de su división al empezar la temporada
        for div in g_season['div'].unique():
            base = ELO_TOP if div in DIVISIONS else ELO_SECOND
            for team in pd.unique(g_season.loc[g_season['div'] == div, ['HomeTeam', 'AwayTeam']].values.ravel()):
                if team in elo:
                    elo[team] = ELO_REGRESSION * elo[team] + (1 - ELO_REGRESSION) * base
                else:
                    elo[team] = base
                team_div[(season, team)] = div
        for date, day in g_season.groupby('date', sort=True):
            for r in day.itertuples():
                pre[(r.date, r.HomeTeam, r.AwayTeam)] = (elo[r.HomeTeam], elo[r.AwayTeam])
            for r in day.itertuples():
                eh, ea = elo[r.HomeTeam], elo[r.AwayTeam]
                result = 1.0 if r.FTHG > r.FTAG else (0.5 if r.FTHG == r.FTAG else 0.0)
                gd = abs(r.FTHG - r.FTAG)
                mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
                delta = ELO_K * mult * (result - expected_home(eh, ea))
                elo[r.HomeTeam], elo[r.AwayTeam] = eh + delta, ea - delta
    return pre, team_div


def previous_season_stats(results: pd.DataFrame, season: str) -> dict:
    """Puntos, goles a favor y en contra por partido de cada equipo en la
    temporada anterior a `season`, y si jugaba en segunda."""
    prev = SEASONS[SEASONS.index(season) - 1] if season in SEASONS[1:] else None
    stats = {}
    if prev is None:
        return stats
    g = results[results.season == prev]
    for side, gf, ga in [('HomeTeam', 'FTHG', 'FTAG'), ('AwayTeam', 'FTAG', 'FTHG')]:
        for r in g[[side, gf, ga, 'div']].itertuples(index=False):
            team, f, a, div = r
            s = stats.setdefault(team, {'pts': 0.0, 'gf': 0.0, 'ga': 0.0, 'n': 0,
                                        'second': div not in DIVISIONS})
            s['pts'] += 3 if f > a else (1 if f == a else 0)
            s['gf'] += f
            s['ga'] += a
            s['n'] += 1
    return stats


# Features para cada partido de StatsBomb
def history_features(meta: list[dict]) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Devuelve:
      team (N, 2, D_HIST_TEAM): por equipo [Elo, puntos/partido, goles a favor,
                                goles en contra (temporada anterior), venía de segunda]
      game (N, D_HIST_GAME):    [diferencia de Elo, victoria local esperada]
      matched: nº de partidos emparejados con football-data
    """
    results = load_all_results()
    pre_elo, _ = compute_elo(results)
    team = np.zeros((len(meta), 2, D_HIST_TEAM), dtype=np.float32)
    game = np.zeros((len(meta), D_HIST_GAME), dtype=np.float32)
    fd_cache, prev_cache, matched = {}, {}, 0
    # Valores por defecto si un equipo no aparece (media de un recién ascendido)
    default_prev = {'pts': 1.1, 'gf': 1.1, 'ga': 1.4, 'second': True}

    for k, m in enumerate(meta):
        div = FD_DIVISION.get(m['competition_id'])
        season = SEASON_CODE.get(m['season_id'])
        eh = ea = ELO_TOP
        prev_h = prev_a = default_prev
        if div and season in SEASONS:
            if (div, season) not in fd_cache:
                fd_cache[(div, season)] = load_fd(div, season)
                prev_cache[season] = previous_season_stats(results, season)
            row = find_fd_match(fd_cache[(div, season)], m)
            if row is not None:
                matched += 1
                eh, ea = pre_elo[(row.date, row.HomeTeam, row.AwayTeam)]
                ps = prev_cache[season]
                prev_h = {**default_prev, **_avg(ps.get(row.HomeTeam))}
                prev_a = {**default_prev, **_avg(ps.get(row.AwayTeam))}
        for side, e, p in [(0, eh, prev_h), (1, ea, prev_a)]:
            team[k, side] = [(e - ELO_TOP) / 200.0, p['pts'] / 3.0, p['gf'] / 3.0,
                             p['ga'] / 3.0, float(p['second'])]
        game[k] = [(eh - ea) / 200.0, expected_home(eh, ea)]
    return team, game, matched


def _avg(s: dict | None) -> dict:
    if not s or s['n'] == 0:
        return {}
    return {'pts': s['pts'] / s['n'], 'gf': s['gf'] / s['n'], 'ga': s['ga'] / s['n'],
            'second': s['second']}


if __name__ == '__main__':
    res = load_all_results()
    pre, _ = compute_elo(res)
    print(f"{len(res)} partidos históricos cargados ({SEASONS[0]} → {SEASONS[-1]})")
    # Elo al final de 2014/15 (antes de empezar 2015/16) de algunos equipos
    last = {}
    for (d, h, a), (eh, ea) in pre.items():
        if str(d) < '2015-07-01':
            last[h], last[a] = eh, ea
    for liga, equipos in [('La Liga', ['Barcelona', 'Real Madrid', 'Ath Madrid', 'Eibar', 'Granada']),
                          ('Serie A', ['Juventus', 'Roma', 'Napoli', 'Bologna', 'Carpi'])]:
        print(liga + ': ' + ', '.join(f'{t} {last.get(t, float("nan")):.0f}' for t in equipos))
