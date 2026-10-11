"""Preparación de los datos de StatsBomb para el modelo (sección 3.2 del paper).

- Un paso de tiempo por evento clave (tiros, faltas, córners, cambios...),
  más una columna inicial con la información previa al partido.
- Filas: los convocados (jueguen o no), los dos equipos y el partido. Con
  convocatoria=False solo entran los que juegan (versión v2).
- Entradas: las seis modalidades del paper (se montan en procesado.py).
- Objetivos: lo que le queda a cada jugador y equipo hasta el final, y el 1X2.

El nivel previo y la historia de football-data solo usan partidos de fechas
anteriores.
"""

import os
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from historial import history_features
# Constantes y procesado con NumPy (compartido con las Lambdas de AWS)
from procesado import (A, ACTIONS, N_ACTIONS, N_POS, position_group, D_PLAYER, D_PLAYER_STR, D_TEAM,
                       D_TEAM_STR_BASE, D_GAME, process_events, build_inputs)


# Configuración
DATA_DIR = os.environ.get('STATSBOMB_DATA', r'D:\04 Master\BigData\TFM\open-data')
CACHE_VERSION = 'v4'          # + '_conv' si se usa la convocatoria completa

# Conjuntos de partidos: lista de (competition_id, season_id)
PRESETS = {
    # 5 grandes ligas 2015/16: temporadas completas (1.551 partidos)
    'top5_1516':   [(11, 27), (2, 27), (12, 27), (7, 27), (9, 27)],
    'laliga_1516': [(11, 27)],
}
# Suavizado del nivel previo hacia la media (en partidos "ficticios")
PLAYER_PRIOR_MATCHES = 3
TEAM_PRIOR_MATCHES = 5


# Lectura de datos
def load_matches(competition_id: int, season_id: int,
                 data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Metadatos de los partidos de una competición-temporada."""
    path = os.path.join(data_dir, 'data', 'matches', str(competition_id),
                        f'{season_id}.json')
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    return pd.DataFrame([{
        'match_id':       m['match_id'],
        'match_date':     m['match_date'],
        'kick_off':       m.get('kick_off') or '18:00:00',
        'match_week':     m.get('match_week'),
        'competition_id': competition_id,
        'season_id':      season_id,
        'home_team':      m['home_team']['home_team_name'],
        'away_team':      m['away_team']['away_team_name'],
        'home_team_id':   m['home_team']['home_team_id'],
        'away_team_id':   m['away_team']['away_team_id'],
        'home_score':     m['home_score'],
        'away_score':     m['away_score'],
    } for m in raw])


def load_events(match_id: int, data_dir: str = DATA_DIR) -> list[dict]:
    path = os.path.join(data_dir, 'data', 'events', f'{match_id}.json')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# Procesado de un partido (la lógica está en procesado.py)
def load_lineup(match_id: int, data_dir: str = DATA_DIR) -> list[dict]:
    path = os.path.join(data_dir, 'data', 'lineups', f'{match_id}.json')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def player_positions(matches: pd.DataFrame, data_dir: str = DATA_DIR) -> dict:
    """Grupo de posición más habitual de cada jugador en las convocatorias de
    la temporada. Sirve para los suplentes que no juegan, cuya convocatoria
    no indica posición."""
    votes = {}
    for mid in matches.match_id:
        for team in load_lineup(mid, data_dir):
            for p in team['lineup']:
                for pos in p['positions']:
                    g = position_group(pos['position'])
                    votes.setdefault(p['player_id'], np.zeros(N_POS, int))[g] += 1
    return {pid: int(v.argmax()) for pid, v in votes.items()}


def build_squad(meta: dict, lineup: list[dict], pos_map: dict) -> list[tuple]:
    """Convocatoria del partido como lista de (player_id, lado, posición).
    Sin posición conocida se asume centrocampista."""
    side_of = {meta['home_team_id']: 0, meta['away_team_id']: 1}
    squad = []
    for team in lineup:
        side = side_of.get(team['team_id'])
        if side is None:
            continue
        for p in team['lineup']:
            if p['positions']:
                pos = position_group(p['positions'][0]['position'])
            else:
                pos = pos_map.get(p['player_id'], 2)
            squad.append((p['player_id'], side, pos))
    return squad


def process_match(meta: dict, data_dir: str = DATA_DIR, pos_map: dict | None = None) -> dict | None:
    """Lee los eventos de un partido del disco y los procesa (procesado.py).
    Con pos_map se añade la convocatoria completa como agentes."""
    squad = None
    if pos_map is not None:
        squad = build_squad(meta, load_lineup(meta['match_id'], data_dir), pos_map)
    return process_events(load_events(meta['match_id'], data_dir), meta, squad)


def _process_job(args):
    meta, data_dir, pos_map = args
    try:
        return process_match(meta, data_dir, pos_map)
    except Exception as e:                          # partido corrupto -> se descarta
        print(f"  Aviso: Error en partido {meta['match_id']}: {e}")
        return None


# Nivel previo (strength) — solo con partidos anteriores
def compute_strength(records: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Recorre los partidos en orden cronológico y, para cada uno, calcula el
    nivel previo de jugadores y equipos con lo acumulado en fechas anteriores.

    Jugador: ritmo por 90' de cada acción, suavizado hacia la media de su
      posición (PLAYER_PRIOR_MATCHES partidos ficticios), nº de partidos,
      minutos medios y % de titularidades.
    Equipo: media por partido de cada acción a favor y en contra (suavizada
      hacia la media de su liga), puntos por partido y nº de partidos.
    """
    player_hist, pos_hist = {}, {}                  # sumas acumuladas
    team_hist, league_hist = {}, {}
    p_str = [None] * len(records)
    t_str = [None] * len(records)

    dates = sorted({r['meta']['match_date'] for r in records})
    by_date = {d: [] for d in dates}
    for i, r in enumerate(records):
        by_date[r['meta']['match_date']].append(i)

    for d in dates:
        # 1) features con lo acumulado antes de esta fecha
        for i in by_date[d]:
            r = records[i]
            feats = np.zeros((len(r['player_ids']), D_PLAYER_STR), dtype=np.float32)
            for k, (pid, pos) in enumerate(zip(r['player_ids'], r['pos'])):
                h = player_hist.get(pid)
                ph = pos_hist.get(pos)
                prior = ph['acts'] / ph['min'] if ph and ph['min'] > 0 else np.zeros(N_ACTIONS)
                acts = h['acts'] if h else 0.0
                mins = h['min'] if h else 0.0
                kmin = PLAYER_PRIOR_MATCHES * 90.0
                rate90 = 90.0 * (acts + prior * kmin) / (mins + kmin)
                feats[k, :N_ACTIONS] = np.log1p(rate90) / 3.0
                apps = h['apps'] if h else 0
                feats[k, N_ACTIONS] = np.log1p(apps) / 3.0
                feats[k, N_ACTIONS + 1] = mins / apps / 90.0 if apps else 0.0
                feats[k, N_ACTIONS + 2] = h['starts'] / apps if apps else 0.0
            p_str[i] = feats

            lg_key = (r['meta']['competition_id'], r['meta']['season_id'])
            lg = league_hist.get(lg_key)
            lg_mean = lg['acts'] / lg['n'] if lg else np.zeros(N_ACTIONS)
            tf = np.zeros((2, D_TEAM_STR_BASE), dtype=np.float32)
            for sd, tid in enumerate([r['meta']['home_team_id'], r['meta']['away_team_id']]):
                h = team_hist.get((lg_key, tid))
                n = h['n'] if h else 0
                k = TEAM_PRIOR_MATCHES
                avg_for = ((h['for'] if h else 0.0) + k * lg_mean) / (n + k)
                avg_against = ((h['against'] if h else 0.0) + k * lg_mean) / (n + k)
                ppg = ((h['pts'] if h else 0.0) + k * 1.37) / (n + k)
                tf[sd, :N_ACTIONS] = np.log1p(avg_for) / 6.0
                tf[sd, N_ACTIONS:2 * N_ACTIONS] = np.log1p(avg_against) / 6.0
                tf[sd, -2] = ppg / 3.0
                tf[sd, -1] = np.log1p(n) / 4.0
            t_str[i] = tf

        # 2) actualizar con los partidos de esta fecha
        for i in by_date[d]:
            r = records[i]
            tot_p = r['counts_p'].sum(axis=1).astype(np.float64)
            mins = r['minutes'][:, -1]
            for k, (pid, pos) in enumerate(zip(r['player_ids'], r['pos'])):
                if mins[k] <= 0:
                    continue
                h = player_hist.setdefault(pid, {'acts': np.zeros(N_ACTIONS), 'min': 0.0,
                                                 'apps': 0, 'starts': 0})
                h['acts'] += tot_p[k]
                h['min'] += mins[k]
                h['apps'] += 1
                h['starts'] += int(r['started'][k])
                ph = pos_hist.setdefault(pos, {'acts': np.zeros(N_ACTIONS), 'min': 0.0})
                ph['acts'] += tot_p[k]
                ph['min'] += mins[k]

            lg_key = (r['meta']['competition_id'], r['meta']['season_id'])
            tot_t = r['counts_t'].sum(axis=1).astype(np.float64)
            goals = tot_t[:, A['goals']]
            lg = league_hist.setdefault(lg_key, {'acts': np.zeros(N_ACTIONS), 'n': 0})
            for sd, tid in enumerate([r['meta']['home_team_id'], r['meta']['away_team_id']]):
                h = team_hist.setdefault((lg_key, tid), {'for': np.zeros(N_ACTIONS),
                                                         'against': np.zeros(N_ACTIONS),
                                                         'pts': 0.0, 'n': 0})
                h['for'] += tot_t[sd]
                h['against'] += tot_t[1 - sd]
                gf, ga = goals[sd], goals[1 - sd]
                h['pts'] += 3 if gf > ga else (1 if gf == ga else 0)
                h['n'] += 1
                lg['acts'] += tot_t[sd]
                lg['n'] += 1
    return p_str, t_str


# Dataset PyTorch
class FootballDataset(Dataset):
    """
    Un ejemplo = un partido completo. Ordenados cronológicamente
    (necesario para el split temporal y el nivel previo).
    """

    def __init__(self, preset: str = 'top5_1516', max_matches: int = None,
                 data_dir: str = DATA_DIR, cache_dir: str = None,
                 num_workers: int = 4, convocatoria: bool = True):
        super().__init__()
        cache_dir = cache_dir or os.path.join(os.path.dirname(data_dir), 'cache_tensores')
        matches = pd.concat([load_matches(c, s, data_dir) for c, s in PRESETS[preset]],
                            ignore_index=True)
        matches = matches.sort_values(['match_date', 'match_id']).reset_index(drop=True)
        if max_matches:
            matches = matches.head(max_matches)

        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir,
                                  f'partidos_{CACHE_VERSION}{"_conv" if convocatoria else ""}'
                                  f'_{preset}_{len(matches)}.pt')
        if os.path.exists(cache_path):
            self.records = torch.load(cache_path, weights_only=False)
            print(f"Dataset cargado de caché: {len(self.records)} partidos")
        else:
            print(f"Procesando {len(matches)} partidos ({preset}, {num_workers} procesos)...")
            pos_map = player_positions(matches, data_dir) if convocatoria else None
            jobs = [(m, data_dir, pos_map) for m in matches.to_dict('records')]
            with Pool(num_workers) as pool:
                self.records = [r for r in pool.imap(_process_job, jobs, chunksize=8)
                                if r is not None]
            torch.save(self.records, cache_path)
            print(f"Dataset listo: {len(self.records)} partidos (caché: {cache_path})")

        self.meta = [r['meta'] for r in self.records]
        self.p_str, self.t_str = compute_strength(self.records)
        self.h_team, self.h_game, matched = history_features(self.meta)
        print(f"Historia previa (football-data): {matched}/{len(self.meta)} partidos emparejados")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        x = build_inputs(self.records[i], self.p_str[i], self.t_str[i],
                         self.h_team[i], self.h_game[i])
        out = {k: torch.from_numpy(v) for k, v in x.items() if k != 'y_outcome'}
        out['y_outcome'] = torch.tensor(x['y_outcome'])
        return out


def collate(batch: list[dict]) -> dict:
    """Agrupa partidos de distinto nº de jugadores (P) y pasos (T) con relleno
    y máscaras: player_mask (B, P_max) y time_mask (B, T_max + 1)."""
    B = len(batch)
    P = max(b['x_player'].shape[0] for b in batch)
    T = max(b['x_player'].shape[1] for b in batch)
    out = {
        'x_player': torch.zeros(B, P, T, D_PLAYER),
        'x_player_str': torch.zeros(B, P, D_PLAYER_STR),
        'x_team': torch.zeros(B, 2, T, D_TEAM),
        'x_team_str': torch.stack([b['x_team_str'] for b in batch]),
        'x_game': torch.zeros(B, T, D_GAME),
        'x_game_ctx': torch.stack([b['x_game_ctx'] for b in batch]),
        'y_player': torch.zeros(B, P, T + 1, N_ACTIONS),
        'y_team': torch.zeros(B, 2, T + 1, N_ACTIONS),
        'y_outcome': torch.stack([b['y_outcome'] for b in batch]),
        'clock': torch.zeros(B, T),
        'player_mask': torch.zeros(B, P, dtype=torch.bool),
        'time_mask': torch.zeros(B, T + 1, dtype=torch.bool),
    }
    for k, b in enumerate(batch):
        p, t = b['x_player'].shape[:2]
        out['x_player'][k, :p, :t] = b['x_player']
        out['x_player_str'][k, :p] = b['x_player_str']
        out['x_team'][k, :, :t] = b['x_team']
        out['x_game'][k, :t] = b['x_game']
        out['y_player'][k, :p, :t + 1] = b['y_player']
        out['y_team'][k, :, :t + 1] = b['y_team']
        out['clock'][k, :t] = b['clock']
        out['player_mask'][k, :p] = True
        out['time_mask'][k, :t + 1] = True
    return out


def split_datasets(dataset: FootballDataset, val_frac: float = 0.15,
                   test_frac: float = 0.15) -> tuple[Subset, Subset, Subset]:
    """Split cronológico train / val / test: se entrena con el pasado y se
    evalúa con partidos posteriores, como ocurriría en producción."""
    n = len(dataset)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_val - n_test
    idx = list(range(n))
    return (Subset(dataset, idx[:n_train]),
            Subset(dataset, idx[n_train:n_train + n_val]),
            Subset(dataset, idx[n_train + n_val:]))


def get_dataloaders(preset='top5_1516', max_matches=None, batch_size=2,
                    data_dir=DATA_DIR, num_workers=4, convocatoria=True):
    """(train_loader, val_loader). El test se reserva para evaluate.py."""
    dataset = FootballDataset(preset, max_matches, data_dir, num_workers=num_workers,
                              convocatoria=convocatoria)
    train_ds, val_ds, _ = split_datasets(dataset)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate))


# Comprobaciones
if __name__ == '__main__':
    ds = FootballDataset('top5_1516', max_matches=40)
    tr, va, te = split_datasets(ds)
    print(f"Split cronológico: train={len(tr)}  val={len(va)}  test={len(te)}")
    x = ds[0]
    for k, v in x.items():
        print(f"  {k:<14}{tuple(v.shape)}")
    ok = sum(int(ds[i]['y_team'][0, 0, A['goals']]) == m['home_score'] and
             int(ds[i]['y_team'][1, 0, A['goals']]) == m['away_score']
             for i, m in enumerate(ds.meta))
    print(f"Goles totales == marcador real: {ok}/{len(ds)}")
    print("Pipeline OK")
