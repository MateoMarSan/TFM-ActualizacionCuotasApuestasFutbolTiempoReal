"""Procesado de los eventos de StatsBomb: las 12 acciones, los pasos de tiempo
y las seis modalidades de entrada.

Solo usa numpy para poder meterlo también en las Lambdas, donde no cabe
PyTorch. Así el entrenamiento y el directo preparan los datos con el mismo
código.
"""

import numpy as np


# Constantes
COMPETITIONS = [11, 2, 12, 7, 9]     # orden del one-hot de competición (+1 "otra")

# Las 12 acciones objetivo del paper
ACTIONS = ['goals', 'assists', 'shots', 'shots_on_target', 'corners', 'passes',
           'accurate_passes', 'tackles', 'fouls', 'yellow_cards', 'red_cards',
           'own_goals']
A = {name: i for i, name in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)

# Eventos clave: cada uno abre un paso de tiempo
KEY_EVENTS = ['Shot', 'Foul Committed', 'Corner', 'Throw-in', 'Goal Kick',
              'Kick Off', 'Substitution', 'Bad Behaviour', 'Offside',
              'Injury Stoppage', 'Half Start', 'Half End', 'Own Goal Against']
KEY_IDX = {name: i for i, name in enumerate(KEY_EVENTS)}
SET_PIECES = {'Corner', 'Throw-in', 'Goal Kick', 'Kick Off'}   # tipos de pase clave

N_POS = 4                            # portero, defensa, centrocampista, delantero

# Historia previa de football-data (historial.py)
D_HIST_TEAM = 5    # Elo, puntos, goles a favor, goles en contra, venía de segunda
D_HIST_GAME = 2    # diferencia de Elo y victoria local esperada según el Elo

# Dimensiones de cada modalidad
D_PLAYER = 1 + N_POS + 1 + 1 + N_ACTIONS + 1        # 20
D_TEAM = 1 + N_ACTIONS + 1 + 1                      # 15
D_GAME = len(KEY_EVENTS) + 1 + 2 + 3 + 2 + 3 + 2    # 26
D_PLAYER_STR = N_ACTIONS + 3                        # 15
# Parte calculada con StatsBomb + parte de historia previa (historial.py)
D_TEAM_STR_BASE = 2 * N_ACTIONS + 2                 # 26
D_GAME_CTX_BASE = len(COMPETITIONS) + 1 + 2         # 8
D_TEAM_STR = D_TEAM_STR_BASE + D_HIST_TEAM          # 31
D_GAME_CTX = D_GAME_CTX_BASE + D_HIST_GAME          # 10


# Utilidades sobre eventos
def position_group(name: str | None) -> int:
    """Posición StatsBomb -> 0 portero, 1 defensa, 2 centrocampista, 3 delantero."""
    n = (name or '').lower()
    if 'goalkeeper' in n:
        return 0
    if 'back' in n:
        return 1
    if 'wing' in n or 'forward' in n or 'striker' in n:
        return 3
    return 2


def key_event_type(e: dict) -> str | None:
    """Tipo de evento clave, o None si el evento no abre un paso de tiempo."""
    t = e['type']['name']
    if t == 'Pass':
        pt = (e['pass'].get('type') or {}).get('name')
        return pt if pt in SET_PIECES else None
    return t if t in KEY_IDX else None


def card_of(e: dict) -> str | None:
    for key in ('foul_committed', 'bad_behaviour'):
        card = (e.get(key, {}).get('card') or {}).get('name')
        if card:
            return card
    return None


def player_actions(e: dict) -> list[int]:
    """Acciones (índices de ACTIONS) que suma el jugador que realiza el evento."""
    t = e['type']['name']
    acts = []
    if t == 'Shot':
        outcome = e['shot']['outcome']['name']
        acts.append(A['shots'])
        if outcome in ('Goal', 'Saved', 'Saved To Post'):
            acts.append(A['shots_on_target'])
        if outcome == 'Goal':
            acts.append(A['goals'])
    elif t == 'Pass':
        p = e['pass']
        acts.append(A['passes'])
        if 'outcome' not in p:                       # sin outcome = pase completado
            acts.append(A['accurate_passes'])
        if p.get('goal_assist'):
            acts.append(A['assists'])
        if (p.get('type') or {}).get('name') == 'Corner':
            acts.append(A['corners'])
    elif t == 'Duel' and (e['duel'].get('type') or {}).get('name') == 'Tackle':
        acts.append(A['tackles'])
    elif t == 'Foul Committed':
        acts.append(A['fouls'])
    elif t == 'Own Goal Against':
        acts.append(A['own_goals'])

    card = card_of(e)
    if card in ('Yellow Card', 'Second Yellow'):
        acts.append(A['yellow_cards'])
    if card in ('Red Card', 'Second Yellow'):
        acts.append(A['red_cards'])
    return acts


# Procesado de un partido
def process_events(events: list[dict], meta: dict, squad: list | None = None) -> dict | None:
    """
    Convierte la lista de eventos de un partido (completa, o hasta el último
    evento recibido si el partido está en juego) en arrays compactos. Las
    entradas y los objetivos del modelo se derivan de ellos en build_inputs.

    squad: lista opcional de (player_id, lado, posición) de los jugadores del
    partido, conocida de antemano. En directo hace falta para que los
    suplentes que aún no han salido existan ya en la rejilla (fuera del
    campo), igual que cuando se procesa el partido completo para entrenar.

    Devuelve None si el partido no tiene alineaciones o eventos clave.
    """
    side_of = {meta['home_team_id']: 0, meta['away_team_id']: 1}

    # Pasos de tiempo: posición de cada evento clave en la lista
    key_pos, key_types, seen_half = [], [], set()
    for i, e in enumerate(events):
        kt = key_event_type(e)
        if kt is None:
            continue
        if kt in ('Half Start', 'Half End'):         # vienen duplicados (uno por equipo)
            if (kt, e['period']) in seen_half:
                continue
            seen_half.add((kt, e['period']))
        key_pos.append(i)
        key_types.append(kt)
    T = len(key_pos)
    if T == 0:
        return None
    key_pos = np.array(key_pos)
    # Cada evento pertenece al paso del primer evento clave igual o posterior:
    # el paso s acumula todo lo ocurrido en (clave s-1, clave s]
    step_of = np.minimum(np.searchsorted(key_pos, np.arange(len(events))), T - 1)

    # Jugadores
    players = {}   # id -> dict(side, pos, entry_step, exit_step, started)

    def add_player(pid, side, pos, entry, started=False):
        if pid not in players:
            players[pid] = {'side': side, 'pos': pos, 'entry': entry,
                            'exit': T, 'started': started}

    for e in events:
        if e['type']['name'] == 'Starting XI':
            side = side_of.get(e['team']['id'])
            if side is None:
                return None
            for p in e['tactics']['lineup']:
                add_player(p['player']['id'], side,
                           position_group(p['position']['name']), 0, True)
    for pid, side, pos in (squad or []):       # resto de jugadores: aún sin entrar
        add_player(int(pid), int(side), int(pos), T)
    if len(players) < 22:
        return None

    counts = []            # (player_id, paso, acción)
    own_goals_for = []     # (lado beneficiado, paso)
    reds = np.zeros((2, T), dtype=np.int8)
    clock = np.zeros(T, dtype=np.float32)
    game = np.zeros((T, D_GAME), dtype=np.float32)
    poss = np.zeros((2, T), dtype=bool)
    involved = []          # (player_id, paso)

    for i, e in enumerate(events):
        s = int(step_of[i])
        team_side = side_of.get((e.get('team') or {}).get('id'))
        pid = (e.get('player') or {}).get('id')
        if pid is not None and team_side is not None:
            add_player(pid, team_side,
                       position_group((e.get('position') or {}).get('name')), T)

        t = e['type']['name']
        if t == 'Substitution' and pid is not None:
            rid = e['substitution']['replacement']['id']
            add_player(rid, team_side,
                       position_group((e.get('position') or {}).get('name')), s)
            players[rid]['entry'] = min(players[rid]['entry'], s)
            players[pid]['exit'] = min(players[pid]['exit'], s)
        if t == 'Own Goal For' and team_side is not None:
            own_goals_for.append((team_side, s))
        if pid is not None:
            for a in player_actions(e):
                counts.append((pid, s, a))
            if card_of(e) in ('Red Card', 'Second Yellow'):
                players[pid]['exit'] = min(players[pid]['exit'], s)
                reds[players[pid]['side'], s] += 1

        # Features del partido en el paso s (tomadas del propio evento clave)
        if i == key_pos[s]:
            clock[s] = e['minute'] + e['second'] / 60.0
            g = game[s]
            g[KEY_IDX[key_types[s]]] = 1.0
            o = len(KEY_EVENTS)
            g[o] = clock[s] / 90.0
            g[o + 1] = e['period'] == 1
            g[o + 2] = e['period'] >= 2
            loc = e.get('location')
            if loc:
                g[o + 3], g[o + 4], g[o + 5] = loc[0] / 120.0, loc[1] / 80.0, 1.0
            if team_side is not None:
                g[o + 6 + team_side] = 1.0
            pt = (e.get('possession_team') or {}).get('id')
            if pt in side_of:
                poss[side_of[pt], s] = True
            if pid is not None:
                involved.append((pid, s))

    # Arrays por jugador
    order = sorted(players, key=lambda p: (players[p]['side'],
                                           not players[p]['started'],
                                           players[p]['pos']))
    idx = {p: k for k, p in enumerate(order)}
    P = len(order)
    counts_p = np.zeros((P, T, N_ACTIONS), dtype=np.int16)
    for pid, s, a in counts:
        counts_p[idx[pid], s, a] += 1
    side = np.array([players[p]['side'] for p in order], dtype=np.int8)
    steps = np.arange(T)
    entry = np.array([players[p]['entry'] for p in order])[:, None]
    exit_ = np.array([players[p]['exit'] for p in order])[:, None]
    on_pitch = (steps >= entry) & (steps < exit_)
    # Minutos jugados: suma de avances del reloj estando en el campo
    # (el reloj vuelve a 45 al empezar la 2ª parte -> incrementos negativos = 0)
    dt = np.diff(clock, prepend=0.0).clip(min=0.0)
    minutes = np.cumsum(on_pitch * dt, axis=1).astype(np.float32)
    inv = np.zeros((P, T), dtype=bool)
    for pid, s in involved:
        if pid in idx:
            inv[idx[pid], s] = True

    # Arrays por equipo
    counts_t = np.zeros((2, T, N_ACTIONS), dtype=np.int16)
    for sd in (0, 1):
        counts_t[sd] = counts_p[side == sd].sum(axis=0)
    for sd, s in own_goals_for:            # los autogoles del rival suman goles
        counts_t[sd, s, A['goals']] += 1

    # Marcador y rojas acumulados -> features del partido
    score = np.cumsum(counts_t[:, :, A['goals']], axis=1)
    reds_acc = np.cumsum(reds, axis=1)
    o = len(KEY_EVENTS) + 8
    game[:, o] = score[0] / 5.0
    game[:, o + 1] = score[1] / 5.0
    game[:, o + 2] = (score[0] - score[1]) / 5.0
    game[:, o + 3] = reds_acc[0] / 2.0
    game[:, o + 4] = reds_acc[1] / 2.0

    # Contexto pre-partido
    ctx = np.zeros(D_GAME_CTX_BASE, dtype=np.float32)
    comp = meta['competition_id']
    ctx[COMPETITIONS.index(comp) if comp in COMPETITIONS else len(COMPETITIONS)] = 1.0
    hh, mm = meta['kick_off'].split(':')[:2]
    ctx[-2] = (int(hh) + int(mm) / 60.0) / 24.0
    ctx[-1] = (meta['match_week'] or 19) / 38.0

    hs, as_ = meta.get('home_score'), meta.get('away_score')
    outcome = None if hs is None or as_ is None else (0 if hs > as_ else (1 if hs == as_ else 2))
    return {
        'meta': meta,
        'player_ids': np.array(order, dtype=np.int64),
        'side': side,
        'pos': np.array([players[p]['pos'] for p in order], dtype=np.int8),
        'started': np.array([players[p]['started'] for p in order], dtype=bool),
        'counts_p': counts_p, 'counts_t': counts_t,
        'on_pitch': on_pitch, 'minutes': minutes, 'involved': inv,
        'poss': poss, 'reds': reds_acc.astype(np.int8),
        'game': game, 'clock': clock, 'ctx': ctx,
        'outcome': outcome,                 # None en directo (resultado aún desconocido)
    }


# Entradas y objetivos del modelo
def build_inputs(r: dict, p_str: np.ndarray, t_str: np.ndarray,
                 h_team: np.ndarray, h_game: np.ndarray) -> dict:
    """
    A partir de un partido procesado (process_events) y de su información
    previa, construye las seis modalidades de entrada y los objetivos.

      p_str (P, D_PLAYER_STR)          nivel previo de los jugadores
      t_str (2, D_TEAM_STR_BASE)       nivel previo de los equipos (StatsBomb)
      h_team (2, D_HIST_TEAM), h_game (D_HIST_GAME,)   historia (football-data)
    """
    P, T = r['on_pitch'].shape

    # Acumulados hasta cada paso (incluido) y totales del partido
    cum_p = np.cumsum(r['counts_p'], axis=1, dtype=np.float32)       # (P, T, 12)
    cum_t = np.cumsum(r['counts_t'], axis=1, dtype=np.float32)       # (2, T, 12)

    x_player = np.zeros((P, T, D_PLAYER), dtype=np.float32)
    x_player[:, :, 0] = (r['side'] == 0)[:, None]                   # es local
    x_player[:, :, 1:1 + N_POS] = np.eye(N_POS, dtype=np.float32)[r['pos']][:, None]
    o = 1 + N_POS
    x_player[:, :, o] = r['on_pitch']
    x_player[:, :, o + 1] = r['minutes'] / 90.0
    x_player[:, :, o + 2:o + 2 + N_ACTIONS] = np.log1p(cum_p) / 4.0
    x_player[:, :, -1] = r['involved']

    x_team = np.zeros((2, T, D_TEAM), dtype=np.float32)
    x_team[0, :, 0] = 1.0                                           # es local
    x_team[:, :, 1:1 + N_ACTIONS] = np.log1p(cum_t) / 6.0
    x_team[:, :, -2] = (11 - r['reds']) / 11.0                       # jugadores en el campo
    x_team[:, :, -1] = r['poss']

    # Objetivo: lo que queda desde cada columna hasta el final.
    # Columna 0 = pre-partido (queda todo); columna s+1 = tras el evento s.
    tot_p, tot_t = cum_p[:, -1:], cum_t[:, -1:]
    y_player = np.concatenate([tot_p, tot_p - cum_p], axis=1)       # (P, T+1, 12)
    y_team = np.concatenate([tot_t, tot_t - cum_t], axis=1)         # (2, T+1, 12)

    return {
        'x_player': x_player,
        'x_player_str': p_str.astype(np.float32),
        'x_team': x_team,
        'x_team_str': np.concatenate([t_str, h_team], axis=1).astype(np.float32),
        'x_game': r['game'],
        'x_game_ctx': np.concatenate([r['ctx'], h_game]).astype(np.float32),
        'y_player': y_player,
        'y_team': y_team,
        'y_outcome': r['outcome'],
        'clock': r['clock'],
    }
