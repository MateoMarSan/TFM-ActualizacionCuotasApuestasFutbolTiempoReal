"""Axial Transformer (Horton y Lucey, 2025, secciones 3.3 y 3.4).

La rejilla tiene H = P + 3 filas (jugadores, los dos equipos y el partido) y
W = T + 1 columnas: la primera con la información previa y una más por cada
evento clave.

En cada capa, cada celda atiende a la vez a su fila (pasos anteriores del
mismo agente) y a su columna (todos los agentes en ese paso), con un solo
softmax y las mismas W_Q, W_K, W_V para las dos direcciones (algoritmo 1 del
paper).

El parámetro ablation permite quitar partes como en la sección 4.3 del paper
(no_agent, no_temporal, no_pregame); en el TFM no se ha llegado a usar.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from procesado import (D_PLAYER, D_PLAYER_STR, D_TEAM, D_TEAM_STR,
                       D_GAME, D_GAME_CTX, N_ACTIONS)

ABLATIONS = ('none', 'no_agent', 'no_temporal', 'no_pregame')


# Atención axial (Algoritmo 1)
class AxialAttention(nn.Module):
    """
    Atención sobre la rejilla G (B, H, W, D). Para cada celda (h, w):
      logits de fila    q_hw · k_hv / √d   para los pasos v permitidos
      logits de columna q_hw · k_gw / √d   para los agentes g permitidos
    y un softmax conjunto sobre ambos. Es lo mismo que re-normalizar
    R_row y R_col con N_row / (N_row + N_col) y N_col / (N_row + N_col).
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)   # W_Q, W_K, W_V compartidas
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, G: torch.Tensor, row_mask: torch.Tensor,
                col_mask: torch.Tensor) -> torch.Tensor:
        """
        G:        (B, H, W, D)
        row_mask: (B, 1, 1, W, W)  True = el paso w puede atender al paso v
        col_mask: (B, 1, H, 1, H)  True = el agente h puede atender al agente g
        """
        B, H, W, D = G.shape
        q, k, v = (self.qkv(G).view(B, H, W, 3, self.n_heads, self.d_head)
                   .permute(3, 0, 4, 1, 2, 5))                  # 3 × (B, n, H, W, d)
        scale = 1.0 / math.sqrt(self.d_head)

        s_row = q @ k.transpose(-1, -2)                         # (B, n, H, W, W)
        s_col = (q.transpose(2, 3) @ k.transpose(2, 3).transpose(-1, -2)
                 ).transpose(2, 3)                              # (B, n, H, W, H)
        s_row = s_row.masked_fill(~row_mask, float('-inf'))
        s_col = s_col.masked_fill(~col_mask, float('-inf'))

        attn = torch.softmax(torch.cat([s_row, s_col], dim=-1) * scale, dim=-1)
        attn = self.dropout(attn)
        a_row, a_col = attn.split([W, H], dim=-1)

        out = a_row @ v                                          # (B, n, H, W, d)
        out = out + (a_col.transpose(2, 3) @ v.transpose(2, 3)).transpose(2, 3)
        return self.out(out.permute(0, 2, 3, 1, 4).reshape(B, H, W, D))


class AxialTransformerLayer(nn.Module):
    """Capa encoder estándar (Vaswani et al. 2017) con atención axial:
    atención + residual + LayerNorm, FFN + residual + LayerNorm."""

    def __init__(self, d_model: int = 128, n_heads: int = 4,
                 d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attn = AxialAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, E, row_mask, col_mask):
        E = self.norm1(E + self.dropout(self.attn(E, row_mask, col_mask)))
        return self.norm2(E + self.dropout(self.ffn(E)))


# Modelo completo
class AxialTransformer(nn.Module):
    """
    Entrada: dict con las seis modalidades + máscaras (ver data_pipeline.collate).
    Salida:  dict con
      log_lambda_player (B, P, W, 12)  log λ de las acciones que quedan por jugador
      log_lambda_team   (B, 2, W, 12)  ídem por equipo
      outcome_logits    (B, W, 3)      resultado final [local, empate, visitante]
    """

    def __init__(self, d_model: int = 128, n_heads: int = 4, d_ff: int = 512,
                 n_layers: int = 4, dropout: float = 0.1, ablation: str = 'none'):
        super().__init__()
        assert ablation in ABLATIONS
        self.ablation = ablation

        # Capas de embedding: una por modalidad
        self.f_player = nn.Linear(D_PLAYER, d_model)
        self.f_player_str = nn.Linear(D_PLAYER_STR, d_model)
        self.f_team = nn.Linear(D_TEAM, d_model)
        self.f_team_str = nn.Linear(D_TEAM_STR, d_model)
        self.f_game = nn.Linear(D_GAME, d_model)
        self.f_game_ctx = nn.Linear(D_GAME_CTX, d_model)

        self.layers = nn.ModuleList([
            AxialTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Capas objetivo
        self.head_player = nn.Linear(d_model, N_ACTIONS)
        self.head_team = nn.Linear(d_model, N_ACTIONS)
        self.head_outcome = nn.Linear(d_model, 3)

    def embed(self, b: dict) -> torch.Tensor:
        """Construye la rejilla E₀ (B, H, W, D)."""
        pre = 0.0 if self.ablation == 'no_pregame' else 1.0
        players = torch.cat([self.f_player_str(pre * b['x_player_str']).unsqueeze(2),
                             self.f_player(b['x_player'])], dim=2)       # (B, P, W, D)
        teams = torch.cat([self.f_team_str(pre * b['x_team_str']).unsqueeze(2),
                           self.f_team(b['x_team'])], dim=2)             # (B, 2, W, D)
        game = torch.cat([self.f_game_ctx(pre * b['x_game_ctx']).unsqueeze(1),
                          self.f_game(b['x_game'])], dim=1).unsqueeze(1) # (B, 1, W, D)
        return torch.cat([players, teams, game], dim=1)

    def masks(self, b: dict, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        time_mask = b['time_mask']                                       # (B, W)
        B = time_mask.shape[0]
        agent_mask = torch.cat([b['player_mask'],
                                torch.ones(B, 3, dtype=torch.bool,
                                           device=time_mask.device)], dim=1)  # (B, H)
        # Fila: solo pasos anteriores (máscara estrictamente triangular) y válidos
        strict_past = torch.ones(W, W, dtype=torch.bool, device=time_mask.device).tril(-1)
        if self.ablation == 'no_temporal':
            strict_past = torch.zeros_like(strict_past)
        row_mask = strict_past & time_mask[:, None, :]                   # (B, W, W)
        # Columna: todos los agentes válidos del paso, incluida la propia celda
        eye = torch.eye(H, dtype=torch.bool, device=time_mask.device)
        col_mask = eye if self.ablation == 'no_agent' else (agent_mask[:, None, :] | eye)
        col_mask = col_mask.expand(B, H, H)                              # (B, H, H)
        return row_mask[:, None, None], col_mask[:, None, :, None, :]

    def forward(self, b: dict) -> dict:
        E = self.embed(b)
        B, H, W, _ = E.shape
        row_mask, col_mask = self.masks(b, H, W)
        for layer in self.layers:
            E = layer(E, row_mask, col_mask)

        P = H - 3
        return {
            'log_lambda_player': self.head_player(E[:, :P]).clamp(max=12.0),
            'log_lambda_team': self.head_team(E[:, P:P + 2]).clamp(max=12.0),
            'outcome_logits': self.head_outcome(E[:, P + 2]),
        }


# Comprobaciones
def _random_batch(B=2, P=26, T=40):
    from data_pipeline import collate
    items = []
    for p in (P, P - 3):
        items.append({
            'x_player': torch.randn(p, T, D_PLAYER),
            'x_player_str': torch.randn(p, D_PLAYER_STR),
            'x_team': torch.randn(2, T, D_TEAM),
            'x_team_str': torch.randn(2, D_TEAM_STR),
            'x_game': torch.randn(T, D_GAME),
            'x_game_ctx': torch.randn(D_GAME_CTX),
            'y_player': torch.zeros(p, T + 1, N_ACTIONS),
            'y_team': torch.zeros(2, T + 1, N_ACTIONS),
            'y_outcome': torch.tensor(0),
            'clock': torch.zeros(T),
        })
    return collate(items[:B])


def _reference_attention(attn: AxialAttention, G, row_mask, col_mask):
    """Algoritmo 1 literal: atención por filas y por columnas por separado,
    re-normalizadas con sus factores N_row y N_col."""
    B, H, W, D = G.shape
    n, d = attn.n_heads, attn.d_head
    q, k, v = attn.qkv(G).view(B, H, W, 3, n, d).permute(3, 0, 4, 1, 2, 5)
    R = torch.zeros(B, n, H, W, d)
    for i in range(H):
        for j in range(W):
            lr = (q[:, :, i, j, None] * k[:, :, i]).sum(-1) / math.sqrt(d)   # (B, n, W)
            lc = (q[:, :, i, j, None] * k[:, :, :, j]).sum(-1) / math.sqrt(d)  # (B, n, H)
            er = lr.exp() * row_mask[:, 0, 0, j][:, None]
            ec = lc.exp() * col_mask[:, 0, i, 0][:, None]
            n_row, n_col = er.sum(-1, keepdim=True), ec.sum(-1, keepdim=True)
            r_row = (er[..., None] * v[:, :, i]).sum(2) / n_row.clamp(min=1e-30)
            r_col = (ec[..., None] * v[:, :, :, j]).sum(2) / n_col
            w_row = n_row / (n_row + n_col)
            R[:, :, i, j] = w_row * r_row + (1 - w_row) * r_col
    return attn.out(R.permute(0, 2, 3, 1, 4).reshape(B, H, W, D))


if __name__ == '__main__':
    torch.manual_seed(0)
    model = AxialTransformer()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros entrenables: {n_params:,}")

    b = _random_batch()
    with torch.no_grad():
        out = model(b)
    for k, v in out.items():
        print(f"  {k:<18}{tuple(v.shape)}")

    # 1) La implementación vectorizada coincide con el Algoritmo 1 literal
    small = AxialAttention(16, 2, dropout=0.0).eval()
    bs = _random_batch(P=4, T=5)
    probe = AxialTransformer(d_model=16, n_heads=2)
    G = torch.randn(2, 4 + 3, 6, 16)
    rm, cm = probe.masks(bs, 7, 6)
    with torch.no_grad():
        diff = (small(G, rm, cm) - _reference_attention(small, G, rm, cm)).abs().max()
    assert diff < 1e-5, diff
    print(f"OK: Atención = Algoritmo 1 del paper (dif. máx {diff:.1e})")

    # 2) Causalidad: cambiar el futuro no altera las predicciones del pasado
    b2 = {k: v.clone() for k, v in b.items()}
    b2['x_player'][:, :, 20:] = torch.randn_like(b2['x_player'][:, :, 20:])
    b2['x_game'][:, 20:] = torch.randn_like(b2['x_game'][:, 20:])
    with torch.no_grad():
        out2 = model(b2)
    # columnas 0..20 = pre-partido + eventos 0..19
    assert torch.allclose(out['outcome_logits'][:, :21], out2['outcome_logits'][:, :21], atol=1e-5)
    assert torch.allclose(out['log_lambda_player'][:, :, :21],
                          out2['log_lambda_player'][:, :, :21], atol=1e-5)
    print("OK: Causalidad temporal verificada")

    # 3) El relleno (jugadores/pasos ficticios) no afecta a los agentes reales
    b3 = {k: v.clone() for k, v in b.items()}
    pad = ~b3['player_mask'][1]
    b3['x_player'][1, pad] = torch.randn_like(b3['x_player'][1, pad])
    with torch.no_grad():
        out3 = model(b3)
    assert torch.allclose(out['outcome_logits'], out3['outcome_logits'], atol=1e-5)
    print("OK: Máscara de relleno verificada")

    for abl in ABLATIONS[1:]:
        with torch.no_grad():
            AxialTransformer(ablation=abl).eval()(b)
    print("OK: Ablaciones ejecutan correctamente")
