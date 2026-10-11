"""Inferencia en el endpoint de SageMaker.

Recibe la rejilla del partido en .npz, pasa el modelo entero y devuelve en
JSON la última columna, la del evento que acaba de pasar. Por la máscara
causal da lo mismo que ir paso a paso.
"""

import io
import json
import os

import numpy as np
import torch

from axial_transformer import AxialTransformer

MODALIDADES = ['x_player', 'x_player_str', 'x_team', 'x_team_str', 'x_game', 'x_game_ctx']


def model_fn(model_dir):
    ckpt = torch.load(os.path.join(model_dir, 'model.pth'), map_location='cpu',
                      weights_only=False)
    a = ckpt['args']
    model = AxialTransformer(a['d_model'], a['n_heads'], a['d_ff'], a['n_layers'],
                             a['dropout'], a['ablation'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    torch.set_num_threads(2)
    return model


def input_fn(body, content_type):
    if content_type != 'application/x-npz':
        raise ValueError(f'Tipo de contenido no admitido: {content_type}')
    datos = np.load(io.BytesIO(body))
    b = {k: torch.from_numpy(datos[k].astype(np.float32)).unsqueeze(0) for k in MODALIDADES}
    P, T = b['x_player'].shape[1:3]
    # Un único partido sin relleno: todas las posiciones son válidas
    b['player_mask'] = torch.ones(1, P, dtype=torch.bool)
    b['time_mask'] = torch.ones(1, T + 1, dtype=torch.bool)
    return b


def predict_fn(b, model):
    with torch.no_grad():
        out = model(b)
    return {
        'p_outcome': out['outcome_logits'][0, -1].softmax(-1).tolist(),
        'lam_team': out['log_lambda_team'][0, :, -1].exp().tolist(),
        'lam_player_goals': out['log_lambda_player'][0, :, -1, 0].exp().tolist(),
        'columnas': int(b['time_mask'].shape[1]),
    }


def output_fn(pred, accept):
    return json.dumps(pred), 'application/json'
