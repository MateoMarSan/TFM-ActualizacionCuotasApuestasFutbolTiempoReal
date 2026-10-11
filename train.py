"""Entrenamiento del Axial Transformer (sección 3.5 del paper).

Pérdida: Poisson para las 12 acciones de jugadores y equipos y entropía
cruzada para el 1X2. El 1X2 lleva peso extra (w_outcome, y w_pre en la columna
previa al partido) porque si no apenas cuenta frente a los pases y el modelo
no aprendía a distinguir al favorito.

Adam con lr 3e-4 y coseno, partición cronológica 70/15/15 y se guarda el
mejor modelo según validación.

  python train.py                              # 5 grandes ligas 2015/16
  python train.py --max_matches 60 --epochs 2  # prueba rápida
  python train.py --sin_convocatoria           # versión v2
  python train.py --calibrar                   # temperatura del 1X2
  python train.py --demo                       # cuotas de un partido de test
"""

import os
import math
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data_pipeline import (FootballDataset, split_datasets, get_dataloaders, collate,
                           DATA_DIR, PRESETS, A)
from axial_transformer import AxialTransformer, ABLATIONS
from odds_generator import OddsGenerator, calibrate


# Pérdida
def poisson_nll(log_lam: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """−log P(k | λ) completo (incluye log k!, así es la log-probabilidad
    del paper con el signo cambiado)."""
    return log_lam.exp() - k * log_lam + torch.lgamma(k + 1)


def compute_loss(out: dict, b: dict, w_outcome: float = 1.0,
                 w_pre: float = 0.0) -> tuple[torch.Tensor, dict]:
    """
    Suma de las pérdidas medias de cada objetivo (12 acciones × jugador,
    12 acciones × equipo y 1X2). Se ignoran jugadores y pasos de relleno.
    """
    time_mask = b['time_mask']                                        # (B, W)
    pmask = (b['player_mask'][:, :, None] & time_mask[:, None, :]).float()   # (B, P, W)
    tmask = time_mask[:, None, :].expand(-1, 2, -1).float()                  # (B, 2, W)

    nll_p = poisson_nll(out['log_lambda_player'], b['y_player'])     # (B, P, W, 12)
    nll_t = poisson_nll(out['log_lambda_team'], b['y_team'])         # (B, 2, W, 12)
    per_action_p = (nll_p * pmask[..., None]).sum((0, 1, 2)) / pmask.sum()
    per_action_t = (nll_t * tmask[..., None]).sum((0, 1, 2)) / tmask.sum()

    logits = out['outcome_logits']                                   # (B, W, 3)
    target = b['y_outcome'][:, None].expand(-1, logits.shape[1])
    ce = F.cross_entropy(logits.transpose(1, 2), target, reduction='none')
    outcome = (ce * time_mask).sum() / time_mask.sum()
    outcome_pre = ce[:, 0].mean()                                     # solo pre-partido

    total = (per_action_p.sum() + per_action_t.sum()
             + w_outcome * outcome + w_pre * outcome_pre)
    parts = {'player': per_action_p.detach(), 'team': per_action_t.detach(),
             'outcome': outcome.detach(), 'outcome_pre': outcome_pre.detach()}
    return total, parts


@torch.no_grad()
def init_output_bias(model: AxialTransformer, loader: DataLoader):
    """Inicializa las cabezas con la media de cada objetivo en el pre-partido
    (log λ medio y log frecuencia de cada resultado). Acelera la convergencia."""
    sum_p = sum_t = 0.0
    n_p = n_t = 0
    counts = torch.zeros(3)
    for b in loader:
        yp = b['y_player'][:, :, 0][b['player_mask']]                 # (n, 12)
        sum_p, n_p = sum_p + yp.sum(0), n_p + len(yp)
        sum_t, n_t = sum_t + b['y_team'][:, :, 0].sum((0, 1)), n_t + 2 * len(b['y_outcome'])
        counts += torch.bincount(b['y_outcome'], minlength=3)
    model.head_player.bias.copy_((sum_p / n_p).clamp(min=1e-3).log())
    model.head_team.bias.copy_((sum_t / n_t).clamp(min=1e-3).log())
    model.head_outcome.bias.copy_((counts / counts.sum()).log())


# Bucles de entrenamiento y validación
def to_device(b: dict, device) -> dict:
    return {k: v.to(device) for k, v in b.items()}


def train_epoch(model, loader, optimizer, device, weights: dict):
    model.train()
    total, n = 0.0, 0
    for b in loader:
        b = to_device(b, device)
        optimizer.zero_grad()
        loss, _ = compute_loss(model(b), b, **weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item() * len(b['y_outcome'])
        n += len(b['y_outcome'])
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, weights: dict = None) -> tuple[float, dict]:
    model.eval()
    total, n = 0.0, 0
    agg = {}
    for b in loader:
        b = to_device(b, device)
        loss, parts = compute_loss(model(b), b, **(weights or {}))
        m = len(b['y_outcome'])
        total += loss.item() * m
        n += m
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v.cpu() * m
    return total / n, {k: v / n for k, v in agg.items()}


# Entrenamiento principal
def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"Dispositivo: {device}  |  ablación: {args.ablation}")

    train_loader, val_loader = get_dataloaders(args.preset, args.max_matches,
                                               args.batch_size, args.data_dir,
                                               args.num_workers, args.convocatoria)
    print(f"Partidos train: {len(train_loader.dataset)}  |  val: {len(val_loader.dataset)}"
          f"  (split cronológico; test reservado)")

    model = AxialTransformer(args.d_model, args.n_heads, args.d_ff, args.n_layers,
                             args.dropout, args.ablation).to(device)
    init_output_bias(model, train_loader)
    print(f"Modelo: {sum(p.numel() for p in model.parameters()):,} parámetros")
    print("Columnas: pérdida total; NLL de los goles que quedan por equipo; "
          "log-loss del 1X2 antes\ndel partido y en todos los eventos "
          "(referencia 1X2: uniforme = 1.0986). Menor es mejor.")

    # Pesos de la pérdida 1X2; la validación usa los mismos para elegir el modelo
    weights = {'w_outcome': args.w_outcome, 'w_pre': args.w_pre}
    print(f"Pesos de la pérdida 1X2: w_outcome={args.w_outcome}, w_pre={args.w_pre}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    print(f"\n{'Época':>5} {'Train':>9} {'Val':>9} {'Goles eq.':>10} "
          f"{'1X2 pre':>8} {'1X2 juego':>9} {'min':>5}")
    print("-" * 63)
    best, bad_epochs = math.inf, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, weights)
        val_loss, parts = evaluate(model, val_loader, device, weights)
        scheduler.step()

        row = {'epoch': epoch, 'train': train_loss, 'val': val_loss,
               'team_goals': float(parts['team'][A['goals']]),
               'outcome_pre': float(parts['outcome_pre']),
               'outcome': float(parts['outcome'])}
        history.append(row)
        print(f"{epoch:>5} {train_loss:>9.3f} {val_loss:>9.3f} {row['team_goals']:>10.4f} "
              f"{row['outcome_pre']:>8.4f} {row['outcome']:>9.4f} "
              f"{(time.time() - t0) / 60:>5.1f}", flush=True)

        if val_loss < best:
            best, bad_epochs = val_loss, 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': val_loss, 'args': vars(args), 'history': history},
                       os.path.join(args.checkpoint_dir, 'best_model.pt'))
            print(f"      OK: checkpoint guardado", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"      Parada temprana: {args.patience} épocas sin mejorar")
                break

    print(f"\nMejor val loss: {best:.3f}")
    return model


# Inferencia
def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Carga un checkpoint. Devuelve (model, args del entrenamiento); los args
    incluyen la temperatura de calibración del 1X2 (1,0 si no se ha calibrado)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    a = {**ckpt['args'], 'temperature': ckpt.get('temperature', 1.0)}
    model = AxialTransformer(a['d_model'], a['n_heads'], a['d_ff'], a['n_layers'],
                             a['dropout'], a['ablation']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, a


@torch.no_grad()
def predict_match(model, dataset: FootballDataset, i: int, device='cpu') -> dict:
    """
    Predicciones de un partido en todas sus columnas (pre-partido + eventos):
      p_outcome (W, 3), lam_team (2, W, 12), lam_player (P, W, 12),
      current_team (2, W, 12) acciones ya hechas, clock (W,) minuto de cada columna.
    """
    item = dataset[i]
    b = to_device(collate([item]), device)
    out = model(b)
    y_team = item['y_team'].numpy()                         # lo que queda
    return {
        'p_outcome': out['outcome_logits'][0].softmax(-1).cpu().numpy(),
        'lam_team': out['log_lambda_team'][0].exp().cpu().numpy(),
        'lam_player': out['log_lambda_player'][0].exp().cpu().numpy(),
        'y_team': y_team,
        'y_player': item['y_player'].numpy(),
        'current_team': y_team[:, :1] - y_team,             # total − lo que queda
        'clock': np.concatenate([[0.0], item['clock'].numpy()]),
    }


def column_at_minute(clock: np.ndarray, minute: float) -> int:
    """Última columna cuyo evento ocurrió antes del minuto dado (0 = pre-partido)."""
    if minute <= 0:
        return 0
    valid = np.where(clock[1:] <= minute)[0]
    return int(valid[-1] + 1) if len(valid) else 0


def calibrate_temperature(args):
    """
    Ajusta la temperatura T del 1X2 minimizando el log-loss antes del partido
    en los partidos de validación (nunca en test) y la guarda en el checkpoint.
    T > 1 suaviza probabilidades demasiado extremas.
    """
    from scipy.optimize import minimize_scalar
    path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    model, a = load_model(path, args.device)
    dataset = FootballDataset(a['preset'], a['max_matches'], a['data_dir'],
                              convocatoria=a.get('convocatoria', False))
    _, val_ds, _ = split_datasets(dataset)
    P = np.array([predict_match(model, dataset, i, args.device)['p_outcome'][0]
                  for i in val_ds.indices])
    y = np.array([dataset.records[i]['outcome'] for i in val_ds.indices])
    log_loss = lambda T: -np.log(calibrate(P, T)[np.arange(len(y)), y]).mean()
    T = minimize_scalar(log_loss, bounds=(0.5, 5.0), method='bounded').x
    print(f"Temperatura: T = {T:.3f}  |  log-loss 1X2 pre-partido en validación "
          f"{log_loss(1.0):.4f} -> {log_loss(T):.4f}")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    ckpt['temperature'] = float(T)
    torch.save(ckpt, path)
    print(f"Guardada en {path}")


def demo_inference(args):
    """Evolución de las cuotas durante un partido real del conjunto de test."""
    model, a = load_model(os.path.join(args.checkpoint_dir, 'best_model.pt'), args.device)
    dataset = FootballDataset(a['preset'], a['max_matches'], a['data_dir'],
                              convocatoria=a.get('convocatoria', False))
    _, _, test_ds = split_datasets(dataset)
    i = test_ds.indices[0]
    m = dataset.meta[i]
    pred = predict_match(model, dataset, i, args.device)
    gen = OddsGenerator(alpha=0.94, margin=0.05, temperature=a['temperature'])

    print(f"\n=== {m['home_team']} vs {m['away_team']} ({m['match_date']}) — "
          f"final {m['home_score']}-{m['away_score']} ===")
    print(f"{'Min':>4} {'Marc.':>6} {'1':>6} {'X':>6} {'2':>6} {'Gol L':>6} {'Gol V':>6} "
          f"{'+2.5':>6} {'λ gol L':>8} {'λ gol V':>8}")
    for minute in [0, 15, 30, 45, 60, 75, 85]:
        c = column_at_minute(pred['clock'], minute)
        o = gen.generate(pred['p_outcome'][c], pred['lam_team'][:, c],
                         pred['current_team'][:, c])
        hg, ag = o['_score']
        print(f"{minute:>4} {hg:>3}-{ag:<2} {o['1X2_home']:>6} {o['1X2_draw']:>6} "
              f"{o['1X2_away']:>6} {o['next_goal_home']:>6} {o['next_goal_away']:>6} "
              f"{o['goals_ou_2.5_over']:>6} {pred['lam_team'][0, c, A['goals']]:>8.3f} "
              f"{pred['lam_team'][1, c, A['goals']]:>8.3f}")

    # Jugadores con más goles esperados antes del partido
    rec = dataset.records[i]
    lam_g = pred['lam_player'][:, 0, A['goals']]
    print("\nGoles esperados pre-partido (top 5 jugadores):")
    for k in np.argsort(-lam_g)[:5]:
        team = m['home_team'] if rec['side'][k] == 0 else m['away_team']
        real = int(pred['y_player'][k, 0, A['goals']])
        print(f"  jugador {rec['player_ids'][k]:>6} ({team}): λ = {lam_g[k]:.3f}  | marcó {real}")


# CLI
def parse_args():
    p = argparse.ArgumentParser(description='Entrena el Axial Transformer (Horton & Lucey 2025)')
    p.add_argument('--preset', default='top5_1516', choices=list(PRESETS))
    p.add_argument('--data_dir', default=DATA_DIR)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--max_matches', type=int, default=None)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=2,
                   help='2 cabe en ~2 GB de RAM; el paper usa 30 en GPU')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--d_ff', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--ablation', default='none', choices=ABLATIONS)
    p.add_argument('--w_outcome', type=float, default=10.0,
                   help='Peso de la pérdida 1X2 (todas las columnas)')
    p.add_argument('--w_pre', type=float, default=10.0,
                   help='Peso extra de la pérdida 1X2 antes del partido')
    p.add_argument('--sin_convocatoria', dest='convocatoria', action='store_false',
                   help='agentes = solo los jugadores que aparecen (modelo v2)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--checkpoint_dir', default=None,
                   help="Por defecto 'checkpoints' (o 'checkpoints_<ablación>')")
    p.add_argument('--demo', action='store_true')
    p.add_argument('--calibrar', action='store_true',
                   help='Ajusta la temperatura del 1X2 con los partidos de validación')
    args = p.parse_args()
    if args.checkpoint_dir is None:
        args.checkpoint_dir = ('checkpoints' if args.ablation == 'none'
                               else f'checkpoints_{args.ablation}')
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.demo:
        demo_inference(args)
    elif args.calibrar:
        calibrate_temperature(args)
    else:
        train(args)
