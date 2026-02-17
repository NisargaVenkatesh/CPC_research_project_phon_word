# --- stdlib ---
import os, time, math, argparse, json, random, statistics, sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- torch & tb ---
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast  # AMP

# safer multiprocessing sharing strategy (avoids FD/shm explosion)
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

def heartbeat(msg: str) -> None:
    print(msg, flush=True)

_THIS = Path(__file__).resolve()
_CODE_DIR = _THIS.parent  
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from src.torch_models._word_models import WordBuilder, WordCPCModel
from src.dataset.word_txt_dataset import WordTxtDataset
from src.dataset.word_txt_prosody_dataset import WordTxtProsodyDataset

try:
    import optuna
except Exception:
    optuna = None  


def set_seed(seed: int = 42):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cli_device: Optional[str] = None) -> torch.device:
    if cli_device:
        return torch.device(cli_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_words(batch) -> List[List[torch.Tensor]]:
    """Normalize dataset item formats to List[List[LongTensor]]."""
    if len(batch) == 0:
        raise RuntimeError("Empty batch")

    first = batch[0]
    if isinstance(first, dict):    # WordTxtProsodyDataset
        words_batch = [b["words"] for b in batch]
    else:                          # WordTxtDataset
        words_batch = batch

    # Light validation
    for utt in words_batch:
        for w in utt:
            if not torch.is_tensor(w):
                raise TypeError("Each word must be a LongTensor of phoneme IDs")
            if w.dtype != torch.long:
                raise TypeError("Phoneme ID tensors must be dtype torch.long")
    return words_batch


def build_dataloaders(args, id_map: Optional[Dict[str, int]]):
    root = Path(args.dataset_root)

    # pin_memory should key off the actual device in use
    pin_memory = (get_device(args.device).type == "cuda")

    if args.use_prosody:
        if WordTxtProsodyDataset is None:
            raise ImportError("word_txt_prosody_dataset.py not found/importable")
        ds_train = WordTxtProsodyDataset(root, id_map=id_map, max_files=args.max_files)
        ds_val   = WordTxtProsodyDataset(root, id_map=id_map, max_files=args.max_files_val)
    else:
        if WordTxtDataset is None:
            raise ImportError("word_txt_dataset.py not found/importable")
        ds_train = WordTxtDataset(
            data_dir=root,
            id_map=id_map,
            split_file=args.train_split,
            max_files=args.max_files
        )
        ds_val = None
        if getattr(args, "val_split", None):
            ds_val = WordTxtDataset(
                data_dir=root,
                id_map=id_map,
                split_file=args.val_split,
                max_files=args.max_files_val
            )

    # ---- DataLoaders ----
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    # safely set prefetch & persistence only if workers > 0
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = args.persistent_workers

    train_loader = DataLoader(
        ds_train, shuffle=True, collate_fn=collate_words, **loader_kwargs
    )

    if ds_val is None:
        # Fallback: use a small copy of train as "val" if none provided
        val_loader = DataLoader(
            ds_train, shuffle=False, collate_fn=collate_words, **loader_kwargs
        )
    else:
        val_loader = DataLoader(
            ds_val, shuffle=False, collate_fn=collate_words, **loader_kwargs
        )

    return train_loader, val_loader


def load_id_map(path: Optional[str]) -> Optional[Dict[str, int]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"id_map file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        m = json.load(f)
    if "<pad>" in m and m["<pad>"] != 0:
        raise ValueError("id_map: <pad> must map to index 0 for padding_idx=0")

    # sanity checks
    idxs = list(m.values())
    if len(set(idxs)) != len(idxs):
        raise ValueError("id_map has duplicate indices")
    if min(idxs) != 0:
        raise ValueError("id_map indices must start at 0 (pad)")
    if sorted(idxs) != list(range(max(idxs)+1)):
        raise ValueError("id_map indices must be contiguous 0..N-1")
    return m


def summarize_id_map(id_map: Dict[str,int], expect: Optional[int]) -> Dict[str, Any]:
    inv = {v:k for k,v in id_map.items()}
    size = max(id_map.values()) + 1
    report = {
        "vocab_size": size,
        "pad_token": inv.get(0, None),
        "has_pad_zero": True if inv.get(0, None) is not None else False,
        "min_index": min(id_map.values()),
        "max_index": max(id_map.values()),
        "contiguous": sorted(id_map.values()) == list(range(size)),
    }
    if expect is not None:
        report["matches_expectation"] = (size == expect)
    return report


# Train / Val steps with logging

# Throughput timer 
_throughput_t0 = None

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    log: SummaryWriter,
    epoch: int,
    log_every: int,
    grad_clip: float,
    args,  # to read --shuffle_word_order and --log_batches
) -> Dict[str, float]:
    global _throughput_t0

    model.train()
    losses: List[float] = []
    batch_times: List[float] = []
    grad_norms: List[float] = []

    num_batches = len(loader)

    for i, words_batch in enumerate(loader, start=1):
        t0 = time.time()

        # ---- optional baseline: shuffle word order within each utterance (TRAINING ONLY) ----
        if getattr(args, "shuffle_word_order", False):
            shuffled = []
            for utt in words_batch:
                order = list(range(len(utt)))
                random.shuffle(order)
                shuffled.append([utt[j] for j in order])
            words_batch = shuffled

        words_batch = [[w.to(device) for w in utt] for utt in words_batch]

        opt.zero_grad(set_to_none=True)
        if scaler is not None and device.type == "cuda":
            with autocast(device_type="cuda", enabled=True):
                loss = model(words_batch)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            else:
                total_norm = torch.norm(
                    torch.stack([p.grad.detach().data.norm(2)
                                 for p in model.parameters() if p.grad is not None]), 2
                )
            scaler.step(opt)
            scaler.update()
        else:
            loss = model(words_batch)
            loss.backward()
            if grad_clip > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            else:
                total_norm = torch.norm(
                    torch.stack([p.grad.detach().data.norm(2)
                                 for p in model.parameters() if p.grad is not None]), 2
                )
            opt.step()

        # --- Throughput probe (prints every --log_every steps if --log_batches true) ---
        if getattr(args, "log_batches_bool", False):
            if _throughput_t0 is None:
                _throughput_t0 = time.time()
            # global step (0-based) over entire training so far
            global_step = (epoch - 1) * num_batches + (i - 1)
            if ((global_step + 1) % log_every) == 0:
                dt = time.time() - _throughput_t0
                bps_items = (log_every * args.batch_size) / max(dt, 1e-6)
                bps_batches = bps_items / max(args.batch_size, 1)
                print(f"[throughput] step={global_step+1} "
                      f"batches/sec≈{bps_batches:.2f} items/sec≈{bps_items:.1f}",
                      flush=True)
                _throughput_t0 = time.time()
        

        dt = time.time() - t0
        batch_times.append(dt)
        grad_norms.append(float(total_norm))
        losses.append(float(loss.item()))

        if (i % log_every) == 0 or i == 1:
            step = epoch * len(loader) + i
            log.add_scalar("train/loss", float(loss.item()), step)
            log.add_scalar("train/grad_norm", float(total_norm), step)
            log.add_scalar("train/batch_time_sec", dt, step)
            heartbeat(f"[train] epoch {epoch} batch {i}/{len(loader)} loss={float(loss.item()):.4f}")

    epoch_stats = {
        "loss_avg": float(sum(losses) / max(1, len(losses))),
        "loss_median": float(statistics.median(losses)) if losses else 0.0,
        "batch_time_median": float(statistics.median(batch_times)) if batch_times else 0.0,
        "grad_norm_median": float(statistics.median(grad_norms)) if grad_norms else 0.0,
    }

    print(
        f"Epoch {epoch} Train Avg: {epoch_stats['loss_avg']:.4f} | "
        f"Median batch {epoch_stats['batch_time_median']:.3f}s | "
        f"Median grad-norm {epoch_stats['grad_norm_median']:.3f}",
        flush=True
    )
    log.add_scalar("train/epoch_avg_loss", epoch_stats["loss_avg"], epoch)
    log.add_scalar("train/epoch_median_batch_time", epoch_stats["batch_time_median"], epoch)
    log.add_scalar("train/epoch_median_grad_norm", epoch_stats["grad_norm_median"], epoch)
    return epoch_stats


def validate(model: nn.Module, loader: DataLoader, device: torch.device, log: SummaryWriter, epoch: int) -> float:
    model.eval()
    total = 0.0
    steps = 0
    t0 = time.time()

    with torch.no_grad():
        for words_batch in loader:
            words_batch = [[w.to(device) for w in utt] for utt in words_batch]
            loss = model(words_batch)
            total += float(loss.item()); steps += 1

    dt = time.time() - t0
    avg = total / max(1, steps)
    print(f"Epoch {epoch} Val   Avg: {avg:.4f} ({dt:.1f}s)", flush=True)
    log.add_scalar("val/epoch_avg_loss", avg, epoch)
    return avg


def parse_args():
    p = argparse.ArgumentParser(description="Train Word-level CPC (masked InfoNCE, AMP, Optuna)")

    # Data
    p.add_argument("--dataset_root", type=str, required=True, help="Dataset root folder")
    p.add_argument("--train_split", type=str, default="Dataset/splits/train.txt")
    p.add_argument("--val_split",   type=str, default="Dataset/splits/val.txt")
    p.add_argument("--id_map_json", type=str, default=None, help="JSON mapping phoneme->id. Requires <pad>:0")
    p.add_argument("--use_prosody", action="store_true", help="Use WordTxtProsodyDataset (expects 'words' key)")
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--max_files_val", type=int, default=None)
    p.add_argument("--expect_vocab_size", type=int, default=66, help="Sanity check for phoneme vocab")

    # Model
    p.add_argument("--vocab_size", type=int, default=66)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--cnn_channels", type=int, default=128)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--lstm_hidden", type=int, default=128)
    p.add_argument("--lstm_layers", type=int, default=1)
    p.add_argument("--wb_dropout", type=float, default=0.1)
    p.add_argument("--wb_pooling", type=str, default="mean", choices=["mean","last"])
    p.add_argument("--use_cnn", action="store_true")

    p.add_argument("--context_hidden", type=int, default=256)
    p.add_argument("--context_layers", type=int, default=1)
    p.add_argument("--prediction_steps", type=int, default=3)
    p.add_argument("--across_batch_negs", action="store_true")

    # Optim
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # safer dataloader knobs
    p.add_argument("--prefetch_factor", type=int, default=1,
                   help="Prefetch per-worker (only if num_workers>0)")
    p.add_argument("--persistent_workers", action="store_true",
                   help="Keep workers alive between epochs (only if num_workers>0)")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, choices=["cpu","cuda","mps"])
    p.add_argument("--run_tag", type=str, default="")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--shuffle_word_order", action="store_true",
                   help="Shuffle word order within each utterance during TRAINING only (baseline)")

    # throughput probe toggles
    p.add_argument("--log_batches", type=str, default="false",
                   help="Print batches/sec during training (true/false)")
    p.add_argument("--log_every_batches", type=int, default=50,
                   help="Alias for --log_every when used with --log_batches=true (kept for compatibility)")

    # Early stopping
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="Patience (in epochs) for early stopping; 0 disables.")

    # Optuna
    p.add_argument("--optuna", action="store_true", help="Run hyperparameter optimization")
    p.add_argument("--study_name", type=str, default="wordcpc_study")
    p.add_argument("--storage", type=str, default="sqlite:///logs/optuna_wordcpc.db")
    p.add_argument("--n_trials", type=int, default=20)
    p.add_argument("--epochs_per_trial", type=int, default=8)

    # Allow locking use_cnn across trials
    p.add_argument(
        "--fix_use_cnn",
        type=lambda x: str(x).lower() in ("true", "1", "yes", "y"),
        default=None,
        help="If set, fixes use_cnn=True/False for all Optuna trials; otherwise use_cnn is tuned."
    )

    return p.parse_args()


def build_model_from_args(args) -> nn.Module:
    wb = WordBuilder(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        cnn_channels=args.cnn_channels,
        kernel_size=args.kernel_size,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.wb_dropout,
        use_cnn=args.use_cnn,
        pooling=args.wb_pooling,
    )
    model = WordCPCModel(
        word_builder=wb,
        context_hidden=args.context_hidden,
        context_layers=args.context_layers,
        prediction_steps=args.prediction_steps,
        use_across_batch_negatives=args.across_batch_negs,
    )
    return model


def single_run(args):
    args.log_batches_bool = str(getattr(args, "log_batches", "false")).lower() in ("1", "true", "yes")

    if hasattr(args, "log_every_batches") and args.log_every_batches != 50 and args.log_every == 50:
        args.log_every = args.log_every_batches

    set_seed(args.seed)
    device = get_device(args.device)
    heartbeat(f"[INFO] Device: {device}")

    # id_map / vocab checks
    id_map = load_id_map(args.id_map_json)
    if id_map is not None:
        vocab_size = max(id_map.values()) + 1
        if args.expect_vocab_size is not None and vocab_size != args.expect_vocab_size:
            heartbeat(f"[WARN] id_map vocab_size={vocab_size} differs from expected {args.expect_vocab_size}")
        args.vocab_size = vocab_size
        summary = summarize_id_map(id_map, args.expect_vocab_size)
        heartbeat(f"[INFO] id_map summary: {summary}")

    # data
    train_loader, val_loader = build_dataloaders(args, id_map)
    heartbeat(f"[INFO] train batches={len(train_loader)} val batches={len(val_loader)}")

    # model
    model = build_model_from_args(args).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    # logging
    ROOT = Path("logs"); ROOT.mkdir(exist_ok=True, parents=True)
    run_name = f"wordcpc_{time.strftime('%Y%m%d_%H%M%S')}{('_'+args.run_tag) if args.run_tag else ''}"
    writer = SummaryWriter(ROOT / run_name)

    best_val = math.inf
    best_path = Path("checkpoints") / f"{run_name}_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    # Early stopping state
    patience = args.early_stop_patience
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, opt, scaler, device, writer, epoch, args.log_every, args.grad_clip, args)
        val_loss = validate(model, val_loader, device, writer, epoch)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "best_val": best_val,
                "id_map": id_map,
            }, best_path)
            heartbeat(f"[INFO] Saved best checkpoint to {best_path} (val={best_val:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if patience and no_improve >= patience:
                heartbeat(f"[STOP] Early stopping at epoch {epoch}; no improvement for {no_improve} epochs.")
                break

    wb_only = Path("checkpoints") / f"{run_name}_wordbuilder.pt"
    torch.save(model.word_builder.state_dict(), wb_only)
    heartbeat(f"[INFO] Saved WordBuilder to {wb_only}")

    writer.flush(); writer.close()


# ---------- Optuna ----------

def suggest_hparams(trial: "optuna.trial.Trial", args) -> Dict[str, Any]:
    space = {}
    space["embed_dim"]        = trial.suggest_categorical("embed_dim", [96, 128, 192])
    space["cnn_channels"]     = trial.suggest_categorical("cnn_channels", [64, 128, 192])
    space["kernel_size"]      = trial.suggest_categorical("kernel_size", [3, 5, 7])
    space["lstm_hidden"]      = trial.suggest_categorical("lstm_hidden", [128, 192, 256])
    space["lstm_layers"]      = trial.suggest_categorical("lstm_layers", [1, 2])
    space["wb_dropout"]       = trial.suggest_float("wb_dropout", 0.0, 0.3)
    space["context_hidden"]   = trial.suggest_categorical("context_hidden", [256, 320, 384])
    space["prediction_steps"] = trial.suggest_categorical("prediction_steps", [2, 4, 6])

    # use_cnn: tune unless fixed by --fix_use_cnn
    if args.fix_use_cnn is not None:
        space["use_cnn"] = bool(args.fix_use_cnn)
    else:
        space["use_cnn"] = trial.suggest_categorical("use_cnn", [True, False])

    space["lr"]               = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
    space["weight_decay"]     = trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True)
    return space


def objective_builder(global_args):
    def objective(trial: "optuna.trial.Trial") -> float:
        args = argparse.Namespace(**vars(global_args))
        # apply suggested params
        hp = suggest_hparams(trial, args)
        for k, v in hp.items():
            setattr(args, k, v)

        set_seed(args.seed)
        device = get_device(args.device)

        id_map = load_id_map(args.id_map_json)
        if id_map is not None:
            args.vocab_size = max(id_map.values()) + 1

        train_loader, val_loader = build_dataloaders(args, id_map)
        heartbeat(f"[trial {trial.number}] start | batches: train={len(train_loader)} val={len(val_loader)} | hp={hp}")

        model = build_model_from_args(args).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = GradScaler(enabled=(device.type == "cuda"))

        # light logging per trial
        tb_dir = Path("logs") / f"optuna_{global_args.study_name}" / f"trial_{trial.number}"
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tb_dir)

        best_val = math.inf
        patience = args.early_stop_patience  # you can pass a small value here if desired
        no_improve = 0

        for epoch in range(1, args.epochs_per_trial + 1):
            train_one_epoch(model, train_loader, opt, scaler, device, writer, epoch, args.log_every, args.grad_clip, args)
            val_loss = validate(model, val_loader, device, writer, epoch)
            trial.report(val_loss, epoch)
            best_val = min(best_val, val_loss)
            heartbeat(f"[trial {trial.number}] epoch {epoch} val={val_loss:.4f} best={best_val:.4f}")

            # pruning
            if trial.should_prune():
                writer.close()
                heartbeat(f"[trial {trial.number}] PRUNED at epoch {epoch}")
                raise optuna.TrialPruned()

            # early stopping inside a trial
            if patience:
                if val_loss > best_val:
                    no_improve += 1
                    if no_improve >= patience:
                        heartbeat(f"[trial {trial.number}] Early stop at epoch {epoch}; no improvement for {no_improve}.")
                        break
                else:
                    no_improve = 0

        writer.close()
        heartbeat(f"[trial {trial.number}] done | best_val={best_val:.4f}")
        return best_val
    return objective


def run_optuna(args):
    if optuna is None:
        raise ImportError("Optuna is not installed. Please pip install optuna.")

    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )
    heartbeat(f"[INFO] Optuna study: {args.study_name} storage={args.storage}")

    objective = objective_builder(args)
    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)

    print("[INFO] Best trial:", flush=True)
    print(f"  value = {study.best_value}", flush=True)
    print(f"  params = {study.best_params}", flush=True)

    # Save best params
    out = Path("logs") / f"{args.study_name}_best_params.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)
    heartbeat(f"[INFO] Saved best params to {out}")


def main():
    args = parse_args()

    if args.optuna:
        run_optuna(args)
    else:
        single_run(args)


if __name__ == "__main__":
    main()
