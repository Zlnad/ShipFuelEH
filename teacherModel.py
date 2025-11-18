"""
使用基于 Transformer 的时序回归模型替换传统树模型，实现船舶燃油效率预测。

核心流程：
1. 读取并清洗 `data/mingxi_0618_0715_with_anomaly.csv`。
2. 将特征序列化为固定窗口，交由 Transformer Encoder 进行建模。
3. 输出训练与测试集的 RMSE / MAE / MAPE / R² 等指标。
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

import Distinguish

try:
    import optuna  # type: ignore

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None  # type: ignore


DEFAULT_RANDOM_SEED = 42
AUTO_TUNE_TRIALS = 10


def set_random_seed(seed: int = DEFAULT_RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================
# 全局数据配置
# ===========================
DATA_FILE_PATH = "data/mingxi_0618_0715_with_anomaly.csv"
TARGET_COLUMN = "MESFOC_nmile"
CANDIDATE_FEATURES = [
    "MERpm",
    "METorque",
    "MEShaftPow",
    "ShipSpdToWater",
    "WindSpd",
    "WindDir",
    "ShipDraughtBow",
    "ShipDraughtStern",
    "hour",
    "minute",
    "dayofweek",
]
TRAIN_SPLIT_RATIO = 0.8


# ===========================
# 配置
# ===========================

@dataclass
class TransformerConfig:
    # 数据相关超参
    seq_len: int = 48
    prediction_horizon: int = 1
    batch_size: int = 256


    # 训练相关超参
    num_epochs: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    seed: int = DEFAULT_RANDOM_SEED

    # 模型相关超参
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.2
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CONFIG = TransformerConfig()


def describe_config(cfg: TransformerConfig) -> str:
    parts = [
        f"seq_len={cfg.seq_len}",
        f"batch_size={cfg.batch_size}",
        f"num_layers={cfg.num_layers}",
        f"d_model={cfg.d_model}",
        f"nhead={cfg.nhead}",
        f"dropout={cfg.dropout}",
        f"lr={cfg.learning_rate}",
        f"epochs={cfg.num_epochs}",
    ]
    return ", ".join(parts)


# ===========================
# 数据处理
# ===========================

def load_and_prepare_dataframe(csv_path: str) -> pd.DataFrame:
    """载入清洗后的困难样本并补充时间特征。"""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"未找到数据文件: {csv_path}")

    hard_df = Distinguish.disHardData(csv_path)
    df = hard_df.copy()

    df["PCTime"] = pd.to_datetime(df["PCTime"])
    df = df.sort_values("PCTime").reset_index(drop=True)
    df["hour"] = df["PCTime"].dt.hour
    df["minute"] = df["PCTime"].dt.minute
    df["dayofweek"] = df["PCTime"].dt.dayofweek
    return df


def build_sequences(
    features: np.ndarray,
    target: np.ndarray,
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """滑动窗口构建 Transformer 所需的序列样本。"""
    X, y = [], []
    total_steps = len(features)
    last_start = total_steps - seq_len - horizon + 1
    if last_start <= 0:
        raise ValueError("样本数量不足以支撑指定的序列长度，请增大数据量或缩短 seq_len。")

    for start in range(last_start):
        end = start + seq_len
        target_idx = end + horizon - 1
        X.append(features[start:end])
        y.append(target[target_idx])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class SequenceDataset(Dataset):
    """简单的 TensorDataset 封装器。"""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.features = torch.from_numpy(sequences)
        self.targets = torch.from_numpy(targets).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]


def resolve_feature_columns(
    df: pd.DataFrame, candidate_features: List[str]
) -> Tuple[List[str], List[str]]:
    """根据可用列过滤特征，返回(可用, 缺失)。"""
    available = [col for col in candidate_features if col in df.columns]
    missing = [col for col in candidate_features if col not in df.columns]
    if missing:
        warnings.warn(f"以下特征缺失，将自动跳过: {missing}")
    if not available:
        raise ValueError("所有候选特征均缺失，无法训练模型。")
    return available, missing


def build_dataloaders_from_dataframe(
    df: pd.DataFrame,
    candidate_features: List[str],
    target_col: str,
    cfg: TransformerConfig,
) -> Tuple[DataLoader, DataLoader, StandardScaler, List[str]]:
    """将数据框转换为训练/测试 DataLoader。"""
    if target_col not in df.columns:
        raise ValueError(f"缺少目标字段: {target_col}")

    available_features, _ = resolve_feature_columns(df, candidate_features)
    scaler = StandardScaler()
    features = scaler.fit_transform(df[available_features])
    target = df[target_col].values.astype(np.float32)

    sequences, labels = build_sequences(
        features,
        target,
        seq_len=cfg.seq_len,
        horizon=cfg.prediction_horizon,
    )
    if len(sequences) < 2:
        raise ValueError("构建的序列样本量不足，无法训练。")

    split_idx = int(len(sequences) * TRAIN_SPLIT_RATIO)
    split_idx = min(max(split_idx, 1), len(sequences) - 1)

    train_ds = SequenceDataset(sequences[:split_idx], labels[:split_idx])
    test_ds = SequenceDataset(sequences[split_idx:], labels[split_idx:])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, test_loader, scaler, available_features


# ===========================
# 模型定义
# ===========================

class PositionalEncoding(nn.Module):
    """标准正弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FuelTransformerRegressor(nn.Module):
    """针对多变量时序回归的 Transformer Encoder。"""

    def __init__(self, input_dim: int, cfg: TransformerConfig):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, cfg.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.pos_encoder = PositionalEncoding(cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.reg_head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        pooled = self.norm(pooled)
        return self.reg_head(pooled).squeeze(-1)


# ===========================
# 训练 & 评估
# ===========================

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.squeeze(-1).to(device)
        preds = model(features)
        loss = criterion(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(features)
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    preds_list, targets_list = [], []
    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.squeeze(-1).to(device)
        preds = model(features)
        loss = criterion(preds, targets)
        total_loss += loss.item() * len(features)
        preds_list.append(preds.cpu().numpy())
        targets_list.append(targets.cpu().numpy())
    y_pred = np.concatenate(preds_list)
    y_true = np.concatenate(targets_list)
    return total_loss / len(dataloader.dataset), y_pred, y_true


def train_transformer_regressor(
    config: TransformerConfig,
    df: pd.DataFrame,
    candidate_features: Optional[List[str]] = None,
    target_col: str = TARGET_COLUMN,
    verbose: bool = True,
) -> Dict[str, object]:
    """根据指定配置训练 Transformer 并返回指标及训练Artifacts。"""
    candidate_features = candidate_features or CANDIDATE_FEATURES
    set_random_seed(config.seed)

    (
        train_loader,
        test_loader,
        scaler,
        used_features,
    ) = build_dataloaders_from_dataframe(df, candidate_features, target_col, config)

    model = FuelTransformerRegressor(input_dim=len(used_features), cfg=config).to(config.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.num_epochs - 5)
    )

    best_rmse = float("inf")
    best_state: Optional[Dict[str, object]] = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.num_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config.device,
            config.grad_clip,
        )
        test_loss, y_pred, y_true = evaluate_model(model, test_loader, criterion, config.device)
        scheduler.step()

        metrics = regression_metrics(y_true, y_pred)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                **metrics,
            }
        )

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "metrics": metrics,
            }

        if verbose:
            print(
                f"[Epoch {epoch:02d}] "
                f"TrainLoss={train_loss:.4f} | TestLoss={test_loss:.4f} | "
                f"RMSE={metrics['rmse']:.4f} | R2={metrics['r2']:.4f}"
            )

    if best_state is None:
        raise RuntimeError("训练过程中未能记录有效结果。")

    model.load_state_dict(best_state["model_state"])  # type: ignore[arg-type]
    _, best_preds, best_targets = evaluate_model(model, test_loader, criterion, config.device)
    final_metrics = regression_metrics(best_targets, best_preds)

    if verbose:
        print("\n" + "=" * 70)
        print("Transformer 模型最终评估")
        print("=" * 70)
        print(f"最佳 Epoch: {best_state['epoch']} | 配置: {describe_config(config)}")
        for key, value in final_metrics.items():
            print(f"{key.upper():>4}: {value:.4f}")

    return {
        "model": model,
        "metrics": final_metrics,
        "history": history,
        "best_epoch": best_state["epoch"],
        "feature_cols": used_features,
        "scaler": scaler,
        "config": config,
    }


def sample_config_from_trial(trial: "optuna.Trial", base_config: TransformerConfig) -> TransformerConfig:
    """根据Optuna Trial对TransformerConfig进行采样。"""
    cfg = replace(base_config)
    cfg.seq_len = trial.suggest_int("seq_len", 24, 72, step=12)
    cfg.batch_size = trial.suggest_categorical("batch_size", [64, 96, 128, 192])
    cfg.num_epochs = trial.suggest_int("num_epochs", 30, 80, step=10)
    cfg.learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    cfg.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    cfg.d_model = trial.suggest_categorical("d_model", [64, 96, 128, 160, 192])
    cfg.nhead = trial.suggest_categorical("nhead", [4, 8])
    if cfg.d_model % cfg.nhead != 0:
        cfg.nhead = 4  # Fallback，保证可整除
    cfg.num_layers = trial.suggest_int("num_layers", 2, 5)
    cfg.dim_feedforward = trial.suggest_categorical(
        "dim_feedforward", [256, 384, 512, 640, 768]
    )
    cfg.dropout = trial.suggest_float("dropout", 0.05, 0.35, step=0.05)
    cfg.grad_clip = trial.suggest_categorical("grad_clip", [0.5, 1.0])
    return cfg


def run_hyperparameter_search(
    df: pd.DataFrame,
    candidate_features: List[str],
    target_col: str,
    base_config: TransformerConfig = CONFIG,
    n_trials: int = AUTO_TUNE_TRIALS,
) -> TransformerConfig:
    """利用 Optuna 进行自动调参，返回最优配置。"""
    if not OPTUNA_AVAILABLE:
        print("⚠️ 未检测到 optuna，跳过自动调参，直接使用默认配置。")
        return base_config

    print(f"启动 Optuna 自动调参，共 {n_trials} 次 trial ...")

    def objective(trial: "optuna.Trial") -> float:
        cfg = sample_config_from_trial(trial, base_config)
        try:
            result = train_transformer_regressor(
                cfg,
                df,
                candidate_features=candidate_features,
                target_col=target_col,
                verbose=False,
            )
        except ValueError as exc:
            raise optuna.TrialPruned(str(exc))

        metrics = result["metrics"]
        trial.set_user_attr("config_dict", asdict(cfg))
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("best_epoch", result["best_epoch"])
        return metrics["rmse"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_trial = study.best_trial
    best_config_dict = best_trial.user_attrs["config_dict"]
    tuned_config = replace(base_config, **best_config_dict)

    print(
        f"自动调参完成: Trial #{best_trial.number} | RMSE={best_trial.value:.4f} | "
        f"R2={best_trial.user_attrs['metrics']['r2']:.4f}"
    )
    print(f"最优配置: {describe_config(tuned_config)}")
    return tuned_config


def run_full_pipeline(auto_tune_trials: int = AUTO_TUNE_TRIALS) -> Dict[str, object]:
    """整体流程：加载数据 -> 自动调参 -> 最终训练与评估。"""
    df = load_and_prepare_dataframe(DATA_FILE_PATH)
    candidate_features = CANDIDATE_FEATURES.copy()

    if auto_tune_trials > 0:
        tuned_config = run_hyperparameter_search(
            df,
            candidate_features,
            TARGET_COLUMN,
            base_config=CONFIG,
            n_trials=auto_tune_trials,
        )
    else:
        tuned_config = CONFIG
        print("跳过自动调参，直接使用默认配置。")

    print("\n开始使用最优配置进行最终训练...")
    return train_transformer_regressor(
        tuned_config,
        df,
        candidate_features=candidate_features,
        target_col=TARGET_COLUMN,
        verbose=True,
    )


if __name__ == "__main__":
    run_full_pipeline(auto_tune_trials=AUTO_TUNE_TRIALS)
