"""
使用基于 Transformer 的时序回归模型替换传统树模型，实现船舶燃油效率预测。

核心流程：
1. 读取并清洗 `data/mingxi_0618_0715_with_anomaly.csv`。
2. 将特征序列化为固定窗口，交由 Transformer Encoder 进行建模。
3. 输出训练与测试集的 RMSE / MAE / MAPE / R² 等指标。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

import Distinguish


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

    # 模型相关超参
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.2
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CONFIG = TransformerConfig()


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


def train_transformer_regressor() -> None:
    """主训练入口。"""
    csv_path = "data/mingxi_0618_0715_with_anomaly.csv"
    df = load_and_prepare_dataframe(csv_path)

    feature_cols: List[str] = [
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
    target_col = "MESFOC_nmile"

    if target_col not in df.columns:
        raise ValueError(f"缺少目标字段: {target_col}")

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        warnings.warn(f"以下特征缺失，将自动从训练中移除: {missing_features}")
        feature_cols = [col for col in feature_cols if col in df.columns]

    if not feature_cols:
        raise ValueError("所有候选特征均缺失，无法训练模型。")

    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols])
    target = df[target_col].values.astype(np.float32)

    sequences, labels = build_sequences(
        features,
        target,
        seq_len=CONFIG.seq_len,
        horizon=CONFIG.prediction_horizon,
    )

    split_idx = int(len(sequences) * 0.8)
    train_ds = SequenceDataset(sequences[:split_idx], labels[:split_idx])
    test_ds = SequenceDataset(sequences[split_idx:], labels[split_idx:])

    train_loader = DataLoader(train_ds, batch_size=CONFIG.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=CONFIG.batch_size, shuffle=False, drop_last=False)

    model = FuelTransformerRegressor(input_dim=len(feature_cols), cfg=CONFIG).to(CONFIG.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, CONFIG.num_epochs - 5)
    )

    best_test_rmse = float("inf")
    best_state = None

    for epoch in range(1, CONFIG.num_epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            CONFIG.device,
            CONFIG.grad_clip,
        )
        test_loss, y_pred, y_true = evaluate_model(model, test_loader, criterion, CONFIG.device)
        scheduler.step()

        metrics = regression_metrics(y_true, y_pred)
        if metrics["rmse"] < best_test_rmse:
            best_test_rmse = metrics["rmse"]
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "metrics": metrics,
            }

        print(
            f"[Epoch {epoch:02d}] "
            f"TrainLoss={train_loss:.4f} | TestLoss={test_loss:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | R2={metrics['r2']:.4f}"
        )

    if best_state is None:
        raise RuntimeError("训练过程中未能记录有效结果。")

    model.load_state_dict(best_state["model_state"])
    print("\n" + "=" * 70)
    print("Transformer 模型最终评估")
    print("=" * 70)

    _, test_preds, test_targets = evaluate_model(model, test_loader, criterion, CONFIG.device)
    final_metrics = regression_metrics(test_targets, test_preds)
    for key, value in final_metrics.items():
        print(f"{key.upper():>4}: {value:.4f}")


if __name__ == "__main__":
    train_transformer_regressor()
