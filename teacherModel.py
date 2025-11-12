import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import Distinguish


file_path = "data/mingxi_0618_0715_with_anomaly.csv"
hardDatas = Distinguish.disHardData(file_path)

df = hardDatas

df['PCTime'] = pd.to_datetime(df['PCTime'])
df['hour'] = df['PCTime'].dt.hour
df['minute'] = df['PCTime'].dt.minute

print(f"数据形状: {df.shape}")
print(f"数据列: {df.columns.tolist()}")

def create_ft_transformer(num_features, embed_dim=128, num_heads=8, num_layers=4, ff_dim=256, dropout_rate=0.1):
    """
    创建FT-Transformer模型（Feature Tokenizer + Transformer）
    这是目前表格数据回归任务中最先进的方法之一
    """
    inputs = keras.Input(shape=(num_features,))
    
    # Feature Tokenization: 为每个特征创建独立的embedding层
    # 将每个特征值投影到embedding空间
    feature_embeddings = []
    for i in range(num_features):
        # 提取单个特征
        feature_slice = layers.Lambda(lambda x, idx=i: x[:, idx:idx+1])(inputs)
        # 为每个特征创建独立的线性投影
        feat_emb = layers.Dense(embed_dim, name=f'feature_emb_{i}')(feature_slice)
        feat_emb = layers.LayerNormalization(name=f'ln_{i}')(feat_emb)
        feature_embeddings.append(feat_emb)
    
    # 堆叠所有特征embeddings形成序列 (batch, num_features, embed_dim)
    x = layers.Concatenate(axis=1, name='concat_features')(feature_embeddings)
    
    # 添加位置编码
    position_embedding_layer = layers.Embedding(
        input_dim=num_features,
        output_dim=embed_dim,
        name='position_embedding'
    )
    # 使用Lambda层动态创建位置编码
    def add_position_encoding(features):
        batch_size = tf.shape(features)[0]
        positions = tf.range(start=0, limit=num_features, delta=1)
        position_embeddings = position_embedding_layer(positions)
        # 扩展维度以匹配batch size: (num_features, embed_dim) -> (1, num_features, embed_dim) -> (batch, num_features, embed_dim)
        position_embeddings = tf.expand_dims(position_embeddings, axis=0)
        position_embeddings = tf.tile(position_embeddings, [batch_size, 1, 1])
        return features + position_embeddings
    
    x = layers.Lambda(add_position_encoding, name='add_position')(x)
    
    # Transformer编码器层
    for i in range(num_layers):
        # Multi-Head Self-Attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
            name=f'mha_{i}'
        )(x, x)
        attn_output = layers.Dropout(dropout_rate, name=f'attn_dropout_{i}')(attn_output)
        x = layers.Add(name=f'attn_add_{i}')([x, attn_output])  # 残差连接
        x = layers.LayerNormalization(name=f'attn_ln_{i}')(x)
        
        # Feed Forward Network
        ffn_output = layers.Dense(ff_dim, activation='gelu', name=f'ffn_dense1_{i}')(x)
        ffn_output = layers.Dropout(dropout_rate, name=f'ffn_dropout1_{i}')(ffn_output)
        ffn_output = layers.Dense(embed_dim, name=f'ffn_dense2_{i}')(ffn_output)
        ffn_output = layers.Dropout(dropout_rate, name=f'ffn_dropout2_{i}')(ffn_output)
        x = layers.Add(name=f'ffn_add_{i}')([x, ffn_output])  # 残差连接
        x = layers.LayerNormalization(name=f'ffn_ln_{i}')(x)
    
    # 全局池化（使用所有特征的聚合）
    x = layers.GlobalAveragePooling1D(name='global_pool')(x)
    
    # 输出层
    x = layers.Dense(ff_dim, activation='gelu', name='output_dense1')(x)
    x = layers.Dropout(dropout_rate, name='output_dropout1')(x)
    x = layers.Dense(ff_dim // 2, activation='gelu', name='output_dense2')(x)
    x = layers.Dropout(dropout_rate * 0.5, name='output_dropout2')(x)
    outputs = layers.Dense(1, name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_resnet_like_model(num_features, embed_dim=256):
    """
    创建ResNet-like架构的回归模型（备选方案）
    使用残差连接和先进的激活函数
    """
    inputs = keras.Input(shape=(num_features,))
    
    # 初始投影
    x = layers.Dense(embed_dim)(inputs)
    x = layers.LayerNormalization()(x)
    
    # ResNet块
    def resnet_block(x, dim, dropout_rate=0.2):
        residual = x
        x = layers.Dense(dim)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('gelu')(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(dim)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        # 如果维度不匹配，使用投影
        if residual.shape[-1] != dim:
            residual = layers.Dense(dim)(residual)
        x = layers.Add()([x, residual])
        x = layers.Activation('gelu')(x)
        return x
    
    # 多个ResNet块
    x = resnet_block(x, embed_dim, 0.2)
    x = resnet_block(x, embed_dim, 0.2)
    x = resnet_block(x, embed_dim // 2, 0.15)
    x = resnet_block(x, embed_dim // 2, 0.15)
    x = resnet_block(x, embed_dim // 4, 0.1)
    
    # 输出层
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def predict_fuel_efficiency():
    #按小时进行时序预测

    features = ['MERpm', 'METorque', 'MEShaftPow', 'ShipSpdToWater',
                'WindSpd', 'WindDir', 'ShipDraughtBow', 'hour']

    X = df[features]
    y = df['MESFOC_nmile']  # 每海里的燃油消耗

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 使用FT-Transformer架构（目前表格数据回归的最佳方法）
    print("构建FT-Transformer模型...")
    model = create_ft_transformer(
        num_features=len(features),
        embed_dim=128,
        num_heads=8,
        num_layers=4,
        ff_dim=256,
        dropout_rate=0.1
    )
    
    # 如果FT-Transformer太复杂，可以使用ResNet-like模型
    # model = create_resnet_like_model(num_features=len(features), embed_dim=256)
    
    # 编译模型 - 使用AdamW优化器（比Adam更先进）
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.01
        ),
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    # 学习率调度器
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        elif epoch < 50:
            return lr * 0.95
        else:
            return lr * 0.9
    
    # 设置回调函数
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        min_delta=1e-6
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-7,
        verbose=1
    )
    
    lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
    
    # 模型检查点
    checkpoint = keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
    
    # 训练模型
    print("开始训练模型...")
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=300,
        batch_size=64,
        verbose=1,
        callbacks=[early_stopping, reduce_lr, lr_scheduler, checkpoint]
    )
    
    # 预测
    y_pred = model.predict(X_test_scaled, verbose=0).flatten()

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100

    print(f"\n老师模型燃油效率预测结果（FT-Transformer）:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # 打印模型摘要
    print(f"\n模型参数数量: {model.count_params():,}")

    return model, scaler, features

predict_fuel_efficiency()