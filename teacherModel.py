import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
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
    # model = xgb.XGBRegressor(
    #     n_estimators=100,
    #     max_depth=6,
    #     learning_rate=0.1,
    #     random_state=42
    # )
    #网格搜索调参
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"老师模型燃油效率预测结果:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")



    return model, scaler, features

predict_fuel_efficiency()