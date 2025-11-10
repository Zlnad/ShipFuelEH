import pandas as pd


# pandas取消限制
# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)
# pd.set_option('display.width',None)
# pd.set_option('display.max_colwidth',None)


def disHardData(file_path):

    df = pd.read_csv(file_path)
    hardData = df[df['Is_Anomaly'] == True]
    return hardData

def disEasyData(file_path):

    df = pd.read_csv(file_path)
    easyData = df[df['Is_Anomaly'] == False]
    return easyData



# result = distinguishH('data/mingxi_0618_0715_with_anomaly.csv')
# print(result)


# resultE = disEasyDatas('data/mingxi_0618_0715_with_anomaly.csv')
# print(resultE)