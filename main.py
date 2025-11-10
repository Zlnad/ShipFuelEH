import Distinguish
import teacherModel

if __name__ == "__main__":
    dataSource = "data/mingxi_0618_0715_with_anomaly.csv"

    easyDatas = Distinguish.disEasyData(dataSource)

    hardDatas = Distinguish.disHardData(dataSource)

    # print(hardDatas)
    #
    # print("---------------------------------------")
    #
    # print(easyDatas)
#教师模型提取时序特征
    # teacherModel.dataprocessing(hardDatas)