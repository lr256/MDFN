import csv
import random
import numpy as np
random.seed(2223)
from openpyxl import load_workbook as lw#读取xlsx
import cv2
import six

class Folds:
    def __init__(self, benignPatients, malignantPatients, numOfFold, dataPath, patients,benignPatients1,benignPatients2,malignantPatients1):
        self.BenignPatients = benignPatients
        self.MalignantPatients = malignantPatients
        self.MalignantPatients1 = malignantPatients1
        self.BenignPatients1 = benignPatients1
        self.BenignPatients2 = benignPatients2
        self.NumOfFold = numOfFold
        self.DataPath = dataPath
        self.Ratio = 0.11
        self.patients = patients

    def LoadSetData(self, benignStart, benignEnd, malignantStart, malignantEnd, dataPercentage = 1.0):
        setDataIndexList = []
        setLabelList = []
        def LoadBenignOrMalignant(start, end, ps):
            numOfPatients = int((end - start) * dataPercentage)
            for i in range(start, start + numOfPatients):
                setDataIndexList.append(ps[i][0])#把index和label区分
                setLabelList.append(ps[i][1:])
        LoadBenignOrMalignant(int(benignStart), int(benignEnd), self.BenignPatients)
        LoadBenignOrMalignant(int(malignantStart), int(malignantEnd), self.MalignantPatients)

        return np.array(setDataIndexList), np.array(setLabelList)
 
    def LoadSetData_5(self, benignStart, benignEnd, malignantStart, malignantEnd, malignantStart1, malignantEnd1,benignStart1, benignEnd1,
                      benignStart2, benignEnd2, dataPercentage=1.0):
        setDataIndexList = []
        setLabelList = []

        def LoadBenignOrMalignant(start, end, ps):
            numOfPatients = int((end - start) * dataPercentage)
            for i in range(start, start + numOfPatients):
                setDataIndexList.append(ps[i][0])  # 把index和label区分
                setLabelList.append(ps[i][1:])

        LoadBenignOrMalignant(int(benignStart), int(benignEnd), self.BenignPatients)
        LoadBenignOrMalignant(int(malignantStart), int(malignantEnd), self.MalignantPatients)
        LoadBenignOrMalignant(int(malignantStart1), int(malignantEnd1), self.MalignantPatients1)
        LoadBenignOrMalignant(int(benignStart1), int(benignEnd1), self.BenignPatients1)
        LoadBenignOrMalignant(int(benignStart2), int(benignEnd2), self.BenignPatients2)
        return np.array(setDataIndexList), np.array(setLabelList)
    def LoadBenignOrMalignant(self, start, end):
        setDataIndexList = []
        setLabelList = []
        ps = self.patients
        numOfPatients = int((end - start) * 1.0)
        for i in range(start, start + numOfPatients):
            setDataIndexList.append(ps[i][0])
            setLabelList.append(ps[i][1:])
        return np.array(setDataIndexList), np.array(setLabelList)


    def NextFold(self, trainDataPercentage = 1.0):
        fold = {}
        whole = {}
        # 如果是二级预测分类就不能设置按良、恶划分数据集

        random.shuffle(self.BenignPatients)#随机打乱顺序
        random.shuffle(self.MalignantPatients)#随机打乱顺序
        random.shuffle(self.MalignantPatients1)  # 随机打乱顺序
        random.shuffle(self.BenignPatients1)  # 随机打乱顺序
        random.shuffle(self.BenignPatients2)  # 随机打乱顺序
        random.shuffle(self.patients)
        whole["DataPath"] = self.DataPath
        whole["ValidationSetDataIndex"], \
        whole["ValidationSetLabel"] = \
            self.LoadSetData_5(0, round(len(self.BenignPatients) * self.Ratio), \
                               0, round(len(self.MalignantPatients) * self.Ratio), \
                               0, round(len(self.MalignantPatients1) * self.Ratio), \
                               0, round(len(self.BenignPatients1) * self.Ratio),\
                               0, round(len(self.BenignPatients2) * self.Ratio))
        fold["DataPath"] = self.DataPath
        fold["TrainSetDataIndex"], \
        fold["TrainSetLabel"] = \
            self.LoadSetData_5(round(len(self.BenignPatients) * self.Ratio), len(self.BenignPatients),\
                               round(len(self.MalignantPatients) * self.Ratio), len(self.MalignantPatients),\
                               round(len(self.MalignantPatients1) * self.Ratio),len(self.MalignantPatients1), \
                               round(len(self.BenignPatients1) * self.Ratio),len(self.BenignPatients1),\
                               round(len(self.BenignPatients2) * self.Ratio),len(self.BenignPatients2))

        self.BenignPatients = self.BenignPatients[
                              round(len(self.BenignPatients) * self.Ratio): len(self.BenignPatients)] + \
                              self.BenignPatients[0: round(len(self.BenignPatients) * self.Ratio)]
        self.MalignantPatients = self.MalignantPatients[
                                 round(len(self.MalignantPatients) * self.Ratio): len(self.MalignantPatients)] + \
                                 self.MalignantPatients[0: round(len(self.MalignantPatients) * self.Ratio)]
        self.MalignantPatients1 = self.MalignantPatients1[
                                  round(len(self.MalignantPatients1) * self.Ratio): len(self.MalignantPatients1)] + \
                                  self.MalignantPatients1[0: round(len(self.MalignantPatients1) * self.Ratio)]
        self.BenignPatients1 = self.BenignPatients1[
                               round(len(self.BenignPatients1) * self.Ratio): len(self.BenignPatients1)] + \
                               self.BenignPatients1[0: round(len(self.BenignPatients1) * self.Ratio)]
        self.BenignPatients2 = self.BenignPatients2[
                                  round(len(self.BenignPatients2) * self.Ratio): len(self.BenignPatients2)] + \
                                  self.BenignPatients2[0: round(len(self.BenignPatients2) * self.Ratio)]
        return fold, whole
  
    
    def GetWholeAsVal(self):
        whole = {}
        random.shuffle(self.patients)
        whole["DataPath"] = self.DataPath
        whole["ValidationSetDataIndex"], \
        whole["ValidationSetLabel"] = \
            self.LoadBenignOrMalignant(0, len(self.patients))
       
        return whole
    def GetWholeAsTest(self):
        whole = {}
        #random.shuffle(self.patients)
        whole["DataPath"] = self.DataPath
        whole["TestSetDataIndex"], \
        whole["TestSetLabel"] = \
            self.LoadBenignOrMalignant(0, len(self.patients))
        return whole


def ReadFolds(paths,ratio2):
    dataPath, infoPath = paths
    patients = []
    if ratio2==0:
        nnnn = 2
    else:
        nnnn=5
    sheet = lw(infoPath).worksheets[0]
    pass_title = False
    for row in sheet.values:
        if not pass_title:
            pass_title = True
            continue
        dataIndex = str(row[0])  # .split(".")[0]
        label = int(row[1])  # 转移5分类

        biaojiwu1= float(row[7])#肿瘤标记物1#+10
        biaojiwu2 = float(row[8])  # 肿瘤标记物1+10
        biaojiwu3 = float(row[9])  # 肿瘤标记物1+10
        biaojiwu4 = float(row[10])  # 肿瘤标记物1+10
        biaojiwu5 = float(row[11])  # 肿瘤标记物1+10
        biaojiwu6 = float(row[12])  # 肿瘤标记物1+10
        biaojiwu7 = float(row[13])  # 肿瘤标记物1+10
        biaojiwu8 = float(row[14])  # 肿瘤标记物1+10
        biaojiwu9 = float(row[15])  # 肿瘤标记物1+10
        biaojiwu10 = float(row[16])  # 肿瘤标记物1+10
        
       
        patients.append((dataIndex,
                         label, 
                         biaojiwu1, biaojiwu2, biaojiwu3, biaojiwu4, biaojiwu5, biaojiwu6, biaojiwu7, biaojiwu8,
                         biaojiwu9, biaojiwu10))  #

    benignPatients = []
    benignPatients1 = []
    benignPatients2 = []
    malignantPatients = []
    malignantPatients1 = []
    
    for patient in patients:#5分类就要考虑按5个分
        if patient[1] == 0:#[1] == 0:
            benignPatients.append(patient)
        elif patient[1] == 1:
            benignPatients1.append(patient)
        elif patient[1] == 2:
            benignPatients2.append(patient)
        elif patient[1] == 3:
            malignantPatients.append(patient)
        else:
            malignantPatients1.append(patient)

    folds = Folds(benignPatients, malignantPatients, 5, dataPath, patients,benignPatients1,benignPatients2,malignantPatients1)
    return folds