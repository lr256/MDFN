import copy
import torch
import argparse
import numpy as np
import torch.nn as nn
from termcolor import colored
from DataUtils.PrintAndPlot import MultiTaskClassificationAnswer
from NeutralNetworkUtils.MDFN import InceptionResNetV2,MDFN,BasicConvolution2D
from StatisticsUtils import ClassificationMetrics,MultipleClassificationMetric
from MachineLearningModel import MachineLearningModel, EvaluateMachineLearningModel
import time
import warnings
from StatisticsUtils import CalculateAUC

warnings.filterwarnings("ignore")
class MultiTaskModel(MachineLearningModel):
    def __init__(self, earlyStoppingPatience, learnRate, batchSize,mode_flag,yuce_level,numclass):
        super().__init__(earlyStoppingPatience, learnRate, batchSize)
        if mode_flag == "B":
            self.Net = InceptionResNetV2(numOfClasses=numclass)
            self.Net.Convolution1A = BasicConvolution2D(1, 32, kernelSize=3, stride=2)  # 1,32
        elif mode_flag == "C":
            self.Net = InceptionResNetV2(numOfClasses=numclass)
            self.Net.Convolution1A = BasicConvolution2D(3, 32, kernelSize=3, stride=2)  # 1,32
        else:
            self.Net = MDFN(numOfClasses=numclass)
        self.LossFunction = nn.CrossEntropyLoss()
    def Train(self,xishu,ratio,mode_flag,yuce_level):
        epoch = 0
        patience = self.EarlyStoppingPatience
        optimizer = torch.optim.Adam(self.Net.parameters(), lr=self.LearnRate)
        numOfInstance = len(self.TrainLabel)
        minLoss = float(0x7FFFFFFF)
        maxAUC =0
        bestValidationAnswer = None
        trainLosses = []
        validationLosses = []
        while patience > 0:
            self.Net.train()
            epoch += 1

            runningLoss = 0.0
            for batchImage, batchLabel, _ in self.TrainLoader:
                batchImage = batchImage.float().cuda()
                optimizer.zero_grad()
                batchLabel1 = batchLabel[:, 0].long().cuda()
                batchLabel_linchuang = batchLabel[:, 1:].float().cuda()
                if mode_flag == "B" or mode_flag == "C":
                    outputClass = self.Net.forward(batchImage, batchLabel_linchuang, ratio)
                    loss = self.LossFunction(outputClass, batchLabel1)
                else:
                    outputClass, outputClass_us, outputClass_cdfi = self.Net.forward(batchImage, batchLabel_linchuang,ratio,flag_feature=0)#mode="B+C")
                    loss = self.LossFunction(outputClass, batchLabel1) + xishu * self.LossFunction(outputClass_cdfi,batchLabel1) + \
                               (xishu) * self.LossFunction(outputClass_us, batchLabel1)

                loss = loss.mean()
                runningLoss += loss.item()
                loss.backward()
                optimizer.step()

            self.Net.eval()
            trainLoss = (runningLoss * self.BatchSize) / numOfInstance
            validationAnswer, validationLoss = self.Evaluate(self.ValidationLoader, None,xishu,ratio,mode_flag,yuce_level)

            trainLosses.append(trainLoss)
            validationLosses.append(validationLoss)
            print("Epoch %d:  (Patience left: %d )\ntrainLoss -> %.3f, valLoss -> %.3f" % (epoch, patience, trainLoss, validationLoss))
            print("Accuracy -> %f" % validationAnswer.Accuracy, end = ", ")
            print("Recall -> %f" % validationAnswer.Recall, end = ", ")
            print("Precision -> %f" % validationAnswer.Precision, end = ", ")
            print("Sensitivity -> %f" % validationAnswer.Sensitivity, end = ", ")
            print("Specificity -> %f" % validationAnswer.Specificity)
            print("f1_score -> %f" % validationAnswer.F1)
            print("PPV -> %f" % validationAnswer.PPV)
            print("NPV -> %f" % validationAnswer.NPV)

            validationAUC, validationFPR, validationTPR = CalculateAUC(validationAnswer.Outputs.numpy(),validationAnswer.Labels.numpy(),
                                                                        needThreshold=True,
                                                                        multi=True,multi_len=5)
            if minLoss > validationLoss:
                patience = self.EarlyStoppingPatience
                minLoss = validationLoss
                maxAUC = validationAUC
                print("AUC ->{}; maxAUC->{}".format(validationAUC, maxAUC))
                bestValidationAnswer = validationAnswer
                self.BestStateDict = copy.deepcopy(self.Net.state_dict())
                print(colored("Better!!!!!!!!!!!!!!!!!!!!!!!!!!!", "green"))
            else:
                patience -= 1
                print("AUC ->{}; maxAUC->{}".format(validationAUC, maxAUC))
                print(colored("Worse!!!!!!!!!!!!!!!!!!!!!!!!!!!", "red"))

        bestValidationAnswer.TrainLosses = trainLosses
        bestValidationAnswer.ValidationLosses = validationLosses
        return bestValidationAnswer,self.BestStateDict

    def Evaluate(self, dataLoader, stateDictionary,xishu,ratio,mode_flag,yuce_level):
        self.Net.eval()
        answer = SingleTaskClassificationAnswer()
        if stateDictionary is not None:
            self.LoadStateDictionary(stateDictionary)
        with torch.no_grad():
            numOfInstance = len(dataLoader.dataset)
            runningLoss = 0.0
            TP = [0] * 5
            FP = [0] * 5
            TN = [0] * 5
            FN = [0] * 5


            for batchImage, batchLabel, batchDataIndex in dataLoader:
                batchImage = batchImage.float().cuda()
                batchLabel1 = batchLabel[:, 0].long().cuda()
                batchLabel_linchuang = batchLabel[:, 1:].float().cuda()
                if mode_flag == "B" or mode_flag == "C":
                    outputClass = self.Net.forward(batchImage, batchLabel_linchuang, ratio)
                    loss = self.LossFunction(outputClass, batchLabel1)
                else:
                    outputClass, outputClass_us, outputClass_cdfi = self.Net.forward(batchImage,batchLabel_linchuang, ratio,flag_feature=0)
                    loss = self.LossFunction(outputClass, batchLabel1) + xishu * self.LossFunction(outputClass_cdfi,batchLabel1) + (xishu) * self.LossFunction(outputClass_us, batchLabel1)
                loss = loss.mean()
                runningLoss += loss.item()
                MultipleClassificationMetric(outputClass, batchLabel1, TP, FP, TN, FN)
                answer.Outputs = torch.cat((answer.Outputs, outputClass.softmax(dim=1).cpu()), dim=0)
                answer.Labels = torch.cat((answer.Labels, batchLabel1.float().cpu()), dim=0)
                answer.DataIndexes += batchDataIndex


            a, r, p, s, s2, p2, n2, f2 = [0] * len(TP), [0] * len(TP), [0] * len(TP), [0] * len(TP), [0] * len(TP), [0] * len(TP), [0] * len(TP), [0] * len(TP)
            for c in range(len(TP)):
                a[c], r[c], p[c], s[c], s2[c], p2[c], n2[c], f2[c] = ClassificationMetrics(TP[c], FP[c], TN[c],FN[c],self.Epsilon)
            answer.Accuracy, answer.Recall, answer.Precision, answer.Specificity, answer.Sensitivity, answer.PPV, answer.NPV, answer.F1 = np.mean(
                a), np.mean(r), np.mean(p), np.mean(s), np.mean(s2), np.mean(p2), np.mean(n2), np.mean(f2)

            loss = (runningLoss * self.BatchSize) / numOfInstance

            if stateDictionary is not None:
                print("batchLabel1:{}".format(batchLabel1))
                print("Accuracy -> %f" % answer.Accuracy, end=", ")
                print("Recall -> %f" % answer.Recall, end=", ")
                print("Precision -> %f" % answer.Precision, end=", ")
                print("Sensitivity -> %f" % answer.Sensitivity, end=", ")
                print("Specificity -> %f" % answer.Specificity)
                print("PPV -> %f" % answer.PPV)
                print("NPV -> %f" % answer.NPV)
                print("f1_score -> %f" % answer.F1)
                testAUC, testFPR, testTPR = CalculateAUC(answer.Outputs.numpy(),answer.Labels.numpy(),needThreshold=True,multi=True, multi_len=5
                print("testAUC -> %f" % testAUC)

            return answer, loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--TrainFolderPath", help = "define the train data folder path", type = str)
    parser.add_argument("--TestFolderPath", help = "define the test data folder path", type = str)
    parser.add_argument("--TrainInfoPath", help = "define the train info path", type = str)
    parser.add_argument("--TestInfoPath", help = "define the test info path", type = str)
    parser.add_argument("--SaveFolderPath", help = "define the save folder path", type = str)
    parser.add_argument("--Name", help = "define the name", type = str)
    args = parser.parse_args()
    args.TrainFolderPath = "/home/lr/data/images/"  # train
    args.TestFolderPath = "/home/lr/data/images/"   # test
    args.SaveFolderPath = "./save_model_3100_generate_4"

    args.TrainInfoPath = "./xlsx/2025-2-19-zhuanyi-train.xlsx"
    args.TestInfoPath = "./xlsx/2025-2-19-zhuanyi-test.xlsx"
    args.Name = "lymph_MDFN_" + time.strftime("%Y-%m-%d %H.%M.%S",time.localtime())
    EvaluateMachineLearningModel(MultiTaskModel, \
                                 args.SaveFolderPath, (args.TrainFolderPath, args.TrainInfoPath),
                                 (args.TestFolderPath, args.TestInfoPath), earlyStoppingPatience = 20,batchSize=64,\
                                 name=args.Name, xishu=0.2, ratio=1,ratio2 =0,mode_flag="B+C",yuce_level=1,numclass=5)
    
