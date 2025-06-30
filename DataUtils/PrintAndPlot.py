import os
import csv
import torch
import numpy as np
import matplotlib.pylab as plot

import sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from StatisticsUtils import CalculateAUC, ClassificationMetrics, MultipleClassificationMetric
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
targetResolutionPerSubFigure = 1080
targetDPI = 200
from sklearn.metrics import f1_score
class MultiTaskClassificationAnswer():
    def __init__(self):
        self.Outputs = torch.Tensor()
        self.Labels = torch.Tensor()
        self.DataIndexes = []
        self.Accuracy = 0
        self.Recall = 0
        self.Precision = 0
        self.Specificity = 0
        self.Sensitivity =0
        self.PPV = 0
        self.NPV = 0
        self.F1 = 0
        self.TrainLosses = None
        self.ValidationLosses = None

def DrawPlots(validationFPRs, validationTPRs, validationAUCs,\
              testFPRs, testTPRs, testAUCs,\
              ensembleFPR, ensembleTPR, ensembleAUC,\
              validationAnswers, saveFolderPath, numOfFold):
    gridSize = 2
    targetFigureSize = (targetResolutionPerSubFigure * gridSize / targetDPI, targetResolutionPerSubFigure * gridSize / targetDPI)
    plot.figure(figsize = targetFigureSize, dpi = targetDPI)
    plot.subplot(gridSize, gridSize, 1)
    for i in range(3):#5
        plot.title("Validation AUC by folds")
        plot.plot(validationFPRs[i], validationTPRs[i], alpha = 0.7, label = ("Fold %d Val AUC = %0.3f" % (i, validationAUCs[i])))
        plot.legend(loc = "lower right")
        plot.plot([0, 1], [0, 1],"r--")
        plot.xlim([0, 1])
        plot.ylim([0, 1.05])
        plot.ylabel("True Positive Rate")
        plot.xlabel("False Positive Rate")

    plot.subplot(gridSize, gridSize, 2)
    for i in range(3):#5
        plot.title("Test AUC by folds")
        plot.plot(testFPRs[i], testTPRs[i], alpha = 0.7, label = ("Fold %d Test AUC = %0.3f" % (i, testAUCs[i])))
        plot.legend(loc = "lower right")
        plot.plot([0, 1], [0, 1],"r--")
        plot.xlim([0, 1])
        plot.ylim([0, 1.05])
        plot.ylabel("True Positive Rate")
        plot.xlabel("False Positive Rate")

    plot.subplot(gridSize, gridSize, 3)
    plot.title("Test AUC by ensemble")
    #plot.plot(ensembleFPR, ensembleTPR, alpha = 0.7, label = "Test AUC = %0.3f" % ensembleAUC)
    plot.plot(ensembleFPR, ensembleTPR, alpha=0.7, label="ROC curve (AUC= %0.3f)" % ensembleAUC)
    plot.legend(loc = "lower right")
    plot.plot([0, 1], [0, 1],"r--")
    plot.xlim([0, 1])
    plot.ylim([0, 1.05])
    plot.ylabel("Sensitivity")
    plot.xlabel("1-Specificity")
  
    plot.savefig(os.path.join(saveFolderPath, "ROCCurvePlot.png"))

    if validationAnswers[0].TrainLosses is None:
        return
    hasLabelLoss = hasattr(validationAnswers[0], "TrainLabelLosses")
    gridSize = 4 if hasLabelLoss else 3
    targetFigureSize = (targetResolutionPerSubFigure * gridSize / targetDPI, targetResolutionPerSubFigure * gridSize / targetDPI)
    plot.figure(figsize = targetFigureSize, dpi = targetDPI)
    for i in range(3):#numOfFold
        plot.subplot(gridSize, gridSize, i + 1)
        plot.title("Fold %d Losses" % i)
        plot.plot(np.array(validationAnswers[i].TrainLosses), label = "Train Loss")
        plot.plot(np.array(validationAnswers[i].ValidationLosses), label = "Validation Loss")
        plot.legend(loc = "upper right")
        plot.xlabel("Epoch")
        plot.ylabel("Loss")

        if hasLabelLoss:
            plot.subplot(gridSize, gridSize, i + 6)
            plot.title("Fold %d Label Losses" % i)
            plot.plot(np.array(validationAnswers[i].TrainLabelLosses), label = "Train Label Loss")
            plot.plot(np.array(validationAnswers[i].ValidationLabelLosses), label = "Validation Label Loss")
            plot.legend(loc = "upper right")
            plot.xlabel("Epoch")
            plot.ylabel("Loss")
    plot.savefig(os.path.join(saveFolderPath, "LossesPlot.png"))



import scipy.stats
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2, n-1)
    return m, m-h, m+h


class MultiTaskClassificationAnswer():
    def __init__(self):
        self.Outputs = torch.Tensor() #* 11
        self.Labels = torch.Tensor()
        self.DataIndexes = []
        self.Accuracy = [0] #* 11
        self.Recall = [0] #* 11
        self.Precision = [0] #* 11
        self.Specificity = [0]# * 11
        self.TrainLosses = None
        self.TrainLabelLosses = None
        self.ValidationLosses = None
        self.ValidationLabelLosses = None

def MultiTaskEnsembleTest(testAnswers, saveFolderPath,multi_len=5):
    foldPredict = np.array([testAnswer.Outputs.numpy() for testAnswer in testAnswers])
    _,len_data,n_class= foldPredict.shape
    rawResults = np.mean(foldPredict, axis=0)  
    label = testAnswers[0].Labels.numpy()
    output = rawResults
    predict = np.argmax(output,1)
    TP = [0]*n_class
    FP = [0] * n_class
    TN = [0] * n_class
    FN = [0] * n_class

    for c in range(rawResults.shape[1]):
        #print(c)
        P = (predict == c).astype(np.int64)#(predict.int() == c).int()
        N = (predict != c).astype(np.int64)#(predict.int() != c).int()
        l = (label == c).astype(np.int64)#(label.int() == c).int()
        TP[c] += np.sum(P * l)#(P * l).sum().item()
        FP[c] += np.sum(P * (1 - l))#(P * (1 - l)).sum().item()
        TN[c] += np.sum(N * (1 - l))#(N * (1 - l)).sum().item()
        FN[c] += np.sum(N * l)#(N * l).sum().item()
   
    a, r, p, s, s2, p2, n2, f2 = [0] * len(TP), [0] * len(TP), [0] * len(TP), [0] * len(TP), [0] * len(TP), [0] * len(TP), [0] * len(TP), [0] * len(TP)
    for c in range(len(TP)):
        a[c], r[c], p[c], s[c], s2[c], p2[c], n2[c], f2[c] = ClassificationMetrics(TP[c], FP[c], TN[c], FN[c])
    accuracy, recall, precision, specificity, sensitivity, PPV, NPV, F1 = np.mean(a), np.mean(r), np.mean(p), np.mean(s), np.mean(s2), np.mean(p2), np.mean(n2), np.mean(f2)
    ensembleAUC, ensembleFPR, ensembleTPR = CalculateAUC(rawResults,label, needThreshold=True,multi=True,multi_len=multi_len) 
    print("\nEnsemble Test Results:")
    print("AUC,%f\nAccuracy,%f\nRecall,%f\nPrecision,%f\nSpecificity,%f\nSensitivity,%f\nPPV,,%f\nNPV,,%f\nF1%f\n" % \
          (ensembleAUC, accuracy, recall, precision, specificity, sensitivity, PPV, NPV, F1))

    foldPredicts = []
    labels = []
   
    with open(os.path.join(saveFolderPath, "TestResults.csv"), mode = "w", newline = "") as csvFile:
        csvWriter = csv.writer(csvFile)
        csvWriter.writerow(["Ensemble Test Results:"])
        csvWriter.writerow(["AUC:", ensembleAUC, "Accuracy:", accuracy, "Recall:",recall,"Precision:", precision, "Specificity:", specificity,"Sensitivity:", sensitivity, "PPV:", PPV, "NPV:", NPV, "F1:", F1])
        csvWriter.writerow(["DataIndex", "Ensembled", "Fold1", "Fold2", "Fold3"])
        for i, dataIndex in enumerate(testAnswers[0].DataIndexes):
            csvWriter.writerow([dataIndex, str(rawResults[i]), str(foldPredict[0][i]), str(foldPredict[1][i]), str(foldPredict[2][i])])
    return ensembleAUC, ensembleFPR, ensembleTPR


def MultiTaskClassificationPrintAndPlot(validationAnswers, testAnswers, saveFolderPath,multi_len=4):
    numOfFold = len(validationAnswers)
    validationAverages = [0] * 9
    testAverages = [0] * 9

    validationAUCs = []
    validationFPRs = []
    validationTPRs = []
    testAUCs = []
    testFPRs = []
    testTPRs = []
    valid_result = [[0] * 3, [0] * 3, [0] * 3, [0] * 3, [0] * 3, [0] * 3, [0] * 3, [0] * 3, [0] * 3]
    test_result = [[0] * 3, [0] * 3, [0] * 3, [0] * 3, [0] * 3, [0] * 3, [0] * 3, [0] * 3, [0] * 3]
    print(",,,Validation,,,,,,Test,,,,")
    print("Fold,Accuracy,Recall,Precision,Specificity,AUC,,Accuracy,Recall,Precision,Specificity,AUC,")
    for i in range(numOfFold):
       
        validationAUC, validationFPR, validationTPR = CalculateAUC(validationAnswers[i].Outputs.numpy(),
                                                                   validationAnswers[i].Labels.numpy(), needThreshold=True,
                                                                   multi=True,multi_len=multi_len) 
        validationAUCs.append(validationAUC)
        validationFPRs.append(validationFPR)
        validationTPRs.append(validationTPR)

        validationAverages[0] += validationAnswers[i].Accuracy
        validationAverages[1] += validationAnswers[i].Recall
        validationAverages[2] += validationAnswers[i].Precision
        validationAverages[3] += validationAnswers[i].Specificity
        validationAverages[4] += validationAUC
        validationAverages[5] += validationAnswers[i].F1
        validationAverages[6] += validationAnswers[i].PPV
        validationAverages[7] += validationAnswers[i].NPV
        validationAverages[8] += validationAnswers[i].Sensitivity

        valid_result[0][i] = validationAUC
        valid_result[1][i] = validationAnswers[i].Accuracy
        valid_result[2][i] = validationAnswers[i].Recall
        valid_result[3][i] = validationAnswers[i].Precision
        valid_result[4][i] = validationAnswers[i].Specificity
        valid_result[5][i] = validationAnswers[i].F1
        valid_result[6][i] = validationAnswers[i].PPV
        valid_result[7][i] = validationAnswers[i].NPV
        valid_result[8][i] = validationAnswers[i].Sensitivity

        print("%d," % i, end="")
        print("%f," % validationAnswers[i].Accuracy, end="")
        print("%f," % validationAnswers[i].Recall, end="")
        print("%f," % validationAnswers[i].Precision, end="")
        print("%f," % validationAnswers[i].Specificity, end="")
        print("%f,," % validationAUC, end="")
        print("%f,," % validationAnswers[i].F1, end="")
        print("%f,," % validationAnswers[i].PPV, end="")
        print("%f,," % validationAnswers[i].NPV, end="")
        print("%f,," % validationAnswers[i].Sensitivity, end="")

        #Test
        testAUC, testFPR, testTPR = CalculateAUC(testAnswers[i].Outputs.numpy(),testAnswers[i].Labels.numpy(),needThreshold=True,multi=True,multi_len=multi_len)
        testAUCs.append(testAUC)
        testFPRs.append(testFPR)
        testTPRs.append(testTPR)

        testAverages[0] += testAnswers[i].Accuracy
        testAverages[1] += testAnswers[i].Recall
        testAverages[2] += testAnswers[i].Precision
        testAverages[3] += testAnswers[i].Specificity
        testAverages[4] += testAUC
        testAverages[5] += testAnswers[i].F1
        testAverages[6] += testAnswers[i].PPV
        testAverages[7] += testAnswers[i].NPV
        testAverages[8] += testAnswers[i].Sensitivity

        test_result[0][i] = testAUC
        test_result[1][i] = testAnswers[i].Accuracy
        test_result[2][i] = testAnswers[i].Recall
        test_result[3][i] = testAnswers[i].Precision
        test_result[4][i] = testAnswers[i].Specificity
        test_result[5][i] = testAnswers[i].F1
        test_result[6][i] = testAnswers[i].PPV
        test_result[7][i] = testAnswers[i].NPV
        test_result[8][i] = testAnswers[i].Sensitivity

        print("%f," % testAnswers[i].Accuracy, end="")
        print("%f," % testAnswers[i].Recall, end="")
        print("%f," % testAnswers[i].Precision, end="")
        print("%f," % testAnswers[i].Specificity, end="")
        print("%f," % testAUC, end="")
        print("%f,," % testAnswers[i].F1, end="")
        print("%f,," % testAnswers[i].PPV, end="")
        print("%f,," % testAnswers[i].NPV, end="")
        print("%f,," % testAnswers[i].Sensitivity, end="")
        print("\n")

    validationAverages = np.array(validationAverages) / numOfFold
    testAverages = np.array(testAverages) / numOfFold

    print("Average,", end = "")
    for v in validationAverages:
        print("%f," % v, end = "")
    print(",", end = "")
    for v in testAverages:
        print("%f," % v, end = "")
    print()
    print("95CI AUC, Average, Accuracy, Recall, Precision, Specificity, F1, PPV, NPV, Sensitivity\n", end="")
    for i in range(9):
        tmp = mean_confidence_interval(valid_result[i], confidence=0.95)
        print("%f " % np.float64(tmp[0]), end="")
        print("(%f, " % np.float64(tmp[1]), end="")
        print(" %f)! " % np.float64(tmp[2]), end="")

    print(" ")
    print("Test 95CI AUC, Average, Accuracy, Recall, Precision, Specificity, F1, PPV, NPV, Sensitivity\n", end="")
    for i in range(9):
        tmp = mean_confidence_interval(test_result[i], confidence=0.95)
        print("%f  " % np.float64(tmp[0]), end="")
        print("(%f, " % np.float64(tmp[1]), end="")
        print(" %f)! " % np.float64(tmp[2]), end="")
    print()


    ensembleAUC, ensembleFPR, ensembleTPR = MultiTaskEnsembleTest(testAnswers, saveFolderPath,multi_len=multi_len)

    DrawPlots(validationFPRs, validationTPRs, validationAUCs,\
              testFPRs, testTPRs, testAUCs,\
              ensembleFPR, ensembleTPR, ensembleAUC,\
              validationAnswers, saveFolderPath, numOfFold)

def ClassificationPrintAndPlot(validationAnswers, testAnswers, saveFolderPath):
    MultiTaskClassificationPrintAndPlot(validationAnswers, testAnswers, saveFolderPath,multi_len=5)
   




