import numpy as np
import math
import random
from sklearn import metrics

def EvlClustering(dataSet, labelsTrue, labelsPred, labelIsNum=False):
    data = []
    labels_True = [np.argmax(labelsTrue[i]) for i in range(len(labelsTrue))]
    if labelIsNum:
        labels_Pred = labelsPred
        data = dataSet
    else:
        labels_Pred = [np.argmax(labelsPred[i]) for i in range(len(labelsPred))]
        for i in range(len(dataSet)):
            data.append(dataSet[i][0:len(dataSet[i])])

    ARI, NMI, HOMO, COMP, VMeasure, FMS, SC = [0, 0, 0, 0, 0, 0, 0]

    if labelsTrue != []:
        # Adjusted Rand index
        ARI = metrics.adjusted_rand_score(labels_True, labels_Pred)
        # Mutual Information based scores
        AMI = metrics.adjusted_mutual_info_score(labels_True, labels_Pred)

        NMI = metrics.normalized_mutual_info_score(labels_True, labels_Pred)
        # 同质性homogeneity：每个群集只包含单个类的成员
        HOMO = metrics.homogeneity_score(labels_True, labels_Pred)
        # 完整性completeness：给定类的所有成员都分配给同一个群集
        COMP = metrics.completeness_score(labels_True, labels_Pred)
        # 两者的调和平均V-measure
        VMeasure = metrics.v_measure_score(labels_True, labels_Pred)
        # Fowlkes-Mallows scores
        FMS = metrics.fowlkes_mallows_score(labels_True, labels_Pred)

        F1We = metrics.f1_score(labels_True,labels_Pred, average='weighted')
        F1Ma = metrics.f1_score(labels_True, labels_Pred, average='macro')
        F1Mi = metrics.f1_score(labels_True, labels_Pred, average='micro')


    # Silhouette Coefficient
    # SC = metrics.silhouette_score(data, labels_Pred, metric='cosine')

    # return [ARI, NMI, HOMO, COMP, VMeasure, FMS, SC]

    return [NMI, AMI, FMS]

