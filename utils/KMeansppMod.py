import numpy as np
from enum import Enum
import math
import random


class SimilarityType(Enum):
    Euclidean = 0
    Mahalanobis = 1
    Cosine = 2
    Hamming = 3
    Neighbour = 4


def KMeansClustering(dataSet, classAmount, patience, similarityType, isInit, formerCluster=None):
    # print("K-Means Started")
    clusteredDataSet = []
    for i in range(classAmount):
        cluster = []
        clusteredDataSet.append(cluster)
    isConvergence = False
    # print("Calculate Init Centers")
    centers = []
    if isInit == True:
        centers = SetInitCenters(dataSet, classAmount, similarityType)
    else:
        assert formerCluster != None, 'If \'isInit\' is not True, \'formerCluster\' should be given!'
        for i in range(classAmount):
            centers.append(ReCalBrayCenter(formerCluster[i]))
        clusteredDataSet = formerCluster

    simMap = []
    for i in range(classAmount):
        tm1 = [i, 0.0]
        simMap.append(tm1)

    iterCount = 0

    while not isConvergence:
        # print("Iter "+ str(iterCount) +" Start")
        lastClusterdDataSet = []
        for item in clusteredDataSet:
            t1 = []
            for strs in item:
                t1.append(strs)
            lastClusterdDataSet.append(t1)

        if iterCount > 0:

            for i in range(classAmount):
                centers[i] = ReCalBrayCenter(clusteredDataSet[i])
            # print("Bray Center Recalculated")

        for i in range(classAmount):
            clusteredDataSet[i].clear()
        centersSet = []

        for center in centers:
            centerSet = center[1:len(center)]
            centersSet.append(centerSet)

        # print("Calculate Similarity")

        for item in dataSet:
            itemSet = item[1:len(item)]

            for j in range(len(centersSet)):
                simMap[j][0] = j
                simMap[j][1] = CalSimilarity(itemSet, centersSet[j], similarityType)

            simMap.sort(key=(lambda x:x[1]))
            clusteredDataSet[int(simMap[0][0])].append(item)

        for i in range(classAmount):
            isConvergence = True
            if len(lastClusterdDataSet[i]) != len(clusteredDataSet[i]):
                # print("Amount not equal")
                isConvergence = False
                break

            for item in lastClusterdDataSet[i]:
                if not clusteredDataSet[i].__contains__(item):
                    # print("Item not equal")
                    isConvergence = False
                    break

        if iterCount >= patience:
            isConvergence = True
            # print("No patience, terminate clustering")

        iterCount += 1
    return clusteredDataSet, iterCount


def CalSimilarity(categorySet1, categorySet2, similarityType):
    if similarityType == SimilarityType.Euclidean:
        cb = [categorySet1, categorySet2]
        cb = [[row[i] for row in cb] for i in range(len(cb[0]))]
        return math.sqrt(math.fsum(map(lambda x:(x[0]-x[1])*(x[0]-x[1]), cb)))
    if similarityType == SimilarityType.Cosine:
        categorySet1 = np.array(categorySet1)
        categorySet2 = np.array(categorySet2)
        return 1 - np.dot(categorySet1,categorySet2) / \
               (np.linalg.norm(categorySet1) * np.linalg.norm(categorySet2))


def SetInitCenters(dataSet, classAmount, similarityType):
    centers = []
    centersCopy = []
    SimValueSet = []
    centers.append(dataSet[random.choice(range(len(dataSet)))])
    centersCopy.append(centers[0])

    iterCount = 0
    while len(centers) < classAmount or len(centers) == 1:
        brayCenter = ReCalBrayCenter(centersCopy)
        brayCenterSet = brayCenter[1:len(brayCenter)]

        maxSimItem = []
        maxSimValue = 0

        for item in dataSet:
            itemSet = item[1:len(item)]
            currentSim = CalSimilarity(brayCenterSet, itemSet, similarityType)

            if currentSim > maxSimValue:
                maxSimValue = currentSim
                maxSimItem = item

        if iterCount >= 100:
            # print("No patience, terminate init")
            maxSimItem = random.choice(dataSet)

        # maxSimItem = random.choice(dataSet)

        if not centers.__contains__(maxSimItem):
            centers.append(maxSimItem)
            centersCopy.append(maxSimItem)
        iterCount += 1

    return centers


def ReCalBrayCenter(cluster):
    cluster = [[row[i] for row in cluster] for i in range(len(cluster[0]))]
    return list(map(lambda x:math.fsum(x)/len(x), cluster))


def ClusterCalibration(currentCluster, formerCluster, formerPseudoY, allY, classes):
    classRecord = [[0 for i in range(len(classes))] for j in range(len(currentCluster))]
    for i in range(len(currentCluster)):
        for j in range(len(currentCluster[i])):
            for k in range(len(classes)):
                if formerPseudoY[currentCluster[i][j][0]] == classes[k]:
                    classRecord[i][k] += 1

    npClassRecord = np.array(classRecord)
    maxRecord = np.argmax(npClassRecord, 1)


    pass




if __name__=="__main__":
    print(KMeansClustering([[1,1,1,1,1],[2,2,2,2,2],[3,1,1,1,2]], 3, 1000, SimilarityType.Euclidean, True))













