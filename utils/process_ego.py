import numpy as np
import copy
import pandas as pd


# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, allY

def load_data(dataSetName):
    classInfo = "data/{}/{}.circles".format(dataSetName, dataSetName)
    adjInfo = "data/{}/{}.edges".format(dataSetName, dataSetName)
    featureInfo = "data/{}/{}.feat".format(dataSetName, dataSetName)

    selectedFeatureNum = 0
    if dataSetName == 'facebook':
        selectedFeatureNum = 5
    else:
        selectedFeatureNum = 0

    featureLines = []
    with open(featureInfo, 'r') as featureFile:
        lines = featureFile.readlines()
        for line in lines:
            lineData = list(map(int, line.strip().split()))
            featureLines.append(lineData)
        # featureLines.sort(key=lambda x: x[0])
    featureCount = len(featureLines)

    circleLines = []
    allUsers = []
    with open(classInfo, 'r') as circleFile:
        lines = circleFile.readlines()
        for line in lines:
            lineData = list(map(int, line.strip().split()))
            circleLines.append(lineData)
            allUsers += lineData[1:]
    classCount = len(circleLines)

    trainItems = []
    for i in range(classCount):
        for j in range(selectedFeatureNum):
            trainItems.append(circleLines[i][j + 1])

    insertIndex = 0
    for i in range(featureCount):
        if featureLines[i][0] in allUsers:
            allUsers.remove(featureLines[i][0])
            line = copy.deepcopy(featureLines[insertIndex])
            featureLines[insertIndex] = copy.deepcopy(featureLines[i])
            featureLines[i] = copy.deepcopy(line)
            insertIndex += 1

    featureLines = featureLines[0:insertIndex]
    featureCount = len(featureLines)

    insertIndex = 0
    for i in range(featureCount):
        if featureLines[i][0] in trainItems:
            trainItems.remove(featureLines[i][0])
            line = copy.deepcopy(featureLines[insertIndex])
            featureLines[insertIndex] = copy.deepcopy(featureLines[i])
            featureLines[i] = copy.deepcopy(line)
            insertIndex += 1

    # adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, allY
    features = copy.deepcopy(np.array(featureLines)[:, 1:])
    classes = np.diag([1 for i in range(classCount)]).tolist()
    allY = [[] for i in range(featureCount)]
    allMask = np.array([True for i in range(featureCount)])

    for i in range(featureCount):
        for j in range(classCount):
            if featureLines[i][0] in circleLines[j][1:]:
                allY[i] = classes[j]
                break

    y_train = [[0 for i in range(classCount)] for i in range(featureCount)]
    y_val = [[0 for i in range(classCount)] for i in range(featureCount)]
    y_test = [[0 for i in range(classCount)] for i in range(featureCount)]
    train_mask = np.array([False for i in range(featureCount)])
    val_mask = np.array([False for i in range(featureCount)])
    test_mask = np.array([False for i in range(featureCount)])

    y_train[0:45] = allY[0:45]
    y_val[45:155] = allY[45:155]
    y_test[300:] = allY[300:]

    train_mask[0:45] = allMask[0:45]
    val_mask[45:155] = allMask[45:155]
    test_mask[300:] = allMask[300:]

    indexes = []
    for feature in featureLines:
        indexes.append(feature[0])

    adj = np.zeros((featureCount, featureCount))
    adjLines = []
    with open(adjInfo, 'r') as adjFile:
        lines = adjFile.readlines()
        for line in lines:
            [i, j] = list(map(int, line.strip().split()))
            try:
                adj[indexes.index(i)][indexes.index(j)] = 1
                adj[indexes.index(j)][indexes.index(i)] = 1
            except:
                continue
            adjLines.append(lineData)

    return [adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, allY]




if __name__ == "__main__":
    load_data("facebook")
