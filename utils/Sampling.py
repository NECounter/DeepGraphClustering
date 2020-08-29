import numpy as np
import random

def SampleFromClusters(clusters = [[]], originData=[[]], SamplesNum = 10):
    # todo: dismish odd clusters
    clustersCount = len(clusters)
    clusterItemCount = [len(clusters[i]) for i in range(clustersCount)]
    clusterItemDiv = [(int(clusterItemCount[i] / SamplesNum) + 1) * SamplesNum - clusterItemCount[i] \
        for i in range(clustersCount)]
    for i in range(clustersCount):
        temp = clusters[i]
        for j in range(clusterItemDiv[i]):
            clusters[i].append(random.choice(temp))
        random.shuffle(clusters[i])

    samplesCountPerCluster = [len(clusters[i]) / SamplesNum for i in range(clustersCount)]
    samples = []
    for i in range(SamplesNum):
        sample = []
        for j in range(clustersCount):
            for k in range(int(samplesCountPerCluster[j])):
                sample.append(originData[int(clusters[j].pop()[0])])
        samples.append(sample)
    
    return samples



    