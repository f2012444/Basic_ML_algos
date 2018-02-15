import scipy.io
import random
import sys



def getkmeansloss(data, clusters, k):
    initialloss = 0.0
    diff = 1.0

    while diff != 0.0:
        cluster_points = []
        currentloss = 0.0
        for i in range(len(data)):
            cluster = 0
            loss = 0
            initialdistance = sys.maxint
            for j in range(len(clusters)):
                dist = getdistance(data[i], clusters[j])
                if dist < initialdistance:
                    initialdistance = dist
                    cluster = j
            currentloss = currentloss + initialdistance
            cluster_points.append(cluster)
        if abs(currentloss - initialloss)  <0.00001:
            return currentloss
        initialloss = currentloss
        #newcenters = clusters
        newCount = [0] * len(clusters)
        newCenters = [[0.0] * len(data[0]) for _ in range(len(clusters))]
        for i in range(len(cluster_points)):
            newCenters[cluster_points[i]] = newCenters[cluster_points[i]] + data[i]
            newCount[cluster_points[i]] = newCount[cluster_points[i]] + 1
        for i in range(len(clusters)):
            if newCount[i] == 0:
                continue
            newCenters[i] = newCenters[i]/newCount[i]


        clusters = newCenters
    return initialloss






def getdistance(data1, data2):
    sum = 0.0
    for i in range(len(data1)):
        val = data1[i]-data2[i]
        sum = sum + square(val)
    return sum

def square(val):
    return val*val


ran_list = []
def main():
    mat = scipy.io.loadmat('/Users/arya/Downloads/5-1/SML/ass3/kmeans_data.mat')
    data = mat['data']
    for k in range(2, 11):
        clusters = list(map(lambda _: random.choice(data), range(k)))
        #print(clusters[0])
        loss = getkmeansloss(data, clusters, k)
        ran_list.append((k, loss))
        print(loss)
    # k-means++
    for k in range(2, 11):
        clusters = []
        clusters.append(data[0])
        for i in range(k-1):
            cluster_point = [0.0] * len(data[0])
            max_dist = -sys.maxint
            for j in range(len(data)):
                for m in range(len(clusters)):
                    distance = getdistance(clusters[m], data[j])
                    if distance > max_dist:
                        cluster_point = data[j]
                        max_dist = distance
            clusters.append(cluster_point)
        #print(clusters[1])
        loss = getkmeansloss(data, clusters, k)
        ran_list.append((k, loss))
        print(loss)

thefile = open('/Users/arya/PycharmProjects/SML-ass3/test.txt', 'w')
for t in ran_list:
    thefile.write(t)





if __name__ == "__main__":
    main()