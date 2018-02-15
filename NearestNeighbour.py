import sys
from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def getKNearestNeighbors(images, true_labels, data_point, k):
    labels = []
    for _ in range(k):
        min_distance = sys.maxint
        nearest_index = 0
        for i in range(len(images)):
            image = images[i]
            cur_distance = 0
            for j in range(len(image)):
                diff = image[j]-data_point[j]
                diff = diff*diff
                cur_distance = cur_distance+diff

            if cur_distance < min_distance:
                min_distance = cur_distance
                nearest_index = i
        labels.append(true_labels[nearest_index])
        #print(true_labels[nearest_index])
        del images[nearest_index]
        del true_labels[nearest_index]

    k_label_counts = [0]*10
    for label in labels:
        k_label_counts[label]+=1

    return k_label_counts.index(max(k_label_counts))


data = "/Users/arya/Downloads/5-1/SML/data"
mndata = MNIST(data)
images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
# 0.919191919192
# 0.949494949495
# 0.949494949495
# 0.959595959596
# 0.949494949495
all_k = [1, 3, 5, 10, 30, 50, 70, 80, 90, 100]
accuracy_vals = []
for k in all_k:
    accurate_count = 0.0
    for i in range(len(test_images[1:250])):
        label = getKNearestNeighbors(images[1:2500], labels[1:2500], test_images[i], k)

        if label == test_labels[i]:
            accurate_count+=1

    accuracy = accurate_count/len(test_images[1:250])
    print("time is ", str(datetime.now()))
    print(accuracy)
    accuracy_vals.append(accuracy)

plt.plot(all_k, accuracy_vals)
plt.show()
plt.savefig("accuracy_kvalue.png")