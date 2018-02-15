import sys
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.utils import shuffle
from datetime import datetime
from scipy import linalg as LA

def getKNearestNeighbors(images, true_labels, data_point, k):
    labels = []
    indices = set()
    for _ in range(k):
        min_distance = sys.maxint
        nearest_index = 0
        for i in range(len(images)):
            if i in indices:
                continue
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
        indices.add(nearest_index)
        #print(true_labels[nearest_index])
        #del images[nearest_index]
        #del true_labels[nearest_index]

    k_label_counts = [0]*2
    for label in labels:
        k_label_counts[label]+=1

    return k_label_counts.index(max(k_label_counts))

def getPCAData(data, no_components = 50):
    np_data = np.array(data)
    np_mean = np_data.mean(axis=0)
    np_data1 = np_data - np_mean
    np_data2 = np_data1.T
    scalar = np.dot(np_data2, np_data1)/len(np_data1)
    e_vals, e_vecs = LA.eig(scalar)

    lis = []
    for i in range(len(np_data1)):
        temp = []
        for j in range(no_components):
            temp.append(np.dot(data[i], e_vecs[j]))
        lis.append(temp)

    return lis
def main():
    mat = scipy.io.loadmat('/Users/arya/Downloads/5-1/SML/ass3/knn_data.mat')
    # data = mat['data']
    images = mat['train_data'].tolist()
    labels = mat['train_label'].tolist()
    final_test_images = mat['test_data'].tolist()
    final_test_labels = mat['test_label'].tolist()
    labels2 = list()
    for i in labels:
        labels2.append(i[0])
    labels = labels2
    final_test_labels2 = list()
    for i in final_test_labels:
        final_test_labels2.append(i[0])
    final_test_labels = final_test_labels2

    labelset = list()
    for i in labels:
        if i not in labelset:
            labelset.append(i)
            print(i)
    images, labels = shuffle(images, labels)
    # images, labels = mndata.load_training()
    # test_images, test_labels = mndata.load_testing()
    # 0.919191919192
    # 0.949494949495
    # 0.949494949495
    # 0.959595959596
    # 0.949494949495
    all_k = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    accuracy_vals = []
    pca_accuracy_vals = []
    for k in all_k:
        accuracy = 0.0
        pca_accuracy = 0.0
        for m in range(5):
            test_data_start = m * 1000
            test_data_end = test_data_start + 999
            train_data_start_1 = 0
            train_data_end_1 = test_data_start
            train_data_start_2 = test_data_end
            train_data_end_2 = len(images)

            test_data = images[test_data_start:test_data_end]
            test_data_labels = labels[test_data_start:test_data_end]
            train_data = []
            train_data_labels = []
            if len(images[train_data_start_1:train_data_end_1]) != 0:
                train_data = images[train_data_start_1:train_data_end_1]
                train_data_labels = labels[train_data_start_1:train_data_end_1]
            if len(labels[train_data_start_2:train_data_end_2]) != 0:
                if len(train_data) != 0:
                    train_data = train_data + (images[train_data_start_2:train_data_end_2])
                    train_data_labels = train_data_labels + labels[train_data_start_2:train_data_end_2]
                else:
                    train_data = (images[train_data_start_2:train_data_end_2])
                    train_data_labels = labels[train_data_start_2:train_data_end_2]
                    # train_data_labels = train_data_labels + labels[train_data_start_2:train_data_end_2]

            # train_data_labels = labels[train_data_start_1:train_data_end_1] + labels[train_data_start_2:train_data_end_2]
            accurate_count = 0.0
            pca_accurate_count = 0.0

            PCA_train_data = getPCAData(train_data, 50)
            PCA_test_data = getPCAData(test_data, 50)
            for i in range(len(test_data)):
                label = getKNearestNeighbors(train_data, train_data_labels, test_data[i], k)
                pcalabel = getKNearestNeighbors(PCA_train_data, train_data_labels, PCA_test_data[i], k)

                if label == test_data_labels[i]:
                    accurate_count += 1

                if pcalabel == test_data_labels[i]:
                    pca_accurate_count += 1
            accuracy += accurate_count / len(test_data)
            pca_accuracy += pca_accurate_count / len(test_data)

            print("time is ", str(datetime.now()))
            print(accurate_count / len(test_data))
            print(pca_accurate_count / len(test_data))
        accuracy_vals.append((k, accuracy / 5))
        pca_accuracy_vals.append((k, pca_accuracy / 5))
    best_k = 1
    temp_acc = 0.0
    for tuple in accuracy_vals:
        k = tuple[0]
        acc = tuple[1]
        if acc > temp_acc:
            temp_acc = acc
            best_k = k

    best_pca_k = 1
    temp_acc = 0.0
    PCA_train_data = getPCAData(images, 50)
    PCA_test_data = getPCAData(final_test_images, 50)
    for tuple in pca_accuracy_vals:
        k = tuple[0]
        acc = tuple[1]
        if acc > temp_acc:
            temp_acc = acc
            best_pca_k = k
    best_k_accurate_count = 0.0
    best_pca_accurate_count = 0.0
    for i in range(len(final_test_images)):
        label = getKNearestNeighbors(images, labels, final_test_images[i], best_k)
        pcalabel = getKNearestNeighbors(PCA_train_data, labels, PCA_test_data[i], best_pca_k)

        if label == final_test_labels[i]:
            best_k_accurate_count += 1

        if pcalabel == final_test_labels[i]:
            best_pca_accurate_count += 1
    best_k_accuracy = best_k_accurate_count / len(final_test_labels)
    best_pca_accuracy = best_pca_accurate_count / len(final_test_labels)

    print("best k is ", best_k)
    print("best pca k is ", best_pca_k)
    print("k acc is ", best_k_accuracy)
    print("pca accuracy is ", best_pca_accuracy)

    # plt.plot(all_k, accuracy_vals)
    # plt.show()
    # plt.savefig("accuracy_kvalue.png")

    plt.plot(all_k, accuracy_vals, 'r', label="knn")
    plt.plot(all_k, pca_accuracy_vals, 'g', label="pca_accuracy")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    plt.savefig("accuracy_kvalue.png")




if __name__ == '__main__':
    main()


