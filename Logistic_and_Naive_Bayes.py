from operator import mul
import math
import random
import matplotlib.pyplot as plt
import sys
from datetime import datetime

def naive_Bayes(train_path, test_path, training_data_fraction):
    text_file_data = open(train_path, "r")
    text_file_labels = open(test_path, "r")
    data = text_file_data.readlines()

    len_data = len(data)
    training_data_length = int(training_data_fraction * len_data)
    train_data = data[0:training_data_length]
    labels = text_file_labels.readlines()

    data_preproces = []
    data_dict = {}
    data_dict_id = {}
    data_list_doc = []
    count = 0
    data_dict_wordtoid = {}

    # data_preproces is lis of input statements
    # data_dict is a counter for each word
    # data_dict_id is a map from word_id to word
    # data_dict_wordtoid is a map from word to word_id, opposite of data_dict_d
    # temp_doc_dict is a counter from each word to the number of times the word is present in that particular doc
    # data_list_doc is the list of all temp_doc_dict

    for line in data:
        line = line.split()
        data_preproces.append(line)
        temp_doc_dict = {}
        for word in line:
            if word not in data_dict:
                data_dict[word] = 1
                data_dict_id[count] = word
                data_dict_wordtoid[word] = count
                count = count + 1
            else:
                val = data_dict.get(word)
                data_dict[word] = val + 1
            if word not in temp_doc_dict:
                temp_doc_dict[word] = 1
            else:
                val = temp_doc_dict.get(word)
                temp_doc_dict[word] = val + 1
        data_list_doc.append(temp_doc_dict)
    true_labels = []
    for label in labels:
        true_labels.append(int(label.split()[1]))

    # doc_count_list is a list of all docs with each doc being a counter from every word_id to it's count in the document


    doc_count_list = []
    num_words = len(data_dict_wordtoid)
    # need to change this to full data_list_doc
    for doc in data_list_doc[0:training_data_length]:
        cur_doc_count = [0] * num_words
        for i in range(num_words):
            word = data_dict_id.get(i)
            if word in doc:
                count = doc.get(word)
                cur_doc_count[i] = count
        doc_count_list.append(cur_doc_count)
    priors, word_counts = naive_Bayes_parameters(doc_count_list[0:training_data_length], true_labels[0:training_data_length])


    predict_accurate_count = 0.0
    for i in range(len(true_labels)):
        predicted_label = naive_Bayes_Predict(priors, word_counts, data_dict_wordtoid, data[i].split(), num_words)
        if predicted_label == true_labels[i]:
            predict_accurate_count+=1

    accuracy = predict_accurate_count/len(true_labels)
    return accuracy



# def logistic_regression_test(weights, test_data, test_labels, threshold = 0.5):
#     accurate_count = 0.0
#     for i in range(len(test_data)):
#         predicted_label = get_logistic_regression_label(weights, test_data[i], test_labels[i])
#         if predicted_label == test_labels[i]:
#             accurate_count+=1
#     accuracy = accurate_count/len(test_data)
#
#     print("accuracy is %", accuracy)
#     return accuracy


def naive_Bayes_parameters(doc_count_list, true_labels):
    priors = [0]*2
    #counts = [0]*2

    vocab_len = len(doc_count_list[0])
    priors[1] = sum(true_labels)
    priors[0] = len(doc_count_list) - sum(true_labels)
    word_count = [[0]*2 for _ in range(vocab_len)]
    #data = [[None] * 5 for _ in range(5)]
    for i in range(len(doc_count_list)):
        true_label = true_labels[i]
        #priors[true_label] +=1
        for j in range(len(doc_count_list[i])):
            val = doc_count_list[i][j]
            #counts[true_label]+=val
            if doc_count_list[i][j]!=0:
                word_count[j][true_label] += 1
    return priors, word_count


def naive_Bayes_Predict(priors, word_counts, data_dict_wordtoid, data_point, vocab_len):


    # Assuming class 0 and calculating probability
    prior_count_0 = float(priors[0])
    prior_count_1 = float(priors[1])
    prior_class_0 = prior_count_0/(priors[0]+priors[1])
    prior_class_1 = 1.0 - prior_class_0
    product_0 = 1.0
    product_1 = 1.0
    log_0 = 0.0
    log_1 = 0.0
    for word in data_point:
        word_0 = 0.0
        word_1 = 0.0

        if word not in data_dict_wordtoid:
            word_0 = 1.0/(vocab_len+prior_count_0)
            word_1 = 1.0/(vocab_len+prior_count_1)
            # print("word not is ", word)
        else:
            id = data_dict_wordtoid.get(word)
            # print("word  is ", word)
            # print("vocab_len is", vocab_len)
            # print("total count zero is ", counts[0])
            # print("total count one is ", counts[1])
            # print("count zero is", word_counts[id][0])
            # print("count one is ", word_counts[id][1])
            word_0 = (word_counts[id][0]+1.0)/(vocab_len+prior_count_0)
            word_1 = (word_counts[id][1]+1.0)/(vocab_len+prior_count_1)
            # print("word_0 is ", word_0)
            # print("word_0 is ", word_1)

        log_0 = log_0 + math.log(word_0)
        log_1 = log_1 + math.log(word_1)
        # product_0 = product_0*word_0
        # product_1 = product_1*word_1
    #print("sentence length is ", len(data_point))
    # print("product_0 is ", product_0)
    # print("product_1 is ", product_1)

    # print("likelihood zero is ", product_0)
    # print("likelihood one is ", product_1)
    log_0 = log_0 + math.log(prior_class_0)
    log_1 = log_1 + math.log(prior_class_1)
    #print("log_0 is ", log_0)
    #print("log_1 is ", log_1)
    #denominator = product_0*prior_class_0+product_1*prior_class_1
    # print("denominator is ", denominator)
    #
    # prob_0 = (product_0)*prior_class_0/denominator
    # prob_1 = (product_1)*prior_class_1/denominator
    if log_0 > log_1:
        return 0
    else:
        return 1
    # if prob_0 > prob_1:
    #     return 0
    # else:
    #     return 1



def logistic_regression(train_path, test_path, training_data_fraction, threshold=0.5, learning_rate = 0.001):
    text_file_data = open(train_path, "r")
    text_file_labels = open(test_path, "r")
    data = text_file_data.readlines()

    len_data = len(data)
    training_data_length = int(training_data_fraction*len_data)
    train_data = data[0:training_data_length]
    labels = text_file_labels.readlines()

    data_preproces = []
    data_dict = {}
    data_dict_id = {}
    data_list_doc = []
    count = 0
    data_dict_wordtoid = {}

    # data_preproces is lis of input statements
    # data_dict is a counter for each word
    # data_dict_id is a map from word_id to word
    # data_dict_wordtoid is a map from word to word_id, opposite of data_dict_d
    # temp_doc_dict is a counter from each word to the number of times the word is present in that particular doc
    # data_list_doc is the list of all temp_doc_dict

    for line in data:
        line = line.split()
        data_preproces.append(line)
        temp_doc_dict = {}
        for word in line:
            if word not in data_dict:
                data_dict[word] = 1
                data_dict_id[count] = word
                data_dict_wordtoid[word] = count
                count = count + 1
            else:
                val = data_dict.get(word)
                data_dict[word] = val + 1
            if word not in temp_doc_dict:
                temp_doc_dict[word] = 1
            else:
                val = temp_doc_dict.get(word)
                temp_doc_dict[word] = val + 1
        data_list_doc.append(temp_doc_dict)
    true_labels = []
    for label in labels:
        true_labels.append(int(label.split()[1]))


    # doc_count_list is a list of all docs with each doc being a counter from every word_id to it's count in the document


    doc_count_list = []
    num_words = len(data_dict_wordtoid)
    for doc in data_list_doc:
        cur_doc_count = [0] * num_words
        for i in range(num_words):
            word = data_dict_id.get(i)
            if word in doc:
                count = doc.get(word)
                cur_doc_count[i] = count
        doc_count_list.append(cur_doc_count)

    data_len = len(data_preproces)

    weights = []
    for i in range(num_words+1):
        #weights.append(random.uniform(0.0, 1.0))
        weights.append(0.0)

    for i in range(training_data_length):
        weights = logistic_regression_train(weights, doc_count_list[i], learning_rate, true_labels[i], 0.5)

    predict_accurate_count = 0.0

    for i in range(len(doc_count_list)):
        predicted_label = get_logistic_regression_label(weights, doc_count_list[i], threshold)
        if predicted_label == true_labels[i]:
            predict_accurate_count+=1

    accuracy = predict_accurate_count/len(true_labels)
    return accuracy


def get_logistic_regression_label(weights, datapoint, threshold):
    # testing time, getting the label from the leasrned weights using sigmoid functions
    datapoint.insert(0, 1)
    transpose = getdotproduct(weights, datapoint)
    if transpose > 0:
        return 1
    else:
        return 0
    # if transpose < 0:
    #     regression_val = 1 - (1 / (1 + math.exp(transpose)))
    # else:
    #     regression_val = 1 / (1 + math.exp(-transpose))
    #
    # if regression_val < threshold:
    #     return 0
    # else:
    #     return 1000

    # transpose = -1 * transpose
    # exp = math.exp(transpose)
    # regression_val = 1 / (1 + exp)
    # predicted_label = 0
    # if regression_val >= threshold:
    #     predicted_label = 1
    # return predicted_label


def logistic_regression_train(weights, datapoint, learning_rate, true_label, threshold = 0.5):
    datapoint.insert(0, 1)
    transpose = getdotproduct(weights, datapoint)
    #transpose = transpose
    if transpose < 0:
        regression_val =  1 - (1 / (1 + math.exp(transpose)))
    else:
        regression_val = 1 / (1 + math.exp(-transpose))

    error = true_label - regression_val
    if error == 0:
        return weights
    for i in range(len(weights)):
        weights[i] = weights[i] + learning_rate*(error)*(datapoint[i])
    return weights



def getdotproduct(weights, datapoint):
    sum = 0.0
    for i in range(len(weights)):
        sum = sum + weights[i]*datapoint[i]
    return sum


def main():
    training_fraction_list = [ 0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    logistic_accuracy = []
    naive_Bayes_accuracy = []
    train_path = "/Users/arya/Downloads/5-1/SML/doc-data.txt"
    test_path = "/Users/arya/Downloads/5-1/SML/doc-label.txt"
    print("time is ", str(datetime.now()))
    for training_fraction in training_fraction_list:
        log_accuracy = 0.0
        naive_accuracy = 0.0
        for _ in range(5):
            learning_rate = 0.0001/training_fraction
            print("learning rate is ", learning_rate)
            logistic_accuracy_temp = logistic_regression(train_path, test_path, training_fraction, 0.5, learning_rate)
            print("logistic temp accuracy is ", logistic_accuracy_temp)
            naive_Bayes_accuracy_temp = naive_Bayes(train_path, test_path, training_fraction)
            log_accuracy+=logistic_accuracy_temp
            naive_accuracy+=naive_Bayes_accuracy_temp

        log_accuracy = log_accuracy/5
        naive_accuracy = naive_accuracy/5
        logistic_accuracy.append(log_accuracy)
        naive_Bayes_accuracy.append(naive_accuracy)
        print("time is ", str(datetime.now()))
        print("training fraction is %", training_fraction)
        print("naive Bayes accuracy is %", naive_accuracy)
        print("logistic_accuracy is %", log_accuracy)
    plt.plot(training_fraction_list, naive_Bayes_accuracy, 'r', label="naive")
    plt.plot(training_fraction_list, logistic_accuracy, 'g', label="logistic-regression")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


if __name__ == "__main__":
        main()


#
# text_file_data = open("/Users/arya/Downloads/5-1/SML/doc-data.txt", "r")
# text_file_labels = open("/Users/arya/Downloads/5-1/SML/doc-label.txt", "r")
# data = text_file_data.readlines()
#
# train_data = data[0:]
# labels = text_file_labels.readlines()
#
# data_preproces = []
# data_dict = {}
# data_dict_id = {}
# data_list_doc = []
# count = 0
# data_dict_wordtoid = {}
#
# for line in data:
#     line = line.split()
#     data_preproces.append(line)
#     temp_doc_dict = {}
#     for word in line:
#         if word not in data_dict:
#             data_dict[word] = 1
#             data_dict_id[count] = word
#             data_dict_wordtoid[word] = count
#             count = count+1
#         else:
#             val = data_dict.get(word)
#             data_dict[word] = val+1
#         if word not in temp_doc_dict:
#             temp_doc_dict[word] = 1
#         else:
#             val = temp_doc_dict.get(word)
#             temp_doc_dict[word] = val+1
#     data_list_doc.append(temp_doc_dict)
# true_labels = []
# for label in labels:
#     true_labels.append(label.split()[1])
#
# doc_count_list = []
# # Changing from 4143 to len(data)
# num_docs = len(data)
# # Changing from 43624 to len(data_dict_wordtoid)
# num_words = len(data_dict_wordtoid)
# for doc in data_list_doc:
#     cur_doc_count = [0]*num_words
#     for i in range(num_words):
#         word = data_dict_id.get(i)
#         if word in doc:
#             count = doc.get(word)
#             cur_doc_count[i] = count
#     doc_count_list.append(cur_doc_count)
#
#
#
# data_len = len(data_preproces)
#
# weights = []
# for i in range(43625):
#     weights.append(random.uniform(0.0, 1.0))
#
# for i in range(len(doc_count_list)):
#     weights = logistic_regression(weights, doc_count_list[i], 0.001, true_labels[i], 0.5)
#
#
# priors, counts, word_counts = naive_Bayes(doc_count_list, true_labels)
#
# predicted_label = naive_Bayes_Predict(priors, counts, word_counts, data_dict_wordtoid, data_point)