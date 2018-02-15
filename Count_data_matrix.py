import csv
#
# with open('/Users/arya/Downloads/5-1/SML/doc-count_init.txt', 'r') as in_file:
#     stripped = (line.strip() for line in in_file)
#     lines = (line.split(",") for line in stripped if line)
#     with open('/Users/arya/Downloads/5-1/SML/count_log.csv', 'w') as out_file:
#         writer = csv.writer(out_file)
#         writer.writerow(('title', 'intro'))
#         writer.writerows(lines)


def getdatamatrix(train_path, test_path):
    text_file_data = open(train_path, "r")
    text_file_labels = open(test_path, "r")
    data = text_file_data.readlines()


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
    h = len(doc_count_list[0])
    w = len(doc_count_list)
    print("w is ", w)
    print("h is", h)
    data_matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(len(doc_count_list)):
        for j in range(len(doc_count_list[i])):
            data_matrix[j][i] = doc_count_list[i][j]
    with open("/Users/arya/Downloads/5-1/SML/doc-count_recent2.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(data_matrix)


def main():
    train_path = "/Users/arya/Downloads/5-1/SML/doc-data.txt"
    test_path = "/Users/arya/Downloads/5-1/SML/doc-label.txt"
    getdatamatrix(train_path, test_path)


if __name__ == "__main__":
    main()