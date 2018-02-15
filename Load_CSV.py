import csv

with open('/Users/arya/Downloads/5-1/SML/doc-count_recent.csv', 'rb') as f:
    reader = csv.reader(f)
    data_as_list = list(reader)

print data_as_list