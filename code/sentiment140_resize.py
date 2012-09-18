import csv

N = 1000
fi = csv.reader(open("Sentiment140/training.1600000.processed.noemoticon.csv"),
                delimiter=",", quotechar="\"")
fo = csv.writer(open("Sentiment140/training.%d.processed.noemoticon.csv" % (N*2), "w"),
                delimiter=",", quotechar="\"")

counts = [0]*5
for row in fi:
    label = int(row[0])    
    if counts[label] < N:
        counts[label] += 1
        fo.writerow(row)
print counts
