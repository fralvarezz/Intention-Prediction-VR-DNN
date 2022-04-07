import csv

fname = "test.csv"
filters = [""]

#TODO: Filter out unnecessary fields

with open(file=fname, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)