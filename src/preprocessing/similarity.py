import csv
import os
import sys
from collections import defaultdict

from pyvis.network import Network

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.settings import OUTPUTS_DIR
from src.preprocessing.cluster import min_max_normalize

csv.field_size_limit(sys.maxsize)


def draw_chart():
    net = Network("1000px", "1000px", bgcolor="#22222")
    colors = {"0": "#3f51b5", "1": "#f44336", "2": "#651fff", "3": "#00e5ff"}
    with open(os.path.join(OUTPUTS_DIR, "similarity.csv")) as f:
        reader = list(csv.reader(f))
        reader.pop(0)
    count_map = defaultdict(int)
    sizes = [0] * (len(reader) + 1)
    for row in reader:
        index, title, cluster, *rows = row
        for idx, r in enumerate(rows, 1):
            if index != str(idx) and r != "0":
                count_map[idx] += 1
    for index, value in count_map.items():
        sizes[index] = value
    normalized = min_max_normalize(sizes)
    for row in reader:
        index, title, cluster, *rows = row
        color = colors[cluster]
        net.add_node(
            index,
            "".join(title.split(" ")[:2]),
            color=color,
            size=20 + (normalized[int(index)] * 20),
        )
    for row in reader:
        index, title, cluster, *rows = row
        for idx, r in enumerate(rows, 1):
            if index != str(idx) and r != "0":
                end_color = colors[reader[idx - 1][2]]
                net.add_edge(index, str(idx), color=end_color, weight=float(r))
    net.set_edge_smooth("dynamic")
    net.show_buttons(filter_=["physics"])
    net.show("nx.html")


def write_similarity():
    docs = []
    vect = TfidfVectorizer()
    with open(os.path.join(OUTPUTS_DIR, "cluster-docs.csv")) as f:
        reader = list(csv.reader(f))
        reader.pop(0)
        for row in reader:
            *remain, context, cluster = row
            docs.append(context)
    tfidf = vect.fit_transform(docs)
    matrix = (tfidf * tfidf.T).A
    listed = matrix.tolist()
    for row in listed:
        df = pd.DataFrame({"index": range(len(row)), "value": row})
        sort = df.sort_values(by="value", ascending=False).head(6)
        top5 = set(sort.index.tolist())
        for i in range(len(row)):
            if i not in top5:
                row[i] = 0
    file = open(os.path.join(OUTPUTS_DIR, "similarity.csv"), "w")
    w = csv.writer(file)
    w.writerow(
        ["index", "title", "category", *list(range(1, len(reader) + 1))]
    )
    for index, row in enumerate(listed):
        w.writerow([index + 1, reader[index][3], reader[index][5], *row])
    file.close()


def main():
    # write_similarity()
    draw_chart()


if __name__ == "__main__":
    main()
