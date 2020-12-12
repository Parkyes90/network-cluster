import csv
import os
import sys
from collections import defaultdict

from pyvis.network import Network

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.settings import OUTPUTS_DIR
from src.preprocessing.cluster import min_max_normalize
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import pandas as pd

csv.field_size_limit(sys.maxsize)


def draw_chart_with_networkx():
    df = pd.read_csv(os.path.join(OUTPUTS_DIR, "for-network-draw.csv"))
    print(df)


def draw_chart():
    net = Network("1600px", "1600px", bgcolor="#22222")
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
            index, index, color=color, size=20 + (normalized[int(index)] * 20),
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
    w.writerow(["index", "title", "cluster", *list(range(1, len(reader) + 1))])
    for index, row in enumerate(listed):
        w.writerow([index + 1, reader[index][3], reader[index][5], *row])
    file.close()


def write_network():
    with open(os.path.join(OUTPUTS_DIR, "similarity.csv")) as f:
        reader = list(csv.reader(f))
    weights = [["index", "title", "cluster", "connected_count"]]
    count_map = defaultdict(int)
    for_draw = [
        [
            "index",
            "source",
            "destination",
            "source_cluster",
            "destination_cluster",
            "source_connected_count",
        ]
    ]
    d_idx = 1
    cluster_map = {}
    for row in reader[1:]:
        index, title, cluster, *normalized = row
        cluster_map[index] = cluster
    for row in reader[1:]:
        index, title, cluster, *normalized = row
        for target_index, value in enumerate(normalized, 1):
            if value != "0" and index != str(target_index):
                count_map[str(target_index)] += 1
    for row in reader[1:]:
        index, title, cluster, *normalized = row
        for target_index, value in enumerate(normalized, 1):
            if value != "0" and index != str(target_index):
                r = [
                    d_idx,
                    index,
                    target_index,
                    cluster,
                    cluster_map[str(target_index)],
                    count_map[str(index)],
                ]
                for_draw.append(r)
                d_idx += 1

    for row in reader[1:]:
        index, title, cluster, *normalized = row
        weights.append([index, title, cluster, count_map[index]])
    with open(os.path.join(OUTPUTS_DIR, "for-network-draw.csv"), "w") as f:
        w = csv.writer(f)
        w.writerows(for_draw)
    with open(os.path.join(OUTPUTS_DIR, "network-detail-draw.csv"), "w") as f:
        w = csv.writer(f)
        w.writerows(weights)


def main():
    # write_similarity()
    # write_network()
    draw_chart_with_networkx()


if __name__ == "__main__":
    main()
