import csv
import os
import sys
from collections import defaultdict

from pyvis.network import Network
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.settings import OUTPUTS_DIR
from src.preprocessing.cluster import min_max_normalize
import pandas as pd
import networkx as nx

csv.field_size_limit(sys.maxsize)


def draw_network():
    color_map = {
        "3": "#F2C34B",
        "2": "#48A985",
        "1": "#3282F4",
        "0": "#C14F49",
    }
    node_shape_map = {
        "3": "^",
        "2": "o",
        "1": "v",
        "0": "d",
    }

    cluster_map = {}
    edge_colors = []
    with open(os.path.join(OUTPUTS_DIR, "for-network-draw.csv")) as f:
        reader = list(csv.reader(f))
    nodes = {(r[1], r[3], r[5]) for r in reader[1:]}
    G = nx.Graph()
    normalized = []
    nodes = list(nodes)
    for _, _, source_connected_count in nodes:
        normalized.append(int(source_connected_count))
    normalized = min_max_normalize(normalized)
    for idx, remain in enumerate(nodes):
        node, cluster, _ = remain
        cluster_map[node] = cluster
        G.add_node(
            node,
            s=node_shape_map[cluster],
            size=30 + 150 * normalized[idx],
            color=color_map[cluster],
        )
    for row in reader[1:]:
        (
            index,
            source,
            destination,
            source_cluster,
            destination_cluster,
            source_connected_count,
        ) = row
        G.add_edge(source, destination)
        edge_colors.append(color_map[destination_cluster])
    options = {
        "linewidths": 0,
        "alpha": 0.8,
    }
    pos = nx.spring_layout(G, k=0.8, iterations=20)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(16, 16))
    ax = plt.gca()
    for idx, edge in enumerate(G.edges()):
        source, target = edge
        rad = 0.4
        arrowprops = dict(
            linewidth=0.1,
            arrowstyle="-",
            color=color_map[cluster_map[source]],
            connectionstyle=f"arc3,rad={rad}",
            alpha=0.5,
        )
        ax.annotate(
            "", xy=pos[source], xytext=pos[target], arrowprops=arrowprops
        )
    nodeShapes = set((aShape[1]["s"] for aShape in G.nodes(data=True)))
    for aShape in nodeShapes:
        node_list = []
        colors = []
        sizes = []
        for sNode in filter(lambda x: x[1]["s"] == aShape, G.nodes(data=True)):
            colors.append(sNode[1]["color"])
            sizes.append(sNode[1]["size"])
            node_list.append(sNode[0])
        nx.draw_networkx_nodes(
            G,
            pos,
            node_shape=aShape,
            nodelist=node_list,
            node_color=colors,
            node_size=sizes,
            **options,
        )
    # nx.draw_networkx_nodes(G, pos, **options)
    #
    plt.axis("off")
    plt.savefig("network.svg", format="svg", dpi=400)
    plt.savefig("network.png", format="png", dpi=400)
    plt.show()


def draw_chart():
    net = Network("1600px", "1600px", bgcolor="#22222")
    colors = {
        "3": "#ffee33",
        "2": "#00a152",
        "1": "#2979ff",
        "0": "#d500f9",
    }
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
            index,
            color=color,
            size=20 + (normalized[int(index)] * 100),
        )
    for row in reader:
        index, title, cluster, *rows = row
        for idx, r in enumerate(rows, 1):
            if index != str(idx) and r != "0":
                end_color = colors[cluster]
                net.add_edge(index, str(idx), color=end_color, weight=float(r))
    net.set_edge_smooth("dynamic")
    net.show_buttons(filter_=["physics"])
    # net.show("nx.html")
    net.show("nx.html")


def write_similarity():
    docs = []
    clusters = []
    vect = TfidfVectorizer()
    with open(os.path.join(OUTPUTS_DIR, "cluster-docs.csv")) as f:
        reader = list(csv.reader(f))
        reader.pop(0)
        for row in reader:
            *remain, context, cluster, cluster_distance = row
            clusters.append(cluster)
            docs.append(context)
    tfidf = vect.fit_transform(docs)
    matrix = (tfidf * tfidf.T).A
    listed = matrix.tolist()
    raw = []
    for idx, row in enumerate(listed):
        raw.append([1 - r for r in row])
        current_cluster = clusters[idx]
        df = pd.DataFrame(
            {"index": range(len(row)), "value": row, "cluster": clusters}
        )
        sort = (
            df.loc[df.cluster == current_cluster]
            .sort_values(by="value", ascending=False)
            .head(5)
        )
        top3 = sort.index.tolist()
        sort = (
            df.loc[df.cluster != current_cluster]
            .sort_values(by="value", ascending=False)
            .head(1)
        )
        top2 = sort.index.tolist()
        top5 = set(top3 + top2)
        for i in range(len(row)):
            if i not in top5:
                row[i] = 0
            else:
                row[i] = 1

    file = open(os.path.join(OUTPUTS_DIR, "similarity.csv"), "w")
    w = csv.writer(file)
    w.writerow(["index", "title", "cluster", *list(range(1, len(reader) + 1))])
    for index, row in enumerate(listed):
        w.writerow([index + 1, reader[index][3], reader[index][5], *row])

    file.close()

    file = open(os.path.join(OUTPUTS_DIR, "similarity-raw.csv"), "w")
    w = csv.writer(file)
    w.writerow(["index", "title", "cluster", *list(range(1, len(reader) + 1))])
    for index, row in enumerate(raw):
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
    # draw_chart()
    draw_network()


if __name__ == "__main__":
    main()
