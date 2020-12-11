import csv
import os
from itertools import combinations
import numpy as np

from src.config.settings import DATA_DIR, OUTPUTS_DIR
from src.preprocessing.cluster import min_max_normalize

format_string = "{}. {}_result.csv"

MORPHS_PATH = os.path.join(DATA_DIR, "morphs")
FUTURES_PATH = os.path.join(DATA_DIR, "futures")


def standard(arr):
    data = np.array(arr)
    std_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    return std_data


def get_cluster_data():
    with open(os.path.join(OUTPUTS_DIR, "cluster-docs.csv")) as f:
        return list(csv.reader(f))[1:]


def get_data(path):
    ret = {}
    vectors = []
    with open(os.path.join(FUTURES_PATH, "future_vector.csv")) as f:
        for vector in list(csv.reader(f))[1:]:
            merged = [f"{vector[0].strip()}: {v.strip()}" for v in vector]
            vectors += merged[1:]
    files = os.listdir(path)
    files = [
        file for file in files if all(["euckr" not in file, "공통" not in file])
    ]
    files.sort()
    for idx, filename in enumerate(files):
        with open(os.path.join(path, filename)) as f:
            reader = list(csv.reader(f))[1:]
            ret[vectors[idx].strip()] = reader
    return ret


def export_vectors(morphs, cluster_data):
    header = ["index", "cate", "year", "title", "cluster", *morphs.keys()]
    ret = [header]
    for item in cluster_data:
        *remain, context, cluster = item
        splited = set(context.split(" "))
        temp = []
        for key, value in morphs.items():
            total = 0
            keywords = set(item[0] for item in value)
            for k in splited:
                if k in keywords:
                    total += 1
            temp.append(total)

        ret.append([*remain, cluster, *temp])

    with open(os.path.join(OUTPUTS_DIR, "future_vectors_raw.csv"), "w") as f:
        reader = csv.writer(f)
        reader.writerows(ret)


def export_normalized_future_vectors():

    with open(os.path.join(OUTPUTS_DIR, "future_vectors_raw.csv")) as f:
        reader = list(csv.reader(f))
    ret = [[] for _ in range(len(reader[0]) - 5)]
    for row in reader[1:]:
        _, _, _, _, _, *count = row
        for idx, v in enumerate(count):
            ret[idx].append(int(v))
    normals = []
    for r in ret:
        temp = min_max_normalize(r)
        normals.append([(t - 0.5) * 2 for t in temp])
    data = [reader[0]]
    for idx, row in enumerate(reader[1:]):
        temp = []
        for i in range(len(row) - 5):
            temp.append(normals[i][idx])
        data.append(row[:5] + temp)
    with open(
        os.path.join(OUTPUTS_DIR, "normalized_future_vectors.csv"), "w"
    ) as f:
        reader = csv.writer(f)
        reader.writerows(data)


def export_normalized_future_cluster_vectors():
    cluster_map = {}
    with open(os.path.join(OUTPUTS_DIR, "normalized_future_vectors.csv")) as f:
        reader = list(csv.reader(f))
    for row in reader[1:]:
        if row[4] not in cluster_map:
            cluster_map[row[4]] = {}
        for idx, value in enumerate(row[5:]):
            if idx not in cluster_map[row[4]]:
                cluster_map[row[4]][idx] = []
            cluster_map[row[4]][idx].append(float(value))
    keys = list(cluster_map.keys())
    ret = [[key] for key in keys]
    for idx in range(len(reader[0]) - 5):
        normals = []
        for value in cluster_map.values():
            normals.append(sum(value[idx]) / len(value[idx]))
        normals = min_max_normalize(normals)
        normals = [(n - 0.5) * 2 for n in normals]
        for i, v in enumerate(normals):
            ret[i].append(v)
    ret.insert(0, ["cluster", *reader[0][5:]])
    with open(
        os.path.join(OUTPUTS_DIR, "normalized_future_cluster_vectors.csv"), "w"
    ) as f:
        w = csv.writer(f)
        w.writerows(ret)


def export_comb(morphs):
    ret = []
    keys = list(morphs.keys())
    for i in range(0, len(keys), 2):
        ret.append(f"{keys[i]}|{keys[i + 1]}")
    comb = list(combinations(ret, 2))
    with open(os.path.join(OUTPUTS_DIR, "future_comb.csv"), "w") as f:
        w = csv.writer(f)
        w.writerows([("가로축", "세로축"), *comb])


def main():
    morphs = get_data(MORPHS_PATH)
    # clusters = get_cluster_data()
    # export_vectors(morphs, clusters)
    # export_normalized_future_vectors()
    # export_normalized_future_cluster_vectors()
    export_comb(morphs)


if __name__ == "__main__":
    main()
