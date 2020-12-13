import csv
import os
import sys

from src.config.settings import DATA_DIR, OUTPUTS_DIR
from src.vectors.pre.morphs import (
    get_cluster_data,
    get_data,
    KEYWORD_PATH,
    get_raw_content,
)

csv.field_size_limit(sys.maxsize)


def main():
    raw_data = get_raw_content()
    cluster = get_cluster_data()
    morphs = get_data(KEYWORD_PATH)
    ret = {}
    for c in cluster:
        index, *_, cluster, _ = c
        context = raw_data[int(index)][4]
        if cluster not in ret:
            ret[cluster] = {}
        for k, v in morphs.items():
            if k not in ret[cluster]:
                ret[cluster][k] = {}
            for keyword in v:
                count = context.count(keyword)
                if keyword not in ret[cluster][k]:
                    ret[cluster][k][keyword] = 0
                ret[cluster][k][keyword] += count
    header = ["클러스터", "미래벡터", "단어", "카운트"]
    data = [header]

    for cluster, values in ret.items():
        for v_name, keywords in values.items():
            for keyword, count in keywords.items():
                data.append([cluster, v_name, keyword, count])
    with open(
        os.path.join(OUTPUTS_DIR, "cluster-keyword-count-map.csv"), "w"
    ) as f:
        writer = csv.writer(f)
        writer.writerows(data)
    with open(
        os.path.join(OUTPUTS_DIR, "cluster-keyword-count-map-euckr.csv"),
        "w",
        encoding="cp949",
    ) as f:
        writer = csv.writer(f)
        writer.writerows(data)
    # print(cluster, morphs)


if __name__ == "__main__":
    main()
