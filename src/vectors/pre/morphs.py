import csv
import os
from itertools import combinations
import numpy as np
import pandas as pd
from bokeh.core.property.dataspec import value
from bokeh.io import show, export_png
from bokeh.io.export import export_svg
from bokeh.models import ColumnDataSource, HoverTool, LinearAxis, Range1d
from bokeh.plotting import figure
from selenium import webdriver

from src.config.settings import DATA_DIR, OUTPUTS_DIR, BASE_DIR
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
    keys = list(morphs.keys())
    vector_header = []
    for i in range(0, len(keys), 2):
        vector_header.append(f"{keys[i]} | {keys[i + 1]}")
    header = ["index", "cate", "year", "title", "cluster", *vector_header]
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
        merge = []
        for i in range(0, len(temp), 2):
            merge.append(temp[i + 1] - temp[i])

        ret.append([*remain, cluster, *merge])
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
        ret.append(f"{keys[i]} | {keys[i + 1]}")
    comb = list(combinations(ret, 2))
    with open(os.path.join(OUTPUTS_DIR, "future_comb.csv"), "w") as f:
        w = csv.writer(f)
        w.writerows([("가로축", "세로축"), *enumerate(comb, 1)])


def draw_vectors():
    driver = webdriver.Chrome(os.path.join(BASE_DIR, "chromedriver"))

    df = pd.read_csv(
        os.path.join(OUTPUTS_DIR, "normalized_future_vectors.csv")
    )
    comb = list(combinations(range(5, len(df.columns)), 2))

    for idx, coord in enumerate(comb, 1):
        x, y = coord
        X = df[df.columns[x]].to_list()
        Y = df[df.columns[y]].to_list()

        tsne_df = pd.DataFrame(
            zip(X, Y), index=range(len(X)), columns=["x_coord", "y_coord"]
        )
        tsne_df["title"] = df["title"].to_list()
        tsne_df["cluster_no"] = df["cluster"].to_list()
        colormap = {0: "#ffee33", 1: "#00a152", 2: "#2979ff", 3: "#d500f9"}
        colors = [colormap[x] for x in tsne_df["cluster_no"]]
        tsne_df["color"] = colors
        plot_data = ColumnDataSource(data=tsne_df.to_dict(orient="list"))
        tsne_plot = figure(
            # title='TSNE Twitter BIO Embeddings',
            plot_width=400,
            plot_height=400,
            active_scroll="wheel_zoom",
            output_backend="webgl",
        )
        tsne_plot.add_tools(HoverTool(tooltips="@title"))
        tsne_plot.circle(
            source=plot_data,
            x="x_coord",
            y="y_coord",
            line_alpha=0.9,
            fill_alpha=0.8,
            size=10,
            fill_color="color",
            line_color="color",
        )
        start_x, end_x = df.columns[x].split("|")
        start_y, end_y = df.columns[y].split("|")
        start_x = start_x.split(":")[1].strip()
        end_x = end_x.split(":")[1].strip()
        start_y = start_y.split(":")[1].strip()
        end_y = end_y.split(":")[1].strip()
        tsne_plot.title.text_font_size = value("16pt")
        tsne_plot.xaxis.visible = True
        tsne_plot.yaxis.visible = True
        tsne_plot.xaxis.axis_label = f"<--- {start_x} | {end_x} --->"
        tsne_plot.yaxis.axis_label = f"<--- {start_y} | {end_y} --->"
        tsne_plot.name = str(idx)
        # tsne_plot.grid.grid_line_color = None
        # tsne_plot.outline_line_color = None
        export_svg(tsne_plot, filename=f"{idx}.svg", webdriver=driver)
        # show(tsne_plot)


def main():
    # morphs = get_data(MORPHS_PATH)
    # clusters = get_cluster_data()
    # export_vectors(morphs, clusters)
    # export_normalized_future_vectors()
    # export_normalized_future_cluster_vectors()
    # export_comb(morphs)
    draw_vectors()


if __name__ == "__main__":
    main()
