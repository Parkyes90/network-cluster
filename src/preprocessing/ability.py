import csv
import json
import os
from collections import defaultdict

from colour import Color
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
import pandas as pd
from plotly import graph_objs as go
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import font_manager

from src.config.settings import ABILITY_DIR
from src.preprocessing.cluster import min_max_normalize

okt = Okt()

stopwords = {
    "의",
    "가",
    "이",
    "은",
    "들",
    "는",
    "좀",
    "잘",
    "걍",
    "과",
    "도",
    "를",
    "으로",
    "자",
    "에",
    "와",
    "한",
    "하다",
    "및",
    "함",
    "긍",
    "끝",
    "수",
}

singles = {
    "줄",
    "꿈",
    "답",
    "힘",
    "관",
    "말",
    "법",
    "글",
    "일",
    "삶",
    "팀",
    "꽃",
    "집",
    "개",
    "끝",
    "왕",
    "탓",
    "책",
    "시",
    "순",
    "옷",
    "폭",
}


def to_csv():
    df = pd.read_excel(
        os.path.join(ABILITY_DIR, "input", "ability.xlsx")
    ).dropna(axis=0)
    df.sentence = df.sentence.str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
    tokens = []
    singles = set()
    for s in df.sentence:
        temp_x = okt.nouns(s)
        temp = []
        for t in temp_x:
            if len(t) > 1:
                temp.append(t)
            if t in singles:
                temp.append(t)
        tokens.append(temp)
    new_df = pd.DataFrame(
        {"type": df.type.to_list(), "words": [" ".join(t) for t in tokens]}
    )
    new_df.to_csv(os.path.join(ABILITY_DIR, "output", "words.csv"))


def modeling(min_count=2):
    df = pd.read_csv(os.path.join(ABILITY_DIR, "output", "words.csv")).dropna(
        axis=0
    )
    df.words = df.words.str.split(" ")
    word_count_map = defaultdict(int)
    for words in df.words:
        for word in words:
            word_count_map[word] += 1
    word_count_df = pd.DataFrame(
        {"word": word_count_map.keys(), "count": word_count_map.values()}
    )
    train_words = []
    filtered_word_df = word_count_df.loc[word_count_df["count"] >= min_count]
    filtered_set = set(filtered_word_df.word.to_list())
    for words in df.words:
        temp = []
        for word in words:
            if word in filtered_set:
                temp.append(word)
        train_words.append(temp)
    model = Word2Vec(
        train_words, size=3, window=5, min_count=min_count, workers=4, sg=0
    )
    vectors = [["word", "x", "y", "z"]]
    networks = [["source", *[word for word in filtered_set]]]
    for word in filtered_set:
        vectors.append([word, *model.wv[word].tolist()])
        temp = [word]
        for w in filtered_set:
            if word != w:
                temp.append(model.wv.similarity(word, w))
            else:
                temp.append(0)
        networks.append(temp)
    with open(os.path.join(ABILITY_DIR, "output", "vectors.csv"), "w") as f:
        w = csv.writer(f)
        w.writerows(vectors)
    with open(os.path.join(ABILITY_DIR, "output", "networks.csv"), "w") as f:
        w = csv.writer(f)
        w.writerows(networks)


def draw_vectors():
    df = pd.read_csv(
        os.path.join(ABILITY_DIR, "output", "vectors.csv")
    ).dropna(axis=0)
    data = go.Scatter3d(
        x=df.x,
        y=df.y,
        z=df.z,
        text=df.word,
        mode="markers+text",
        marker={
            "size": 12,
            "color": df.z,
            "colorscale": "Viridis",
            "opacity": 0.7,
        },
    )
    fig = go.Figure(data=[data])
    fig.write_image("vectors.png", width=2400, height=2400, scale=4)
    fig.write_image("vectors.svg", width=2400, height=2400, scale=4)
    # fig.show()
    # trace = graph_objs.Scatter3d(
    #     x=df.x.to_list(),  # <-- Put your data instead
    #     y=df.y.to_list(),  # <-- Put your data instead
    #     z=df.z.to_list(),  # <-- Put your data instead
    #     mode="markers",
    #     marker={"size": 10, "opacity": 0.8},
    # )
    # layout = graph_objs.Layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})
    #
    # data = [trace]
    #
    # plot_figure = graph_objs.Figure(data=data, layout=layout)


def write_similarity_csv():
    df = pd.read_csv(
        os.path.join(ABILITY_DIR, "output", "networks.csv")
    ).dropna(axis=0)
    rank = len(df) // 10
    ret = [df.columns.tolist()]
    for idx, row in df.iterrows():
        word, *r = row.tolist()
        row_df = pd.DataFrame({"value": [float(i) for i in r]})
        sort = row_df.value.sort_values(ascending=False).head(rank)
        temp = []
        indexes = set(sort.index.to_list())
        for index, _ in row_df.iterrows():
            if index in indexes:
                temp.append(1)
            else:
                temp.append(0)
        ret.append([word, *temp])
    with open(
        os.path.join(ABILITY_DIR, "output", "network_for_draw.csv"), "w"
    ) as f:
        w = csv.writer(f)
        w.writerows(ret)


def draw_network():
    with open(
        os.path.join(ABILITY_DIR, "output", "network_for_draw.csv")
    ) as f:
        reader = list(csv.reader(f))
    with open(os.path.join(ABILITY_DIR, "output", "network_helper.json")) as f:
        helper = json.load(f)
    header = reader[0]
    G = nx.Graph()
    labels = {}
    edge_colors = []
    node_sizes = []
    node_colors = []
    for row in reader[1:]:
        # print(row[0])
        G.add_node(row[0])
        node_sizes.append(float(helper[row[0]]["size"]))
        node_colors.append(helper[row[0]]["color"])
        labels[row[0]] = row[0]
    for row in reader[1:]:
        source = row[0]
        for index, r in enumerate(row[1:], 1):
            if r == "1":
                G.add_edge(source, header[index])
                edge_colors.append(helper[header[index]]["color"])
    pos = nx.spring_layout(G)
    plt.figure(figsize=(24, 24))
    ax = plt.gca()
    options = {
        "linewidths": 0,
        "alpha": 0.8,
        "node_size": node_sizes,
        "node_color": node_colors,
    }
    for idx, edge in enumerate(G.edges()):
        source, target = edge
        rad = 0.4
        arrowprops = dict(
            linewidth=0.2,
            arrowstyle="-",
            color=helper[target]["color"],
            connectionstyle=f"arc3,rad={rad}",
            alpha=0.6,
        )
        ax.annotate(
            "", xy=pos[source], xytext=pos[target], arrowprops=arrowprops
        )

    nx.draw_networkx_nodes(G, pos, **options)
    nx.draw_networkx_labels(G, pos, labels=labels, font_family="NanumGothic")

    plt.axis("off")

    plt.savefig("network.svg", format="svg", dpi=400)
    plt.savefig("network.png", format="png", dpi=400)
    # plt.show()


def write_count():
    with open(
        os.path.join(ABILITY_DIR, "output", "network_for_draw.csv")
    ) as f:
        reader = list(csv.reader(f))
    header = reader[0]
    count_map = defaultdict(int)
    for row in reader[1:]:
        _, *targets = row
        for idx, target in enumerate(targets, 1):
            if target == "1":
                count_map[header[idx]] += 1
    normalized = min_max_normalize(count_map.values())
    size_map = defaultdict(int)
    for idx, key in enumerate(count_map.keys()):
        size_map[key] = 30 + 150 * normalized[idx]
    blue = Color("blue")
    green = Color("green")
    colors = list(blue.range_to(green, len(header) - 1))
    color_map = {}
    sorted_size_map = sorted(
        size_map.items(), key=lambda x: x[1], reverse=True
    )
    for idx, key_value in enumerate(sorted_size_map):
        k, _ = key_value
        color_map[k] = colors[idx].get_hex()
    ret = {}
    for idx, row in enumerate(reader[1:]):
        word = row[0]
        color = color_map[word]
        size = size_map[word]
        ret[word] = {"color": color, "size": size, "count": count_map[word]}
    with open(
        os.path.join(ABILITY_DIR, "output", "network_helper.json"), "w"
    ) as f:
        json.dump(ret, f)


if __name__ == "__main__":
    # to_csv()
    # modeling(10)
    # draw_vectors()
    # write_similarity_csv()
    # write_count()
    draw_network()
