import os
import pandas as pd
import numpy as np
from bokeh.io.export import export_svg, export_png
from gensim.models import Word2Vec
from selenium import webdriver
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from src.config.settings import BASE_DIR, OUTPUTS_DIR
import os.path
import pickle
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource, value


def min_max_normalize(lst):
    normalized = []

    for v in lst:
        try:
            normalized_num = (v - min(lst)) / (max(lst) - min(lst))
        except ZeroDivisionError:
            normalized_num = 0

        normalized.append(normalized_num)

    return normalized


model = Word2Vec.load(
    os.path.join(BASE_DIR, "data", "downloads", "embedding.save")
)


def get_tokens(context):
    return [c for c in context.split(" ") if len(c) > 1]


def get_sentence_mean_vector(morphs):
    vector = []
    for i in morphs:
        try:
            vector.append(model.wv[i])
        except KeyError:
            pass
    try:
        return np.mean(vector, axis=0)
    except IndexError:
        pass


def draw_chart(df):
    driver = webdriver.Chrome(os.path.join(BASE_DIR, "chromedriver"))
    X = df["wv"].to_list()
    y = df["cluster"].to_list()
    tsne_filepath = "tsne3000.pkl"

    if not os.path.exists(tsne_filepath):
        tsne = TSNE(random_state=42)
        tsne_points = tsne.fit_transform(X)
        with open(tsne_filepath, "wb+") as f:
            pickle.dump(tsne_points, f)
    else:  # Cache Hits!
        with open(tsne_filepath, "rb") as f:
            tsne_points = pickle.load(f)
    tsne_df = pd.DataFrame(
        tsne_points, index=range(len(X)), columns=["x_coord", "y_coord"]
    )

    tsne_df["title"] = df["title"].to_list()
    tsne_df["tokens_len"] = df["tokens_len"].to_list()
    tsne_df["cluster_no"] = y
    colormap = {0: "#ffee33", 1: "#00a152", 2: "#2979ff", 3: "#d500f9"}
    colors = [colormap[x] for x in tsne_df["cluster_no"]]
    tsne_df["color"] = colors
    normalized = min_max_normalize(tsne_df.tokens_len.to_list())
    tsne_df["radius"] = [5 + x * 10 for x in normalized]
    plot_data = ColumnDataSource(data=tsne_df.to_dict(orient="list"))
    tsne_plot = figure(
        # title='TSNE Twitter BIO Embeddings',
        plot_width=1200,
        plot_height=1200,
        active_scroll="wheel_zoom",
        output_backend="svg",
    )
    tsne_plot.add_tools(HoverTool(tooltips="@title"))
    tsne_plot.circle(
        source=plot_data,
        x="x_coord",
        y="y_coord",
        line_alpha=0.6,
        fill_alpha=0.6,
        size="radius",
        fill_color="color",
        line_color="color",
    )
    tsne_plot.title.text_font_size = value("16pt")
    tsne_plot.xaxis.visible = True
    tsne_plot.yaxis.visible = True
    tsne_plot.background_fill_color = None
    tsne_plot.border_fill_color = None
    tsne_plot.grid.grid_line_color = None
    tsne_plot.outline_line_color = None
    # tsne_plot.grid.grid_line_color = None
    # tsne_plot.outline_line_color = None
    show(tsne_plot)
    tsne_plot.toolbar.logo = None
    tsne_plot.toolbar_location = None
    export_svg(
        tsne_plot, filename=f"cluster.svg", webdriver=driver,
    )
    export_png(
        tsne_plot, filename=f"cluster.png", webdriver=driver,
    )


def read_word_vector_docs():
    df = pd.read_csv(
        os.path.join(OUTPUTS_DIR, "filtered-word-vector-docs.csv")
    )
    df["tokens"] = df["context"].map(get_tokens)
    df["tokens_len"] = df["tokens"].map(len)
    df["wv"] = df["tokens"].map(get_sentence_mean_vector)
    word_vectors = df.wv.to_list()
    num_clusters = 4
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)
    df["cluster"] = idx
    return df


def main():
    df = read_word_vector_docs()
    draw_chart(df)
    del df["wv"]
    del df["tokens_len"]
    del df["tokens"]
    df.to_csv(os.path.join(OUTPUTS_DIR, "cluster-docs.csv"), index=False)


if __name__ == "__main__":
    main()
