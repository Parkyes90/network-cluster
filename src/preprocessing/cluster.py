import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from src.config.settings import BASE_DIR, OUTPUTS_DIR
import os.path
import pickle
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, value
from bokeh.palettes import brewer

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
    tsne_df["cluster_no"] = y
    colors = brewer["Spectral"][len(tsne_df["cluster_no"].unique())]
    colormap = {i: colors[i] for i in tsne_df["cluster_no"].unique()}
    colors = [colormap[x] for x in tsne_df["cluster_no"]]
    tsne_df["color"] = colors
    plot_data = ColumnDataSource(data=tsne_df.to_dict(orient="list"))
    tsne_plot = figure(
        # title='TSNE Twitter BIO Embeddings',
        plot_width=650,
        plot_height=650,
        active_scroll="wheel_zoom",
        output_backend="webgl",
    )
    tsne_plot.add_tools(HoverTool(tooltips="@title"))
    tsne_plot.circle(
        source=plot_data,
        x="x_coord",
        y="y_coord",
        line_alpha=0.3,
        fill_alpha=0.2,
        size=10,
        fill_color="color",
        line_color="color",
    )
    tsne_plot.title.text_font_size = value("16pt")
    tsne_plot.xaxis.visible = False
    tsne_plot.yaxis.visible = False
    tsne_plot.grid.grid_line_color = None
    tsne_plot.outline_line_color = None
    show(tsne_plot)


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
    print(idx)
    df["cluster"] = idx
    return df


def main():
    df = read_word_vector_docs()
    # draw_chart(df)


if __name__ == "__main__":
    main()
