import csv
import os
from collections import defaultdict

from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
import pandas as pd
from src.config.settings import ABILITY_DIR

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
            temp.append(model.wv.similarity(word, w))
        networks.append(temp)
    with open(os.path.join(ABILITY_DIR, "output", "vectors.csv"), "w") as f:
        w = csv.writer(f)
        w.writerows(vectors)
    with open(os.path.join(ABILITY_DIR, "output", "networks.csv"), "w") as f:
        w = csv.writer(f)
        w.writerows(networks)


def draw_vectors():
    pass


if __name__ == "__main__":
    # to_csv()
    # modeling()
    draw_vectors()
