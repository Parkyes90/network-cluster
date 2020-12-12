import csv
import os
import sys

from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.settings import OUTPUTS_DIR

csv.field_size_limit(sys.maxsize)

otk = Okt()


def get_papers():
    with open(os.path.join(OUTPUTS_DIR, "news-papers.csv")) as f:
        reader = csv.reader(f)
        return list(reader)


def remove_no_context_rows():
    file = open(
        os.path.join(OUTPUTS_DIR, "filtered-word-vector-docs.csv"), "w"
    )
    w = csv.writer(file)
    w.writerow(["index", "cate", "year", "title", "context"])
    count = 1
    with open(os.path.join(OUTPUTS_DIR, "word-vector-docs.csv")) as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            _, *remain, context = row
            if len(context) > 10:
                w.writerow([count, *remain, context])
                count += 1


def process_nouns(row):
    vectorizer = TfidfVectorizer()
    *remain, context = row
    lines = context.split("\n")
    noun_lines = []
    for line in lines:
        nouns = otk.nouns(line)
        if nouns:
            noun_lines.append(" ".join(nouns))
    words = []
    if len(noun_lines) > 300:
        try:
            train = vectorizer.fit_transform(noun_lines)
            terms = vectorizer.get_feature_names()
            sums = train.sum(axis=0)
            for col, term in enumerate(terms):
                words.append((term, sums[0, col]))
        except ValueError:
            pass
    words.sort(key=lambda x: x[1], reverse=True)
    words = [w[0] for w in words][:300]
    print(remain, len(noun_lines))

    return [*remain, " ".join(words)]


def process():
    papers = get_papers()
    papers.pop(0)
    docs = []
    for paper in papers:
        docs.append(process_nouns(paper))
    with open(
        os.path.join(OUTPUTS_DIR, "word-vector-docs.csv"),
        "w",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerow(["index", "cate", "year", "title", "context"])
        for doc in docs:
            w.writerow(doc)


def main():
    process()
    remove_no_context_rows()


if __name__ == "__main__":
    main()
