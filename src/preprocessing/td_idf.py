import csv
import os
import sys
from math import log

import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.settings import OUTPUTS_DIR

csv.field_size_limit(sys.maxsize)

otk = Okt()


def get_papers():
    with open(os.path.join(OUTPUTS_DIR, "news-papers.csv")) as f:
        reader = csv.reader(f)
        return list(reader)


def count_all():
    papers = get_papers()
    papers.pop(0)
    ret = set()
    for paper in papers:
        *remain, context = paper
        nouns = otk.nouns(context)

        for n in nouns:
            ret.add(n)
        print(remain, len(ret))
    return len(ret)


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
    if len(noun_lines) > 0:
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


def write_reindex_raw_data():
    raw_map = {}
    with open(os.path.join(OUTPUTS_DIR, "news-papers.csv")) as f:
        raw = list(csv.reader(f))
        for r in raw[1:]:
            raw_map[r[3]] = r
    ret = [raw[0]]

    with open(os.path.join(OUTPUTS_DIR, "filtered-word-vector-docs.csv")) as f:
        reader = list(csv.reader(f))
    for r in reader[1:]:
        idx, *remain = r
        ridx, *ra = raw_map[r[3]]
        ret.append([idx, *ra])
    with open(
        os.path.join(OUTPUTS_DIR, "index-raw-papers.csv"),
        "w",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)
        w.writerows(ret)


def write_td_idf_by_doc():
    papers = get_papers()[1:]
    f = open(
        os.path.join(OUTPUTS_DIR, "tf-df-idf.csv"), "w", encoding="utf-8",
    )
    writer = csv.writer(f)
    writer.writerow(["doc_index", "word", "tf", "df", "idf"])
    for paper in papers:
        idx, *_, context = paper
        lines = context.split("\n")
        noun_lines = []
        for line in lines:
            nouns = otk.nouns(line)
            if nouns:
                noun_lines.append(" ".join(nouns))
        total_docs_count = len(noun_lines)
        words_set = set()
        for line in noun_lines:
            nouns = line.split()
            for noun in nouns:
                words_set.add(noun)
        for word in words_set:
            tf = 0
            df = 0
            for li in noun_lines:
                target_nouns = li.split()
                temp_tf_count = target_nouns.count(word)
                tf += temp_tf_count
                if temp_tf_count > 0:
                    df += 1
            idf = log(total_docs_count / (df + 1))
            df = df / total_docs_count
            row = [idx, word, tf, df, idf]
            writer.writerow(row)
        print(idx)
    f.close()


def write_word_count_by_doc():
    f = open(
        os.path.join(OUTPUTS_DIR, "word-count-by-doc.csv"),
        "w",
        encoding="utf-8",
    )
    writer = csv.writer(f)
    writer.writerow(["doc_index", "word", "count"])
    with open(os.path.join(OUTPUTS_DIR, "tf-df-idf.csv")) as f:
        reader = csv.reader(f)
        for idx, line in enumerate(reader):
            if idx == 0:
                continue
            doc_index, word, count, *_ = line

            writer.writerow([doc_index, word, count])
    f.close()


def main():
    # count_all()
    # process()
    write_td_idf_by_doc()
    # write_word_count_by_doc()
    # remove_no_context_rows()
    # write_reindex_raw_data()


if __name__ == "__main__":
    main()
