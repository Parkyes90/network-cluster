import csv
import os
import sys

from src.config.settings import DATA_DIR, OUTPUTS_DIR

csv.field_size_limit(sys.maxsize)


def main():
    with open(
        os.path.join(DATA_DIR, "scrapy", "no-dep-news-content.csv")
    ) as f:
        news = list(csv.reader(f))
    with open(os.path.join(OUTPUTS_DIR, "index-papers.csv")) as f:
        papers = list(csv.reader(f))
    ret = [news[0]]
    for idx, row in enumerate(news[1:] + papers[1:], 1):
        ret.append([idx, *row[1:]])
    with open(os.path.join(OUTPUTS_DIR, "news-papers.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(ret)


if __name__ == "__main__":
    main()
