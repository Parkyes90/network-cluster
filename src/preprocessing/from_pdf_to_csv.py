import csv
import multiprocessing
import os
import re
import sys
from src.config.settings import PAPERS_DIR, OUTPUTS_DIR
import pdftotext

csv.field_size_limit(sys.maxsize)


def get_years():
    return os.listdir(PAPERS_DIR)


def walk_papers(year):
    return os.listdir(os.path.join(PAPERS_DIR, year))


def read_pdf(path):
    file = open(path, "rb")
    pdf = pdftotext.PDF(file)
    text = ""
    for page in pdf:
        text += page
    file.close()
    return text


def write_row(item):
    paper, year = item
    paper = str(paper)
    path = os.path.join(PAPERS_DIR, year, paper)
    title = paper.split(".")[0]
    context = read_pdf(path)
    context = re.sub(r"\(cid:\d{1,10}\)", "", context)
    return ["논문", year, title, context]


def to_csv():
    file = open(os.path.join(OUTPUTS_DIR, "papers.csv"), "w")
    w = csv.writer(file)
    w.writerow(["cate", "year", "title", "context"])
    years = get_years()
    for year in years:
        papers = walk_papers(year)
        papers = [(paper, year) for paper in papers]
        with multiprocessing.Pool(
            processes=multiprocessing.cpu_count() * 2
        ) as pool:
            data = pool.map(write_row, papers)
        for row in data:
            w.writerow(row)
    file.close()


def apply_index_to_csv():
    file = open(os.path.join(OUTPUTS_DIR, "index-papers.csv"), "w")
    w = csv.writer(file)
    w.writerow(["index", "cate", "year", "title", "context"])
    with open(os.path.join(OUTPUTS_DIR, "papers.csv")) as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            w.writerow([index, *row])

    file.close()


def main():
    to_csv()
    apply_index_to_csv()


if __name__ == "__main__":
    main()
