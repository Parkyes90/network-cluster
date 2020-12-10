import csv
import multiprocessing
import os
import re
from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from src.config.settings import PAPERS_DIR, OUTPUTS_DIR


def get_years():
    return os.listdir(PAPERS_DIR)


def walk_papers(year):
    return os.listdir(os.path.join(PAPERS_DIR, year))


def read_pdf(path):
    output = StringIO()
    file = open(path, "rb")
    parser = PDFParser(file)
    doc = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    device = TextConverter(rsrcmgr, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)
    print(f"{path} READ")
    return str(output.getvalue())


def write_row(item):
    paper, year = item
    paper = str(paper)
    path = os.path.join(PAPERS_DIR, year, paper)
    title = paper.split(".")[0]
    try:
        context = read_pdf(path)
        context = re.sub(r"\(cid:\d{1,10}\)", "", context)

        return ["논문", year, title, context]
    except TypeError:
        return ["논문", year, title, ""]


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
        # for paper in papers:
        #     paper = str(paper)
        #     path = os.path.join(PAPERS_DIR, year, paper)
        #     context = read_pdf(path)
        #     title = paper.split(".")[0]
        #     w.writerow(["논문", year, title, context])
    file.close()


def main():
    to_csv()


if __name__ == "__main__":
    main()
