import csv
import os

from src.config.settings import DATA_DIR

MORPHS_PATH = os.path.join(DATA_DIR, "morphs")
FUTURES_PATH = os.path.join(DATA_DIR, "futures")


def get_data(path):
    ret = []

    files = os.listdir(path)
    files = [file for file in files if "euckr" not in file]
    for filename in files:
        with open(os.path.join(path, filename)) as f:
            reader = list(csv.reader(f))
            reader.pop(0)
            ret += reader
    return ret


def main():
    morphs = get_data(MORPHS_PATH)
    futures = get_data(FUTURES_PATH)
    print(morphs, futures)


if __name__ == "__main__":
    main()
