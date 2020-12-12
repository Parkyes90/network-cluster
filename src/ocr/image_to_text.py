import os
import subprocess
import re
from src.config.settings import DATA_DIR


def main():
    years = os.listdir(os.path.join(DATA_DIR, "pdfs"))
    for year in years:
        files = os.listdir(os.path.join(DATA_DIR, "pdfs", year))
        output_year_dir = os.path.join(DATA_DIR, "converted_pdfs", year)
        if not os.path.isdir(output_year_dir):
            os.mkdir(output_year_dir)
        for file in files:
            filepath = os.path.join(DATA_DIR, "pdfs", year, file)
            output_path = os.path.join(output_year_dir, file)
            print(f"ocrmypdf {filepath} {output_path}")
            output = subprocess.getoutput(f"ocrmypdf {filepath} {output_path}")
            if not re.search(
                "PriorOcrFoundError: page already has text!", output
            ):
                print(filepath, output_year_dir, "mv")
                mv_output = subprocess.getoutput(
                    f"mv '{filepath}' {output_year_dir}"
                )
                print(mv_output)

                print("Uploaded scanned pdf")
            else:
                print("Uploaded digital pdf")


if __name__ == "__main__":
    main()
