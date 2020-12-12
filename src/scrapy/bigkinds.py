import csv
import os
import time

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait

from src.config.settings import BASE_DIR, DATA_DIR

LIST_URL = "https://www.bigkinds.or.kr/v2/news/search.do"


def write_data_ids():
    driver = webdriver.Chrome(os.path.join(BASE_DIR, "chromedriver"))

    # f = open(os.path.join(DATA_DIR, "scrapy", "news-content.csv"), "w")
    # writer = csv.writer(f)
    driver.get(LIST_URL)
    tab = driver.find_element_by_css_selector(
        "[data-parent='#news-analysis-accordion']"
    )
    tab.click()
    page = 2
    # 미래 학교 교육
    # 1995-01-01 부터 2020-12-12
    # 칼럼,기고문,기고,사설,논설
    # 미래,학교,교육,예측
    input("검색조건 설정을 완료하셨습니까?")
    # writer.writerow(["index", "year", "title", "context"])
    index = 1
    while True:
        data_ids = driver.find_elements_by_css_selector(
            "#news-results .news-item"
        )
        for data_id in data_ids:
            action = ActionChains(driver)
            action.move_to_element(data_id).perform()
            title = data_id.find_element_by_tag_name("h4").text
            year = data_id.find_element_by_class_name(
                "news-item__date"
            ).text.split("/")[0]
            data_id.click()
            modal = driver.find_element_by_id("news-detail-modal")
            time.sleep(1)
            context = modal.find_element_by_class_name("modal-body").text
            print(index, year, title)
            # writer.writerow([index, year, title, context])
            close_button = modal.find_element_by_class_name("close")
            close_button.click()
            index += 1

        next_page = WebDriverWait(driver, 10).until(
            expected_conditions.presence_of_element_located(
                (By.CSS_SELECTOR, f"[data-page='{page}']")
            )
        )
        action = ActionChains(driver)
        action.move_to_element(next_page).perform()
        next_page.click()
        time.sleep(10)
        page += 1
        if page > 100:
            break
    # f.close()


def filter_dup_data():
    with open(os.path.join(DATA_DIR, "scrapy", "news-content.csv")) as f:
        reader = list(csv.reader(f))
    print(reader[0])
    header = [*reader[0]]
    header.insert(1, "cate")
    ret = [header]
    for idx, row in enumerate(reader[101:], 1):
        _, *remain = row
        ret.append([idx, "news", *remain])

    # print(len(reader[0] + reader[101:]))
    with open(
        os.path.join(DATA_DIR, "scrapy", "no-dep-news-content.csv"), "w"
    ) as f:
        writer = csv.writer(f)
        writer.writerows(ret)


def main():
    filter_dup_data()


if __name__ == "__main__":
    main()
