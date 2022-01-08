import selenium
import numpy as np
import time

from selenium import webdriver
import random
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options)
question = input("Are we looking at your account ?:\n")
print("0k")
if (question=="no"):
    account_name = input("what is the account name?:\n")
    log = Login("real.rahoolio", "rahulnairiscool")
    pub = public_account(account_name)
    pub.insta_page2()
    nav = insta_navigator()
    followers_button = driver.find_element_by_xpath("//*[@id=\"react-root\"]/section/main/div/header/section/ul/li[2]/a")
    followers_button.click()
    log.password_insta_logger_other()
    nav.not_now()
    time.sleep(1)
    nav.comparer()
else:
    log = Login(#your password
      , #your username)
    log.insta_page()
    log.password_insta_logger()
    nav = insta_navigator()
    nav.comparer()

driver.quit()
