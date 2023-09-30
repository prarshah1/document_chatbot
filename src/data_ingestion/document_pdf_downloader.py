import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
import os
import requests

page=1
# Set up Chrome webdriver with headless option
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')

# If running on a server or without a display, add the following options:
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-gpu')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument(f"--profile-directory=Profile{page}")

# Set up Chrome service
s = ChromeService(executable_path='../../resources/chromedriver/chromedriver')
browser = webdriver.Chrome(service=s, options=chrome_options)
browser.quit()