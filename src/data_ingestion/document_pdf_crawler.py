import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
import os
import requests


for page in range(8, 336):
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

    # Initialize the webdriver
    browser = webdriver.Chrome(service=s, options=chrome_options)
    # browser.quit()

    url = f"https://docs.nvidia.com/search/index.html?facet.mimetype[]=pdf&page={page}&sort=relevance&term=pdf"
    print(f"\n\n Page: {page}\n")

    browser.get(url)
    time.sleep(5)
    html = browser.page_source

    # Parse the HTML content
    soup = BeautifulSoup(html, 'html')

    # # Find the section containing the links
    # a = soup.find('section', 'wrapper')

    # Create the 'pdfs' directory if it doesn't exist
    os.makedirs('pdfs', exist_ok=True)

    # Loop through the links and download the PDFs
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.endswith('.pdf'):
            pdf_url = href

            # Get the filename from the URL
            filename = os.path.join('pdfs', str(pdf_url).replace("https://docs.nvidia.com/", ""))
            os.makedirs(filename.replace(os.path.basename(filename), ""), exist_ok=True)

            # Download the PDF
            with open(filename, 'wb') as f:
                pdf_content = requests.get(pdf_url)
                f.write(pdf_content.content)
                print(f"Downloaded: {filename}")

    browser.quit()
    time.sleep(5)


# excluded - broken links
# pdfs/developer/docs/drive/drive-os/latest/linux/sdk/DRIVE_OS_Free_Open_Source_Supplemental_License_Catalog.pdf
# pdfs/developer/docs/drive/drive-os/latest/tensorrt/pdf/NVIDIA_DRIVE_OS_6_0_TensorRT_8_4_11_Developer_Guide.pdf
# pdfs/developer/docs/drive/drive-os/latest/linux/sdk/api_reference/es_spec_3.1.pdf
# pdfs/developer/docs/drive/drive-os/latest/es_full_spec_2.0.25.pdf
# pdfs/developer/docs/drive/drive-os/latest/es_spec_3.0.3.pdf
# pdfs/developer/docs/drive/drive-os/latest/es_spec_3.1.pdf
# pdfs/developer/docs/drive/drive-os/latest/es_spec_3.2.withchanges.pdf
# pdfs/developer/docs/drive/drive-os/latest/GLSL_ES_Specification_1.0.17.pdf
# pdfs/developer/docs/drive/drive-os/latest/GLSL_ES_Specification_3.00.4.pdf
# pdfs/developer/docs/drive/drive-os/latest/GLSL_ES_Specification_3.10.pdf
# pdfs/developer/docs/drive/drive-os/latest/GLSL_ES_Specification_3.20.withchanges.pdf
# pdfs/cuda/pdf/Ampere_Compatibility_Guide.pdf
# pdfs/cuda/pdf/Nsight_Eclipse_Plugins_Installtion_Guide.pdf
# pdfs/cuda/pdf/CUDA-Occupancy-Calculator.pdf
# pdfs/cuda/pdf/Ampere_Tuning_Guide.pdf
# pdfs/nvshmem/pdf/NVSHMEM-Release-Notes.pdf
# pdfs/nvshmem/pdf/NVSHMEM-API-Reference.pdf
# pdfs/nvshmem/pdf/NVSHMEM-Installation-Guide.pdf
# pdfs/nvshmem/pdf/NVSHMEM-Archived.pdf
# pdfs/nvshmem/pdf/NVSHMEM-SLA.pdf
# pdfs/grid/4.5/pdf/grid-vgpu-release-notes-huawei-uvp.pdf
# pdfs/grid/4.4/pdf/grid-vgpu-release-notes-huawei-uvp.pdf
# pdfs/grid/4.8/pdf/grid-vgpu-release-notes-huawei-uvp.pdf
# pdfs/grid/4.7/pdf/grid-vgpu-release-notes-huawei-uvp.pdf