import cv2
import numpy as np
import pytesseract
import requests
from bs4 import BeautifulSoup
from PIL import Image
import urllib.request
import os

# Function to detect tables in an image (from previous code)
def detect_table(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
    
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    table_grid = cv2.add(horizontal_lines, vertical_lines)
    contours, _ = cv2.findContours(table_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    table_data = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:
            roi = img[y:y+h, x:x+w]
            pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(pil_img, config='--psm 6').strip()
            if text:
                table_data.append({
                    'position': (x, y, w, h),
                    'text': text
                })
    
    table_data.sort(key=lambda x: (x['position'][1], x['position'][0]))
    
    print("Detected table values:")
    current_row = -1
    for cell in table_data:
        if current_row != cell['position'][1]//20:
            current_row = cell['position'][1]//20
            print("\nRow:", end=" ")
        print(f"'{cell['text']}' ", end=" ")
    
    cv2.imwrite('table_detected.jpg', table_grid)
    return table_data

# Function to crawl Naver Smart Store and extract images
def crawl_naver_smartstore_images(url, save_folder="naver_images"):
    # Set headers to mimic a browser
    # headers = {
    #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    # }
    
    # Send request to the URL
    # response = requests.get(url, headers=headers)
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to access URL: {response.status_code}")
        return []
    
    # Parse HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all image tags
    img_tags = soup.find_all('img')
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    image_urls = []
    for idx, img in enumerate(img_tags):
        src = img.get('src')
        if src and 'http' in src:  # Ensure it's a valid URL
            image_urls.append(src)
            try:
                # Download the image
                img_path = os.path.join(save_folder, f'image_{idx}.jpg')
                urllib.request.urlretrieve(src, img_path)
                print(f"Downloaded: {img_path}")
            except Exception as e:
                print(f"Failed to download {src}: {str(e)}")
    
    return image_urls

# Combined function to handle both tasks
def process_image_and_webpage(image_path=None, url=None):
    results = {}
    
    # Detect table in image if provided
    if image_path:
        print("Processing image for table detection...")
        table_data = detect_table(image_path)
        results['table_data'] = table_data
    
    # Crawl webpage for images if URL provided
    if url:
        print(f"\nCrawling {url} for images...")
        image_urls = crawl_naver_smartstore_images(url)
        results['image_urls'] = image_urls
    
    return results

# Example usage
if __name__ == "__main__":
    # Replace with your local image path if you want table detection
    image_path = "table_image.jpg"  # Optional
    url = "https://smartstore.naver.com/fmarketss/products/10855965721"
    
    results = process_image_and_webpage(
        image_path=None,  # Set to None if you only want web crawling
        url=url
    )
    
    # Print results
    if 'image_urls' in results:
        print("\nExtracted image URLs:")
        for url in results['image_urls']:
            print(url)
    if 'table_data' in results and not results['table_data']:
        print("\nNo table data detected in the image.")