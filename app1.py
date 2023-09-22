import cv2
import face_recognition
from flask import Flask, request, jsonify
import requests
import json
import numpy as np
from multiprocessing import Pool

app = Flask(__name__)

# Load face encodings into memory for caching
face_encodings_cache = {}

# Function to fetch data from an external API with error handling
def fetch_data_from_api(api_url, headers):
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()  # Raise an exception for non-200 HTTP status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f'Error: RequestException - {e}')
        return None
    except json.JSONDecodeError as e:
        print(f'Error: JSONDecodeError - {e}')
        return None

# Function to fetch image URLs from API data
def get_image_urls_from_api_data(api_data):
    image_urls = []
    if api_data and 'data' in api_data:
        driver_data = api_data['data']
        image_urls = [driver.get('driver_photo') for driver in driver_data if driver.get('driver_photo')]
        print(image_urls)
    return image_urls

# Function to load an image from a URL or local file path with error handling
def load_image(image_path):
    try:
        if image_path.startswith('http://') or image_path.startswith('https://'):
            headers = {"User-Agent": "XY"}
            response = requests.get(image_path, headers=headers)
            response.raise_for_status()
            image_content = np.frombuffer(response.content, np.uint8)
            image = cv2.imdecode(image_content, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(image_path)

        if image is not None:
            return image
        else:
            print(f'Error: Failed to load image from path or URL: {image_path}')
            return None
    except Exception as e:
        print(f'Error: {e}')
        return None

# Function to find face encodings in an image with error handling and caching
def find_face_encodings(image_path):
    # Check if the face encoding is already cached
    if image_path in face_encodings_cache:
        return face_encodings_cache[image_path]

    image = load_image(image_path)
    if image is None:
        return None

    face_enc = face_recognition.face_encodings(image)
    if face_enc:
        # Cache the face encoding for future use
        face_encodings_cache[image_path] = face_enc[0]
        return face_enc[0]
    else:
        print(f'Error: No face found in the image: {image_path}')
        return None

# Function to compare image_url1 with multiple image URLs and return matches
def compare_images(image_url1, image_urls):
    matches = []
    image_1 = find_face_encodings(image_url1)

    if image_1 is None:
        return matches

    # Use a multiprocessing Pool for parallel processing
    with Pool() as pool:
        results = pool.map(find_face_encodings, image_urls)

    for image_2 in results:
        if image_2 is not None:
            is_same = face_recognition.compare_faces([image_1], image_2)[0]

            if is_same:
                distance = face_recognition.face_distance([image_1], image_2)
                accuracy = round((1 - distance[0]) * 100, 2)
                print(accuracy)
                if accuracy > 55:
                    response = {
                        'message': 'The images are the same.',
                        'accuracy': accuracy
                    }
                    return response  # Stop execution and return the response immediately

    # If no match found, return the default response
    return {'message': 'No matching images found.'}

# Function to filter driver data based on a matching image URL
def filter_driver_data(driver_data, image_url):
    filtered_data = []
    image_1 = find_face_encodings(image_url)
    if image_1 is None:
        return filtered_data
    for driver in driver_data:
        driver_image_url = driver.get('driver_photo')
        if driver_image_url:
            image_2 = find_face_encodings(driver_image_url)
            if image_2 is not None:
                is_same = face_recognition.compare_faces([image_1], image_2)[0]
                if is_same:
                    distance = face_recognition.face_distance([image_1], image_2)
                    accuracy = round((1 - distance[0]) * 100, 2)
                    print(accuracy)
                    if accuracy > 55:
                        filtered_data.append(driver)
    return filtered_data

# This is route function, call when '/compare_faces'
@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    data = request.get_json()
    if 'image_url1' not in data:
        return jsonify({'error': 'Please provide image URL for image_url1'}), 400
    image_url1 = data['image_url1']
    api_url = 'https://zeclipseinfomedia.com/dms/driverList.php'  # Replace with the actual API URL
    headers = {"User-Agent": "XY"}
    try:
        data_from_api = fetch_data_from_api(api_url, headers)
    except Exception as e:
        return jsonify({'error': f'Failed to fetch data from the API: {e}'}), 500
    driver_data = data_from_api.get('data', [])
    if not driver_data:
        return jsonify({'error': 'No valid driver data found in the API response'}), 400
    matching_drivers = filter_driver_data(driver_data, image_url1)
    if not matching_drivers:
        return jsonify({'message': 'No matching drivers found.'})
    return jsonify({'matching_drivers': matching_drivers})
if __name__ == '__main__':
    app.run(debug=True)
