from flask import Flask, request, jsonify, send_file
import os
import pandas as pd
import openpyxl
import requests
import base64
import pytz
from datetime import datetime
from ytmusicapi import YTMusic
from pytube import YouTube
import json
import re
from PIL import Image, ImageDraw, ImageFont
import io
import cv2 as cv
import numpy as np
import mediapipe as mp
import fitz  # PyMuPDF
import urllib.parse 
from docx import Document

app = Flask(__name__)
ytmusic = YTMusic()
 
image_with_landmark = None

def create_app():
    app = Flask(__name__, static_folder='uploads', static_url_path='/uploads')
    app.config['UPLOAD_FOLDER'] = '/app/uploads/'
    upload_folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    return app

app = create_app()

@app.route('/', methods=['GET'])
def homepage():
    return "Homepage"

@app.before_request
def before_request():
    pass  # Placeholder for future before request actions

@app.after_request
def after_request(response):
    return response

def parse_key_pair_values(value, data_type, allow_empty):
    if isinstance(value, float) and pd.isna(value):
        return None  # Handle NaN values for numeric columns

    value = str(value).strip()
    if value == '':
        return '' if allow_empty else None

    try:
        if data_type == 'int':
            return int(value)
        elif data_type == 'float':
            return float(value)
        elif data_type == 'date':
            return pd.to_datetime(value).strftime('%Y-%m-%d')
        elif data_type == 'array':
            return [item.strip() for item in value.split('|')]
        elif data_type == 'keyvalue':
            key, val = value.split(':', 1)
            return {key.strip(): val.strip()}
        elif data_type == 'arraykeyvalue':
            key_value_pairs = value.split('|')
            return {pair.split(':', 1)[0].strip(): pair.split(':', 1)[1].strip() for pair in key_value_pairs}
        else:
            return value
    except ValueError:
        return None

@app.route('/csvimport', methods=['POST'])
def csv_import():
    file = request.files.get('file')
    allow_empty = request.args.get('allowempty', 'false').lower() == 'true'

    if not file or file.filename == '':
        return jsonify({"error": "No file part"}), 400

    try:
        filename = file.filename
        if filename.endswith('.csv'):
            df = pd.read_csv(file, keep_default_na=False)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file, keep_default_na=False)
        else:
            return jsonify({"error": "File format not supported"}), 400

        data_types = df.iloc[0].to_dict()
        df = df[1:]

        processed_records = []
        for _, row in df.iterrows():
            processed_record = {}
            for key, value in row.items():
                data_type = data_types.get(key, 'string')
                parsed_value = parse_key_pair_values(value, data_type, allow_empty)
                if parsed_value is not None:
                    processed_record[key] = parsed_value
            processed_records.append(processed_record)

        return jsonify(processed_records)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/get_redirected_url', methods=['GET'])
def get_redirected_url():
    original_url = request.args.get('url')
    host = request.args.get('host')

    if not original_url or not host:
        return jsonify({"error": "Missing url or host parameter"}), 400

    try:
        # Make a request to the original URL
        response = requests.get(original_url, allow_redirects=True)
        redirected_url = response.url

        # Construct the new URL without '?url='
        final_url = f'{host}{redirected_url}'

        return jsonify({"final_url": final_url})

    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
#########################################################################################################################
# Predefined credentials
credentials = {
    'bonus': {
        'client_id': 'EzUvevnZSHaqckAel0XQLw', 
        'client_secret': 'iD92OPk0Vq9DYtk45nqarHtfKIy7HR7d', 
        'account_id': 'GawvZa-MTg2lj5xA2EW9yg',
        'client_username': 'it@vizsoft.in',
        'timezone': 'Asia/Kuwait'
    },
    'drip': {
        'client_id': 'S2K8AXVQ7OJvLB59Giirg', 
        'client_secret': 'TTbyJqB50X85sFgC9n3QzDgicN3hlj0l', 
        'account_id': 'GawvZa-MTg2lj5xA2EW9yg',
        'client_username': 'it@vizsoft.in',
        'timezone': 'Asia/Kuwait'
    },
}

@app.route('/get_zoom_token', methods=['GET'])
def get_zoom_token():
    client_name = request.args.get('client_name')
    
    if client_name not in credentials:
        return jsonify({"error": "Invalid client name"}), 400

    client_id = credentials[client_name]['client_id']
    client_secret = credentials[client_name]['client_secret']
    account_id = credentials[client_name]['account_id']

    # Encode the client credentials
    encoded_credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    body = {
        "grant_type": "account_credentials",
        "account_id": account_id
    }

    try:
        response = requests.post('https://zoom.us/oauth/token', headers=headers, data=body)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

@app.route('/create_zoom_meeting', methods=['POST'])
def create_zoom_meeting():
    data = request.json
    client_name = data.get('client_name')

    if client_name not in credentials:
        return jsonify({"error": "Invalid client name"}), 400

    # Required fields
    topic = data.get('topic')
    input_start_time = data.get('start_time')  # e.g., "2023-12-21 22:00:00.000"
    duration = data.get('duration')
    agenda = data.get('agenda')

    if not all([topic, input_start_time, duration, agenda]):
        return jsonify({"error": "Missing required parameters"}), 400

    client_details = credentials[client_name]

    # Format the input time for Zoom
    start_time = format_start_time_for_zoom(input_start_time)

    # Generate OAuth token
    oauth_token = get_oauth_token(client_details)
    if 'error' in oauth_token:
        return oauth_token

    # Zoom create meeting endpoint
    zoom_create_meeting_url = f'https://api.zoom.us/v2/users/{client_details["client_username"]}/meetings'

    headers = {
        "Authorization": f"Bearer {oauth_token['access_token']}",
        "Content-Type": "application/json"
    }

    meeting_data = {
        "topic": topic,
        "type": 2,  # Scheduled meeting
        "start_time": start_time,
        "duration": duration,
        "timezone": client_details['timezone'],
        "agenda": agenda
    }

    try:
        response = requests.post(zoom_create_meeting_url, headers=headers, json=meeting_data)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

def format_start_time_for_zoom(input_time):
    # Format: 'YYYY-MM-DD HH:MM:SS.mmm' to 'YYYY-MM-DDTHH:MM:SS'
    return input_time.split('.')[0].replace(' ', 'T')

def get_oauth_token(client_details):
    client_id = client_details['client_id']
    client_secret = client_details['client_secret']
    account_id = client_details['account_id']

    # Encode the client credentials
    encoded_credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    body = {
        "grant_type": "account_credentials",
        "account_id": account_id
    }

    try:
        response = requests.post('https://zoom.us/oauth/token', headers=headers, data=body)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}

@app.route('/update_zoom_meeting', methods=['PATCH'])
def update_zoom_meeting():
    client_name = request.headers.get('client_name')
    meeting_id = request.headers.get('meeting_id')
    data = request.json

    if not client_name or not meeting_id:
        return jsonify({"error": "Missing client name or meeting ID"}), 400

    if client_name not in credentials:
        return jsonify({"error": "Invalid client name"}), 400

    client_details = credentials[client_name]

    # Generate OAuth token
    oauth_token = get_oauth_token(client_details)
    if 'error' in oauth_token:
        return oauth_token

    # Zoom update meeting endpoint
    zoom_update_meeting_url = f'https://api.zoom.us/v2/meetings/{meeting_id}'

    headers = {
        "Authorization": f"Bearer {oauth_token['access_token']}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.patch(zoom_update_meeting_url, headers=headers, json=data)
        response.raise_for_status()  # Raises an error for HTTP error responses

        # If the response is successful but contains no content
        if response.status_code == 204:
            return jsonify({"message": "Meeting data updated successfully"})

        # If the response contains content, return it
        return jsonify(response.json())
    except requests.RequestException as e:
        # Log the response text for debugging
        error_message = f"RequestException: {str(e)}, Response: {e.response.text if e.response else 'No response'}"
        return jsonify({"error": error_message}), 500
    except ValueError as e:
        # JSON parsing error
        return jsonify({"error": f"JSON parsing error: {str(e)}"}), 500

@app.route('/get_zoom_meeting', methods=['GET'])
def get_zoom_meeting():
    client_name = request.args.get('client_name')
    meeting_id = request.args.get('meeting_id')

    if not client_name or not meeting_id:
        return jsonify({"error": "Missing client name or meeting ID"}), 400

    if client_name not in credentials:
        return jsonify({"error": "Invalid client name"}), 400

    client_details = credentials[client_name]

    # Generate OAuth token
    oauth_token = get_oauth_token(client_details)
    if 'error' in oauth_token:
        return oauth_token

    # Zoom get meeting details endpoint
    zoom_meeting_details_url = f'https://api.zoom.us/v2/meetings/{meeting_id}'

    headers = {
        "Authorization": f"Bearer {oauth_token['access_token']}"
    }

    try:
        response = requests.get(zoom_meeting_details_url, headers=headers)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500
        
@app.route('/delete_zoom_meeting', methods=['DELETE'])
def delete_zoom_meeting():
    client_name = request.args.get('client_name')
    meeting_id = request.args.get('meeting_id')

    if not client_name or not meeting_id:
        return jsonify({"error": "Missing client name or meeting ID"}), 400

    if client_name not in credentials:
        return jsonify({"error": "Invalid client name"}), 400

    client_details = credentials[client_name]

    # Generate OAuth token
    oauth_token = get_oauth_token(client_details)
    if 'error' in oauth_token:
        return oauth_token

    # Zoom delete meeting endpoint
    zoom_delete_meeting_url = f'https://api.zoom.us/v2/meetings/{meeting_id}'

    headers = {
        "Authorization": f"Bearer {oauth_token['access_token']}"
    }

    try:
        response = requests.delete(zoom_delete_meeting_url, headers=headers)
        response.raise_for_status()
        return jsonify({"message": "Meeting deleted successfully"})
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

#############################################################################################################
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    category = request.args.get('category')  # New query parameter for category

    if not query:
        return "No query provided", 400

    results = ytmusic.search(query)
    if category:
        filtered_results = [result for result in results if result.get('category') == category]
        return jsonify(filtered_results)
    return jsonify(results)

    
    
@app.route('/get_streams', methods=['GET'])
def get_streams():
    video_id = request.args.get('video_id')
    if not video_id:
        return jsonify({'error': 'No video ID provided'}), 400

    try:
        yt = YouTube(f'https://www.youtube.com/watch?v={video_id}')
        streams = yt.streams.filter(progressive=True).all()
        stream_data = [{'itag': s.itag, 'mime_type': s.mime_type, 'resolution': s.resolution, 'fps': s.fps} for s in streams]
        return jsonify(stream_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_download_links', methods=['GET'])
def get_download_links():
    video_id = request.args.get('video_id')
    if not video_id:
        return jsonify({'error': 'No video ID provided'}), 400

    try:
        yt = YouTube(f'https://www.youtube.com/watch?v={video_id}')
        video_streams = yt.streams.filter(progressive=True, file_extension='mp4').all()
        audio_streams = yt.streams.filter(only_audio=True, file_extension='mp4').all()

        video_links = [{'itag': s.itag, 'type': 'video', 'mime_type': s.mime_type, 'resolution': s.resolution, 'download_url': s.url} for s in video_streams]
        audio_links = [{'itag': s.itag, 'type': 'audio', 'mime_type': s.mime_type, 'bitrate': s.abr, 'download_url': s.url} for s in audio_streams]

        return jsonify({'video': video_links, 'audio': audio_links})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('q')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    url = f'https://suggestqueries.google.com/complete/search?client=youtube&ds=yt&q={query}'
    response = requests.get(url)
    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch suggestions'}), 500

    try:
        suggestions = response.json()[1]
        terms = [suggestion[0] for suggestion in suggestions]
        return jsonify(terms)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


############################################################################################################################


@app.route('/landmark_detection', methods=['POST'])
def landmark_detection():
    if 'image' not in request.files:
        return jsonify({'error': 'Image is required for landmark detection'}), 400
    
    image = request.files['image']
    return perform_landmark_detection(image)

@app.route('/openai_processing', methods=['POST'])
def openai_processing():
    openai_api_key = request.form['openai_api_key']
    user_content = request.form['data']
    system_content = request.form.get('system_content', 'You are an AI capable of processing data')

    return call_openai_api(system_content, user_content, openai_api_key)

@app.route('/faceapi_processing', methods=['POST'])
def faceapi_processing():
    if 'image' not in request.files:
        return jsonify({'error': 'Image is required for faceapi processing'}), 400

    image = request.files['image']
    landmark_result = perform_landmark_detection(image)
    serialized_data = json.dumps(landmark_result)
    openai_api_key = request.form['openai_api_key']
    system_content = request.form.get('system_content', 'You are an AI capable of processing data')

    return call_openai_api(system_content, serialized_data, openai_api_key)

def perform_landmark_detection(image):
    luxand_response = requests.post(
        'https://api.luxand.cloud/photo/landmarks',
        headers={'token': '5acc11ec40f9441284ce5f90c0467087'},
        files={'photo': image}
    )
    return luxand_response.json()

def call_openai_api(system_content, user_content, api_key):
    openai_response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        },
        json={
            'model': 'gpt-3.5-turbo',
            'messages': [
                {
                    'role': 'system',
                    'content': system_content
                },
                {
                    'role': 'user',
                    'content': user_content
                }
            ]
        }
    )
    return openai_response.json()

if __name__ == '__main__':
    app.run(debug=True)
##################################################################################################################################

@app.route('/format-json', methods=['POST'])
def format_json():
    try:
        # Get the raw string from the request data
        data = request.data.decode('utf-8')

        # Replace escaped newlines and quotes
        formatted_data = data.replace('\\n', '\n').replace('\\"', '"')

        # Convert the string to a JSON object
        json_data = json.loads(formatted_data)

        # Return the pretty-printed JSON
        return jsonify(json_data), 200
    except Exception as e:
        return jsonify({"error": "Invalid JSON format", "details": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
###################################################################################################################################

@app.route('/serialize_json', methods=['POST'])
def serialize_json():
    # Get JSON data from the request
    json_data = request.get_json()

    # Convert JSON object to string
    serialized_data = str(json_data).replace("'", "\"").replace(" ", "")

    # Return serialized JSON as a string
    return jsonify({'serialized_json': serialized_data})

if __name__ == '__main__':
    app.run(debug=True)

##################################################################################################################################
@app.route('/analyze', methods=['POST'])
def analyze_input():
    input_text = request.json['input_text']
    sections = input_text.split('###')
    
    output = {}
    default_section = None
    
    for section in sections:
        if ':' in section:
            section_name, section_content = section.split(':', 1)
            output[section_name.strip()] = {"Description": section_content.strip()}
        elif default_section is None:
            default_section = section.strip()
    
    if default_section:
        output["Default"] = {"Description": default_section}
    
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
##################################################################################################################################

@app.route('/convert-to-jpeg', methods=['POST'])
def convert_to_jpeg():
    # Check if the request has the file part
    if 'file' not in request.files:
        return {'message': 'No file part in the request'}, 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return {'message': 'No selected file'}, 400

    try:
        # Convert the image to JPEG
        image = Image.open(file)
        # If the image is already a JPEG, we don't need to convert it
        if image.format == 'JPEG':
            output = io.BytesIO()
            file.save(output)
            output.seek(0)
        else:
            output = io.BytesIO()
            image.save(output, format='JPEG')
            output.seek(0)

        return send_file(output, mimetype='image/jpeg')
    except Exception as e:
        return {'message': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)

##########################################################################################################################
    
# Function to detect face and find out landmarks using Mediapipe lib
def detect_landmarks(image_path):
    # Load image
    image = cv.imread(image_path)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Initialize MediaPipe face detection and face landmark models
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

    # Detect faces in the image
    results_detection = face_detection.process(image_rgb)
    if results_detection.detections:
        for detection in results_detection.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * width), int(bboxC.ymin * height), \
                         int(bboxC.width * width), int(bboxC.height * height)

            # Crop face region
            face_region = image[y:y+h, x:x+w]

            # Detect face landmarks
            results_landmarks = face_mesh.process(cv.cvtColor(face_region, cv.COLOR_BGR2RGB))
            if results_landmarks.multi_face_landmarks:
                landmark_coords = []
                for face_landmarks in results_landmarks.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        landmark_coords.append((cx, cy))

                # Draw landmarks on the image
                for landmark in landmark_coords:
                    cv.circle(face_region, landmark, 1, (0, 255, 0), -1)

                # Save the image with landmarks drawn
                global image_with_landmark
                image_with_landmark = 'temp_image_with_landmarks.jpg'
                cv.imwrite(image_with_landmark, face_region)

                return {"success": True, "image": image_with_landmark, "landmark_coordinates": landmark_coords}

    return {"success": False, "message": "No human face detected in the image."}

@app.route('/detect_landmarks', methods=['POST'])
def detect_landmarks_api():
    # Check if an image file is present in the request
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image found in the request."}), 400

    image_file = request.files['image']

    # Check if the file is a valid image
    if image_file.filename == '':
        return jsonify({"success": False, "message": "Empty image filename."}), 400
    if not image_file:
        return jsonify({"success": False, "message": "Invalid image file."}), 400

    try:
        # Save the image temporarily
        image_path = 'temp_image.jpg'
        image_file.save(image_path)

        # Detect face landmarks
        result = detect_landmarks(image_path)

        # Delete the temporary image file
        os.remove(image_path)

        if result["success"]:
            # Return the processed landmark coordinates
            response_json = {"landmark_coordinates": result["landmark_coordinates"]}
            return jsonify(response_json), 200
        else:
            return jsonify(result), 400

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    
###################################################################################################
# Api is to retrive the landmarked image
@app.route('/get_landmark_image', methods=['GET'])
def get_landmark_image():
    # Path to the image file
    global image_with_landmark

    # Return the image file
    return send_file(image_with_landmark, mimetype='image/jpg')

if __name__ == '__main__':
    app.run(debug=True)

###################################################################################################
#Api is to clean the landmark coordinates from json fromat to text

@app.route('/clean_json', methods=['POST'])
def clean_json():
    # Get the JSON data from the request
    json_data = request.get_json()

    # Extract landmark coordinates
    landmark_coordinates = json_data.get('landmark_coordinates')

    # Prepare the text output with coordinates as list of lists
    text_output = "{"
    for coordinates in landmark_coordinates:
        text_output += f"[{coordinates[0]}, {coordinates[1]}], "
    text_output = text_output.rstrip(", ")  # remove the last comma and space
    text_output += "}"

    # Return the text output
    return text_output

if __name__ == '__main__':
    app.run(debug=True)

###################################################################################################
#Api to check folder status in vdocipher

@app.route('/folder_status', methods=['POST'])
def folder_status():
    data = request.json
    folder_name = data.get('name')

    if not folder_name:
        return jsonify({"error": "Folder name is required"}), 400

    url = "https://dev.vdocipher.com/api/videos/folders/search"
    payload = json.dumps({"name": folder_name})
    headers = {
        'content-type': 'application/json',
        'Authorization': 'Apisecret vBEWfxrM2S60wYiLfpyNT2vD5PNvuKKWmCXJCeyJY0Y02ZCXoqEIUcXvs7xzAg74'
    }

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


###################################################################################################
# Api to Send opt via OPTLESS to user Whatsapp number

@app.route('/send_opt_whatsapp', methods=['POST'])
def send_otp():
    data = request.json
    phoneNumber = data.get('phoneNumber')

    if not phoneNumber:
        return jsonify({"error": "Phone number is required"}), 400

    url = "https://auth.otpless.app/auth/otp/v1/send"

    payload = json.dumps({"phoneNumber": phoneNumber,
                          "otpLength": 6,
                          "channel": "WHATSAPP",
                          "expiry": 300})
    
    headers = {'clientId': '9pfotv2x',
               'clientSecret': 'mcko5gfabcctjyep',
               'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

###################################################################################################
# Api to Resend opt via OPTLESS to user Whatsapp number

@app.route('/resend_opt_whatsapp', methods=['POST'])
def resend_otp():
    data = request.json
    orderId = data.get('orderId')

    if not orderId:
        return jsonify({"error": "Order Id is required"}), 400

    url = "https://auth.otpless.app/auth/otp/v1/resend"

    payload = json.dumps({"orderId": orderId,})
    
    headers = {'clientId': '9pfotv2x',
               'clientSecret': 'mcko5gfabcctjyep',
               'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

###################################################################################################
# Api to Verify opt sentby OPTLESS to user Whatsapp number

@app.route('/verify_opt_whatsapp', methods=['POST'])
def verify_otp():
    data = request.json
    orderId = data.get('orderId')
    otp = data.get('otp')
    phoneNumber = data.get('phoneNumber')

    if not orderId:
        return jsonify({"error": "Order Id is required"}), 400
    elif not otp:
        return jsonify({"error": "Otp is required"}), 400
    elif not phoneNumber:
        return jsonify({"error": "phone number is required"}), 400

    url = "https://auth.otpless.app/auth/otp/v1/verify"

    payload = json.dumps({"orderId": orderId,
                          "otp": otp,
                          "phoneNumber": phoneNumber})
    
    headers = {'clientId': '9pfotv2x',
               'clientSecret': 'mcko5gfabcctjyep',
               'Content-Type': 'application/json'}

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

###############################################################################################################################
# Api for vdoCipher to List all items present in parent folder
# pass the folder_id with url 

@app.route('/list_folders/<folder_id>', methods=['GET'])
def list_all_folders(folder_id):


    url = f"https://dev.vdocipher.com/api/videos/folders/{folder_id}"

    headers = {
        'Authorization': 'Apisecret vBEWfxrM2S60wYiLfpyNT2vD5PNvuKKWmCXJCeyJY0Y02ZCXoqEIUcXvs7xzAg74'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
###################################################################################################################################
API_SECRET = 'DMuGqc1l6UtISGaXFurzhyWF07MYJRNNELUtS6jxfFyJ26i88gzKrE5Yo27KqKlh'
VDO_API_URL = 'https://dev.vdocipher.com/api/videos/{videoid}/files'
VDO_FILE_URL = 'https://dev.vdocipher.com/api/videos/{videoid}/files/{fileid}'

@app.route('/videos/<videoid>/files', methods=['GET'])
def get_video_files(videoid):
    headers = {
        'Authorization': f'Apisecret {API_SECRET}'
    }

    url = VDO_API_URL.format(videoid=videoid)
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()

        # Check for the filter parameter
        filter_data = request.args.get('filter', 'yes')
        if filter_data == 'yes':
            filtered_data = [item for item in data if item.get('encryption_type') == 'original']
        else:
            filtered_data = data

        # Check for the downloadlink parameter
        downloadlink = request.args.get('downloadlink', 'no')
        if filter_data == 'no' and downloadlink == 'yes':
            return jsonify({"error": "Cannot use downloadlink when filter is set to 'no'"}), 400

        if downloadlink == 'yes' and filter_data == 'yes':
            if filtered_data:
                fileid = filtered_data[0]['id']
                file_url = VDO_FILE_URL.format(videoid=videoid, fileid=fileid)
                file_response = requests.get(file_url, headers=headers)

                if file_response.status_code == 200:
                    file_data = file_response.json()
                    filtered_data[0]['file_info'] = file_data
                else:
                    return jsonify({"error": file_response.text}), file_response.status_code

        # Return the filtered data or the entire data
        if filter_data == 'yes' and not filtered_data:
            return jsonify({"error": "no data"}), 404
        return jsonify(filtered_data), 200
    else:
        return jsonify({"error": response.text}), response.status_code

if __name__ == '__main__':
    app.run(debug=True)

########################################################################
# API to fetch videos tagwise and paginate them

@app.route('/videos_tagwise/tag=<tagname>/pageno=<pageno>/limit=<limit>', methods=['GET'])
def list_videos_tagwise(tagname,pageno,limit):


    url = f"https://dev.vdocipher.com/api/videos?tags={tagname}&page={pageno}&limit={limit}"

    headers = {
        'Authorization': 'Apisecret vBEWfxrM2S60wYiLfpyNT2vD5PNvuKKWmCXJCeyJY0Y02ZCXoqEIUcXvs7xzAg74'
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return jsonify(response.json()), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

########################################################################
# API to update the tags of videos

@app.route('/update_video_tags', methods=['PUT'])
def update_video_tags():

    data = request.json
    videos = data.get('videos', [])
    tags = data.get('tags', [])

    if not data or 'videos' not in data or 'tags' not in data:
        return jsonify({"error": "Invalid request data"}), 400
    
    url = "https://dev.vdocipher.com/api/videos/tags"

    headers = {
        'content-type': 'application/json',
        'Authorization': 'Apisecret vBEWfxrM2S60wYiLfpyNT2vD5PNvuKKWmCXJCeyJY0Y02ZCXoqEIUcXvs7xzAg74'
    }

    payload = json.dumps({"videos": videos,"tags": tags})


    response = requests.put(url, data=payload, headers=headers)

    if response.status_code == 200:
        return jsonify({"message": "Tags updated successfully", "response": response.json()}), 200
    else:
        return jsonify({"error": "Failed to update tags", "response": response.json()}), response.status_code

########################################################################
# API to convert .PDF, .docx files to image(.jpeg).

# Configuration
BACKBLAZE_BUCKET_ID = '3da90956953fa79b92240d1f'
BACKBLAZE_BUCKET_NAME = 'storagevizsoft'
BACKBLAZE_AUTH_URL = 'https://api.backblazeb2.com/b2api/v2/b2_authorize_account'
KEY_ID = '004d9965f7b24df0000000005'
APP_KEY = 'K004Tw4DUcnSIq4jiQ/ZXjZisfAv684'
LOCAL_STORAGE = './local_files'

# Helper: Convert PDF to Images
def convert_pdf_to_images(file_path):
    images = []
    with fitz.open(file_path) as pdf_document:
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images

# Helper: Convert DOCX to Images
def convert_docx_to_images(file_path):
    images = []
    doc = Document(file_path)
    font_path = "/path/to/your/font.ttf"  # Replace with a path to a .ttf font file
    font = ImageFont.truetype(font_path, size=20) if os.path.exists(font_path) else ImageFont.load_default()

    # Page dimensions
    page_width = 800
    page_height = 600
    margin = 20
    line_spacing = 30

    current_height = margin
    current_page = Image.new("RGB", (page_width, page_height), color="white")
    draw = ImageDraw.Draw(current_page)

    for paragraph in doc.paragraphs:
        # Split text into lines that fit the width of the page
        lines = text_wrap(paragraph.text, draw, font, page_width - 2 * margin)

        for line in lines:
            if current_height + line_spacing > page_height - margin:
                # Save the current page and start a new one
                images.append(current_page)
                current_page = Image.new("RGB", (page_width, page_height), color="white")
                draw = ImageDraw.Draw(current_page)
                current_height = margin

            # Draw the text line on the page
            draw.text((margin, current_height), line, fill="black", font=font)
            current_height += line_spacing

    # Save the last page
    images.append(current_page)
    return images

# Helper: Text Wrapping
def text_wrap(text, draw, font, max_width):
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        width = draw.textbbox((0, 0), test_line, font=font)[2]
        if width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines

# Helper: Merge Images Vertically
def merge_images_vertically(images):
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)

    merged_image = Image.new("RGB", (max_width, total_height), color="white")
    y_offset = 0
    for img in images:
        merged_image.paste(img, (0, y_offset))
        y_offset += img.height

    return merged_image

# Helper: Authorize Backblaze
def authorize_backblaze():
    response = requests.get(BACKBLAZE_AUTH_URL, auth=(KEY_ID, APP_KEY))
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to authorize Backblaze")

# Helper: Upload file to Backblaze
def upload_to_backblaze(auth_data, file_path, file_name):
    # Sanitize the file name
    sanitized_file_name = file_name.replace(" ", "_")  # Replace spaces with underscores

    # Get upload URL
    api_url = auth_data['apiUrl']
    auth_token = auth_data['authorizationToken']
    upload_url_resp = requests.post(
        f"{api_url}/b2api/v2/b2_get_upload_url",
        headers={'Authorization': auth_token},
        json={'bucketId': BACKBLAZE_BUCKET_ID}
    )
    if upload_url_resp.status_code != 200:
        raise Exception(f"Failed to get upload URL: {upload_url_resp.content.decode()}")

    upload_url_data = upload_url_resp.json()
    upload_url = upload_url_data['uploadUrl']
    upload_auth_token = upload_url_data['authorizationToken']

    # Upload file
    with open(file_path, 'rb') as file_data:
        headers = {
            'Authorization': upload_auth_token,
            'X-Bz-File-Name': sanitized_file_name,
            'Content-Type': 'b2/x-auto',
            'X-Bz-Content-Sha1': 'do_not_verify'
        }
        upload_resp = requests.post(upload_url, headers=headers, data=file_data)
        if upload_resp.status_code == 200:
            return upload_resp.json()
        else:
            raise Exception(f"Failed to upload file: {upload_resp.content.decode()}")


# Helper: Generate public link for Backblaze file
def generate_backblaze_public_link(auth_data, file_name):
    download_url = auth_data['downloadUrl']
    return f"{download_url}/file/{BACKBLAZE_BUCKET_NAME}/{file_name}"

@app.route('/convert_to_images', methods=['POST'])
def convert_to_images():
    uploaded_file = request.files.get('file')
    if not uploaded_file:
        return jsonify({"error": "File is required"}), 400

    file_extension = os.path.splitext(uploaded_file.filename)[1].lower()
    if file_extension not in ['.pdf', '.docx']:
        return jsonify({"error": "Unsupported file type. Only PDF and DOCX are supported"}), 400

    if not os.path.exists(LOCAL_STORAGE):
        os.makedirs(LOCAL_STORAGE)
    
    # Save the original file with its sanitized name
    sanitized_file_name = uploaded_file.filename.replace(" ", "_")  # Replace spaces with underscores
    file_path = os.path.join(LOCAL_STORAGE, sanitized_file_name)
    uploaded_file.save(file_path)

    # Generate the merged file name using the sanitized original file name
    merged_image_name = f"{os.path.splitext(sanitized_file_name)[0]}_merged.jpeg"
    merged_image_path = os.path.join(LOCAL_STORAGE, merged_image_name)

    try:
        # Convert the file to images
        if file_extension == '.pdf':
            images = convert_pdf_to_images(file_path)
        elif file_extension == '.docx':
            images = convert_docx_to_images(file_path)

        # Merge the images into a single image
        merged_image = merge_images_vertically(images)
        merged_image.save(merged_image_path, format="JPEG")

        # Authorize and upload files to Backblaze
        auth_data = authorize_backblaze()

        # Upload the original file
        original_file_link = generate_backblaze_public_link(
            auth_data, sanitized_file_name
        )
        upload_to_backblaze(auth_data, file_path, sanitized_file_name)

        # Upload the merged image
        merged_image_link = generate_backblaze_public_link(
            auth_data, merged_image_name
        )
        upload_to_backblaze(auth_data, merged_image_path, merged_image_name)

        return jsonify({
            "message": "File Conversion successful to Image",
            "original_file_link": original_file_link,
            "public_link": merged_image_link
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up local files
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(merged_image_path):
            os.remove(merged_image_path)


if __name__ == '__main__':
    app.run()