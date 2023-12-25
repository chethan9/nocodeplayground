from flask import Flask, request, jsonify
import os
import pandas as pd
import openpyxl
import requests
import base64
import pytz
from datetime import datetime
from ytmusicapi import YTMusic
from pytube import YouTube


app = Flask(__name__)
ytmusic = YTMusic()

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
    if not query:
        return "No query provided", 400
    results = ytmusic.search(query)
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
