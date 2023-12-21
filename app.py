from flask import Flask, request, jsonify
import os
import pandas as pd
import openpyxl
import requests
import base64
import pytz
from datetime import datetime


app = Flask(__name__)

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
    'client_name_1': {
        'client_id': 'S2K8AXVQ7OJvLB59Giirg', 
        'client_secret': 'TTbyJqB50X85sFgC9n3QzDgicN3hlj0l', 
        'account_id': 'GawvZa-MTg2lj5xA2EW9yg',
        'client_username': 'it@vizsoft.in',
        'timezone': 'Asia/Kuwait'
    },
    # Add other clients as needed
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

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/create_zoom_meeting', methods=['POST'])
def create_zoom_meeting():
    data = request.json
    client_name = data.get('client_name')

    if client_name not in credentials:
        return jsonify({"error": "Invalid client name"}), 400

    # Required fields
    topic = data.get('topic')
    input_start_time = data.get('start_time')  # e.g., "13 October 2023 at 14:07:37 UTC+3"
    duration = data.get('duration')
    agenda = data.get('agenda')

    if not all([topic, input_start_time, duration, agenda]):
        return jsonify({"error": "Missing required parameters"}), 400

    client_details = credentials[client_name]
    timezone = client_details.get('timezone', 'UTC')

    # Parse the input time and adjust to UTC
    start_time = parse_input_time(input_start_time)

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
        "timezone": timezone,
        "agenda": agenda
    }

    try:
        response = requests.post(zoom_create_meeting_url, headers=headers, json=meeting_data)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

def parse_input_time(input_time):
    # Remove any prefix before the date
    date_str = re.sub(r'^.*?\d{1,2} ', '', input_time)

    # Parse the date and time
    dt = datetime.strptime(date_str, '%B %Y at %H:%M:%S UTC%z')
    
    # Convert to UTC and format for Zoom
    utc_dt = dt.astimezone(pytz.utc)
    return utc_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
