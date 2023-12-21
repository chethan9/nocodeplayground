from flask import Flask, request, jsonify
import os
import pandas as pd
import openpyxl
import requests
import base64

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
    'vizsoft': {'client_id': 'S2K8AXVQ7OJvLB59Giirg', 'client_secret': 'TTbyJqB50X85sFgC9n3QzDgicN3hlj0l', 'account_id': 'S2K8AXVQ7OJvLB59Giirg'},
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
