from datetime import datetime, timedelta
from flask import Flask, request, jsonify, session
from instagrapi import Client
from jsmin import jsmin
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, quote
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
from css_html_js_minify import process_single_css_file as minify_css
from cssutils import parseString
import cssutils
import jwt
import logging
import os
import PTN
import random
import requests
import string
import time
import uuid
from htmlmin import minify as htmlmin
import re
from flask import make_response
from twilio.rest import Client
import json
import pandas as pd
import openpyxl


app = Flask(__name__)

# Get the current directory
current_directory = os.getcwd()

# Set the log file path in the root directory
log_file_path = os.path.join(current_directory, 'api.log')
# Configure logging
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_app():
    app = Flask(__name__, static_folder='uploads', static_url_path='/uploads')
    app.config['UPLOAD_FOLDER'] = '/app/uploads/'
    upload_folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    return app


app = create_app()

@app.before_request
def before_request():
    # This code will be executed before each request
    app.logger.info('Request: %s %s', request.method, request.url)
    app.logger.info('Headers: %s', request.headers)
    app.logger.info('Body: %s', request.get_data())


@app.after_request
def after_request(response):
    # This code will be executed after each request
    app.logger.info('Response: %s', response.status)
    return response

@app.route('/', methods=['GET'])
def homepage():
    return "Homepage"
#####################################################################################################


def parse_key_pair_values(value, allow_empty):
    """Parse values into different types based on their format."""
    if not isinstance(value, str):  # Return non-string values as-is
        return value

    # Handle empty strings
    if value.strip() == '':
        return '' if allow_empty else None

    # Convert to integer or float if possible
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        pass

    # Convert to boolean
    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'

    # Parse as list or dictionary
    if '|' in value:
        parts = value.split('|')
        if ':' in value:
            attributes = {}
            for part in parts:
                if ':' in part:
                    key, val = part.split(':', 1)
                    attributes[key.strip()] = val.strip()
            return attributes
        else:
            return [part.strip() for part in parts]

    # Default to return string
    return value.strip()

@app.route('/csvimport', methods=['POST'])
def csv_import():
    # Retrieve file and parameters from the request
    file = request.files.get('file')
    collection_name = request.args.get('collectionName')
    api_url = request.args.get('api_url')
    firebase_push = request.args.get('firebasepush', 'no').lower()
    collection_setting = request.args.get('collectionSetting')
    allow_empty = request.args.get('allowempty', 'false').lower() == 'true'
    import_method = request.args.get('importMethod', 'addUpdate')

    # Validate the inputs
    if not file or file.filename == '':
        return jsonify({"error": "No file part"}), 400
    if not collection_name or not api_url or not collection_setting:
        return jsonify({"error": "Missing collectionName, api_url, or collectionSetting parameters"}), 400

    try:
        # Determine file type and read into a DataFrame
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file, keep_default_na=False)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file, keep_default_na=False)
        else:
            return jsonify({"error": "File format not supported"}), 400

        # Convert DataFrame to JSON with custom parsing
        records = df.to_dict(orient='records')
        processed_records = []
        for record in records:
            processed_record = {}
            for key, value in record.items():
                parsed_value = parse_key_pair_values(value, allow_empty)
                if parsed_value is not None:
                    processed_record[key] = parsed_value
            processed_records.append(processed_record)

        json_data = json.dumps(processed_records)  # Convert to JSON string

        # If firebasepush is 'yes', make the API call
        if firebase_push == 'yes':
            params = {
                "collectionName": collection_name,
                "collectionSetting": collection_setting,
                "importMethod": import_method
            }
            response = requests.post(api_url, params=params, json=json.loads(json_data), headers={'Content-Type': 'application/json'})

            # Return the Cloud Function's response
            return jsonify(response.json()), response.status_code
        else:
            # If firebasepush is 'no', return the JSON data
            return jsonify(json.loads(json_data))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
