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

def parse_value(value, dtype, allow_empty):
    if dtype == 'integer':
        return int(value)
    elif dtype == 'boolean':
        return value.lower() == 'true'
    elif dtype == 'array':
        return [item.strip() for item in value.split('|')]
    elif dtype == 'string':
        return str(value)
    else:
        return value

@app.route('/csvimport', methods=['POST'])
def csv_import():
    file = request.files.get('file')
    collection_name = request.args.get('collectionName')
    api_url = request.args.get('api_url')
    firebase_push = request.args.get('firebasepush', 'no').lower()
    collection_setting = request.args.get('collectionSetting')
    allow_empty = request.args.get('allowempty', 'false').lower() == 'true'
    import_method = request.args.get('importMethod', 'addUpdate')

    if not file or file.filename == '':
        return jsonify({"error": "No file part"}), 400
    if not collection_name or not api_url or not collection_setting:
        return jsonify({"error": "Missing collectionName, api_url, or collectionSetting parameters"}), 400

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file, header=[0, 1], keep_default_na=False)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file, header=[0, 1], keep_default_na=False)
        else:
            return jsonify({"error": "File format not supported"}), 400

        if df.shape[0] < 2 or df.shape[1] != df.iloc[0].apply(str.lower).isin({'string', 'integer', 'boolean', 'array'}).all():
            return jsonify({"error": "Invalid import template"}), 400
        
        dtype_row = df.iloc[0]
        df = df[1:]
        df.reset_index(drop=True, inplace=True)
        df.columns = df.columns.get_level_values(0)
        
        records = df.to_dict(orient='records')
        processed_records = []
        for record in records:
            processed_record = {}
            for key, value in record.items():
                dtype = dtype_row[key].lower()
                if value or allow_empty:
                    processed_record[key] = parse_value(value, dtype, allow_empty)
            processed_records.append(processed_record)

        json_data = json.dumps(processed_records)

        if firebase_push == 'yes':
            params = {
                "collectionName": collection_name,
                "collectionSetting": collection_setting,
                "importMethod": import_method
            }
            response = requests.post(api_url, params=params, json=json.loads(json_data), headers={'Content-Type': 'application/json'})
            return jsonify(response.json()), response.status_code
        else:
            return jsonify(json.loads(json_data))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
