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
import chardet



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

def parse_key_pair_values(value, data_type, allow_empty):
    if value.strip() == '':
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
            # Process as an array of key-value pairs
            key_value_pairs = [item.split(':', 1) for item in value.split('|')]
            return [dict((k.strip(), v.strip()) for k, v in key_value_pairs)]
        else:
            return value.strip()
    except ValueError:
        return None  # or handle the error as you see fit

@app.route('/csvimport', methods=['POST'])
def csv_import():
    file = request.files.get('file')
    allow_empty = request.args.get('allowempty', 'false').lower() == 'true'

    if not file or file.filename == '':
        return jsonify({"error": "No file part"}), 400

    try:
        filename = file.filename
        # Determine file type and read into a DataFrame
        if filename.endswith('.csv'):
            df = pd.read_csv(file, keep_default_na=False)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file, keep_default_na=False)
        else:
            return jsonify({"error": "File format not supported"}), 400

        # Extract data types from the second row
        data_types = df.iloc[0].to_dict()
        df = df[1:]  # Remove the data type row

        # Process each record
        processed_records = []
        for _, row in df.iterrows():
            processed_record = {}
            for key, value in row.items():
                data_type = data_types.get(key, 'string')  # Default to string
                parsed_value = parse_key_pair_values(str(value), data_type, allow_empty)
                if parsed_value is not None:
                    processed_record[key] = parsed_value
            processed_records.append(processed_record)

        return jsonify(processed_records)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
