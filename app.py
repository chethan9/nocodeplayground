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


@app.route('/tor', methods=['GET'])
def tor():
    try:
        name = request.args.get('name')
        url = "https://2torrentz2eu.in/beta2/search.php?torrent-query=" + name
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        rows = table.find_all('tr')
        data = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 5:
                continue  # Skip rows with fewer than 5 td elements
            download_button = [col.find('button', class_='ui blue basic button') for col in cols]
            download_link = []
            for button in download_button:
                if button:
                    onclick_text = button.get('onclick')
                    link = onclick_text.split("'")[1]
                    full_link = "https://2torrentz2eu.in/beta2/page.php?url=" + link
                    download_link.append(full_link)
            # Remove empty strings from download_link
            download_link = [link for link in download_link if link]
            cols = [col.text.strip() for col in cols]
            # Create a dictionary for each row
            row_dict = {
                "Title": cols[0],
                "Seeds": int(cols[1]),
                "Leeches": int(cols[2]),
                "Size": cols[3],
                "Date": cols[4],
                "Download": download_link[0] if download_link else None
            }
            data.append(row_dict)
        response = make_response(jsonify({"movies": data}), 200)
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response
    except Exception as e:
        response = make_response(jsonify({"error": str(e)}), 500)
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response


@app.route('/magnet', methods=['GET'])
def magnet():
    url = request.args.get('url')
    parsed_url = urlparse(url)
    query = parse_qs(parsed_url.query)
    query['url'] = [quote(query['url'][0], safe='')]  # URL-encode the value of the "url" query parameter
    encoded_url = urlunparse(parsed_url._replace(query=urlencode(query, doseq=True)))
    response = requests.get(encoded_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    magnet_link_tag = soup.find('a', class_='download-button', id='magnet')
    if magnet_link_tag:
        magnet_link = magnet_link_tag.get('href')
        if 'magnet:?xt=' in magnet_link:
            return jsonify({'magnet_link': magnet_link})
        else:
            return jsonify({'error': "The link does not contain magnet"}), 404
    else:
        return jsonify({'error': "Could not find the 'Open Magnet' button"}), 404

@app.route('/parse', methods=['GET'])
def parse():
    filename = request.args.get('filename')
    info = PTN.parse(filename)
    return jsonify(info)


@app.route('/tor2', methods=['GET'])
def tor2():
    try:
        name = request.args.get('name')
        url = "https://2torrentz2eu.in/beta2/search.php?torrent-query=" + name
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        rows = table.find_all('tr')
        data = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 5:
                continue  # Skip rows with fewer than 5 td elements
            seeds = int(cols[1].text.strip())
            if seeds == 0:
                continue  # Skip torrents with zero seeds
            download_button = [col.find('button', class_='ui blue basic button') for col in cols]
            download_link = []
            for button in download_button:
                if button:
                    onclick_text = button.get('onclick')
                    link = onclick_text.split("'")[1]
                    full_link = "https://2torrentz2eu.in/beta2/page.php?url=" + link
                    download_link.append(full_link)
            # Remove empty strings from download_link
            download_link = [link for link in download_link if link]
            if not download_link:
                continue  # Skip torrents without a download link
            # Get magnet link
            magnet_link_response = requests.get(download_link[0])
            magnet_soup = BeautifulSoup(magnet_link_response.text, 'html.parser')
            magnet_link_tag = magnet_soup.find('a', class_='download-button', id='magnet')
            if not magnet_link_tag or 'magnet:?xt=' not in magnet_link_tag.get('href'):
                continue  # Skip torrents without a magnet link
            cols = [col.text.strip() for col in cols]
            # Create a dictionary for each row
            row_dict = {
                "Title": cols[0],
                "Seeds": seeds,
                "Leeches": int(cols[2]),
                "Size": cols[3],
                "Date": cols[4],
                "Download": download_link[0]
            }
            data.append(row_dict)
        response = make_response(jsonify({"movies": data}), 200)
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response
    except Exception as e:
        response = make_response(jsonify({"error": str(e)}), 500)
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

@app.route('/movie', methods=['GET'])
def movie():
    try:
        name = request.args.get('name')
        url = "https://2torrentz2eu.in/beta2/search.php?torrent-query=" + name
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        rows = table.find_all('tr')
        data = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 5:
                continue  # Skip rows with fewer than 5 td elements
            seeds = int(cols[1].text.strip())
            if seeds == 0:
                continue  # Skip torrents with zero seeds
            download_button = [col.find('button', class_='ui blue basic button') for col in cols]
            download_link = []
            for button in download_button:
                if button:
                    onclick_text = button.get('onclick')
                    link = onclick_text.split("'")[1]
                    full_link = "https://2torrentz2eu.in/beta2/page.php?url=" + link
                    download_link.append(full_link)
            # Remove empty strings from download_link
            download_link = [link for link in download_link if link]
            cols = [col.text.strip() for col in cols]
            title = cols[0]
            parsed_title = PTN.parse(title)  # Parse the title
            # Create a dictionary for each row
            row_dict = {
                "mTitle": title,
                "mSeeds": seeds,
                "mLeeches": int(cols[2]),
                "mSize": cols[3],
                "mDate": cols[4],
                "mDownload": download_link[0]
            }
            # Merge parsed title into row_dict
            row_dict.update(parsed_title)
            data.append(row_dict)
        response = make_response(jsonify({"movies": data}), 200)
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response
    except Exception as e:
        response = make_response(jsonify({"error": str(e)}), 500)
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

@app.route('/1337', methods=['GET'])
def leet():
    try:
        name = request.args.get('name')
        url = "https://www.1377x.to/search/" + name + "/1/"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')
        rows = table.find_all('tr')
        data = []
        for row in rows[1:]:  # Skip the header row
            cols = row.find_all('td')
            seeds = int(cols[1].text.strip())
            if seeds == 0:
                continue  # Skip torrents with zero seeds
            title_link = cols[0].find('a', href=True, class_=False)  # Exclude the icon link
            title = title_link.text
            download_link = "https://www.1377x.to" + title_link.get('href')
            # Get magnet link
            magnet_response = requests.get(download_link)
            magnet_soup = BeautifulSoup(magnet_response.text, 'html.parser')
            magnet_link_tag = magnet_soup.find('a', href=lambda href: href and href.startswith('magnet:?'))
            if not magnet_link_tag:
                continue  # Skip torrents without a magnet link
            magnet_link = magnet_link_tag.get('href')
            # Create a dictionary for each row
            row_dict = {
                "Title": title,
                "Seeds": seeds,
                "Leeches": int(cols[2].text.strip()),
                "Size": cols[4].text,  # Don't strip "Size"
                "Date": cols[3].text,  # Don't strip "Date"
                "Uploader": cols[5].text,
                "Download": download_link,
                "Magnet": magnet_link
            }
            data.append(row_dict)
        response = make_response(jsonify({"movies": data}), 200)
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response
    except Exception as e:
        response = make_response(jsonify({"error": str(e)}), 500)
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response


@app.route('/info', methods=['GET'])
def info():
    name = request.args.get('name')
    api_key = "0308f0a9278f09cbd10fe7441ccc6664"  # replace with your actual TMDB API key

    # Make API call to TMDB for movies
    movie_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={name}"
    movie_response = requests.get(movie_url)
    movie_data = movie_response.json()

    # Make API call to TMDB for TV shows
    tv_url = f"https://api.themoviedb.org/3/search/tv?api_key={api_key}&query={name}"
    tv_response = requests.get(tv_url)
    tv_data = tv_response.json()

    # Combine the results
    results = []
    for movie in movie_data.get('results', []):
        poster_path = movie.get('poster_path')
        if poster_path is None:
            poster_path = "https://i.ibb.co/72MHwtr/No-Image-Placeholder.jpg"
        else:
            poster_path = f"https://www.themoviedb.org/t/p/w94_and_h141_bestv2{poster_path}"
        results.append({
            'name': movie.get('title'),
            'poster_image_path': poster_path,
            'overview': movie.get('overview'),
            'release_date': movie.get('release_date'),
            'type': 'Movie',
            'popularity': movie.get('popularity', 0),
            'original_language': movie.get('original_language')
        })

    for tv_show in tv_data.get('results', []):
        poster_path = tv_show.get('poster_path')
        if poster_path is None:
            poster_path = "https://i.ibb.co/72MHwtr/No-Image-Placeholder.jpg"
        else:
            poster_path = f"https://www.themoviedb.org/t/p/w94_and_h141_bestv2{poster_path}"
        results.append({
            'name': tv_show.get('name'),
            'poster_image_path': poster_path,
            'overview': tv_show.get('overview'),
            'release_date': tv_show.get('first_air_date'),
            'type': 'TV Show',
            'popularity': tv_show.get('popularity', 0),
            'original_language': tv_show.get('original_language')
        })

    # Sort the results by popularity and prioritize items with images
    results.sort(key=lambda x: (x['poster_image_path'] == "https://i.ibb.co/72MHwtr/No-Image-Placeholder.jpg", -x['popularity']))

    return jsonify({'info': results})


@app.route('/zcreate', methods=['POST'])
def create_zoom_meeting():
    user_id = request.json.get('user_id')
    api_key = request.json.get('api_key')
    api_secret = request.json.get('api_secret')
    topic = request.json.get('topic')
    start_time = request.json.get('start_time')  # Expected format: '2023-07-21T10:00:00'
    duration = request.json.get('duration')  # Expected in minutes
    agenda = request.json.get('agenda')
    password = request.json.get('password')

    # Generate a JWT
    payload = {
        'iss': api_key,
        'exp': datetime.now() + timedelta(minutes=15)
    }
    token = jwt.encode(payload, api_secret, algorithm='HS256')

    # Create the meeting
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
        'topic': topic,
        'type': 2,  # Scheduled meeting
        'start_time': start_time,
        'duration': duration,
        'timezone': 'Asia/Kuwait',
        'agenda': agenda,
        'password': password
    }
    response = requests.post(f'https://api.zoom.us/v2/users/{user_id}/meetings', headers=headers, json=data)

    # Return the meeting info
    return jsonify(response.json())

@app.route('/zupdate', methods=['PATCH'])
def update_zoom_meeting():
    meeting_id = request.json.get('meeting_id')
    api_key = request.json.get('api_key')
    api_secret = request.json.get('api_secret')
    topic = request.json.get('topic')
    start_time = request.json.get('start_time')  # Expected format: '2023-07-21T10:00:00'
    duration = request.json.get('duration')  # Expected in minutes
    agenda = request.json.get('agenda')
    password = request.json.get('password')

    # Generate a JWT
    payload = {
        'iss': api_key,
        'exp': datetime.now() + timedelta(minutes=15)
    }
    token = jwt.encode(payload, api_secret, algorithm='HS256')

    # Update the meeting
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
        'topic': topic,
        'type': 2,  # Scheduled meeting
        'start_time': start_time,
        'duration': duration,
        'timezone': 'Asia/Kuwait',
        'agenda': agenda,
        'password': password
    }
    response = requests.patch(f'https://api.zoom.us/v2/meetings/{meeting_id}', headers=headers, json=data)

    # Check the response
    if response.status_code == 204:
        return '', 204
    else:
        return jsonify(response.json()), response.status_code

@app.route('/zinfo', methods=['GET'])
def get_zoom_meeting():
    meeting_id = request.args.get('meeting_id')
    api_key = request.args.get('api_key')
    api_secret = request.args.get('api_secret')

    # Generate a JWT
    payload = {
        'iss': api_key,
        'exp': datetime.now() + timedelta(minutes=15)
    }
    token = jwt.encode(payload, api_secret, algorithm='HS256')

    # Get the meeting info
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    response = requests.get(f'https://api.zoom.us/v2/meetings/{meeting_id}', headers=headers)

    # Return the meeting infod
    return jsonify(response.json())

@app.route('/logs', methods=['GET'])
def get_logs():
    start_date = request.args.get('start_date')  # Get the start date from query parameter
    end_date = request.args.get('end_date')  # Get the end date from query parameter

    # Check if both start date and end date are provided
    if start_date and end_date:
        logs = []
        with open(log_file_path, 'r') as file:
            for line in file:
                log_data = line.strip().split(' - ')
                log_timestamp = log_data[0]
                log_level = log_data[1]
                log_message = log_data[2]
                if start_date <= log_timestamp <= end_date:
                    logs.append({
                        'timestamp': log_timestamp,
                        'level': log_level,
                        'message': log_message
                    })
        return jsonify(logs)

    else:
        return 'Please provide both start_date and end_date query parameters.', 400


@app.route('/freebird', methods=['POST'])
def freebird():
    # Step 1: Receive a Request
    magnet_link = request.json['magnet_link']

    # Step 2: Generate Authorization Token
    token = '6G7ZWHULQ7WXTTX6DD4CJKGA3OYY6F7HMXHYVL6JS6KXO3YSZAJQ'  # replace with your actual token

    # Step 3: Add Magnet to Real-Debrid
    headers = {'Authorization': 'Bearer ' + token}
    data = {'magnet': magnet_link}
    response = requests.post('https://api.real-debrid.com/rest/1.0/torrents/addMagnet', headers=headers, data=data)

    # Step 4: Retrieve file list and filter for video files
    torrent_id = response.json()['id']
    response = requests.get(f'https://api.real-debrid.com/rest/1.0/torrents/info/{torrent_id}', headers=headers)
    files = response.json()['files']
    video_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpeg', '.3gp', '.webm', '.vob', '.rm', '.swf', '.asf', '.ts', '.mpg', '.m4v', '.ogv', '.qt', '.rmvb', '.mts']
    video_files = [file['id'] for file in files if any(file['path'].lower().endswith(ext) for ext in video_formats)]

    # Step 5: Select video files to download
    data = {'files': ','.join(map(str, video_files))}
    response = requests.post(f'https://api.real-debrid.com/rest/1.0/torrents/selectFiles/{torrent_id}', headers=headers, data=data)

    # Step 6: Wait for Download to Complete
    while True:
        response = requests.get(f'https://api.real-debrid.com/rest/1.0/torrents/info/{torrent_id}', headers=headers)
        if response.json()['status'] == 'downloaded':
            break

    # Step 7: Get Download Links and Parse Titles
    links = response.json()['links']
    download_links = []
    for i, link in enumerate(links, start=1):
        data = {'link': link}
        response = requests.post('https://api.real-debrid.com/rest/1.0/unrestrict/link', headers=headers, data=data)
        download_link = response.json()['download']
        parsed_title = PTN.parse(download_link.split('/')[-1])
        download_links.append({
            'id': i,
            'download_link': download_link,
            'title': parsed_title.get('title'),
            'year': parsed_title.get('year'),
            'resolution': parsed_title.get('resolution'),
            'codec': parsed_title.get('codec'),
            'encoder': parsed_title.get('group'),
            'filetype': parsed_title.get('container'),
            'quality': parsed_title.get('quality'),
            'size': parsed_title.get('size'),
        })

    # Step 8: Return Download Links
    return {'download_links': download_links}
####################################################################################################################################################

@app.route('/send_sms', methods=['POST'])
def send_sms():
    try:
        # Get your Twilio credentials from environment variables
        account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
        auth_token = os.environ.get('TWILIO_AUTH_TOKEN')
        twilio_phone_number = os.environ.get('TWILIO_PHONE_NUMBER')

        # Initialize Twilio client
        client = Client(account_sid, auth_token)

        # Get parameters from the request
        message_body = request.json.get('message')
        to_numbers = request.json.get('to_numbers')

        # Ensure to_numbers is a list
        if not isinstance(to_numbers, list):
            to_numbers = [to_numbers]

        # Send SMS to each number
        for number in to_numbers:
            client.messages.create(body=message_body, from_=twilio_phone_number, to=number)

        return jsonify({"status": "success", "message": "SMS sent successfully!"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


#####################################################################################################################################################
API_ENDPOINT = "https://dev.vdocipher.com/api/videos/{}/otp"
API_SECRET_KEYS = "vBEWfxrM2S60wYiLfpyNT2vD5PNvuKKWmCXJCeyJY0Y02ZCXoqEIUcXvs7xzAg74"  # Replace this with your actual API Secret Key

@app.route('/get_otp_playback', methods=['POST'])
def get_otp_playback():
    try:
        video_id = request.json.get('video_id')
        name = request.json.get('name', '')  # Default to empty string if not provided
        user_id = request.json.get('userId', '')  # Default to empty string if not provided
        ttl = request.json.get('ttl', 300)
        
        if not video_id:
            return jsonify({"status": "error", "message": "video_id is required!"}), 400

        # Only check the format of user_id if it is provided
        if user_id and (len(user_id) > 36 or not re.match("^[a-zA-Z0-9_-]+$", user_id)):
            return jsonify({"status": "error", "message": "Valid userId is required!"}), 400

        annotation = json.dumps([{
            'type': 'rtext',
            'text': ' {}'.format(name),
            'alpha': '0.60',
            'color': '0x000000',
            'size': '15',
            'interval': '30000',
            'skip': '15000'
        }])

        headers = {
            'Accept': 'application/json',
            'Authorization': 'Apisecret ' + API_SECRET_KEYS,
            'Content-Type': 'application/json'
        }

        data = {
            "ttl": ttl,
            "annotate": annotation if name else "",  # Only send annotation if name is provided
            "userId": user_id
        }

        response = requests.post(API_ENDPOINT.format(video_id), headers=headers, json=data)
        response_data = response.json()
        
        if response.status_code != 200:
            return jsonify({"status": "error", "message": response_data.get('message', 'Unknown error')}), response.status_code

        return jsonify({
            "otp": response_data["otp"],
            "playbackInfo": response_data["playbackInfo"]
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


########################################################################################################################################################
VDO_CIPHER_API_URL = "https://dev.vdocipher.com/api/videos"
VDO_CIPHER_API_SECRET = "vBEWfxrM2S60wYiLfpyNT2vD5PNvuKKWmCXJCeyJY0Y02ZCXoqEIUcXvs7xzAg74"  # Replace with your actual secret

@app.route("/create_video", methods=["PUT"])
def create_video():
    title = request.args.get("title")
    folder_id = request.args.get("folderId")
    tags_string = request.args.get("tags")  # A comma-separated list of tags

    url = f"{VDO_CIPHER_API_URL}?title={title}"
    if folder_id:
        url += f"&folderId={folder_id}"

    headers = {
        "Authorization": f"Apisecret {VDO_CIPHER_API_SECRET}"
    }

    create_response = requests.put(url, headers=headers)
    video_data = create_response.json()
    
    if tags_string and 'videoId' in video_data:
        tags = tags_string.split(',')  # Split the string into an array of tags
        tag_url = "https://dev.vdocipher.com/api/videos/tags"
        tag_payload = {
            "videos": [video_data["videoId"]],
            "tags": tags
        }
        tag_headers = {
            "Authorization": f"Apisecret {VDO_CIPHER_API_SECRET}",
            "Content-Type": "application/json"
        }
        requests.post(tag_url, json=tag_payload, headers=tag_headers)

    return jsonify(video_data)

if __name__ == "__main__":
    app.run(debug=True)

##############################################################################################################################################################

VDO_CIPHER_API_SECRET = "vBEWfxrM2S60wYiLfpyNT2vD5PNvuKKWmCXJCeyJY0Y02ZCXoqEIUcXvs7xzAg74"  # replace with your actual secret
VDO_CIPHER_VIDEO_API_URL = "https://dev.vdocipher.com/api/videos/"

@app.route("/video_status/<videoID>", methods=["GET"])
def video_status(videoID):
    # Create the URL with the videoID
    url = f"{VDO_CIPHER_VIDEO_API_URL}{videoID}"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Apisecret {VDO_CIPHER_API_SECRET}",
        "Content-Type": "application/json"
    }

    # Make the GET request
    response = requests.get(url, headers=headers)

    # Forward the response
    return jsonify(response.json())

if __name__ == "__main__":
    app.run(debug=True)
    
##############################################################################################################################################################
VDO_CIPHER_API_SECRET = "vBEWfxrM2S60wYiLfpyNT2vD5PNvuKKWmCXJCeyJY0Y02ZCXoqEIUcXvs7xzAg74"
VDO_CIPHER_VIDEO_API_URL = "https://dev.vdocipher.com/api/videos/"

@app.route("/delete_videos", methods=["DELETE"])
def delete_videos():
    video_ids = request.args.get('videos')
    
    if not video_ids:
        return jsonify({"status": "error", "message": "No video IDs provided!"}), 400
    
    url = f"{VDO_CIPHER_API_URL}?videos={video_ids}"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Apisecret {VDO_CIPHER_API_SECRET}",
        "Content-Type": "application/json"
    }

    # Make the DELETE request
    response = requests.delete(url, headers=headers)

    # Forward the response
    return jsonify(response.json())

##############################################################################################################################################################
VDO_CIPHER_API_SECRET = "vBEWfxrM2S60wYiLfpyNT2vD5PNvuKKWmCXJCeyJY0Y02ZCXoqEIUcXvs7xzAg74"
VDO_CIPHER_VIDEO_API_URL = "https://dev.vdocipher.com/api/videos/"

@app.route("/rename_video/<videoID>", methods=["POST"])
def rename_video(videoID):
    # Get the new title and description from the request
    new_title = request.json.get('title')
    new_description = request.json.get('description')  # Get the optional description
    
    if not new_title:
        return jsonify({"status": "error", "message": "title is required"}), 400

    # Create the URL with the videoID
    url = f"{VDO_CIPHER_VIDEO_API_URL}{videoID}"

    headers = {
        "Accept": "application/json",
        "Authorization": f"Apisecret {VDO_CIPHER_API_SECRET}",
        "Content-Type": "application/json"
    }

    data = {
        "title": new_title
    }

    # If description is provided, add it to the data
    if new_description:
        data["description"] = new_description

    # Make the POST request
    response = requests.post(url, headers=headers, json=data)

    # Forward the response
    return jsonify(response.json())
#################################################################################################################################################

#VDO_CIPHER_API_SECRET = "fk1MyponnRSFWrSPYIOGu5DacEfUgy5H1fsshq6ny8jWk0bdCbUkbKuHI92WLrRZ"
VDO_CIPHER_API_SECRET = "vBEWfxrM2S60wYiLfpyNT2vD5PNvuKKWmCXJCeyJY0Y02ZCXoqEIUcXvs7xzAg74"
VDO_CIPHER_VIDEO_API_URL = "https://dev.vdocipher.com/api/videos/"

@app.route('/poster/<videoID>', methods=['POST'])
def poster(videoID):
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file:
        # Temporary save the file
        filepath = os.path.join("/tmp", secure_filename(file.filename))
        file.save(filepath)

        # Open the file for reading
        with open(filepath, 'rb') as f:
            files = {'file': (file.filename, f, file.mimetype)}
            response = requests.post(
                f"https://dev.vdocipher.com/api/videos/{videoID}/files",
                headers={
                    'Authorization': f'Apisecret {VDO_CIPHER_API_SECRET}'
                },
                files=files
            )

        # Clean up: Delete the temporarily saved file
        os.remove(filepath)

        return jsonify(response.json())
##############################################################################################################################

#API_KEY = "XY3B4WH5dtdco6pyh6BNOm1aqlfPFPicu2IzQNO07wnCzDhfqIGSg2jM4sBVebtG"
API_KEY = "vBEWfxrM2S60wYiLfpyNT2vD5PNvuKKWmCXJCeyJY0Y02ZCXoqEIUcXvs7xzAg74"
VDOCIPHER_ENDPOINT = "https://dev.vdocipher.com/api/videos"

@app.route('/create_folder', methods=['POST'])
def create_folder():
    data = request.json
    folder_name = data.get('folderName', '')
    parent_folder_id = data.get('parentFolderID', '')

    headers = {
        'Accept': 'application/json',
        'Authorization': 'Apisecret ' + API_KEY,
        'Content-Type': 'application/json'
    }

    payload = {
        "name": folder_name,
        "parent": parent_folder_id
    }

    response = requests.post(f"{VDOCIPHER_ENDPOINT}/folders", headers=headers, json=payload)

    return jsonify(response.json()), response.status_code

@app.route('/get_folder/<folderID>', methods=['GET'])
def get_folder(folderID):
    headers = {
        'Accept': 'application/json',
        'Authorization': 'Apisecret ' + API_KEY
    }

    response = requests.get(f"{VDOCIPHER_ENDPOINT}/folders/{folderID}", headers=headers)

    return jsonify(response.json()), response.status_code

@app.route('/get_videos/<folderID>', methods=['GET'])
def get_videos(folderID):
    headers = {
        'Accept': 'application/json',
        'Authorization': 'Apisecret ' + API_KEY
    }

    response = requests.get(f"{VDOCIPHER_ENDPOINT}?folderId={folderID}", headers=headers)

    return jsonify(response.json()), response.status_code
################################################################################################################################

VDO_CIPHER_API_URL = "https://dev.vdocipher.com/api/videos"
VDO_CIPHER_API_SECRET = "vBEWfxrM2S60wYiLfpyNT2vD5PNvuKKWmCXJCeyJY0Y02ZCXoqEIUcXvs7xzAg74"  # Replace with your actual secret

# Existing /create_video endpoint...

@app.route("/fetch_videos", methods=["GET"])
def fetch_videos():
    tag = request.args.get("tag")

    if not tag:
        return jsonify({"error": "Tag is required"}), 400

    url = f"{VDO_CIPHER_API_URL}?tags={tag}"
    headers = {
        "Authorization": f"Apisecret {VDO_CIPHER_API_SECRET}",
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers)
    return jsonify(response.json())

if __name__ == "__main__":
    app.run(debug=True)

##############################################################################################################################

VDO_CIPHER_API_URL = "https://dev.vdocipher.com/api/videos"
VDO_CIPHER_API_SECRET = "vBEWfxrM2S60wYiLfpyNT2vD5PNvuKKWmCXJCeyJY0Y02ZCXoqEIUcXvs7xzAg74"  # Replace with your actual secret


@app.route("/videos_in_folder/<folder_id>", methods=["GET"])
def videos_in_folder(folder_id):
    url = f"{VDO_CIPHER_API_URL}/folders/{folder_id}"
    headers = {
        "Authorization": f"Apisecret {VDO_CIPHER_API_SECRET}",
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers)
    return jsonify(response.json())

if __name__ == "__main__":
    app.run(debug=True)

#############################################################################################################################


def parse_key_pair_values(value, allow_empty):
    """Parse values into lists or dictionaries, or keep as strings based on their format."""
    if not isinstance(value, str):  # Return non-string values as-is
        return value

    if value.strip() == '':
        return '' if allow_empty else None

    if '|' in value:
        # Split by pipe and further process each part
        parts = value.split('|')
        if ':' in value:  # Check for key-value pairs
            attributes = {}
            for part in parts:
                if ':' in part:
                    key, val = part.split(':', 1)
                    attributes[key.strip()] = val.strip()
            return attributes
        else:  # Handle as a list
            return [part.strip() for part in parts]

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
    import_method = request.args.get('importMethod', 'addUpdate')  # Default to 'addUpdate'

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
            # Construct the API URL with additional parameters, including importMethod
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
################################################################################################################################################

@app.route('/upload_image', methods=['POST'])
def upload_image():
    # Set API credentials
    api_token = 'c5b3d2f0510eb3ba3689ca78dd8ed7c5826b8'
    api_key = 'c5b3d2f0510eb3ba3689ca78dd8ed7c5826b8'
    api_email = 'it@vizsoft.in'

    # Extract the file, optional endpoint, and project from the request
    file = request.files['file']
    endpoint = request.form.get('endpoint')
    project = request.form.get('project')

    # Prepare the headers and files for the API call to Cloudflare
    headers = {
        'Authorization': f'Bearer {api_token}',
        'X-Auth-Key': api_key,
        'X-Auth-Email': api_email
    }
    files = {
        'file': (file.filename, file, file.content_type)
    }

    # Make the API call to Cloudflare
    response = requests.post(
        "https://api.cloudflare.com/client/v4/accounts/04fe94c75741abc6c2ee2bf26c54875a/images/v1",
        headers=headers,
        files=files
    )

    # Parse the response JSON
    response_json = response.json()

    # Modify the response JSON to include 'project' inside 'result'
    if project and 'result' in response_json:
        response_json['result']['project'] = project

    # Check if an additional endpoint is provided
    if endpoint:
        # Forward the modified response to the specified endpoint
        forwarded_response = requests.post(endpoint, json=response_json)
        return jsonify(forwarded_response.json()), forwarded_response.status_code

    # Return the modified response
    return jsonify(response_json), response.status_code

if __name__ == '__main__':
    app.run(debug=False)
