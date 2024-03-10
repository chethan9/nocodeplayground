# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Set environment variables for Twilio
ENV TWILIO_ACCOUNT_SID=AC5e038d5961399cc3fc7827956b04c1a7
ENV TWILIO_AUTH_TOKEN=e06d89c46c79d0f7396eec6aeed109d8
ENV TWILIO_PHONE_NUMBER=+18146377360

# Set the working directory in the container to /app
WORKDIR /app

# FFmpeg, Firefox, and geckodriver installation
RUN apt-get update && \
    apt-get install -y ffmpeg firefox-esr wget unzip gnupg2 && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev && \
    python3 -m pip install --upgrade pip setuptools wheel && \
    mkdir -p /var/lib/data && \
    wget https://github.com/mozilla/geckodriver/releases/download/v0.29.1/geckodriver-v0.29.1-linux64.tar.gz -P /var/lib/data && \
    tar -xvzf /var/lib/data/geckodriver-v0.29.1-linux64.tar.gz -C /var/lib/data && \
    rm /var/lib/data/geckodriver-v0.29.1-linux64.tar.gz && \
    chmod +x /var/lib/data/geckodriver && \
    ln -s /var/lib/data/geckodriver /usr/local/bin/geckodriver && \
    apt-get remove -y wget unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Add current directory files to /app in container
ADD . /app

# Install necessary packages, including dlib
RUN pip install --no-cache-dir dlib

# Install other Python packages
RUN pip install --no-cache-dir flask werkzeug beautifulsoup4 pytube ytmusicapi requests chardet ffmpeg-python gunicorn parse-torrent-title pyjwt selenium cinemagoer jsmin css_html_js_minify cssutils htmlmin instaloader instagrapi Pillow>=8.1.1 twilio pandas openpyxl

# Run app.py (Flask server) when the container launches
CMD gunicorn --bind 0.0.0.0:$PORT app:app
