# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV TWILIO_ACCOUNT_SID=AC5e038d5961399cc3fc7827956b04c1a7
ENV TWILIO_AUTH_TOKEN=e06d89c46c79d0f7396eec6aeed109d8
ENV TWILIO_PHONE_NUMBER=+18146377360

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    firefox-esr \
    wget \
    unzip \
    gnupg2 \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install yt-dlp
RUN wget https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -O /usr/local/bin/yt-dlp \
    && chmod a+rx /usr/local/bin/yt-dlp

# Install geckodriver
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.29.1/geckodriver-v0.29.1-linux64.tar.gz \
    && tar -xvzf geckodriver-v0.29.1-linux64.tar.gz \
    && mv geckodriver /usr/local/bin/ \
    && rm geckodriver-v0.29.1-linux64.tar.gz

# Upgrade pip and install Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
    flask \
    werkzeug \
    beautifulsoup4 \
    pytube \
    ytmusicapi \
    requests \
    chardet \
    ffmpeg-python \
    gunicorn \
    parse-torrent-title \
    pyjwt \
    selenium \
    cinemagoer \
    jsmin \
    css_html_js_minify \
    cssutils \
    htmlmin \
    instaloader \
    instagrapi \
    Pillow>=8.1.1 \
    twilio \
    pandas \
    openpyxl \
    numpy \
    opencv-python \
    mediapipe

# Add current directory files to /app in container
COPY . /app

# Run app.py (Flask server) when the container launches
CMD gunicorn --bind 0.0.0.0:$PORT app:app
