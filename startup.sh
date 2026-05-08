#!/bin/bash
# Azure App Service Startup Script for Veritas
# Oryx extracts compressed output to a temp directory but doesn't always
# set the working directory correctly. This script finds the app root
# and ensures gunicorn can locate the wsgi module.

# Try Oryx's APP_PATH first, then search for wsgi.py in /tmp
if [ -d "$APP_PATH" ] && [ -f "$APP_PATH/wsgi.py" ]; then
    APP_DIR="$APP_PATH"
elif [ -f "/home/site/wwwroot/wsgi.py" ]; then
    APP_DIR="/home/site/wwwroot"
else
    APP_DIR=$(dirname "$(find /tmp -maxdepth 2 -name 'wsgi.py' -type f 2>/dev/null | head -1)" 2>/dev/null)
fi

if [ -z "$APP_DIR" ] || [ ! -f "$APP_DIR/wsgi.py" ]; then
    echo "ERROR: Could not find wsgi.py in any expected location"
    exit 1
fi

echo "App directory: $APP_DIR"
cd "$APP_DIR"
export PYTHONPATH="$APP_DIR:$PYTHONPATH"

exec gunicorn --bind=0.0.0.0:8000 --timeout 600 wsgi:app
