FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dépendances système (xgboost, build) + TShark
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 build-essential gcc git curl ca-certificates \
    tshark \
  && rm -rf /var/lib/apt/lists/*

# (tshark installe dumpcap avec les bonnes capabilities setcap, donc pas besoin de root extra)
WORKDIR /app

# Déps web (Flask, Socket.IO, eventlet, gunicorn)
COPY requirementsweb.txt /app/requirementsweb.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirementsweb.txt

# Déps ML
COPY requirementsml.txt /app/requirementsml.txt
RUN pip install --no-cache-dir -r /app/requirementsml.txt

# Code
COPY app.py /app/app.py

EXPOSE 5000

# Production: Gunicorn + eventlet (Socket.IO)
CMD ["gunicorn","--chdir","/app","-k","eventlet","-w","1","--log-level","info","-b","0.0.0.0:5000","app:app"]
