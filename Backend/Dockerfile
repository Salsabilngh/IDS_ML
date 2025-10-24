FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dépendances système (+ tshark)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 build-essential gcc git curl ca-certificates tshark \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installe les deps Python (noms à adapter si besoin)
COPY requirementsweb.txt /app/requirementsweb.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirementsweb.txt

COPY requirementsml.txt /app/requirementsml.txt
RUN pip install --no-cache-dir -r /app/requirementsml.txt

# Code de l'API (ajoute d'autres fichiers si nécessaire)
COPY app.py /app/app.py

EXPOSE 5000
CMD ["gunicorn","--chdir","/app","-k","eventlet","-w","1","--log-level","info","-b","0.0.0.0:5000","app:app"]
