# --- Build stage ---
FROM node:18-alpine AS build
WORKDIR /app

# copies minimales pour tirer parti du cache
COPY package*.json ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

# copie du reste et build
COPY . .
RUN npm run build

# --- Runtime stage ---
FROM nginx:alpine
# Nginx SPA config inline (pas de nginx.conf copié)
RUN printf 'server {\n  listen 80;\n  server_name _;\n  root /usr/share/nginx/html;\n  index index.html;\n  location / {\n    try_files $uri /index.html;\n  }\n}\n' > /etc/nginx/conf.d/default.conf

COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
