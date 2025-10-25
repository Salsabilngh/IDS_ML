# --- Build stage ---
FROM node:18-alpine AS build
WORKDIR /app

# copier les manifestes puis installer (ci si possible, sinon install)
COPY package*.json ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

# copier tout le frontend et builder
COPY . .
RUN npm run build

# --- Runtime stage ---
FROM nginx:alpine

# écrire la conf Nginx SPA directement (plus besoin de copier nginx.conf)
RUN printf 'server {\n\
  listen 80;\n\
  server_name _;\n\
  root /usr/share/nginx/html;\n\
  index index.html;\n\
  location / {\n\
    try_files $uri /index.html;\n\
  }\n\
}\n' > /etc/nginx/conf.d/default.conf

# servir le build
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
