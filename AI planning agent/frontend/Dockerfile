FROM nginx:alpine

# Copy custom Nginx configuration with API reverse proxy
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the frontend files to Nginx's public directory
COPY . /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Run Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]
