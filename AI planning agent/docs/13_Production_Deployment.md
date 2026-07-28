# 13. Production Deployment Guide

Transitioning the AI Planning Agent from a local development environment to a production-grade infrastructure requires shifting from lightweight runtime servers to containerized, high-availability setups.

---

## 1. Production Architecture Overview
In production, the application is divided into three scalable tiers:

```
[ Frontend: CDN / Static Storage ]
               │
               ▼  (HTTP REST)
[ Backend Container: FastAPI + Gunicorn/Uvicorn ]  <──>  [ Relational DB: Neon PostgreSQL ]
               │
               ▼  (HTTP API / SSL)
[ AI Model Container: Dedicated GPU VM / Private Ollama Endpoint ]
```

---

## 2. Containerization (Dockerfile Blueprint)
To run the backend inside a production container:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate Prisma Client Python bindings
RUN prisma generate

EXPOSE 8000

# Start server using Uvicorn workers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## 3. High-Performance LLM Deployment (Gpu Hosting)
Running Ollama on a CPU local server is highly latent. For production:

1. **Dedicated VM Host:** Deploy Ollama on a cloud VM instance (AWS EC2, Google Cloud Compute Engine, or RunPod) equipped with an NVIDIA GPU (e.g. A10G, T4, or L4).
2. **Concurrency Configuration:** Start Ollama with environment variables tuned for concurrent requests:
   * `OLLAMA_NUM_PARALLEL=4`: Allows processing multiple planner requests in parallel.
   * `OLLAMA_MAX_LOADED_MODELS=1`: Pin the target model (`llama3`) in GPU memory to prevent model reload latency.

---

## 4. Serving the Frontend
* **Do NOT use `python -m http.server` in production.** It is a single-threaded blocker.
* **CDNs:** Host the static `frontend/` files on a Content Delivery Network (Vercel, Netlify, Cloudflare Pages, or Google Cloud Storage with CDN routing).
* **Reverse Proxy:** Alternatively, serve them via an Nginx block that proxies API requests:
  ```nginx
  server {
      listen 80;
      server_name planner.yourdomain.com;

      # Serve static frontend assets
      location / {
          root /usr/share/nginx/html;
          index index.html workspace.html;
      }

      # Proxy backend requests to FastAPI
      location /api/v1/ {
          proxy_pass http://localhost:8000;
          proxy_set_header Host $host;
          proxy_set_header X-Real-IP $remote_addr;
      }
  }
  ```

---

## 5. DB Migration Pipeline
For continuous integration and delivery (CI/CD):
1. **Never run `prisma db push` in production.** It can lead to accidental schema adjustments or data truncation.
2. **Prisma Migrations:** Generate and track SQL migration files:
   ```bash
   prisma migrate dev --name init
   ```
3. **CI/CD Execution:** In your CD pipeline (e.g., GitHub Actions), apply pending migrations before launching the new backend container:
   ```bash
   prisma migrate deploy
   ```
