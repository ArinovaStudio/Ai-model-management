# 09. Setup Guide

Follow this guide to configure and run the AI Planning Agent locally.

---

## 1. Prerequisites
* **Python 3.11 to 3.14+** (Note: Python 3.14 emits compatibility warnings from Prisma Pydantic V1 bindings, which are safe to ignore).
* **Node.js & npm** (To run the development server for the static frontend).
* **Ollama Server** (Must be installed and running locally, with the target model pulled).

---

## 2. Installation Steps

### Step 1: Clone the Repository & Configure Environment Variables
Create a `.env` file in the root directory:
```env
OLLAMA_BASE_URL="http://localhost:11434" # Or your active tunnel endpoint
OLLAMA_MODEL="llama3"                   # Model pulled in Ollama
DATABASE_URL="postgresql://username:password@localhost:5432/db" # Or your Neon DB endpoint
```

### Step 2: Install Python Dependencies
Set up your virtual environment and install packages:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Initialize Prisma Client
Run Prisma commands to configure the client and apply schema definitions to the database:
```bash
prisma db push
prisma generate
```

---

## 3. Running the Application

### Step 1: Start the Backend (via Telemetry Launcher)
Start the FastAPI server through the profiling launcher script to enable database query latency and resource tracking:
```bash
python profile_launcher.py
```
The backend server will start on `http://127.0.0.1:8000`.

### Step 2: Start the Frontend Developer Server
In a separate terminal window, start the developer server for the static frontend:
```bash
cd frontend
python -m http.server 3000
```
Open your browser and navigate to `http://localhost:3000` to access the planning workspace interface.
