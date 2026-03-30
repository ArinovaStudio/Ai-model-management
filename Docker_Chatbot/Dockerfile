# Use the Debian Trixie slim Python 3.10 image
FROM python:3.10-slim-trixie

WORKDIR /app

# Install system dependencies (required for thefuzz and other packages)
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the specific spaCy NLP model
RUN python -m spacy download en_core_web_sm

# Copy the rest of the app code (including schema.prisma)
COPY . .

# Generate the Prisma Python Client
RUN prisma generate

# Expose the API port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]