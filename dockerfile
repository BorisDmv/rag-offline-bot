# Use an official lightweight Python base image
FROM python:3.9-slim

# Install system dependencies (build-essential, etc.) needed for some Python packages
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Then install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# Copy application code
COPY . .

# Expose the port Flask will run on
EXPOSE 9090

# Use Gunicorn to serve the Flask app (assumes app factory or app object in app.py)
CMD ["gunicorn", "-c", "gunicorn_config.py", "rag-flask-bot:app"]