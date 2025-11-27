# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the port Render uses
ENV PORT=5000

# Command to run the app using Gunicorn
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000"]
