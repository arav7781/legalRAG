FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY req.txt .
RUN pip install --no-cache-dir -r req.txt

# Copy application code
COPY app.py .

# Create tmp directory for temporary files
RUN mkdir -p /tmp && chmod 777 /tmp

# Expose port
EXPOSE 7860

# Set environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "app.py"]