FROM python:3.11-slim

# Install wkhtmltopdf for PDF generation
RUN apt-get update && apt-get install -y \
    wkhtmltopdf \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variable for wkhtmltopdf path
ENV WKHTMLTOPDF_PATH=/usr/bin/wkhtmltopdf

# Run the script
CMD ["python", "scripts/generate_report.py"]
