# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Ensure data folders exist
RUN mkdir -p /app/data/latest /app/data/raw

# Install cron
RUN apt-get update && apt-get install -y cron && apt-get clean

# Copy cron schedule
COPY cronjob /etc/cron.d/trinetra-cron

# Permissions
RUN chmod 0644 /etc/cron.d/trinetra-cron && crontab /etc/cron.d/trinetra-cron

# Expose port for web visualization
EXPOSE 8080

# Run both: cron (in background) + Flask server for static site
CMD ["sh", "-c", "cron && python3 -m http.server 8080 --directory /app/static"]
