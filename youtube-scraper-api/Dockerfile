# Use a stable and lightweight Python version
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# --- THIS IS THE NEW PART ---
# Install system dependencies required by the weasyprint PDF library
RUN apt-get update && apt-get install -y \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libharfbuzz0b

# First, upgrade pip itself to the latest version
RUN pip install --upgrade pip

# Now, copy and install the required Python libraries with the new pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python script into the container
COPY main.py .

# Command to run your script as a web server when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
