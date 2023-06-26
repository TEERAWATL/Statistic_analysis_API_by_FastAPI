# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose a port for the Flask application
EXPOSE 8000

# Run the application:
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]