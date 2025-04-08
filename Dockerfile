# Use an official Python image as base
FROM python:3.11-slim

# Set environment variables to ensure output is logged instantly
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt if you have one
# (If you don't have one, you can create one from your dependencies)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose port 5000 for the Flask server
EXPOSE 5000

# Define the command to run the Flask server
# Assuming your Flask app is in app.py and using app.run()
CMD ["python", "flask_server.py"]
