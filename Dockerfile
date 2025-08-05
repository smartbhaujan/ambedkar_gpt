# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy required files before installing dependencies
COPY requirements.txt ./

# Install dependencies system-wide (no virtual environment)
RUN pip install --upgrade pip && \
    pip install flask flask_cors gunicorn && \
    pip install -r requirements.txt 

    # ! Note (RUN Flask in local)
    # RUN python -m venv myenv && \
    # . myenv/bin/activate && \
    # pip install --upgrade pip && \
    # pip install flask flask_cors gunicorn && \
    # pip install -r requirements.txt 
# Copy application files after dependencies (optimized layer caching)
COPY . .

# Ensure .env file is copied (and not ignored)
COPY .env ./

# Expose port 8080
EXPOSE 8080

# Start Gunicorn directly (without virtual environment)
# CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "run:app"]
CMD ["python", "run.py"]
