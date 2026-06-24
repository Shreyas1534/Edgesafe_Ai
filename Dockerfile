# 1. Use a highly stable version of Python for AI/ML (Python 3.11)
FROM python:3.11-slim

# 2. Set the working directory inside the server
WORKDIR /app

# 3. Install core graphical system dependencies just in case
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your requirements file
COPY requirements.txt .

# 5. Install all base Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 6. THE MAGIC FIX: Forcefully rip out the GUI version of OpenCV and install the Headless version
RUN pip uninstall -y opencv-python && \
    pip install --no-cache-dir opencv-python-headless

# 7. Copy the rest of your application code
COPY . .

# 8. CRITICAL: Prevent PyTorch from spawning too many threads and causing a SIGSEGV
ENV OMP_NUM_THREADS=1

# 9. Run the native Flask server directly (bypassing Gunicorn completely)
CMD ["python", "app.py"]