FROM python:3.11-slim

# System deps: OpenCV, git (for pip installs from git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx libgomp1 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so the layer is cached when only code changes
COPY REATS/requirements.txt /app/requirements.txt

# Install Python deps (blinker conflict is a Debian-only issue; in Docker we skip it)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full REATS package
COPY REATS/ /app/REATS/

# Expose both services
EXPOSE 8501 7860

# Default: the Streamlit dashboard.
# docker-compose overrides this per service.
CMD ["streamlit", "run", "REATS/modules/module_d_dashboard.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
