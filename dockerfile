# Use official Python image with Debian (has required build tools)
FROM python:3.11-slim-bookworm

# Install system dependencies required by PyMuPDF
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff6 \
    libwebp7 \
    libopenjp2-7 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
# Use pre-built wheel for PyMuPDF to avoid compilation
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create temp directory for PDF processing
RUN mkdir -p /tmp/tafsir_temp

# Expose port (Render uses PORT env variable)
EXPOSE 8000

# Run the application
CMD ["python", "render_ui.py"]