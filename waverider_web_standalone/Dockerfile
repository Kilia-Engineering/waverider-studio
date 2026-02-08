FROM python:3.11-slim

# System deps for CadQuery (OpenCascade) and Gmsh
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglu1-mesa \
    libxft2 \
    libxinerama1 \
    libxcursor1 \
    libxrender1 \
    libfontconfig1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "app:app"]
