FROM python:3.10-slim

# Install OpenJDK 17 and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk \
    apt-utils \
    gcc \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for OpenJDK 17
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Start the Spark job
CMD ["python", "spark_job.py"]