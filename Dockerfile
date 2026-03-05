# ─── Stage 1: Build dependencies ───
FROM python:3.12-alpine AS builder

WORKDIR /app

# Install build deps needed for compiled packages (cryptography, etc.)
RUN apk add --no-cache gcc musl-dev libffi-dev openssl-dev

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─── Stage 2: Runtime (Cloud Run) ───
FROM python:3.12-alpine

WORKDIR /app

# Install only runtime system deps
RUN apk add --no-cache libffi openssl

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Cloud Run injects PORT env var (default 8080)
ENV PORT=8080
EXPOSE 8080

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
