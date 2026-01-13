FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency definition
COPY pyproject.toml uv.lock ./
# Copy local dependency directory structure
COPY slate_client ./slate_client

# Install dependencies
# --frozen ensures we stick to uv.lock
RUN uv sync --frozen

# Copy application code
COPY app ./app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
