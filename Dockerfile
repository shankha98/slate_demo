FROM python:3.12-slim

# Install uv (updated lightweight package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency definition
COPY pyproject.toml uv.lock ./
# Copy local dependency directory structure
# Note: slate-client is referenced as ./slate_client/python in pyproject.toml
COPY slate_client ./slate_client

# Install dependencies
# --frozen ensures we stick to uv.lock
# --no-dev excludes dev dependencies (ruff, pytest, etc.) for production
RUN uv sync --frozen --no-dev

# Copy application code
COPY app ./app

# Expose port
EXPOSE 8000

# Run the application
# Use shell form to allow variable expansion if needed, though exec form is preferred.
# Removed env file dependency, configuration is now runtime or defaults.
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
