# Slate Python Client

This is the Python client for the [Slate](https://github.com/rice-ai-hq/slate) cognitive architecture framework.

## Installation

```bash
pip install slate-client
```

## Usage

```python
from slate_client import CortexClient

client = CortexClient(address="localhost:50051", token="your-auth-token")

# Store memory
response = client.focus("Hello world")
print(f"Stored: {response.id}")

# Retrieve context
items = client.drift()
for item in items.items:
    print(item.content)
```

See [examples](./examples) for more advanced usage.
