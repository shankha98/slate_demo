#!/bin/bash
# Install dependencies first: pip install -r requirements.txt
python3 -m grpc_tools.protoc -I../../crates/server/proto --python_out=src/slate_client --grpc_python_out=src/slate_client slate.proto
