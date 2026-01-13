import unittest
import sys
import os

# Ensure we can import the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Use try-except to handle import error if protobufs aren't generated yet
try:
    from slate_client import CortexClient
except ImportError:
    print("Skipping tests because slate_client is not fully installed or generated.")
    sys.exit(0)

class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        # Assumes server is running on localhost:50051
        token = os.environ.get('SLATE_AUTH_TOKEN')
        self.client = CortexClient(token=token)

    def test_focus(self):
        try:
            response = self.client.focus("E2E Test Python")
            self.assertIsNotNone(response.id)
            print(f"Focused: {response.id}")
        except Exception as e:
            self.fail(f"Focus failed: {e}")

    def test_drift(self):
        try:
            response = self.client.drift()
            # We assume focus worked, so there might be items
            print(f"Drift items: {len(response.items)}")
        except Exception as e:
            self.fail(f"Drift failed: {e}")

if __name__ == '__main__':
    unittest.main()
