import sys
import os
sys.path.append(os.path.dirname(__file__))

import grpc
try:
    from . import slate_pb2
    from . import slate_pb2_grpc
except ImportError as e:
    print(f"Slate Client Import Error: {e}")
    # pass

class CortexClient:
    def __init__(self, address='localhost:50051', token=None, run_id='default'):
        self.channel = grpc.insecure_channel(address)
        self.stub = slate_pb2_grpc.CortexStub(self.channel)
        self.token = token
        self.run_id = run_id

    def _metadata(self):
        if self.token:
            return [('authorization', f'{self.token}')]
        return []

    def focus(self, content):
        return self.stub.Focus(
            slate_pb2.FocusRequest(content=content, run_id=self.run_id),
            metadata=self._metadata()
        )

    def drift(self):
        return self.stub.Drift(
            slate_pb2.DriftRequest(run_id=self.run_id),
            metadata=self._metadata()
        )

    def commit(self, input, outcome, reasoning="", action="", agent_id="user"):
        trace = slate_pb2.Trace(
            input=input,
            outcome=outcome,
            reasoning=reasoning,
            action=action,
            agent_id=agent_id,
            embedding=[0.0] * 768,
            run_id=self.run_id
        )
        return self.stub.Commit(trace, metadata=self._metadata())

    def reminisce(self, query_text, limit=5, filter=None):
        req = slate_pb2.RecallRequest(
            embedding=[0.0] * 768,
            limit=limit,
            query_text=query_text,
            filter=filter if filter else "",
            run_id=self.run_id
        )
        return self.stub.Reminisce(req, metadata=self._metadata())

    def trigger(self, skill_name):
        req = slate_pb2.ReflexRequest(skill_name=skill_name)
        return self.stub.Trigger(req, metadata=self._metadata())

    def delete_run(self):
        req = slate_pb2.RunRequest(run_id=self.run_id)
        return self.stub.DeleteRun(req, metadata=self._metadata())
