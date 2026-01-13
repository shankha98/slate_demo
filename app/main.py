import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRoute, Mount
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.agents import AgentLogger, MultiAgentSystem


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Print endpoints on startup
    print("\nServer endpoints:")
    for route in app.routes:
        if isinstance(route, APIRoute):
            methods = ", ".join(route.methods)
            print(f"  {methods} {route.path} -> {route.name}")
        elif isinstance(route, Mount):
            print(f"  Mount {route.path} -> {route.name}")
        else:
            # WebSocket or other
            path = getattr(route, "path", str(route))
            name = getattr(route, "name", "unknown")
            print(f"  {path} -> {name} ({type(route).__name__})")
    yield


app = FastAPI(title="Slate Multi-Agent Demo", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    run_id = str(uuid.uuid4())[:8]  # Generate a unique run_id for this session

    async def log_callback(data: dict):
        await websocket.send_json(data)

    logger = AgentLogger(log_callback)
    agent_system = MultiAgentSystem(run_id=run_id, logger=logger)

    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                if payload.get("type") == "message":
                    user_message = payload.get("content")
                    if user_message:
                        # Process message
                        response = await agent_system.process_message(user_message)
                        # Send final response
                        await websocket.send_json(
                            {"type": "final_response", "content": response}
                        )
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Error: {e}")
        manager.disconnect(websocket)
