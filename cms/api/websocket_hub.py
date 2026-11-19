"""Real-Time WebSocket Event Hub & SSE Telemetry Streamer for CMS.

Broadcasts live crowd counts, pose safety events (fall, collapse, panic),
and polygonal zone breach transitions to all connected security dashboards.
"""

import asyncio
import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Set

from fastapi import WebSocket


class WebSocketHub:
    """Manages active WebSocket dashboard clients and broadcasts telemetry."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.lock = asyncio.Lock()
        self.main_loop: Optional[asyncio.AbstractEventLoop] = None
        self.recent_events: List[Dict[str, Any]] = []

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the primary FastAPI asyncio event loop."""
        self.main_loop = loop

    async def connect(self, websocket: WebSocket) -> None:
        """Accept new WebSocket client and store connection."""
        await websocket.accept()
        async with self.lock:
            self.active_connections.add(websocket)
        # Send initial handshake with recent events
        try:
            await websocket.send_text(
                json.dumps({
                    "event_type": "handshake",
                    "status": "connected",
                    "timestamp": time.time(),
                    "recent_events": self.recent_events[-15:],
                })
            )
        except Exception:
            pass

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove disconnected client."""
        async with self.lock:
            self.active_connections.discard(websocket)

    async def broadcast(self, event_data: Dict[str, Any]) -> None:
        """Broadcast payload to all active WebSocket clients."""
        if "timestamp" not in event_data:
            event_data["timestamp"] = time.time()

        # Buffer recent events
        self.recent_events.append(event_data)
        if len(self.recent_events) > 100:
            self.recent_events.pop(0)

        message_str = json.dumps(event_data)
        async with self.lock:
            stale_connections = set()
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_str)
                except Exception:
                    stale_connections.add(connection)
            for dead in stale_connections:
                self.active_connections.discard(dead)

    def broadcast_sync(self, event_data: Dict[str, Any]) -> None:
        """Safely schedule broadcast from background worker threads."""
        if self.main_loop and self.main_loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(event_data), self.main_loop)

    async def event_generator(self) -> AsyncIterator[str]:
        """SSE fallback generator streaming events as server-sent events."""
        last_index = len(self.recent_events)
        while True:
            await asyncio.sleep(0.5)
            if len(self.recent_events) > last_index:
                for idx in range(last_index, len(self.recent_events)):
                    data = json.dumps(self.recent_events[idx])
                    yield f"data: {data}\n\n"
                last_index = len(self.recent_events)
            else:
                # Heartbeat ping
                yield f": ping {time.time()}\n\n"


# Shared singleton instance
ws_hub = WebSocketHub()
