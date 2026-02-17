"""
WebSocket Visualization Server for Live Trading Frontend.

Broadcasts real-time activation data from the trading bridge
to connected frontend clients.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from queue import Queue, Empty
from typing import Set, Dict, Any, Optional

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = None

logger = logging.getLogger(__name__)


@dataclass
class VisualizationMessage:
    """
    Complete visualization message sent to frontend.
    """
    timestamp: float

    # Market state
    market: Dict[str, Any] = field(default_factory=dict)

    # Position state
    position: Dict[str, Any] = field(default_factory=dict)

    # Observation breakdown
    observation: Dict[str, Any] = field(default_factory=dict)

    # Analyst activations
    analyst: Dict[str, Any] = field(default_factory=dict)

    # Agent activations
    agent: Dict[str, Any] = field(default_factory=dict)

    # Decision
    decision: Dict[str, Any] = field(default_factory=dict)

    # Feature values (all 54 features per timeframe)
    features: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), default=str)


class VisualizationServer:
    """
    WebSocket server that broadcasts activation data to frontend clients.

    The bridge queues data via queue_data(), and this server broadcasts
    it to all connected WebSocket clients.
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8765,
        broadcast_interval: float = 0.05  # 50ms between broadcasts
    ):
        """
        Initialize visualization server.

        Args:
            host: Host address to bind to
            port: Port number for WebSocket server
            broadcast_interval: Minimum time between broadcasts (seconds)
        """
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError(
                "websockets library not installed. "
                "Install with: pip install websockets"
            )

        self.host = host
        self.port = port
        self.broadcast_interval = broadcast_interval

        # Connected clients
        self.clients: Set[WebSocketServerProtocol] = set()

        # Data queue (thread-safe)
        self.data_queue: Queue = Queue()

        # Control flags
        self._running = False
        self._last_broadcast = 0.0

        # Stats
        self.messages_sent = 0
        self.clients_connected = 0

        logger.info(f"VisualizationServer initialized on {host}:{port}")

    async def handler(self, websocket: WebSocketServerProtocol) -> None:
        """
        Handle a new WebSocket client connection.

        Args:
            websocket: The WebSocket connection
        """
        # Register client
        self.clients.add(websocket)
        self.clients_connected = len(self.clients)
        client_id = id(websocket)

        logger.info(f"Client connected: {client_id} (total: {self.clients_connected})")

        try:
            # Send welcome message with server info
            welcome = {
                'type': 'welcome',
                'server': 'VisualizationServer',
                'version': '1.0',
                'timestamp': time.time(),
            }
            await websocket.send(json.dumps(welcome))

            # Listen for client messages (e.g., pause/resume commands)
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client {client_id}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} connection closed")
        finally:
            # Unregister client
            self.clients.discard(websocket)
            self.clients_connected = len(self.clients)
            logger.info(f"Client disconnected: {client_id} (total: {self.clients_connected})")

    async def _handle_client_message(
        self,
        websocket: WebSocketServerProtocol,
        data: Dict[str, Any]
    ) -> None:
        """
        Handle incoming messages from clients.

        Args:
            websocket: The client WebSocket
            data: Parsed JSON data from client
        """
        msg_type = data.get('type', '')

        if msg_type == 'ping':
            # Respond with pong
            await websocket.send(json.dumps({
                'type': 'pong',
                'timestamp': time.time()
            }))
        elif msg_type == 'status':
            # Send server status
            await websocket.send(json.dumps({
                'type': 'status',
                'running': self._running,
                'clients': self.clients_connected,
                'messages_sent': self.messages_sent,
                'timestamp': time.time()
            }))
        else:
            logger.debug(f"Unknown message type: {msg_type}")

    async def broadcast(self, data: Dict[str, Any]) -> None:
        """
        Broadcast data to all connected clients.

        Args:
            data: Dictionary to broadcast as JSON
        """
        if not self.clients:
            return

        message = json.dumps(data, default=str)

        # Send to all clients concurrently
        tasks = [
            asyncio.create_task(self._send_to_client(client, message))
            for client in self.clients.copy()
        ]

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            self.messages_sent += 1

    async def _send_to_client(
        self,
        client: WebSocketServerProtocol,
        message: str
    ) -> None:
        """
        Send message to a single client, handling errors.

        Args:
            client: WebSocket client
            message: JSON string to send
        """
        try:
            await client.send(message)
        except websockets.exceptions.ConnectionClosed:
            self.clients.discard(client)
        except Exception as e:
            logger.warning(f"Error sending to client: {e}")
            self.clients.discard(client)

    def queue_data(self, data: Dict[str, Any]) -> None:
        """
        Queue visualization data to be broadcast.

        This method is called by the MT5 bridge from a different thread.

        Args:
            data: Visualization data dictionary
        """
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = time.time()

        # Mark as activation data
        data['type'] = 'activation'

        self.data_queue.put(data)

    async def _broadcast_loop(self) -> None:
        """
        Main loop that pulls data from queue and broadcasts to clients.
        """
        logger.info("Starting broadcast loop")

        while self._running:
            try:
                # Non-blocking check for new data
                try:
                    data = self.data_queue.get_nowait()

                    # Throttle broadcasts
                    now = time.time()
                    if now - self._last_broadcast >= self.broadcast_interval:
                        await self.broadcast(data)
                        self._last_broadcast = now

                except Empty:
                    pass

                # Small sleep to prevent busy-waiting
                await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Broadcast loop error: {e}")
                await asyncio.sleep(0.1)

        logger.info("Broadcast loop stopped")

    async def run(self) -> None:
        """
        Start the WebSocket server and broadcast loop.

        This is the main entry point for running the server.
        """
        self._running = True

        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")

        async with websockets.serve(
            self.handler,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        ):
            # Run broadcast loop
            await self._broadcast_loop()

    def stop(self) -> None:
        """
        Stop the server.
        """
        self._running = False
        logger.info("VisualizationServer stopping")


def create_visualization_server(
    host: str = 'localhost',
    port: int = 8765
) -> VisualizationServer:
    """
    Factory function to create a visualization server.

    Args:
        host: Host address
        port: Port number

    Returns:
        Configured VisualizationServer instance
    """
    return VisualizationServer(host=host, port=port)


# Standalone test
if __name__ == '__main__':
    import random

    logging.basicConfig(level=logging.INFO)

    async def test_server():
        """Test the visualization server with dummy data."""
        server = VisualizationServer(port=8765)

        # Simulate queueing data every second
        async def dummy_data_producer():
            while server._running:
                server.queue_data({
                    'timestamp': time.time(),
                    'market': {
                        'price': 2034.56 + random.uniform(-2, 2),
                        'atr': 5.23,
                    },
                    'decision': {
                        'action': random.choice([0, 1, 2]),
                        'action_name': random.choice(['FLAT', 'LONG', 'SHORT']),
                        'confidence': random.uniform(0.3, 0.9),
                    },
                    'analyst': {
                        'probabilities': {
                            'p_up': random.uniform(0.3, 0.7),
                            'p_down': random.uniform(0.3, 0.7),
                        }
                    }
                })
                await asyncio.sleep(1.0)

        # Run server and producer concurrently
        await asyncio.gather(
            server.run(),
            dummy_data_producer()
        )

    print("Starting test server on ws://localhost:8765")
    print("Connect with: wscat -c ws://localhost:8765")
    asyncio.run(test_server())
