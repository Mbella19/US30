import { useState, useEffect, useCallback, useRef } from 'react';

const DEFAULT_WS_URL = 'ws://localhost:8765';

export function useVisualizationData(wsUrl = DEFAULT_WS_URL) {
  const [data, setData] = useState(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const [messageCount, setMessageCount] = useState(0);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setConnected(true);
        setError(null);
        console.log('Connected to visualization server');
      };

      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          if (parsed.type === 'activation') {
            setData(parsed);
            setMessageCount(c => c + 1);
          }
        } catch (e) {
          console.error('Failed to parse message:', e);
        }
      };

      ws.onerror = (e) => {
        setError('WebSocket error');
        console.error('WebSocket error:', e);
      };

      ws.onclose = () => {
        setConnected(false);
        wsRef.current = null;

        // Attempt reconnection after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Attempting to reconnect...');
          connect();
        }, 3000);
      };

      wsRef.current = ws;
    } catch (e) {
      setError(`Connection failed: ${e.message}`);
    }
  }, [wsUrl]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return {
    data,
    connected,
    error,
    messageCount,
    reconnect: connect,
  };
}

export default useVisualizationData;
