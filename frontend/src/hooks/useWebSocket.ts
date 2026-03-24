import { useEffect, useRef, useCallback, type Dispatch } from 'react';
import type { ChatAction, PipelineStep } from '../types';

const GREETING_PREFIX = 'Hello! I am your Metallurgic';

export function useWebSocket(sessionId: string, dispatch: Dispatch<ChatAction>) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const messageQueue = useRef<string[]>([]);
  const greetingSent = useRef(false);

  const flushQueue = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      while (messageQueue.current.length > 0) {
        const msg = messageQueue.current.shift()!;
        wsRef.current.send(msg);
      }
    }
  }, []);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws/${sessionId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      dispatch({ type: 'SET_WS_READY', payload: true });
      flushQueue();
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      dispatch({ type: 'SET_TYPING', payload: false });

      if (data.type === 'message') {
        // Skip duplicate greeting on reconnect
        const isGreeting = data.content?.startsWith(GREETING_PREFIX);
        if (isGreeting && greetingSent.current) {
          // Still update state from greeting message, just don't add it again
        } else {
          if (isGreeting) greetingSent.current = true;
          dispatch({
            type: 'ADD_MESSAGE',
            payload: {
              id: crypto.randomUUID(),
              role: 'assistant',
              content: data.content,
              timestamp: Date.now(),
            },
          });
        }
      } else if (data.type === 'status') {
        if (data.content === 'pipeline_start') {
          dispatch({ type: 'PIPELINE_START' });
        } else {
          dispatch({
            type: 'ADD_MESSAGE',
            payload: {
              id: crypto.randomUUID(),
              role: 'status',
              content: data.content,
              timestamp: Date.now(),
            },
          });
        }
      }

      if (data.state) {
        dispatch({ type: 'SET_STEP', payload: data.state as PipelineStep });
        if (data.state === 'prediction') {
          dispatch({ type: 'PIPELINE_COMPLETE' });
        }
        if (data.state === 'requirements') {
          dispatch({ type: 'PIPELINE_COMPLETE' });
        }
      }
    };

    ws.onclose = () => {
      dispatch({ type: 'SET_WS_READY', payload: false });
      reconnectTimer.current = setTimeout(connect, 3000);
    };
  }, [sessionId, dispatch, flushQueue]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback((message: string) => {
    const payload = JSON.stringify({ message });
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(payload);
    } else {
      messageQueue.current.push(payload);
    }
  }, []);

  const sendCancel = useCallback(() => {
    const payload = JSON.stringify({ type: 'cancel' });
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(payload);
    }
    dispatch({ type: 'PIPELINE_COMPLETE' });
    dispatch({
      type: 'ADD_TOAST',
      payload: {
        id: crypto.randomUUID(),
        type: 'info',
        message: 'Pipeline cancellation requested...',
      },
    });
  }, [dispatch]);

  const sendReset = useCallback(() => {
    const payload = JSON.stringify({ type: 'reset' });
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(payload);
    }
    greetingSent.current = false;
    dispatch({ type: 'RESET' });
  }, [dispatch]);

  const sendSelectModel = useCallback((modelPath: string) => {
    const payload = JSON.stringify({ type: 'select_model', model_path: modelPath });
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(payload);
    }
  }, []);

  return { send, sendCancel, sendReset, sendSelectModel };
}
