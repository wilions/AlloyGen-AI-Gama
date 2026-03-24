import { useState, useCallback, useRef, type Dispatch } from 'react';
import type { ChatAction } from '../types';
import { uploadFile } from '../lib/api';

export function useFileUpload(
  sessionId: string,
  send: (msg: string) => void,
  dispatch: Dispatch<ChatAction>,
) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) setSelectedFile(file);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) setSelectedFile(file);
  }, []);

  const upload = useCallback(async () => {
    if (!selectedFile) return;
    setUploadStatus('uploading');
    try {
      const result = await uploadFile(sessionId, selectedFile);
      if (result.status === 'success') {
        setUploadStatus('success');
        dispatch({ type: 'SET_UPLOADED_FILENAME', payload: result.filename });
        dispatch({ type: 'SET_TYPING', payload: true });
        dispatch({
          type: 'ADD_TOAST',
          payload: {
            id: crypto.randomUUID(),
            type: 'success',
            message: `${result.filename} uploaded successfully`,
          },
        });
        send('[SYSTEM]: File uploaded successfully.');
      } else {
        setUploadStatus('error');
        dispatch({
          type: 'ADD_TOAST',
          payload: {
            id: crypto.randomUUID(),
            type: 'error',
            message: 'Upload failed. Please try again.',
          },
        });
      }
    } catch (err) {
      setUploadStatus('error');
      const message = err instanceof Error ? err.message : 'Upload failed';
      dispatch({
        type: 'ADD_TOAST',
        payload: {
          id: crypto.randomUUID(),
          type: 'error',
          message,
        },
      });
    }
  }, [selectedFile, sessionId, send, dispatch]);

  const reset = useCallback(() => {
    setSelectedFile(null);
    setUploadStatus('idle');
    if (fileInputRef.current) fileInputRef.current.value = '';
  }, []);

  return {
    isDragging,
    selectedFile,
    uploadStatus,
    fileInputRef,
    handleDragOver,
    handleDragLeave,
    handleDrop,
    handleFileSelect,
    upload,
    reset,
  };
}
