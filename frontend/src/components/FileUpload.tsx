import { useEffect } from 'react';
import { CloudUpload, Upload, Check, Loader2, Database } from 'lucide-react';
import { useChat } from '../context/ChatContext';
import { useFileUpload } from '../hooks/useFileUpload';

interface FileUploadProps {
  send: (msg: string) => void;
}

export function FileUpload({ send }: FileUploadProps) {
  const { state, dispatch } = useChat();
  const {
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
  } = useFileUpload(state.sessionId, send, dispatch);

  useEffect(() => {
    if (!state.uploadedFilename) {
      reset();
    }
  }, [state.uploadedFilename, reset]);

  return (
    <div>
      <h2 className="text-[13px] font-semibold text-secondary uppercase tracking-wider mb-4 flex items-center gap-2">
        <Database className="w-4 h-4 text-primary" />
        Dataset
      </h2>

      <div
        className={`border border-dashed rounded-xl p-6 text-center cursor-pointer transition-all bg-surface-subtle relative ${
          isDragging
            ? 'border-primary bg-primary/5'
            : 'border-glass-border hover:border-primary/40 hover:bg-surface-hover'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <CloudUpload
          className={`w-7 h-7 mx-auto mb-2 transition-colors ${
            isDragging ? 'text-primary' : 'text-secondary/50'
          }`}
        />
        <p className="text-[12px] text-secondary leading-relaxed">
          {selectedFile ? (
            <span className="text-primary">{selectedFile.name}</span>
          ) : (
            'Drop CSV/Excel here or click to browse'
          )}
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          className="hidden"
          onChange={handleFileSelect}
        />
      </div>

      {uploadStatus !== 'idle' && (
        <p
          className={`mt-2 text-[11px] text-center ${
            uploadStatus === 'success'
              ? 'text-success'
              : uploadStatus === 'error'
                ? 'text-error'
                : 'text-secondary'
          }`}
        >
          {uploadStatus === 'success'
            ? 'File uploaded successfully.'
            : uploadStatus === 'error'
              ? 'Upload failed. Please try again.'
              : 'Uploading...'}
        </p>
      )}

      <button
        onClick={upload}
        disabled={!selectedFile || uploadStatus === 'uploading' || uploadStatus === 'success'}
        className="w-full mt-3 py-2.5 bg-primary text-white border-none rounded-xl text-[13px] font-semibold cursor-pointer
          transition-all flex justify-center items-center gap-2
          hover:bg-primary-hover hover:shadow-[0_4px_16px_var(--color-primary-glow)]
          disabled:bg-secondary/30 disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none"
      >
        {uploadStatus === 'uploading' ? (
          <>
            <Loader2 className="w-3.5 h-3.5 animate-spin" /> Uploading...
          </>
        ) : uploadStatus === 'success' ? (
          <>
            <Check className="w-3.5 h-3.5" /> Uploaded
          </>
        ) : (
          <>
            <Upload className="w-3.5 h-3.5" /> Upload Data
          </>
        )}
      </button>
    </div>
  );
}
