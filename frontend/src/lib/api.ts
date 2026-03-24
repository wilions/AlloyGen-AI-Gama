export async function uploadFile(
  sessionId: string,
  file: File,
): Promise<{ status: string; filename: string; message: string }> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('session_id', sessionId);

  const res = await fetch('/upload', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || `Upload failed (${res.status})`);
  }

  return res.json();
}

export async function fetchModels(): Promise<import('../types').ModelInfo[]> {
  const res = await fetch('/models');
  if (!res.ok) throw new Error('Failed to fetch models');
  const data = await res.json();
  return data.models;
}

export async function batchPredict(
  sessionId: string,
  file: File,
): Promise<{ status: string; message: string; download_path?: string }> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('session_id', sessionId);

  const res = await fetch('/batch-predict', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errorData.detail || `Batch prediction failed (${res.status})`);
  }

  return res.json();
}
