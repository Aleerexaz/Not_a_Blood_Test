const DEFAULT_BASE = 'http://127.0.0.1:8000';
export const API_BASE = import.meta.env.VITE_API_BASE || DEFAULT_BASE;

export async function predictImage(file) {
  const form = new FormData();
  form.append('image', file);
  const res = await fetch(`${API_BASE}/predict?use_weights=true`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return await res.json();
}

export async function health() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return await res.json();
}

export async function scannerStatus() {
  const res = await fetch(`${API_BASE}/scanner/status`);
  if (!res.ok) throw new Error('Scanner status check failed');
  return await res.json();
}

export async function captureFromScanner() {
  const res = await fetch(`${API_BASE}/scanner/capture`, {
    method: 'POST',
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Scanner capture failed: ${text}`);
  }
  return await res.json();
}

export async function captureAndPredict() {
  const res = await fetch(`${API_BASE}/scanner/capture-and-predict?use_weights=true`, {
    method: 'POST',
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Scanner capture and prediction failed: ${text}`);
  }
  return await res.json();
}

export async function openScannerDemo() {
  const res = await fetch(`${API_BASE}/scanner/open-demo`, { method: 'POST' });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Open demo failed: ${text}`);
  }
  return await res.json();
}

export async function consumeLatestAndPredict() {
  const res = await fetch(`${API_BASE}/scanner/consume-and-predict?use_weights=true`, { method: 'POST' });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`No latest image yet: ${text}`);
  }
  return await res.json();
}

export async function captureMultiAndPredict(count = 3) {
  const res = await fetch(`${API_BASE}/scanner/capture-multi-and-predict?count=${count}&use_weights=true`, {
    method: 'POST',
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Multi-scan failed: ${text}`);
  }
  return await res.json();
}
