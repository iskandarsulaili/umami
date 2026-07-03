// ML Service client: proxies requests to the ML sidecar service on port 8001
const ML_API_URL = process.env.ML_API_URL || 'http://localhost:8001';

export interface MLPrediction {
  page: string;
  probability: number;
}

export interface MLRecommendation {
  page: string;
  score: number;
  base_similarity: number;
}

export async function fetchFromML<T>(endpoint: string, body: Record<string, any>): Promise<T> {
  const response = await fetch(`${ML_API_URL}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    cache: 'no-cache',
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'ML service error' }));
    throw new Error(error.detail || `ML service returned ${response.status}`);
  }

  return response.json();
}
