import { parseRequest } from '@/lib/request';
import { json, unauthorized, serverError } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';
import { fetchFromML } from '@/lib/ml-client';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { websiteId, sessionPages, sessionFeatures, topK, mode, semanticWeight, embeddingModel, embeddingMode, embeddingApiUrl, embeddingApiKey } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'journeys'))) {
    return unauthorized();
  }

  try {
    const mlBody: any = {
      website_id: websiteId,
      session_pages: sessionPages || [],
      session_features: sessionFeatures || {},
      top_k: topK || 20,
      mode: mode || 'token',
      semantic_weight: semanticWeight ?? 0.6,
      embedding_model: embeddingModel || 'minilm',
      embedding_mode: embeddingMode || 'local',
    };
    // Only send URL/key if cloud mode is active
    if (embeddingMode === 'cloud') {
      if (embeddingApiUrl) mlBody.embedding_api_url = embeddingApiUrl;
      if (embeddingApiKey) mlBody.embedding_api_key = embeddingApiKey;
    }
    const data = await fetchFromML('/recommend', mlBody);
    return json(data);
  } catch (e: any) {
    return serverError(e);
  }
}
