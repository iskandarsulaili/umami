import { parseRequest } from '@/lib/request';
import { json, unauthorized, serverError } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';
import { fetchFromML } from '@/lib/ml-client';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { websiteId, sessionPages, sessionFeatures, topK, mode, semanticWeight } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'journeys'))) {
    return unauthorized();
  }

  try {
    const data = await fetchFromML('/recommend', {
      website_id: websiteId,
      session_pages: sessionPages || [],
      session_features: sessionFeatures || {},
      top_k: topK || 20,
      mode: mode || 'token',
      semantic_weight: semanticWeight ?? 0.6,
    });
    return json(data);
  } catch (e: any) {
    return serverError(e);
  }
}
