import { parseRequest } from '@/lib/request';
import { json, unauthorized } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';
import { fetchFromML } from '@/lib/ml-client';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);

  if (error) {
    return error();
  }

  const { websiteId, sessionPages, topK, useTransformer } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'journeys'))) {
    return unauthorized();
  }

  try {
    const data = await fetchFromML('/predict/next-page', {
      website_id: websiteId,
      session_pages: sessionPages || [],
      top_k: topK || 10,
      use_transformer: useTransformer || false,
    });

    return json(data);
  } catch (e) {
    return json({ error: e.message, predictions: [] });
  }
}
