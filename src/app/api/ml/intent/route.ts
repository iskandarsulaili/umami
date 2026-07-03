import { parseRequest } from '@/lib/request';
import { json, unauthorized, serverError } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';
import { fetchFromML } from '@/lib/ml-client';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { websiteId, sessionPages, sessionFeatures } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'journeys'))) {
    return unauthorized();
  }

  try {
    const data = await fetchFromML('/predict/intent', {
      website_id: websiteId,
      session_pages: sessionPages || [],
      session_features: sessionFeatures || {},
    });
    return json(data);
  } catch (e: any) {
    return serverError(e);
  }
}
