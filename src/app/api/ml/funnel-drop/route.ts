import { parseRequest } from '@/lib/request';
import { json, unauthorized, serverError } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';
import { fetchFromML } from '@/lib/ml-client';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { websiteId, session, threshold } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'funnels'))) {
    return unauthorized();
  }

  try {
    const data = await fetchFromML('/predict/funnel-drop', {
      website_id: websiteId,
      session: session || {},
      threshold: threshold || 0.5,
    });
    return json(data);
  } catch (e: any) {
    return serverError(e);
  }
}
