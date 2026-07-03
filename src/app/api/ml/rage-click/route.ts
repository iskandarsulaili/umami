import { parseRequest } from '@/lib/request';
import { json, unauthorized, serverError } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';
import { fetchFromML } from '@/lib/ml-client';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { websiteId, sessionId, clicks } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'heatmaps'))) {
    return unauthorized();
  }

  try {
    let data;
    if (sessionId) {
      data = await fetchFromML('/predict/rage-click', {
        website_id: websiteId,
        session_id: sessionId,
      });
    } else {
      data = await fetchFromML('/predict/rage-click', {
        website_id: websiteId,
        clicks: clicks || [],
      });
    }
    return json(data);
  } catch (e: any) {
    return serverError(e);
  }
}
