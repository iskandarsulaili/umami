import { parseRequest } from '@/lib/request';
import { json, unauthorized } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';
import { fetchFromML } from '@/lib/ml-client';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { websiteId, sessionId } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'journeys'))) {
    return unauthorized();
  }

  try {
    const data = await fetchFromML('/predict/session-replay', {
      website_id: websiteId,
      session_id: sessionId,
    });
    return json(data);
  } catch (e: any) {
    return json({ severity: 'none', error_loops: 0, dead_clicks: 0, form_struggles: 0, rapid_navigation: 0, total_events: 0 });
  }
}
