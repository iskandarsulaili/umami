import { parseRequest } from '@/lib/request';
import { json, unauthorized } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';
import { fetchFromML } from '@/lib/ml-client';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { websiteId, page, session, reward } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'journeys'))) {
    return unauthorized();
  }

  try {
    const data = await fetchFromML('/recommend/bandit/reward', {
      website_id: websiteId,
      page,
      session: session || {},
      reward,
    });
    return json(data);
  } catch (e: any) {
    return json({ status: 'error', message: String(e.message || e) });
  }
}
