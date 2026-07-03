import { parseRequest } from '@/lib/request';
import { json, unauthorized } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';
import { fetchFromML } from '@/lib/ml-client';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { websiteId, session, availablePages, topK } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'journeys'))) {
    return unauthorized();
  }

  try {
    const data = await fetchFromML('/recommend/bandit', {
      website_id: websiteId,
      session: session || {},
      available_pages: availablePages || [],
      top_k: topK || 5,
    });
    return json(data);
  } catch (e: any) {
    return json({ error: String(e.message || e), recommendations: [] });
  }
}
