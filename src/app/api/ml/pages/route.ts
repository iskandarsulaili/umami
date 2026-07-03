import { parseRequest } from '@/lib/request';
import { json, unauthorized } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';
import { getPageviewMetrics } from '@/queries/sql';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { websiteId } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'journeys'))) {
    return unauthorized();
  }

  try {
    const data = await getPageviewMetrics(websiteId, { type: 'url', limit: 20 }, {});
    const pages = (data || []).map((row: any) => row.x).filter(Boolean);
    return json({ pages });
  } catch (e: any) {
    return json({ pages: [] });
  }
}
