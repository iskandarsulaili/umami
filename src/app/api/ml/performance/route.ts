import { parseRequest } from '@/lib/request';
import { json, unauthorized } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';
import { getPerformanceStats } from '@/queries/sql';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { websiteId } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'performance'))) {
    return unauthorized();
  }

  try {
    const data = await getPerformanceStats(websiteId, {});
    return json(data);
  } catch (e: any) {
    return json({ error: String(e.message || e) });
  }
}
