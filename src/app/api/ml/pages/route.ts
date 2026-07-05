import { parseRequest } from '@/lib/request';
import { json, unauthorized, serverError } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';
import prisma from '@/lib/prisma';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { websiteId } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'journeys'))) {
    return unauthorized();
  }

  try {
    const rows: Array<{ url_path: string }> = await prisma.rawQuery(
      `SELECT DISTINCT url_path
       FROM website_event
       WHERE website_id = {{websiteId::uuid}}
         AND url_path IS NOT NULL
         AND url_path != ''
       ORDER BY url_path
       LIMIT 20`,
      { websiteId },
    );
    const pages = (rows || []).map((r: any) => r.url_path).filter(Boolean);
    return json({ pages });
  } catch (e: any) {
    return serverError(e);
  }
}
