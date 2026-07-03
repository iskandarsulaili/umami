import { parseRequest } from '@/lib/request';
import { json, unauthorized, serverError } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';

const ML_API_URL = process.env.ML_API_URL || 'http://localhost:8001';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { websiteId } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'journeys'))) {
    return unauthorized();
  }

  try {
    const response = await fetch(`${ML_API_URL}/ab-test/results`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      cache: 'no-cache',
    });
    const data = await response.json();
    return json(data);
  } catch (e: any) {
    return serverError(e);
  }
}
