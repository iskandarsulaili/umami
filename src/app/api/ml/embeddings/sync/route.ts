import { parseRequest } from '@/lib/request';
import { json, unauthorized, serverError } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';

const ML_API_URL = process.env.ML_API_URL || 'http://localhost:8001';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { websiteId, model, mode } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'settings'))) {
    return unauthorized();
  }

  try {
    const response = await fetch(`${ML_API_URL}/embeddings/sync`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        website_id: websiteId,
        model: model || 'intfloat/multilingual-e5-small',
        mode: mode || 'local',
      }),
      cache: 'no-cache',
    });
    const data = await response.json();
    return json(data);
  } catch (e: any) {
    return serverError(e);
  }
}
