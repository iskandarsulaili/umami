import { parseRequest } from '@/lib/request';
import { json, unauthorized } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';
import { fetchFromML } from '@/lib/ml-client';

export async function GET(request: Request) {
  const { auth, error } = await parseRequest(request);
  if (error) return error();

  try {
    const data = await fetchFromML('/health', {});
    return json(data);
  } catch (e: any) {
    return json({ status: 'unavailable', error: String(e.message || e) });
  }
}

export async function POST(request: Request) {
  return GET(request);
}
