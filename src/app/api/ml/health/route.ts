import { parseRequest } from '@/lib/request';
import { json, unauthorized, serverError } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';

const ML_API_URL = process.env.ML_API_URL || 'http://localhost:8001';

export async function GET(request: Request) {
  const { auth, error } = await parseRequest(request);
  if (error) return error();

  try {
    const response = await fetch(`${ML_API_URL}/health`, {
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

export async function POST(request: Request) {
  return GET(request);
}
