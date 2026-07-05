import { parseRequest } from '@/lib/request';
import { json, unauthorized, serverError } from '@/lib/response';
import prisma from '@/lib/prisma';

export async function GET(request: Request) {
  const { auth, error } = await parseRequest(request);
  if (error) return error();

  const url = new URL(request.url);
  const key = url.searchParams.get('key');
  if (!key) return json({ error: 'key param required' });

  try {
    const rows = await prisma.rawQuery(
      `SELECT pref_value FROM user_preferences
       WHERE user_id = {{userId::uuid}} AND pref_key = {{key}}
       LIMIT 1`,
      { userId: auth.user.id, key },
    );
    return json({ value: rows?.[0]?.pref_value || null });
  } catch (e: any) {
    return serverError(e);
  }
}

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { key, value } = body;
  if (!key) return json({ error: 'key required' });

  try {
    await prisma.rawQuery(
      `INSERT INTO user_preferences (user_id, pref_key, pref_value, updated_at)
       VALUES ({{userId::uuid}}, {{key}}, {{value}}, NOW())
       ON CONFLICT (user_id, pref_key)
       DO UPDATE SET pref_value = {{value}}, updated_at = NOW()`,
      { userId: auth.user.id, key, value: String(value ?? '') },
    );
    return json({ saved: true });
  } catch (e: any) {
    return serverError(e);
  }
}
