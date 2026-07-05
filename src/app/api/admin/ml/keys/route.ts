import { parseRequest } from '@/lib/request';
import { json, unauthorized, serverError } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';
import prisma from '@/lib/prisma';
import crypto from 'crypto';

function hashKey(key: string): string {
  return crypto.createHash('sha256').update(key).digest('hex');
}

function generateKey(): { full: string; prefix: string; hash: string } {
  const raw = crypto.randomBytes(32).toString('hex');
  const prefix = raw.slice(0, 8);
  return { full: `umami_ml_${raw}`, prefix, hash: hashKey(raw) };
}

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { websiteId, name } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'settings'))) {
    return unauthorized();
  }

  try {
    const { full, prefix, hash } = generateKey();
    await prisma.rawQuery(
      `INSERT INTO ml_api_keys (website_id, name, key_prefix, key_hash)
       VALUES ({{websiteId::uuid}}, {{name}}, {{prefix}}, {{hash}})`,
      { websiteId, name: name || 'default', prefix, hash },
    );
    return json({ key: full, prefix, name: name || 'default' });
  } catch (e: any) {
    return serverError(e);
  }
}

export async function GET(request: Request) {
  const { auth, error } = await parseRequest(request);
  if (error) return error();

  const url = new URL(request.url);
  const websiteId = url.searchParams.get('websiteId');

  if (!websiteId || !(await canViewWebsiteSection(auth, websiteId, 'settings'))) {
    return unauthorized();
  }

  try {
    const rows = await prisma.rawQuery(
      `SELECT id, name, key_prefix, active, created_at, last_used_at
       FROM ml_api_keys
       WHERE website_id = {{websiteId::uuid}}
       ORDER BY created_at DESC`,
      { websiteId },
    );
    return json({ keys: rows || [] });
  } catch (e: any) {
    return serverError(e);
  }
}

export async function DELETE(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { websiteId, keyId } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'settings'))) {
    return unauthorized();
  }

  try {
    await prisma.rawQuery(
      `UPDATE ml_api_keys SET active = false WHERE id = {{keyId::uuid}} AND website_id = {{websiteId::uuid}}`,
      { websiteId, keyId },
    );
    return json({ revoked: true });
  } catch (e: any) {
    return serverError(e);
  }
}
