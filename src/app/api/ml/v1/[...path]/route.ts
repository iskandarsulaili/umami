import { json, serverError } from '@/lib/response';
import prisma from '@/lib/prisma';
import crypto from 'crypto';

const ML_API_URL = process.env.ML_API_URL || 'http://localhost:8001';

function hashKey(key: string): string {
  return crypto.createHash('sha256').update(key).digest('hex');
}

async function validateApiKey(request: Request): Promise<{ websiteId: string } | Response> {
  const apiKey = request.headers.get('x-api-key');
  if (!apiKey) {
    return Response.json(
      { error: { message: 'x-api-key header required', code: 'unauthorized', status: 401 } },
      { status: 401 },
    );
  }
  // Extract the random part after "umami_ml_"
  const parts = apiKey.split('umami_ml_');
  if (parts.length !== 2 || !parts[1]) {
    return Response.json(
      { error: { message: 'Invalid API key format', code: 'unauthorized', status: 401 } },
      { status: 401 },
    );
  }

  const rawKey = parts[1];
  const hash = hashKey(rawKey);
  const prefix = rawKey.slice(0, 8);

  try {
    const rows = await prisma.rawQuery(
      `SELECT website_id FROM ml_api_keys
       WHERE key_prefix = {{prefix}}
         AND key_hash = {{hash}}
         AND active = true
       LIMIT 1`,
      { prefix, hash },
    );
    if (!rows || rows.length === 0) {
      return Response.json(
        { error: { message: 'Invalid or revoked API key', code: 'unauthorized', status: 401 } },
        { status: 401 },
      );
    }
    // Update last_used_at
    await prisma.rawQuery(
      `UPDATE ml_api_keys SET last_used_at = NOW() WHERE key_prefix = {{prefix}}`,
      { prefix },
    );
    return { websiteId: rows[0].website_id };
  } catch (e: any) {
    return Response.json(
      { error: { message: 'Authentication failed', code: 'server-error', status: 500 } },
      { status: 500 },
    );
  }
}

async function proxyToML(endpoint: string, body: any, websiteId: string) {
  const response = await fetch(`${ML_API_URL}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, website_id: websiteId }),
    cache: 'no-cache',
  });
  const data = await response.json();
  return json(data);
}

export async function POST(request: Request) {
  const url = new URL(request.url);
  const path = url.pathname.replace('/api/ml/v1', '');

  // Validate API key first
  const auth = await validateApiKey(request);
  if (auth instanceof Response) return auth;

  let body: any;
  try {
    body = await request.json();
  } catch {
    body = {};
  }

  try {
    switch (path) {
      case '/predict/next-page':
        return proxyToML('/predict/next-page', {
          session_pages: body.sessionPages || body.session_pages || [],
          top_k: body.topK || body.top_k || 10,
          use_transformer: body.useTransformer || body.use_transformer || false,
        }, auth.websiteId);

      case '/predict/intent':
        return proxyToML('/predict/intent', {
          session_pages: body.sessionPages || body.session_pages || [],
          session_features: body.sessionFeatures || body.session_features || {},
        }, auth.websiteId);

      case '/predict/funnel-drop':
        return proxyToML('/predict/funnel-drop', {
          session: body.session || {},
          threshold: body.threshold ?? 0.5,
        }, auth.websiteId);

      case '/predict/rage-click':
        return proxyToML('/predict/rage-click', {
          session_id: body.sessionId || body.session_id || null,
          clicks: body.clicks || [],
        }, auth.websiteId);

      case '/predict/cluster':
        return proxyToML('/predict/cluster', {
          session: body.session || {},
        }, auth.websiteId);

      case '/predict/session-replay':
        return proxyToML('/predict/session-replay', {
          session_id: body.sessionId || body.session_id || '',
        }, auth.websiteId);

      case '/recommend':
        return proxyToML('/recommend', {
          session_pages: body.sessionPages || body.session_pages || [],
          session_features: body.sessionFeatures || body.session_features || {},
          top_k: body.topK || body.top_k || 20,
          mode: body.mode || 'token',
          semantic_weight: body.semanticWeight ?? body.semantic_weight ?? 0.6,
        }, auth.websiteId);

      default:
        return json({ error: `Unknown endpoint: ${path}` });
    }
  } catch (e: any) {
    return serverError(e);
  }
}
