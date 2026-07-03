import { parseRequest } from '@/lib/request';
import { json, unauthorized } from '@/lib/response';
import { canViewWebsiteSection } from '@/permissions';
import { fetchFromML } from '@/lib/ml-client';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request);
  if (error) return error();

  const { websiteId, days = 30 } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'ai-insights'))) {
    return unauthorized();
  }

  const endDate = new Date().toISOString();
  const startDate = new Date(Date.now() - days * 86400000).toISOString();

  const results: Record<string, any> = {};

  // Train all models sequentially
  const models = [
    { name: 'next-page', endpoint: '/train/next-page', body: { website_id: websiteId, start_date: startDate, end_date: endDate, min_session_length: 2, use_transformer: true, epochs: 10 } },
    { name: 'funnel', endpoint: '/train/funnel', body: { website_id: websiteId, start_date: startDate, end_date: endDate } },
    { name: 'intent', endpoint: '/train/intent', body: { website_id: websiteId, start_date: startDate, end_date: endDate } },
    { name: 'recommender', endpoint: '/train/recommender', body: { website_id: websiteId, start_date: startDate, end_date: endDate, epochs: 10 } },
    { name: 'rage-click', endpoint: '/train/rage-click', body: { website_id: websiteId, start_date: startDate, end_date: endDate } },
  ];

  for (const model of models) {
    try {
      const data = await fetchFromML(model.endpoint, model.body);
      results[model.name] = { status: 'ok', data };
    } catch (e: any) {
      results[model.name] = { status: 'error', message: String(e.message || e) };
    }
  }

  return json({ status: 'complete', results });
}
