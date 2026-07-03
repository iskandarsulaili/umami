import { getQueryFilters, parseRequest } from '@/lib/request';
import { json, unauthorized } from '@/lib/response';
import { reportResultSchema } from '@/lib/schema';
import { canViewWebsiteSection } from '@/permissions';
import { getJourney } from '@/queries/sql';
import { fetchFromML } from '@/lib/ml-client';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request, reportResultSchema);

  if (error) {
    return error();
  }

  const { websiteId, parameters, filters } = body;
  const { eventType } = parameters;

  if (!(await canViewWebsiteSection(auth, websiteId, 'journeys'))) {
    return unauthorized();
  }

  if (eventType) {
    filters.eventType = eventType;
  }

  const queryFilters = await getQueryFilters(filters, websiteId);

  const data = await getJourney(websiteId, parameters, queryFilters);

  // Enrich with ML predictions (best-effort, non-blocking)
  let mlPredictions = null;
  try {
    const topPath = data?.[0]?.items?.[0];
    if (topPath) {
      mlPredictions = await fetchFromML('/predict/next-page', {
        website_id: websiteId,
        session_pages: [topPath],
        top_k: 5,
        use_transformer: false,
      });
    }
  } catch {
    // ML service unavailable — return SQL data only
  }

  return json(data);
}
