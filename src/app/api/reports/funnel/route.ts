import { getQueryFilters, parseRequest, setWebsiteDate } from '@/lib/request';
import { json, unauthorized } from '@/lib/response';
import { reportResultSchema } from '@/lib/schema';
import { canViewWebsiteSection } from '@/permissions';
import { type FunnelParameters, getFunnel } from '@/queries/sql';
import { fetchFromML } from '@/lib/ml-client';

export async function POST(request: Request) {
  const { auth, body, error } = await parseRequest(request, reportResultSchema);

  if (error) {
    return error();
  }

  const { websiteId } = body;

  if (!(await canViewWebsiteSection(auth, websiteId, 'funnels'))) {
    return unauthorized();
  }

  const parameters = await setWebsiteDate(websiteId, body.parameters);
  const filters = await getQueryFilters(body.filters, websiteId);

  const data = await getFunnel(websiteId, parameters as FunnelParameters, filters);

  // Enrich each funnel step with ML drop-off probability
  let mlPredictions = null;
  try {
    const session = {
      session_depth: 0,
      device: 'desktop',
      browser: 'Chrome',
      country: 'US',
      hour: new Date().getHours(),
      is_weekend: [0, 6].includes(new Date().getDay()),
      pages_seen: 1,
      time_on_site: 0,
    };
    mlPredictions = await fetchFromML('/predict/funnel-drop', {
      website_id: websiteId,
      session,
      threshold: 0.5,
    });
  } catch {
    // ML service unavailable
  }

  return json({
    funnel: data,
    ml: mlPredictions ? { drop_off_probability: mlPredictions } : null,
  });
}
