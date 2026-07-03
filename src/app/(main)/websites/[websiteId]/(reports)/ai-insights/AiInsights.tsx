'use client';
import { Column, Row, Text } from '@umami/react-zen';
import { useMessages, useDateRange, useApi } from '@/components/hooks';
import { useEffect, useState, useCallback } from 'react';
import { LoadingPanel } from '@/components/common/LoadingPanel';
interface Recommendation {
  page: string;
  score: number;
  base_similarity: number;
}

interface NextPagePrediction {
  page: string;
  probability: number;
}

interface IntentResult {
  intent: string;
  confidence: number;
  details?: string;
}

interface FunnelDropResult {
  will_continue: boolean;
  probability: number;
}

interface RageClickResult {
  frustration_type: string;
  confidence: number;
  severity: string;
  details?: string;
}

interface ArchetypeResult {
  archetype: string;
  description: string;
  probability: number;
}

interface BanditResult {
  page: string;
  score: number;
  arm_stats: { n_pulls: number; ctr: number };
}

interface ABTestResult {
  experiment_id: string;
  name: string;
  variants: Array<{
    variant_id: string;
    n: number;
    conversion_rate: number;
    lift_pct: number;
    significant: boolean;
  }>;
}

interface ReplayResult {
  severity: string;
  error_loops: number;
  dead_clicks: number;
  form_struggles: number;
  rapid_navigation: number;
  total_events: number;
}

export function AiInsights({ websiteId }: { websiteId: string }) {
  const { t, labels } = useMessages();
  const { post } = useApi();
  const { dateRange: { startDate, endDate } } = useDateRange();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [nextPages, setNextPages] = useState<NextPagePrediction[]>([]);
  const [intent, setIntent] = useState<IntentResult | null>(null);
  const [funnelDrop, setFunnelDrop] = useState<FunnelDropResult | null>(null);
  const [rageClick, setRageClick] = useState<RageClickResult | null>(null);
  const [archetype, setArchetype] = useState<ArchetypeResult | null>(null);
  const [banditRecs, setBanditRecs] = useState<BanditResult[]>([]);
  const [abTests, setAbTests] = useState<ABTestResult[]>([]);
  const [replay, setReplay] = useState<ReplayResult | null>(null);

  const loadInsights = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const session = {
        current_page: '/',
        session_depth: 1,
        device: 'desktop',
        browser: 'Chrome',
        country: 'US',
        hour: new Date().getHours(),
        is_weekend: [0, 6].includes(new Date().getDay()),
        pages_seen: 1,
        time_on_site: 0,
        has_cart: false,
        has_checkout: false,
      };

      const [
        recData, npData, intentData, funnelData,
        rageData, clusterData, banditData, abData,
      ] = await Promise.allSettled([
        post('/ml/recommend', { websiteId, sessionPages: [], sessionFeatures: {}, topK: 10 }),
        post('/ml/next-page', { websiteId, sessionPages: ['/'], topK: 10, useTransformer: false }),
        post('/ml/intent', { websiteId, sessionPages: ['/'], sessionFeatures: session }),
        post('/ml/funnel-drop', { websiteId, session, threshold: 0.5 }),
        post('/ml/rage-click', { websiteId, clicks: [] }),
        post('/ml/recommend/bandit', { websiteId, session, available_pages: ['/products', '/blog', '/pricing', '/signup', '/about'], top_k: 5 }),
        post('/ml/ab-test/results', {}),
      ]);

      if (recData.status === 'fulfilled') setRecommendations(recData.value?.recommendations || []);
      if (npData.status === 'fulfilled') setNextPages(npData.value?.predictions || []);
      if (intentData.status === 'fulfilled') setIntent(intentData.value);
      if (funnelData.status === 'fulfilled') setFunnelDrop(funnelData.value);
      if (rageData.status === 'fulfilled') setRageClick(rageData.value);
      if (clusterData.status === 'fulfilled') setArchetype(clusterData.value);
      if (banditData.status === 'fulfilled') setBanditRecs(banditData.value?.recommendations || []);
      if (abData.status === 'fulfilled') {
        const results = abData.value;
        setAbTests(Object.values(results).filter((r: any) => r?.experiment_id));
      }
    } catch (e: any) {
      setError(e.message || 'ML service unavailable');
    } finally {
      setLoading(false);
    }
  }, [websiteId, post]);

  useEffect(() => { loadInsights(); }, [loadInsights]);

  if (loading) {
    return <LoadingPanel isLoading={true} height="100%" />;
  }

  return (
    <Column gap="3">
      {error && (
        <Row justifyContent="center" padding="3">
          <Text color="red" size="sm">{error}</Text>
        </Row>
      )}

      {!error && (
        <Row gap="4" wrap>
          {/* Panel 1: Recommended Pages */}
          <Column gap minWidth="280px" flex={1}>
            <Text size="lg" fontWeight="bold">Recommended Pages</Text>
            <Column gap="1">
              {recommendations.map((rec, i) => (
                <Row key={i} gap="2" alignItems="center"
                  style={{ padding: '8px 12px', background: `rgba(59, 130, 246, ${Math.max(0.05, rec.score)})`, borderRadius: '8px' }}>
                  <Text color="blue" fontWeight="bold" minWidth="20px">{i + 1}.</Text>
                  <Column flex={1}>
                    <Text size="sm">{rec.page}</Text>
                    <Text size="xs" color="gray">similarity: {(rec.score * 100).toFixed(0)}%</Text>
                  </Column>
                </Row>
              ))}
              {recommendations.length === 0 && <Text color="gray" size="sm">Train the recommender model to see suggestions</Text>}
            </Column>
          </Column>

          {/* Panel 2: Predicted Next Pages */}
          <Column gap minWidth="280px" flex={1}>
            <Text size="lg" fontWeight="bold">Predicted Next Pages</Text>
            <Column gap="1">
              {nextPages.map((np, i) => (
                <Row key={i} gap="2" alignItems="center"
                  style={{ padding: '8px 12px', background: `rgba(16, 185, 129, ${Math.max(0.05, np.probability)})`, borderRadius: '8px' }}>
                  <Text color="green" fontWeight="bold" minWidth="20px">{i + 1}.</Text>
                  <Column flex={1}>
                    <Text size="sm">{np.page}</Text>
                    <Text size="xs" color="gray">probability: {(np.probability * 100).toFixed(1)}%</Text>
                  </Column>
                </Row>
              ))}
              {nextPages.length === 0 && <Text color="gray" size="sm">Train the next-page predictor to see predictions</Text>}
            </Column>
          </Column>

          {/* Panel 3: Session Intent */}
          <Column gap minWidth="280px" flex={1}>
            <Text size="lg" fontWeight="bold">Session Intent</Text>
            {intent ? (
              <Column gap="1" style={{ padding: '12px', background: 'rgba(139, 92, 246, 0.08)', borderRadius: '8px' }}>
                <Row gap="2" alignItems="center">
                  <Text color="purple" fontWeight="bold">Intent:</Text>
                  <Text>{intent.intent}</Text>
                </Row>
                <Row gap="2" alignItems="center">
                  <Text color="purple" fontWeight="bold">Confidence:</Text>
                  <Text>{(intent.confidence * 100).toFixed(0)}%</Text>
                </Row>
                {intent.details && <Text size="sm" color="gray">{intent.details}</Text>}
              </Column>
            ) : (
              <Text color="gray" size="sm">Train the intent classifier to see predictions</Text>
            )}
          </Column>

          {/* Panel 4: Funnel Drop-off Risk */}
          <Column gap minWidth="280px" flex={1}>
            <Text size="lg" fontWeight="bold">Funnel Drop-off Risk</Text>
            {funnelDrop ? (
              <Column gap="1" style={{ padding: '12px', background: funnelDrop.will_continue ? 'rgba(16, 185, 129, 0.08)' : 'rgba(239, 68, 68, 0.08)', borderRadius: '8px' }}>
                <Row gap="2" alignItems="center">
                  <Text fontWeight="bold">Status:</Text>
                  <Text color={funnelDrop.will_continue ? 'green' : 'red'}>
                    {funnelDrop.will_continue ? 'Likely to continue' : 'At risk of dropping off'}
                  </Text>
                </Row>
                <Row gap="2" alignItems="center">
                  <Text fontWeight="bold">Probability:</Text>
                  <Text>{(funnelDrop.probability * 100).toFixed(0)}%</Text>
                </Row>
              </Column>
            ) : (
              <Text color="gray" size="sm">Train the funnel predictor to see predictions</Text>
            )}
          </Column>

          {/* Panel 5: Frustration Detection */}
          <Column gap minWidth="280px" flex={1}>
            <Text size="lg" fontWeight="bold">Frustration Detection</Text>
            {rageClick && rageClick.frustration_type !== 'none' ? (
              <Column gap="1" style={{ padding: '12px', background: rageClick.severity === 'high' ? 'rgba(239, 68, 68, 0.08)' : 'rgba(245, 158, 11, 0.08)', borderRadius: '8px' }}>
                <Row gap="2" alignItems="center">
                  <Text fontWeight="bold">Signal:</Text>
                  <Text color={rageClick.severity === 'high' ? 'red' : 'orange'}>{rageClick.frustration_type}</Text>
                </Row>
                <Row gap="2" alignItems="center">
                  <Text fontWeight="bold">Confidence:</Text>
                  <Text>{(rageClick.confidence * 100).toFixed(0)}%</Text>
                </Row>
                <Row gap="2" alignItems="center">
                  <Text fontWeight="bold">Severity:</Text>
                  <Text>{rageClick.severity}</Text>
                </Row>
                {rageClick.details && <Text size="sm" color="gray">{rageClick.details}</Text>}
              </Column>
            ) : (
              <Text color="gray" size="sm">No frustration signals detected in current session</Text>
            )}
          </Column>

          {/* Panel 6: Visitor Archetype */}
          <Column gap minWidth="280px" flex={1}>
            <Text size="lg" fontWeight="bold">Visitor Archetype</Text>
            {archetype && archetype.archetype !== 'unclassified' ? (
              <Column gap="1" style={{ padding: '12px', background: 'rgba(236, 72, 153, 0.08)', borderRadius: '8px' }}>
                <Row gap="2" alignItems="center">
                  <Text color="pink" fontWeight="bold">Archetype:</Text>
                  <Text>{archetype.archetype}</Text>
                </Row>
                <Text size="sm" color="gray">{archetype.description}</Text>
                <Row gap="2" alignItems="center">
                  <Text fontWeight="bold">Confidence:</Text>
                  <Text>{(archetype.probability * 100).toFixed(0)}%</Text>
                </Row>
              </Column>
            ) : (
              <Text color="gray" size="sm">Train the journey clusterer to see archetype predictions</Text>
            )}
          </Column>

          {/* Panel 7: Bandit Recommendations */}
          <Column gap minWidth="280px" flex={1}>
            <Text size="lg" fontWeight="bold">Bandit Recommendations</Text>
            <Column gap="1">
              {banditRecs.map((br, i) => (
                <Row key={i} gap="2" alignItems="center"
                  style={{ padding: '8px 12px', background: `rgba(245, 158, 11, ${Math.max(0.05, br.score)})`, borderRadius: '8px' }}>
                  <Text color="orange" fontWeight="bold" minWidth="20px">{i + 1}.</Text>
                  <Column flex={1}>
                    <Text size="sm">{br.page}</Text>
                    <Text size="xs" color="gray">score: {br.score.toFixed(3)} | pulls: {br.arm_stats?.n_pulls || 0}</Text>
                  </Column>
                </Row>
              ))}
              {banditRecs.length === 0 && <Text color="gray" size="sm">Initialize the bandit to see recommendations</Text>}
            </Column>
          </Column>

          {/* Panel 8: A/B Test Results */}
          <Column gap minWidth="280px" flex={1}>
            <Text size="lg" fontWeight="bold">A/B Test Results</Text>
            {abTests.length > 0 ? abTests.map((test) => (
              <Column key={test.experiment_id} gap="1" style={{ padding: '12px', background: 'rgba(6, 182, 212, 0.08)', borderRadius: '8px', marginBottom: '8px' }}>
                <Text fontWeight="bold" size="sm">{test.name}</Text>
                {test.variants?.map((v) => (
                  <Row key={v.variant_id} gap="2" alignItems="center" style={{ padding: '4px 0' }}>
                    <Text size="sm" fontWeight="bold" minWidth="80px">{v.variant_id}</Text>
                    <Text size="sm">{(v.conversion_rate * 100).toFixed(1)}%</Text>
                    {v.lift_pct !== 0 && (
                      <Text size="xs" color={v.lift_pct > 0 ? 'green' : 'red'}>
                        {v.lift_pct > 0 ? '+' : ''}{v.lift_pct.toFixed(1)}%
                      </Text>
                    )}
                    {v.significant && <Text size="xs" color="green">✓ significant</Text>}
                  </Row>
                ))}
              </Column>
            )) : (
              <Text color="gray" size="sm">Create an A/B test to see results</Text>
            )}
          </Column>

          {/* Panel 9: Session Replay Analysis */}
          <Column gap minWidth="280px" flex={1}>
            <Text size="lg" fontWeight="bold">Session Replay Analysis</Text>
            {replay ? (
              <Column gap="1" style={{ padding: '12px', background: replay.severity === 'high' ? 'rgba(239, 68, 68, 0.08)' : 'rgba(107, 114, 128, 0.08)', borderRadius: '8px' }}>
                <Row gap="2" alignItems="center">
                  <Text fontWeight="bold">Severity:</Text>
                  <Text color={replay.severity === 'high' ? 'red' : replay.severity === 'medium' ? 'orange' : 'gray'}>{replay.severity}</Text>
                </Row>
                <Row gap="2" alignItems="center"><Text fontWeight="bold">Error loops:</Text><Text>{replay.error_loops}</Text></Row>
                <Row gap="2" alignItems="center"><Text fontWeight="bold">Dead clicks:</Text><Text>{replay.dead_clicks}</Text></Row>
                <Row gap="2" alignItems="center"><Text fontWeight="bold">Form struggles:</Text><Text>{replay.form_struggles}</Text></Row>
                <Row gap="2" alignItems="center"><Text fontWeight="bold">Rapid nav:</Text><Text>{replay.rapid_navigation}</Text></Row>
                <Text size="xs" color="gray">{replay.total_events} events analyzed</Text>
              </Column>
            ) : (
              <Text color="gray" size="sm">Enable session recording to see replay analysis</Text>
            )}
          </Column>
        </Row>
      )}
    </Column>
  );
}
