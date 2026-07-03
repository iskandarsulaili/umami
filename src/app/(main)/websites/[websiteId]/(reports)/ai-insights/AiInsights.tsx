'use client';
import { Column, Grid, Row, Text } from '@umami/react-zen';
import { useMessages, useDateRange, useApi } from '@/components/hooks';
import { useEffect, useState, useCallback } from 'react';
import { LoadingPanel } from '@/components/common/LoadingPanel';
import { Panel } from '@/components/common/Panel';

interface Recommendation {
  page: string;
  score: number;
}

interface NextPagePrediction {
  page: string;
  probability: number;
}

interface IntentResult {
  intent: string;
  confidence: number;
}

interface FunnelDropResult {
  will_continue: boolean;
  probability: number;
}

interface RageClickResult {
  frustration_type: string;
  confidence: number;
  severity: string;
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

interface MLHealth {
  status: string;
  gpu: { device: string; count: number; memory: Array<{ free: number; total: number; device: string }> };
  models: Record<string, { loaded: boolean; trained: boolean }>;
}

interface PerfStats {
  lcp: number;
  inp: number;
  cls: number;
  fcp: number;
  ttfb: number;
  count: number;
}

function fmtMs(ms: number): string {
  if (ms == null || ms <= 0) return '-';
  if (ms >= 1000) return (ms / 1000).toFixed(1) + 's';
  return Math.round(ms) + 'ms';
}

function fmtPct(val: number): string {
  if (val == null) return '?%';
  return (val * 100).toFixed(0) + '%';
}

function perfRating(val: number, metric: string): { label: string; color: string } {
  if (val == null || val <= 0) return { label: '-', color: 'muted' };
  const thresholds: Record<string, [number, number]> = {
    lcp: [2500, 4000], inp: [200, 500], cls: [0.1, 0.25], fcp: [1800, 3000], ttfb: [800, 1800],
  };
  const [good, poor] = thresholds[metric] || [Infinity, Infinity];
  if (val <= good) return { label: 'good', color: 'success' };
  if (val <= poor) return { label: 'needsImprovement', color: 'warning' };
  return { label: 'poor', color: 'danger' };
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
  const [mlHealth, setMlHealth] = useState<MLHealth | null>(null);
  const [perfStats, setPerfStats] = useState<PerfStats | null>(null);
  const [availablePages, setAvailablePages] = useState<string[]>([]);

  const loadInsights = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const session = {
        current_page: '/', session_depth: 1, device: 'desktop',
        browser: 'Chrome', country: 'US', hour: new Date().getHours(),
        is_weekend: [0, 6].includes(new Date().getDay()),
        pages_seen: 1, time_on_site: 0, has_cart: false, has_checkout: false,
      };

      const results = await Promise.allSettled([
        post('/ml/recommend', { websiteId, sessionPages: [], sessionFeatures: {}, topK: 10 }),
        post('/ml/next-page', { websiteId, sessionPages: ['/'], topK: 10, useTransformer: false }),
        post('/ml/intent', { websiteId, sessionPages: ['/'], sessionFeatures: session }),
        post('/ml/funnel-drop', { websiteId, session, threshold: 0.5 }),
        post('/ml/rage-click', { websiteId, clicks: [] }),
        post('/ml/pages', { websiteId }),
        post('/ml/ab-test/results', {}),
        post('/ml/performance', { websiteId }),
        post('/ml/health', {}),
      ]);

      const [recData, npData, intentData, funnelData, rageData, pagesData, abData, perfData, healthData] = results;

      if (recData?.status === 'fulfilled') setRecommendations(recData.value?.recommendations || []);
      if (npData?.status === 'fulfilled') setNextPages(npData.value?.predictions || []);
      if (intentData?.status === 'fulfilled') setIntent(intentData.value);
      if (funnelData?.status === 'fulfilled') setFunnelDrop(funnelData.value);
      if (rageData?.status === 'fulfilled') setRageClick(rageData.value);
      if (pagesData?.status === 'fulfilled') {
        const p = pagesData.value?.pages || [];
        setAvailablePages(p);
        if (p.length > 0) {
          post('/ml/recommend/bandit', { websiteId, session, available_pages: p, top_k: 5 })
            .then(d => setBanditRecs(d?.recommendations || []))
            .catch(() => {});
        }
      }
      if (abData?.status === 'fulfilled') {
        const d = abData.value;
        if (d && typeof d === 'object' && !Array.isArray(d)) setAbTests(Object.values(d).filter((r: any) => r?.experiment_id));
      }
      if (perfData?.status === 'fulfilled') setPerfStats(perfData.value);
      if (healthData?.status === 'fulfilled') setMlHealth(healthData.value);
    } catch (e: any) {
      setError(e.message || 'ML service unavailable');
    } finally {
      setLoading(false);
    }
  }, [websiteId, post]);

  useEffect(() => { loadInsights(); }, [loadInsights]);

  const trainedCount = mlHealth?.models ? Object.values(mlHealth.models).filter((m: any) => m.trained).length : 0;
  const totalModels = mlHealth?.models ? Object.keys(mlHealth.models).length : 0;
  const gpuMem = mlHealth?.gpu?.memory?.[0];
  const gpuFreeGb = gpuMem ? (gpuMem.free / 1073741824).toFixed(1) : '?';
  const gpuTotalGb = gpuMem ? (gpuMem.total / 1073741824).toFixed(1) : '?';

  const perfMetrics = [
    { key: 'lcp', label: t(labels.lcpFull), val: perfStats?.lcp, unit: 'ms' },
    { key: 'inp', label: t(labels.inpFull), val: perfStats?.inp, unit: 'ms' },
    { key: 'cls', label: t(labels.clsFull), val: perfStats?.cls, unit: '' },
    { key: 'fcp', label: t(labels.fcpFull), val: perfStats?.fcp, unit: 'ms' },
    { key: 'ttfb', label: t(labels.ttfbFull), val: perfStats?.ttfb, unit: 'ms' },
  ];

  return (
    <LoadingPanel data={recommendations.length || nextPages.length || intent || funnelDrop || rageClick || archetype || banditRecs.length || abTests.length || replay || perfStats || mlHealth} isLoading={loading} error={error}>
      <Column gap="3" paddingY="4">

        {/* Summary Stats Bar */}
        <Grid columns={{ base: '1fr 1fr', md: 'repeat(5, 1fr)' }} gap="2">
          <Column padding="3" borderRadius backgroundColor="surface-raised" gap="1">
            <Text size="xs" color="muted" transform="uppercase">{t(labels.mlModels)}</Text>
            <Text size="xl" weight="bold">{trainedCount}/{totalModels}</Text>
            <Text size="xs" color="muted">{t(labels.trained)}</Text>
          </Column>
          <Column padding="3" borderRadius backgroundColor="surface-raised" gap="1">
            <Text size="xs" color="muted" transform="uppercase">{t(labels.gpu)}</Text>
            <Text size="xl" weight="bold">{mlHealth?.gpu?.device || t(labels.nA)}</Text>
            <Text size="xs" color="muted">{t(labels.gbFree, { free: gpuFreeGb, total: gpuTotalGb })}</Text>
          </Column>
          <Column padding="3" borderRadius backgroundColor="surface-raised" gap="1">
            <Text size="xs" color="muted" transform="uppercase">{t(labels.lcp)}</Text>
            <Text size="xl" weight="bold" color={perfRating(perfStats?.lcp ?? 0, 'lcp').color}>{fmtMs(perfStats?.lcp ?? 0)}</Text>
            <Text size="xs" color={perfRating(perfStats?.lcp ?? 0, 'lcp').color}>{t(labels[perfRating(perfStats?.lcp ?? 0, 'lcp').label as keyof typeof labels])}</Text>
          </Column>
          <Column padding="3" borderRadius backgroundColor="surface-raised" gap="1">
            <Text size="xs" color="muted" transform="uppercase">{t(labels.inp)}</Text>
            <Text size="xl" weight="bold" color={perfRating(perfStats?.inp ?? 0, 'inp').color}>{fmtMs(perfStats?.inp ?? 0)}</Text>
            <Text size="xs" color={perfRating(perfStats?.inp ?? 0, 'inp').color}>{t(labels[perfRating(perfStats?.inp ?? 0, 'inp').label as keyof typeof labels])}</Text>
          </Column>
          <Column padding="3" borderRadius backgroundColor="surface-raised" gap="1">
            <Text size="xs" color="muted" transform="uppercase">{t(labels.cls)}</Text>
            <Text size="xl" weight="bold" color={perfRating(perfStats?.cls ?? 0, 'cls').color}>{(perfStats?.cls ?? 0).toFixed(3)}</Text>
            <Text size="xs" color={perfRating(perfStats?.cls ?? 0, 'cls').color}>{t(labels[perfRating(perfStats?.cls ?? 0, 'cls').label as keyof typeof labels])}</Text>
          </Column>
        </Grid>

        {/* Row 1: Recommendations + Next Pages */}
        <Grid columns={{ base: '1fr', md: '1fr 1fr' }} gap="3">
          <Panel title={t(labels.recommendedPages)} description={t(labels.recommendedPagesDesc)}>
            <Column gap="1">
              {recommendations.length > 0 ? recommendations.map((rec, i) => (
                <Row key={i} gap="2" alignItems="center" paddingY="1" paddingX="2" borderRadius backgroundColor={i % 2 === 0 ? 'surface-raised' : undefined}>
                  <Text weight="bold" color="primary" minWidth="24px">{i + 1}.</Text>
                  <Column flex={1}>
                    <Text size="sm">{rec.page}</Text>
                    <Text size="xs" color="muted">{t(labels.similarity)}: {rec.score != null ? (rec.score * 100).toFixed(0) : '?'}%</Text>
                  </Column>
                </Row>
              )) : <Text color="muted" size="sm">{t(labels.trainRecommender)}</Text>}
            </Column>
          </Panel>

          <Panel title={t(labels.predictedNextPages)} description={t(labels.predictedNextPagesDesc)}>
            <Column gap="1">
              {nextPages.length > 0 ? nextPages.map((np, i) => (
                <Row key={i} gap="2" alignItems="center" paddingY="1" paddingX="2" borderRadius backgroundColor={i % 2 === 0 ? 'surface-raised' : undefined}>
                  <Text weight="bold" color="primary" minWidth="24px">{i + 1}.</Text>
                  <Column flex={1}>
                    <Text size="sm">{np.page}</Text>
                    <Text size="xs" color="muted">{t(labels.probability)}: {np.probability != null ? (np.probability * 100).toFixed(1) : '?'}%</Text>
                  </Column>
                </Row>
              )) : <Text color="muted" size="sm">{t(labels.trainNextPage)}</Text>}
            </Column>
          </Panel>
        </Grid>

        {/* Row 2: Intent + Funnel Drop-off + Frustration */}
        <Grid columns={{ base: '1fr', md: '1fr 1fr 1fr' }} gap="3">
          <Panel title={t(labels.sessionIntent)} description={t(labels.sessionIntentDesc)}>
            {intent ? (
              <Column gap="1" padding="2">
                <Row gap="2" alignItems="center"><Text weight="bold">{t(labels.intent)}:</Text><Text>{intent.intent}</Text></Row>
                <Row gap="2" alignItems="center"><Text weight="bold">{t(labels.confidence)}:</Text><Text>{fmtPct(intent.confidence)}</Text></Row>
              </Column>
            ) : <Text color="muted" size="sm">{t(labels.trainIntent)}</Text>}
          </Panel>

          <Panel title={t(labels.funnelDropoffRisk)} description={t(labels.funnelDropoffRiskDesc)}>
            {funnelDrop ? (
              <Column gap="1" padding="2">
                <Row gap="2" alignItems="center">
                  <Text weight="bold">{t(labels.status)}:</Text>
                  <Text color={funnelDrop.will_continue ? 'success' : 'danger'}>
                    {funnelDrop.will_continue ? t(labels.likelyToContinue) : t(labels.atRiskOfDropping)}
                  </Text>
                </Row>
                <Row gap="2" alignItems="center"><Text weight="bold">{t(labels.probability)}:</Text><Text>{fmtPct(funnelDrop.probability)}</Text></Row>
              </Column>
            ) : <Text color="muted" size="sm">{t(labels.trainFunnel)}</Text>}
          </Panel>

          <Panel title={t(labels.frustrationDetection)} description={t(labels.frustrationDetectionDesc)}>
            {rageClick && rageClick.frustration_type !== 'none' ? (
              <Column gap="1" padding="2">
                <Row gap="2" alignItems="center">
                  <Text weight="bold">{t(labels.signal)}:</Text>
                  <Text color={rageClick.severity === 'high' ? 'danger' : 'warning'}>{rageClick.frustration_type}</Text>
                </Row>
                <Row gap="2" alignItems="center"><Text weight="bold">{t(labels.confidence)}:</Text><Text>{fmtPct(rageClick.confidence)}</Text></Row>
                <Row gap="2" alignItems="center"><Text weight="bold">{t(labels.severity)}:</Text><Text>{rageClick.severity}</Text></Row>
              </Column>
            ) : <Text color="muted" size="sm">{t(labels.noFrustrationSignals)}</Text>}
          </Panel>
        </Grid>

        {/* Row 3: Archetype + Bandit + Replay */}
        <Grid columns={{ base: '1fr', md: '1fr 1fr 1fr' }} gap="3">
          <Panel title={t(labels.visitorArchetype)} description={t(labels.visitorArchetypeDesc)}>
            {archetype && archetype.archetype !== 'unclassified' ? (
              <Column gap="1" padding="2">
                <Row gap="2" alignItems="center"><Text weight="bold">{t(labels.archetype)}:</Text><Text>{archetype.archetype}</Text></Row>
                <Text size="sm" color="muted">{archetype.description}</Text>
                <Row gap="2" alignItems="center"><Text weight="bold">{t(labels.confidence)}:</Text><Text>{fmtPct(archetype.probability)}</Text></Row>
              </Column>
            ) : <Text color="muted" size="sm">{t(labels.trainClusterer)}</Text>}
          </Panel>

          <Panel title={t(labels.banditRecommendations)} description={t(labels.banditRecommendationsDesc)}>
            <Column gap="1">
              {banditRecs.length > 0 ? banditRecs.map((br, i) => (
                <Row key={i} gap="2" alignItems="center" paddingY="1" paddingX="2" borderRadius backgroundColor={i % 2 === 0 ? 'surface-raised' : undefined}>
                  <Text weight="bold" color="primary" minWidth="24px">{i + 1}.</Text>
                  <Column flex={1}>
                    <Text size="sm">{br.page}</Text>
                    <Text size="xs" color="muted">{t(labels.score)}: {br.score != null ? br.score.toFixed(3) : '?'} | {t(labels.pulls)}: {br.arm_stats?.n_pulls || 0}</Text>
                  </Column>
                </Row>
              )) : <Text color="muted" size="sm">{t(labels.initBandit)}</Text>}
            </Column>
          </Panel>

          <Panel title={t(labels.sessionReplayAnalysis)} description={t(labels.sessionReplayAnalysisDesc)}>
            {replay ? (
              <Column gap="1" padding="2">
                <Row gap="2" alignItems="center">
                  <Text weight="bold">{t(labels.severity)}:</Text>
                  <Text color={replay.severity === 'high' ? 'danger' : replay.severity === 'medium' ? 'warning' : 'muted'}>{replay.severity}</Text>
                </Row>
                <Row gap="2" alignItems="center"><Text weight="bold">{t(labels.errorLoops)}:</Text><Text>{replay.error_loops}</Text></Row>
                <Row gap="2" alignItems="center"><Text weight="bold">{t(labels.deadClicks)}:</Text><Text>{replay.dead_clicks}</Text></Row>
                <Row gap="2" alignItems="center"><Text weight="bold">{t(labels.formStruggles)}:</Text><Text>{replay.form_struggles}</Text></Row>
                <Row gap="2" alignItems="center"><Text weight="bold">{t(labels.rapidNav)}:</Text><Text>{replay.rapid_navigation}</Text></Row>
                <Text size="xs" color="muted">{t(labels.eventsAnalyzed, { n: replay.total_events })}</Text>
              </Column>
            ) : <Text color="muted" size="sm">{t(labels.enableRecording)}</Text>}
          </Panel>
        </Grid>

        {/* Row 4: Performance Metrics */}
        <Panel title={t(labels.webVitalsPerformance)} description={t(labels.webVitalsPerformanceDesc)}>
          <Grid columns={{ base: '1fr 1fr', md: 'repeat(5, 1fr)' }} gap="2">
            {perfMetrics.map(m => {
              const rating = perfRating(m.val ?? 0, m.key);
              return (
                <Column key={m.key} padding="3" borderRadius backgroundColor="surface-raised" gap="1">
                  <Text size="xs" color="muted">{m.label}</Text>
                  <Text size="lg" weight="bold" color={rating.color}>
                    {m.val != null && m.val > 0 ? (m.unit === 'ms' ? fmtMs(m.val) : m.val.toFixed(3)) : '-'}
                  </Text>
                  <Text size="xs" color={rating.color}>{t(labels[rating.label as keyof typeof labels])}</Text>
                </Column>
              );
            })}
          </Grid>
          {perfStats?.count != null && perfStats.count > 0 && (
            <Text size="xs" color="muted" paddingTop="2">{t(labels.basedOnSamples, { n: perfStats.count })}</Text>
          )}
        </Panel>

        {/* Row 5: A/B Test Results */}
        {abTests.length > 0 && (
          <Panel title={t(labels.abTestResults)} description={t(labels.abTestResultsDesc)}>
            <Column gap="2">
              {abTests.map((test) => (
                <Column key={test.experiment_id} gap="1" padding="2" borderRadius backgroundColor="surface-raised">
                  <Text weight="bold" size="sm">{test.name}</Text>
                  {test.variants?.map((v) => (
                    <Row key={v.variant_id} gap="3" alignItems="center" paddingY="1">
                      <Text size="sm" weight="bold" minWidth="100px">{v.variant_id}</Text>
                      <Text size="sm">{(v.conversion_rate != null ? (v.conversion_rate * 100).toFixed(1) : '?')}%</Text>
                      {v.lift_pct != null && v.lift_pct !== 0 && (
                        <Text size="xs" color={v.lift_pct > 0 ? 'success' : 'danger'}>
                          {v.lift_pct > 0 ? '+' : ''}{v.lift_pct.toFixed(1)}%
                        </Text>
                      )}
                      {v.significant && <Text size="xs" color="success">{t(labels.significant)}</Text>}
                    </Row>
                  ))}
                </Column>
              ))}
            </Column>
          </Panel>
        )}

        {/* Row 6: ML System Health */}
        <Panel title={t(labels.mlSystemHealth)} description={t(labels.mlSystemHealthDesc)}>
          <Grid columns={{ base: '1fr', md: '1fr 1fr' }} gap="3">
            <Column gap="1" padding="2">
              <Text weight="bold" size="sm">{t(labels.gpu)}</Text>
              <Row gap="2" alignItems="center"><Text size="sm" color="muted">{t(labels.device)}:</Text><Text size="sm">{mlHealth?.gpu?.device || t(labels.nA)}</Text></Row>
              <Row gap="2" alignItems="center"><Text size="sm" color="muted">{t(labels.count)}:</Text><Text size="sm">{mlHealth?.gpu?.count || 0}</Text></Row>
              <Row gap="2" alignItems="center"><Text size="sm" color="muted">{t(labels.memory)}:</Text><Text size="sm">{t(labels.gbFree, { free: gpuFreeGb, total: gpuTotalGb })}</Text></Row>
            </Column>
            <Column gap="1" padding="2">
              <Text weight="bold" size="sm">{t(labels.models)}</Text>
              {mlHealth?.models ? Object.entries(mlHealth.models).map(([name, m]: [string, any]) => (
                <Row key={name} gap="2" alignItems="center">
                  <Text size="sm" color="muted" minWidth="180px">{name}</Text>
                  <Text size="xs" color={m.trained ? 'success' : 'warning'}>{m.trained ? t(labels.trained) : t(labels.untrained)}</Text>
                </Row>
              )) : <Text size="sm" color="muted">{t(labels.mlServiceNotAvailable)}</Text>}
            </Column>
          </Grid>
        </Panel>

      </Column>
    </LoadingPanel>
  );
}
