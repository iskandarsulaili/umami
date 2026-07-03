'use client';
import { Column, Row, Text, Icon } from '@umami/react-zen';
import { useMessages, useDateRange, useApi } from '@/components/hooks';
import { useEffect, useState, useCallback } from 'react';
import { LoadingPanel } from '@/components/common/LoadingPanel';
import { target, targetBlank } from '@/lib/react';

interface Recommendation {
  page: string;
  score: number;
  base_similarity: number;
}

interface NextPagePrediction {
  page: string;
  probability: number;
}

export function AiInsights({ websiteId }: { websiteId: string }) {
  const { t, labels } = useMessages();
  const { post } = useApi();
  const { dateRange: { startDate, endDate } } = useDateRange();

  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [nextPages, setNextPages] = useState<NextPagePrediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadInsights = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [recData, npData] = await Promise.all([
        post('/ml/recommend', {
          websiteId,
          sessionPages: [],
          sessionFeatures: {},
          topK: 10,
        }),
        post('/ml/next-page', {
          websiteId,
          sessionPages: ['/'],
          topK: 10,
          useTransformer: false,
        }),
      ]);
      setRecommendations(recData?.recommendations || []);
      setNextPages(npData?.predictions || []);
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
    <Column gap="2">
      {error && (
        <Row justifyContent="center" padding="3">
          <Text color="red" size="sm">{error}</Text>
        </Row>
      )}

      {!error && (
        <Row gap="4" wrap>
          <Column gap minWidth="300px" flex={1}>
            <Text size="lg" fontWeight="bold">Recommended Pages</Text>
            <Column gap="1">
              {recommendations.map((rec, i) => (
                <Row key={i} gap="2" alignItems="center"
                  style={{
                    padding: '8px 12px',
                    background: `rgba(59, 130, 246, ${Math.max(0.05, rec.score)})`,
                    borderRadius: '8px',
                  }}
                >
                  <Text color="blue" fontWeight="bold" minWidth="20px">{i + 1}.</Text>
                  <Column flex={1}>
                    <Text size="sm">{rec.page}</Text>
                    <Text size="xs" color="gray">similarity: {(rec.score * 100).toFixed(0)}%</Text>
                  </Column>
                </Row>
              ))}
              {recommendations.length === 0 && (
                <Text color="gray" size="sm">Train the recommender model to see suggestions</Text>
              )}
            </Column>
          </Column>

          <Column gap minWidth="300px" flex={1}>
            <Text size="lg" fontWeight="bold">Predicted Next Pages</Text>
            <Column gap="1">
              {nextPages.map((np, i) => (
                <Row key={i} gap="2" alignItems="center"
                  style={{
                    padding: '8px 12px',
                    background: `rgba(16, 185, 129, ${Math.max(0.05, np.probability)})`,
                    borderRadius: '8px',
                  }}
                >
                  <Text color="green" fontWeight="bold" minWidth="20px">{i + 1}.</Text>
                  <Column flex={1}>
                    <Text size="sm">{np.page}</Text>
                    <Text size="xs" color="gray">probability: {(np.probability * 100).toFixed(1)}%</Text>
                  </Column>
                </Row>
              ))}
              {nextPages.length === 0 && (
                <Text color="gray" size="sm">Train the next-page predictor to see predictions</Text>
              )}
            </Column>
          </Column>
        </Row>
      )}
    </Column>
  );
}
