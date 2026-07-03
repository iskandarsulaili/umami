'use client';
import { Column, Grid, Row, Text } from '@umami/react-zen';
import { useMessages } from '@/components/hooks';
import { Panel } from '@/components/common/Panel';
import { WebsiteControls } from '@/app/(main)/websites/[websiteId]/WebsiteControls';
import { AiInsights } from './AiInsights';

export function AiInsightsPage({ websiteId }: { websiteId: string }) {
  const { t, labels } = useMessages();

  return (
    <Column gap>
      <WebsiteControls websiteId={websiteId} />
      <Panel allowFullscreen>
        <AiInsights websiteId={websiteId} />
      </Panel>
    </Column>
  );
}
