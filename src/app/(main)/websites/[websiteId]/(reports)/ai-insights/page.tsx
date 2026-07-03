import type { Metadata } from 'next';
import { AiInsightsPage } from './AiInsightsPage';

export default async function ({ params }: { params: Promise<{ websiteId: string }> }) {
  const { websiteId } = await params;
  return <AiInsightsPage websiteId={websiteId} />;
}

export const metadata: Metadata = {
  title: 'AI Insights',
};
