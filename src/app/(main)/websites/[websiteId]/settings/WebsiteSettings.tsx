import { Column } from '@umami/react-zen';
import { Panel } from '@/components/common/Panel';
import { WebsiteData } from './WebsiteData';
import { WebsiteEditForm } from './WebsiteEditForm';
import { WebsiteReplaySettings } from './WebsiteReplaySettings';
import { WebsiteShareForm } from './WebsiteShareForm';
import { WebsiteTrackingCode } from './WebsiteTrackingCode';
import { MlApiKeys } from './MlApiKeys';

export function WebsiteSettings({ websiteId }: { websiteId: string; openExternal?: boolean }) {
  return (
    <Column gap="6">
      <Panel>
        <WebsiteEditForm websiteId={websiteId} />
      </Panel>
      <Panel>
        <WebsiteTrackingCode websiteId={websiteId} />
      </Panel>
      <Panel>
        <WebsiteReplaySettings websiteId={websiteId} />
      </Panel>
      <Panel>
        <WebsiteShareForm websiteId={websiteId} />
      </Panel>
      <Panel title="ML API Keys" description="Generate API keys for third-party access to your trained ML models">
        <MlApiKeys websiteId={websiteId} />
      </Panel>
      <Panel>
        <WebsiteData websiteId={websiteId} />
      </Panel>
    </Column>
  );
}
