import { useState, useEffect } from 'react';
import { Column, Row, Text, Button, Form, FormButtons, FormSubmitButton, TextField } from '@umami/react-zen';
import { useApi } from '@/components/hooks';

export function MlApiKeys({ websiteId }: { websiteId: string }) {
  const { get, post, del } = useApi();
  const [keys, setKeys] = useState<any[]>([]);
  const [newKey, setNewKey] = useState<string | null>(null);
  const [keyName, setKeyName] = useState('');
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);

  const loadKeys = async () => {
    setLoading(true);
    try {
      const data = await get(`/admin/ml/keys?websiteId=${websiteId}`);
      setKeys(data?.keys || []);
    } catch { /* ignore */ }
    setLoading(false);
  };

  useEffect(() => { loadKeys(); }, [websiteId]);

  const createKey = async () => {
    setCreating(true);
    setNewKey(null);
    try {
      const data = await post('/admin/ml/keys', { websiteId, name: keyName || 'default' });
      setNewKey(data?.key || null);
      setKeyName('');
      loadKeys();
    } catch { /* ignore */ }
    setCreating(false);
  };

  const revokeKey = async (keyId: string) => {
    try {
      await del('/admin/ml/keys', { websiteId, keyId });
      loadKeys();
    } catch { /* ignore */ }
  };

  const exampleKey = newKey ? `umami_ml_${newKey.slice(0, 8)}...` : 'umami_ml_YOUR_KEY...';

  return (
    <Column gap="3">
      <Text size="sm" color="muted">
        ML API keys allow third-party services to access your website's trained ML models
        (next-page prediction, intent classification, recommendations, etc.)
        without requiring a Umami login session.
      </Text>

      <Form onSubmit={createKey}>
        <Row gap="2" alignItems="end">
          <FormInput
            label="Key Name"
            name="name"
            value={keyName}
            onChange={setKeyName}
            placeholder="e.g. Production API"
          />
          <FormSubmitButton isLoading={creating}>Generate New Key</FormSubmitButton>
        </Row>
      </Form>

      {newKey && (
        <Column padding="2" borderRadius backgroundColor="surface-raised" gap="1">
          <Text weight="bold" size="sm" color="warning">Save this key — it won't be shown again!</Text>
          <Text size="sm" selectable>{newKey}</Text>
          <Text size="xs" color="muted">Usage: curl -H "x-api-key: YOUR_KEY" https://sense.first8marketing.com/api/ml/v1/recommend</Text>
        </Column>
      )}

      <Column gap="2">
        <Text weight="bold" size="sm">Active API Keys</Text>
        {loading ? (
          <Text color="muted" size="sm">Loading...</Text>
        ) : keys.length === 0 ? (
          <Text color="muted" size="sm">No API keys created yet.</Text>
        ) : keys.map((k: any) => (
          <Row key={k.id} gap="2" alignItems="center" paddingY="1" paddingX="2" borderRadius backgroundColor="surface-raised">
            <Column gap="1" style={{ flex: 1 }}>
              <Row gap="2" alignItems="center">
                <Text size="sm" weight="bold">{k.name}</Text>
                <Text size="xs" color="muted">umami_ml_{k.key_prefix}...</Text>
                <Text size="xs" color={k.active ? 'success' : 'danger'}>{k.active ? 'active' : 'revoked'}</Text>
              </Row>
              <Text size="xs" color="muted">Created: {new Date(k.created_at).toLocaleDateString()}</Text>
            </Column>
            {k.active && (
              <Button variant="quiet" onPress={() => revokeKey(k.id)}>Revoke</Button>
            )}
          </Row>
        ))}
      </Column>

      <Text weight="bold" size="sm" paddingTop="2">API Endpoints</Text>
      <Text size="xs" color="muted">
POST /api/ml/v1/recommend          - Get recommendations{'\n'}
POST /api/ml/v1/predict/next-page  - Next page prediction{'\n'}
POST /api/ml/v1/predict/intent     - Session intent classification{'\n'}
POST /api/ml/v1/predict/funnel-drop- Funnel drop-off risk{'\n'}
POST /api/ml/v1/predict/rage-click - Frustration detection{'\n'}
POST /api/ml/v1/predict/cluster    - Visitor archetype{'\n'}
POST /api/ml/v1/predict/session-replay - Replay analysis{'\n'}
{'\n'}
Headers: x-api-key: YOUR_KEY{'\n'}
Body:    json with snake_case or camelCase keys
      </Text>
    </Column>
  );
}
