import React from 'react';
import type { ModernCanonicalRecord, SentimentAnalysisArtifact } from '../../types/dataset';
import {
  calculateSentimentMetrics,
  getPlatformEmptyState,
  type PlatformFilter,
} from '../../utils/analytics';
import { MetricCard } from '../common/MetricCard';
import { PlatformEmptyState } from '../common/PlatformEmptyState';
import { TemporalSentimentRiver } from '../sentiment/TemporalSentimentRiver';
import { TopicSentimentMatrix } from '../sentiment/TopicSentimentMatrix';
import { ChannelPolarityGauges } from '../sentiment/ChannelPolarityGauges';

interface SentimentDashboardProps {
  filteredRecords: ModernCanonicalRecord[];
  artifact: SentimentAnalysisArtifact | null;
  platformFilter?: PlatformFilter;
  stages?: Record<string, unknown> | null;
  onResetFilter?: (filter: PlatformFilter) => void;
}

export const SentimentDashboard: React.FC<SentimentDashboardProps> = ({
  filteredRecords,
  artifact: _artifact,
  platformFilter = 'all',
  stages,
  onResetFilter,
}) => {
  const emptyState = getPlatformEmptyState(platformFilter, filteredRecords.length, stages);

  if (emptyState.isEmpty) {
    return (
      <div className="dashboard-section">
        <div className="section-header">
          <h2 className="section-title">Sentimental Analysis</h2>
        </div>
        <PlatformEmptyState info={emptyState} onResetFilter={onResetFilter} />
      </div>
    );
  }

  const metrics = calculateSentimentMetrics(filteredRecords);
  const hasAssessed = metrics.assessedCount > 0;
  const isOfflineUndetermined = filteredRecords.length > 0 && metrics.assessedCount === 0;

  return (
    <div className="dashboard-section">
      {/* Heading */}
      <div className="section-header" style={{ marginBottom: '16px' }}>
        <h2 className="section-title">Sentimental Analysis</h2>
      </div>

      {/* Honest Model Availability / Offline Notice */}
      {isOfflineUndetermined && (
        <div className="card-panel" style={{ marginBottom: '24px', backgroundColor: '#FFFBEB', borderLeft: '4px solid #F59E0B' }}>
          <div style={{ fontSize: '13px', fontWeight: 600, color: '#92400E', marginBottom: '4px' }}>
            Offline Execution Notice: Model Inference Unavailable
          </div>
          <div style={{ fontSize: '13px', color: '#B45309', lineHeight: '1.5' }}>
            {filteredRecords.length} posts were collected, but sentiment classification could not be completed because the required transformer model weights (<code>cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual</code>) were unavailable offline.
            All {filteredRecords.length} records are honestly recorded with sentiment status <code>undetermined</code>.
          </div>
        </div>
      )}

      <div className="metrics-grid" style={{ marginBottom: '20px' }}>
        <MetricCard
          label="Total Assessed Records"
          value={metrics.assessedCount}
        />
        <MetricCard
          label="Positive Sentiment"
          value={
            metrics.distribution.positivePct !== null
              ? `${metrics.distribution.positivePct}%`
              : null
          }
          unavailableText="0%"
        />
        <MetricCard
          label="Neutral / Technical"
          value={
            metrics.distribution.neutralPct !== null
              ? `${metrics.distribution.neutralPct}%`
              : null
          }
          unavailableText="0%"
        />
        <MetricCard
          label="Negative Sentiment"
          value={
            metrics.distribution.negativePct !== null
              ? `${metrics.distribution.negativePct}%`
              : null
          }
          unavailableText="0%"
        />
        <MetricCard
          label="Net Polarity Index"
          value={
            metrics.netSentiment !== null
              ? metrics.netSentiment > 0
                ? `+${metrics.netSentiment}`
                : `${metrics.netSentiment}`
              : null
          }
          unavailableText="Unavailable"
        />
      </div>

      {hasAssessed ? (
        <>
          <TemporalSentimentRiver records={filteredRecords} />

          {/* Middle Row: Cross-Topic Sentiment Matrix & Dual-Channel Polarity Gauges */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
              gap: '20px',
              marginBottom: '24px',
            }}
          >
            <TopicSentimentMatrix records={filteredRecords} />
            <ChannelPolarityGauges platformBreakdown={metrics.platformBreakdown} />
          </div>
        </>
      ) : (
        /* Diagnostic State when running offline without model inference */
        <div className="card-panel" style={{ marginTop: '24px', padding: '24px' }}>
          <div className="panel-title" style={{ marginBottom: '16px' }}>Public Mood Distribution &amp; Classification Status</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '24px', flexWrap: 'wrap', padding: '12px 0' }}>
            <div style={{ width: '130px', height: '130px', flexShrink: 0 }}>
              <svg width="130" height="130" viewBox="0 0 130 130">
                <circle cx="65" cy="65" r="50" fill="none" stroke="#E2E8F0" strokeWidth="14" />
                <circle cx="65" cy="65" r="50" fill="none" stroke="#CBD5E1" strokeWidth="14" strokeDasharray="6 4" />
                <text x="65" y="60" textAnchor="middle" dominantBaseline="central" fontSize="18" fontWeight="700" fill="#64748B">
                  0%
                </text>
                <text x="65" y="78" textAnchor="middle" dominantBaseline="central" fontSize="11" fontWeight="500" fill="#94A3B8">
                  Assessed
                </text>
              </svg>
            </div>
            <div style={{ flex: '1 1 300px' }}>
              <div style={{ fontSize: '14px', fontWeight: 600, color: '#334155', marginBottom: '6px' }}>
                100% Undetermined (Weights Uncached Locally)
              </div>
              <p style={{ fontSize: '13px', color: '#64748B', lineHeight: '1.5', marginBottom: '8px' }}>
                All <strong>{metrics.undeterminedCount} active records</strong> returned status: <code>undetermined</code>. The multilingual RoBERTa transformer weights (<code>twitter-xlm-roberta-base-sentiment-multilingual</code>, ~1.1 GB) were not pre-cached in the local offline environment.
              </p>
              <div style={{ fontSize: '12px', color: '#475569', backgroundColor: '#F8FAFC', padding: '8px 12px', borderRadius: '4px', border: '1px solid #E2E8F0' }}>
                Zero artificial mood percentages (e.g. 60% positive) were synthesized. Scores remain strictly unassigned until model weights are cached.
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
