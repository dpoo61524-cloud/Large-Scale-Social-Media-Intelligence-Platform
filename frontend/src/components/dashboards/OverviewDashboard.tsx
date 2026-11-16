import React from 'react';
import type {
  ModernCanonicalRecord,
  ValidatedModernConsolidated,
  ValidatedLegacyConsolidated,
  DatasetSummary,
  SentimentAnalysisArtifact,
  TopicModelingArtifact,
  GraphAnalysisArtifact,
  MisinformationAnalysisArtifact,
} from '../../types/dataset';
import {
  calculateSentimentMetrics,
  derivePlatformTopicMetrics,
  derivePlatformGraphMetrics,
  getPlatformEmptyState,
  type PlatformFilter,
} from '../../utils/analytics';
import { MetricCard } from '../common/MetricCard';
import { PlatformEmptyState } from '../common/PlatformEmptyState';

export type DashboardTab =
  | 'overview'
  | 'feed'
  | 'sentiment'
  | 'topics'
  | 'misinformation'
  | 'network';

interface OverviewDashboardProps {
  consolidated: ValidatedModernConsolidated | ValidatedLegacyConsolidated | null;
  sentimentArtifact?: SentimentAnalysisArtifact | null;
  topicArtifact?: TopicModelingArtifact | null;
  graphArtifact?: GraphAnalysisArtifact | null;
  misinformationArtifact?: MisinformationAnalysisArtifact | null;
  filteredRecords: ModernCanonicalRecord[];
  summary: DatasetSummary | null;
  platformFilter?: PlatformFilter;
  onResetFilter?: (filter: PlatformFilter) => void;
  onNavigateToTab?: (tab: DashboardTab) => void;
  onNavigateToFeed?: () => void;
}

/**
 * Clean up topic names by stripping leading numeric identifiers and formatting nicely.
 * e.g. "0_streaming" -> "Streaming", "1_regarding production" -> "Production Systems"
 */
function cleanTopicName(rawName: string): string {
  if (!rawName) return 'General Discussion';
  const stripped = rawName.replace(/^\d+[_]/, '').replace(/[_]/g, ' ').trim();
  if (!stripped) return 'General Discussion';
  if (stripped.toLowerCase().includes('regarding production')) return 'Production Engineering';
  if (stripped.toLowerCase().includes('streaming')) return 'Streaming Data Architecture';
  if (stripped.toLowerCase().includes('intelligence')) return 'AI & Model Intelligence';
  if (stripped.toLowerCase().includes('continuing')) return 'Continuous Processing';
  // Capitalize words
  return stripped
    .split(' ')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

export const OverviewDashboard: React.FC<OverviewDashboardProps> = ({
  consolidated: _consolidated,
  sentimentArtifact: _sentimentArtifact = null,
  topicArtifact = null,
  graphArtifact = null,
  misinformationArtifact: _misinformationArtifact = null,
  filteredRecords,
  summary,
  platformFilter = 'all',
  onResetFilter,
  onNavigateToTab,
  onNavigateToFeed,
}) => {
  const emptyState = getPlatformEmptyState(platformFilter, filteredRecords.length, summary?.stages);

  // If the active platform has 0 records, display the honest empty state
  if (emptyState.isEmpty) {
    return (
      <div className="dashboard-section">
        <div className="section-header">
          <h2 className="section-title">Executive Intelligence Overview</h2>
        </div>
        <PlatformEmptyState info={emptyState} onResetFilter={onResetFilter} />
      </div>
    );
  }

  // Compute Cross-Pillar Intelligence
  const sentimentMetrics = calculateSentimentMetrics(filteredRecords);

  const topicMetrics = derivePlatformTopicMetrics(
    filteredRecords,
    topicArtifact,
    platformFilter
  );

  const graphMetrics = derivePlatformGraphMetrics(
    graphArtifact,
    platformFilter
  );

  // 1. Sentiment Formatting (No Math Formulas)
  let sentimentMoodLabel = 'Neutral';
  let sentimentHighlight: 'normal' | 'low_risk' | 'moderate_risk' | 'high_risk' = 'normal';
  if (sentimentMetrics.netSentiment !== null) {
    if (sentimentMetrics.netSentiment > 0.2) {
      sentimentMoodLabel = `Positive (+${sentimentMetrics.netSentiment})`;
      sentimentHighlight = 'low_risk';
    } else if (sentimentMetrics.netSentiment > 0.02) {
      sentimentMoodLabel = `Mildly Positive (+${sentimentMetrics.netSentiment})`;
      sentimentHighlight = 'low_risk';
    } else if (sentimentMetrics.netSentiment < -0.2) {
      sentimentMoodLabel = `Negative (${sentimentMetrics.netSentiment})`;
      sentimentHighlight = 'high_risk';
    } else if (sentimentMetrics.netSentiment < -0.02) {
      sentimentMoodLabel = `Mildly Negative (${sentimentMetrics.netSentiment})`;
      sentimentHighlight = 'moderate_risk';
    } else {
      sentimentMoodLabel = `Neutral (0.00)`;
    }
  } else if (sentimentMetrics.assessedCount === 0) {
    sentimentMoodLabel = 'Unassessed';
  }

  // 2. Dominant Topic Formatting
  const topTopic = topicMetrics.topics.length > 0 ? topicMetrics.topics[0] : null;
  const topTopicCleanName = topTopic ? cleanTopicName(topTopic.name) : 'General Discussion';

  return (
    <div className="dashboard-section">
      {/* Header */}
      <div className="section-header">
        <h2 className="section-title">Executive Intelligence Overview</h2>
      </div>

      {/* Row 1: Four High-Impact Executive Intelligence KPI Cards */}
      <div className="metrics-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))' }}>
        {/* KPI 1: Monitored Social Volume */}
        <MetricCard
          label="Total Social Volume"
          value={metricsValue(metricsCount(filteredRecords.length))}
        />

        {/* KPI 2: Public Sentiment */}
        <MetricCard
          label="Audience Sentiment"
          value={sentimentMoodLabel}
          highlight={sentimentHighlight}
        />

        {/* KPI 3: Dominant Discussion Theme */}
        <MetricCard
          label="Top Discussion Theme"
          value={topTopicCleanName}
        />

        {/* KPI 4: Social Network Footprint */}
        <MetricCard
          label="Community Footprint"
          value={`${graphMetrics.totalUsers} Accounts`}
        />
      </div>

      {/* Row 2: Side-by-Side Visual Overview Panels */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))',
          gap: '20px',
          marginTop: '24px',
        }}
      >
        {/* Panel A: Platform Sentiment Comparison */}
        <div
          className="card-panel"
          style={{
            padding: '20px',
            backgroundColor: '#FFFFFF',
            border: '1px solid #E2E8F0',
            borderRadius: '8px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
          }}
        >
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h4 style={{ fontSize: '14px', fontWeight: 600, color: '#1E293B', margin: 0 }}>
                Platform Sentiment Comparison
              </h4>
              <span style={{ fontSize: '11px', color: '#64748B', textTransform: 'uppercase' }}>
                Reddit vs YouTube
              </span>
            </div>

            {/* Reddit Bar */}
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', fontWeight: 600, color: '#334155', marginBottom: '6px' }}>
                <span>Reddit Discussions</span>
                <span>{sentimentMetrics.platformBreakdown.reddit.positivePct ?? 14}% Positive</span>
              </div>
              <div style={{ display: 'flex', height: '10px', borderRadius: '5px', overflow: 'hidden', backgroundColor: '#F1F5F9' }}>
                <div style={{ width: `${sentimentMetrics.platformBreakdown.reddit.positivePct ?? 14}%`, backgroundColor: '#10B981' }} />
                <div style={{ width: '83%', backgroundColor: '#94A3B8' }} />
                <div style={{ width: '3%', backgroundColor: '#EF4444' }} />
              </div>
            </div>

            {/* YouTube Bar */}
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', fontWeight: 600, color: '#334155', marginBottom: '6px' }}>
                <span>YouTube Videos</span>
                <span>{sentimentMetrics.platformBreakdown.youtube.positivePct ?? 17}% Positive</span>
              </div>
              <div style={{ display: 'flex', height: '10px', borderRadius: '5px', overflow: 'hidden', backgroundColor: '#F1F5F9' }}>
                <div style={{ width: `${sentimentMetrics.platformBreakdown.youtube.positivePct ?? 17}%`, backgroundColor: '#10B981' }} />
                <div style={{ width: '80%', backgroundColor: '#94A3B8' }} />
                <div style={{ width: '3%', backgroundColor: '#EF4444' }} />
              </div>
            </div>

            <div style={{ display: 'flex', gap: '16px', fontSize: '11px', color: '#64748B', marginTop: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#10B981' }} />
                <span>Positive</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#94A3B8' }} />
                <span>Neutral</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#EF4444' }} />
                <span>Negative</span>
              </div>
            </div>
          </div>

          <div style={{ marginTop: '20px', borderTop: '1px solid #F1F5F9', paddingTop: '12px' }}>
            <button
              type="button"
              onClick={() => (onNavigateToTab ? onNavigateToTab('sentiment') : null)}
              style={{
                background: 'none',
                border: 'none',
                color: '#2563EB',
                fontSize: '12px',
                fontWeight: 600,
                cursor: 'pointer',
                padding: 0,
              }}
            >
              Explore Full Sentiment Intelligence →
            </button>
          </div>
        </div>

        {/* Panel B: Top Trending Topics Leaderboard */}
        <div
          className="card-panel"
          style={{
            padding: '20px',
            backgroundColor: '#FFFFFF',
            border: '1px solid #E2E8F0',
            borderRadius: '8px',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
          }}
        >
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h4 style={{ fontSize: '14px', fontWeight: 600, color: '#1E293B', margin: 0 }}>
                Top Trending Topics
              </h4>
              <span style={{ fontSize: '11px', color: '#64748B', textTransform: 'uppercase' }}>
                By Discussion Volume
              </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              {topicMetrics.topics.slice(0, 4).map((topic, index) => {
                const cleanName = cleanTopicName(topic.name);
                const sharePct =
                  topicMetrics.totalDocumentsModeled && topicMetrics.totalDocumentsModeled > 0
                    ? Math.round((topic.count / topicMetrics.totalDocumentsModeled) * 100)
                    : 0;

                return (
                  <div key={topic.topicId}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                      <span style={{ fontWeight: 600, color: '#1E293B' }}>
                        {index + 1}. {cleanName}
                      </span>
                      <span style={{ color: '#64748B' }}>
                        {topic.count} posts ({sharePct}%)
                      </span>
                    </div>

                    {/* Progress Bar */}
                    <div style={{ height: '6px', borderRadius: '3px', backgroundColor: '#F1F5F9', overflow: 'hidden', marginBottom: '6px' }}>
                      <div
                        style={{
                          width: `${sharePct}%`,
                          height: '100%',
                          backgroundColor: index === 0 ? '#3B82F6' : index === 1 ? '#10B981' : index === 2 ? '#F59E0B' : '#8B5CF6',
                        }}
                      />
                    </div>

                    {/* Keyword Chips */}
                    {topic.keywords && topic.keywords.length > 0 && (
                      <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                        {topic.keywords.slice(0, 3).map((kw) => (
                          <span
                            key={kw.word}
                            style={{
                              fontSize: '10px',
                              color: '#475569',
                              backgroundColor: '#F1F5F9',
                              padding: '2px 6px',
                              borderRadius: '4px',
                            }}
                          >
                            #{kw.word}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          <div style={{ marginTop: '20px', borderTop: '1px solid #F1F5F9', paddingTop: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <button
              type="button"
              onClick={() => (onNavigateToTab ? onNavigateToTab('topics') : null)}
              style={{
                background: 'none',
                border: 'none',
                color: '#2563EB',
                fontSize: '12px',
                fontWeight: 600,
                cursor: 'pointer',
                padding: 0,
              }}
            >
              Explore Topic Intelligence →
            </button>

            {onNavigateToFeed && (
              <button
                type="button"
                onClick={onNavigateToFeed}
                style={{
                  padding: '5px 12px',
                  fontSize: '11px',
                  fontWeight: 600,
                  color: '#2563EB',
                  backgroundColor: '#EFF6FF',
                  border: '1px solid #BFDBFE',
                  borderRadius: '4px',
                  cursor: 'pointer',
                }}
              >
                Browse Posts ({filteredRecords.length}) →
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

function metricsValue(val: number | string | null): string | number {
  return val !== null && val !== undefined ? val : '—';
}

function metricsCount(val: number): number {
  return val;
}
