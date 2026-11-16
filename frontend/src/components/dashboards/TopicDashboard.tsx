import React from 'react';
import type { ModernCanonicalRecord, TopicModelingArtifact, DatasetSummary } from '../../types/dataset';
import {
  derivePlatformTopicMetrics,
  getPlatformEmptyState,
  type PlatformFilter,
} from '../../utils/analytics';
import { MetricCard } from '../common/MetricCard';
import { PlatformEmptyState } from '../common/PlatformEmptyState';
import { TopicSankeyFlow } from '../topics/TopicSankeyFlow';
import { getTopicMeta, cleanKeyword } from '../../utils/topicHelpers';

interface TopicDashboardProps {
  topicArtifact: TopicModelingArtifact | null;
  filteredRecords: ModernCanonicalRecord[];
  summary: DatasetSummary | null;
  platformFilter?: PlatformFilter;
  onResetFilter?: (filter: PlatformFilter) => void;
}

export const TopicDashboard: React.FC<TopicDashboardProps> = ({
  topicArtifact,
  filteredRecords,
  summary,
  platformFilter = 'all',
  onResetFilter,
}) => {
  const emptyState = getPlatformEmptyState(platformFilter, filteredRecords.length, summary?.stages);

  if (emptyState.isEmpty) {
    return (
      <div className="dashboard-section">
        <div className="section-header">
          <h2 className="section-title">Topic Modeling & Semantic Intelligence</h2>
        </div>
        <PlatformEmptyState info={emptyState} onResetFilter={onResetFilter} />
      </div>
    );
  }

  const metrics = derivePlatformTopicMetrics(
    filteredRecords,
    topicArtifact,
    platformFilter
  );

  const sortedTopics = [...metrics.topics].sort((a, b) => a.topicId - b.topicId);

  const weightCeiling = 0.20;

  return (
    <div className="dashboard-section">
      <div className="section-header">
        <h2 className="section-title">Topic Modeling & Semantic Intelligence</h2>
      </div>

      <div className="metrics-grid" style={{ marginBottom: '16px' }}>
        <MetricCard
          label="Documents Clustered"
          value={metrics.totalDocumentsModeled}
        />
        <MetricCard
          label="Identified Clusters"
          value={metrics.topics.length}
        />
        <MetricCard
          label="Outlier Documents"
          value={metrics.outlierCount}
        />
      </div>

      {sortedTopics.length > 0 ? (
        <>
          <TopicSankeyFlow records={filteredRecords} topics={sortedTopics} />

          <div className="topics-grid">
            {sortedTopics.map((t) => {
              const meta = getTopicMeta(t.name, t.topicId);
              const pctOfTotal = Math.round(
                (t.count / Math.max(metrics.totalDocumentsModeled, 1)) * 100
              );

              return (
                <div
                  key={t.topicId}
                  className="topic-card"
                  style={{
                    backgroundColor: '#FFFFFF',
                    border: '1px solid #E2E8D0',
                    borderTop: `4px solid ${meta.themeColor}`,
                    borderRadius: '8px',
                    boxShadow: '0 2px 6px rgba(6, 78, 59, 0.05)',
                    padding: '16px 18px',
                  }}
                >
                  <div className="topic-header">
                    <span
                      className="topic-id-badge font-mono"
                      style={{
                        backgroundColor: meta.badgeBg,
                        color: meta.themeColor,
                        border: '1px solid #E2E8D0',
                        fontWeight: 700,
                        padding: '2px 8px',
                        borderRadius: '4px',
                      }}
                    >
                      Topic #{t.topicId}
                    </span>
                    <span
                      className="topic-count-badge font-mono"
                      style={{ color: '#4D6B5D', fontWeight: 600, fontSize: '11px' }}
                    >
                      {t.count} doc{t.count === 1 ? '' : 's'} ({pctOfTotal}%)
                    </span>
                  </div>

                  <div style={{ marginTop: '8px' }}>
                    <div
                      className="topic-name"
                      style={{
                        fontSize: '15px',
                        fontWeight: 700,
                        color: '#064E3B',
                        lineHeight: '1.3',
                      }}
                    >
                      {meta.title}
                    </div>
                  </div>

                  <div className="topic-keywords-container" style={{ marginTop: '14px' }}>
                    <span
                      className="keywords-label"
                      style={{
                        fontSize: '10.5px',
                        textTransform: 'uppercase',
                        letterSpacing: '0.05em',
                        color: '#4B6358',
                        fontWeight: 700,
                      }}
                    >
                      Key Discussion Keywords:
                    </span>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '3px', marginTop: '6px' }}>
                      {t.keywords.slice(0, 5).map((kw, kidx) => {
                        const cleaned = cleanKeyword(kw.word);
                        const w = kw.weight !== null ? kw.weight : 0.5;

                        const fillRatio = Math.min(1.0, Math.max(0.12, w / weightCeiling));

                        const radius = 7.5;
                        const circumference = 2 * Math.PI * radius; // ~47.12
                        const dashFilled = (fillRatio * circumference).toFixed(1);
                        const dashEmpty = (circumference - parseFloat(dashFilled)).toFixed(1);

                        return (
                          <div
                            key={kidx}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                              padding: '5px 0',
                              borderBottom:
                                kidx < t.keywords.slice(0, 5).length - 1
                                  ? '1px solid #FEF9C3'
                                  : 'none',
                              fontSize: '12px',
                            }}
                          >
                            <div
                              style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: '8px',
                                minWidth: 0,
                                flex: 1,
                              }}
                            >
                              {/* Option C: Radial Micro-Gauge Ring - Progress strictly equals score */}
                              <svg
                                width="20"
                                height="20"
                                viewBox="0 0 20 20"
                                style={{ flexShrink: 0, transform: 'rotate(-90deg)' }}
                                aria-label={`Score ${w.toFixed(2)}`}
                              >
                                {/* Background Ring Track */}
                                <circle
                                  cx="10"
                                  cy="10"
                                  r={radius}
                                  fill="none"
                                  stroke="#FEF9C3"
                                  strokeWidth="2.6"
                                />
                                {/* Filled Radial Arc */}
                                <circle
                                  cx="10"
                                  cy="10"
                                  r={radius}
                                  fill="none"
                                  stroke={meta.themeColor}
                                  strokeWidth="2.6"
                                  strokeDasharray={`${dashFilled} ${dashEmpty}`}
                                  strokeLinecap="round"
                                />
                              </svg>

                              <span
                                title={cleaned}
                                style={{
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  whiteSpace: 'nowrap',
                                  fontFamily: 'monospace',
                                  color: '#064E3B',
                                  fontWeight: 600,
                                  fontSize: '12px',
                                }}
                              >
                                {cleaned}
                              </span>
                            </div>

                            <span
                              style={{
                                fontSize: '11px',
                                color: '#064E3B',
                                fontFamily: 'monospace',
                                fontWeight: 700,
                                paddingLeft: '10px',
                                flexShrink: 0,
                              }}
                            >
                              {w.toFixed(2)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      ) : (
        <div className="card-panel">
          <div className="empty-panel-notice">
            No distinct topic clusters were discovered for the active platform records.
          </div>
        </div>
      )}
    </div>
  );
};
