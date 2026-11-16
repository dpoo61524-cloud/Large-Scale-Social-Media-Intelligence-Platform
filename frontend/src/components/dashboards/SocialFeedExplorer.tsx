import React, { useState, useMemo } from 'react';
import type { ModernCanonicalRecord } from '../../types/dataset';
import {
  filterAndSortFeedRecords,
  getPlatformEmptyState,
  type PlatformFilter,
} from '../../utils/analytics';
import { PlatformEmptyState } from '../common/PlatformEmptyState';
import { getTopicMeta } from '../../utils/topicHelpers';

interface SocialFeedExplorerProps {
  records: ModernCanonicalRecord[];
  platformFilter?: PlatformFilter;
  stages?: Record<string, unknown> | null;
  onResetFilter?: (filter: PlatformFilter) => void;
}

export const SocialFeedExplorer: React.FC<SocialFeedExplorerProps> = ({
  records,
  platformFilter = 'all',
  stages,
  onResetFilter,
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [platform, setPlatform] = useState<PlatformFilter>(platformFilter);
  const [sentimentFilter, setSentimentFilter] = useState<'all' | 'positive' | 'neutral' | 'negative' | 'undetermined'>('all');
  const [sortBy, setSortBy] = useState<'default' | 'newest' | 'score' | 'credibility'>('default');

  const redditCount = useMemo(() => records.filter((r) => r.platform === 'reddit').length, [records]);
  const youtubeCount = useMemo(() => records.filter((r) => r.platform === 'youtube').length, [records]);

  const platformRecords = useMemo(() => {
    if (platform === 'all') return records;
    return records.filter((r) => r && r.platform === platform);
  }, [records, platform]);

  const emptyState = getPlatformEmptyState(platform, platformRecords.length, stages);

  // Apply search, filters, and sort
  const displayedRecords = useMemo(() => {
    return filterAndSortFeedRecords(platformRecords, {
      query: searchQuery,
      sentimentFilter,
      sortBy,
    });
  }, [platformRecords, searchQuery, sentimentFilter, sortBy]);

  // If the active platform has 0 records (e.g. YouTube skipped), display honest empty state
  if (emptyState.isEmpty) {
    return (
      <div className="dashboard-section">
        <div className="section-header">
          <h2 className="section-title">Social Media Feed & Comment Explorer</h2>
        </div>
        <PlatformEmptyState info={emptyState} onResetFilter={onResetFilter} />
      </div>
    );
  }

  const formatTimestamp = (iso: string | null): string => {
    if (!iso) return 'Not available';
    try {
      const d = new Date(iso);
      if (isNaN(d.getTime())) return iso;
      return d.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        timeZoneName: 'short',
      });
    } catch {
      return iso;
    }
  };

  const clearAllFilters = () => {
    setSearchQuery('');
    setPlatform('all');
    setSentimentFilter('all');
    setSortBy('default');
  };

  const hasActiveFilters = searchQuery !== '' || platform !== 'all' || sentimentFilter !== 'all' || sortBy !== 'default';

  return (
    <div className="dashboard-section">
      <div className="section-header">
        <h2 className="section-title">Social Media Feed & Comment Explorer</h2>
      </div>

      <div className="card-panel" style={{ marginBottom: '20px', padding: '16px 20px', backgroundColor: '#FFFFFF' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ flex: '1 1 320px', position: 'relative' }}>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search post titles, content, or usernames (e.g. 'RoBERTa', 'model')..."
              style={{
                width: '100%',
                padding: '9px 36px 9px 12px',
                fontSize: '13px',
                borderRadius: '6px',
                border: '1px solid #CBD5E1',
                outline: 'none',
              }}
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => setSearchQuery('')}
                style={{
                  position: 'absolute',
                  right: '10px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  background: 'none',
                  border: 'none',
                  color: '#94A3B8',
                  cursor: 'pointer',
                  fontSize: '14px',
                  fontWeight: 600,
                }}
                title="Clear search"
              >
                ✕
              </button>
            )}
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
            <select
              value={platform}
              onChange={(e) => setPlatform(e.target.value as PlatformFilter)}
              style={{
                padding: '8px 12px',
                fontSize: '12px',
                borderRadius: '6px',
                border: '1px solid #CBD5E1',
                backgroundColor: '#F8FAFC',
                color: '#1E293B',
                cursor: 'pointer',
                fontWeight: 600,
              }}
            >
              <option value="all">All Channels ({records.length})</option>
              <option value="reddit">Reddit ({redditCount})</option>
              <option value="youtube">YouTube ({youtubeCount})</option>
            </select>

            <select
              value={sentimentFilter}
              onChange={(e) => setSentimentFilter(e.target.value as typeof sentimentFilter)}
              style={{
                padding: '8px 12px',
                fontSize: '12px',
                borderRadius: '6px',
                border: '1px solid #CBD5E1',
                backgroundColor: '#FFFFFF',
                color: '#334155',
                cursor: 'pointer',
              }}
            >
              <option value="all">All Sentiments</option>
              <option value="positive">Positive</option>
              <option value="neutral">Neutral</option>
              <option value="negative">Negative</option>
              <option value="undetermined">Undetermined (Offline)</option>
            </select>

            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
              style={{
                padding: '8px 12px',
                fontSize: '12px',
                borderRadius: '6px',
                border: '1px solid #CBD5E1',
                backgroundColor: '#FFFFFF',
                color: '#334155',
                cursor: 'pointer',
              }}
            >
              <option value="default">Ingestion Order</option>
              <option value="newest">Newest Published</option>
              <option value="score">Highest Upvotes/Score</option>
              <option value="credibility">Credibility Score</option>
            </select>

            {hasActiveFilters && (
              <button
                type="button"
                onClick={clearAllFilters}
                style={{
                  padding: '7px 12px',
                  fontSize: '12px',
                  borderRadius: '6px',
                  border: '1px solid #E2E8F0',
                  backgroundColor: '#F1F5F9',
                  color: '#475569',
                  cursor: 'pointer',
                }}
              >
                Reset Filters
              </button>
            )}
          </div>
        </div>

        <div style={{ marginTop: '12px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '12px', color: '#64748B' }}>
          <span>
            Showing <strong>{displayedRecords.length}</strong> of <strong>{records.length}</strong> canonical records
            {platformFilter !== 'all' && ` (${platformFilter})`}
          </span>
          {hasActiveFilters && (
            <span style={{ color: '#2563EB', fontWeight: 500 }}>
              Filters applied
            </span>
          )}
        </div>
      </div>

      {displayedRecords.length === 0 && (
        <div className="card-panel" style={{ padding: '40px 20px', textAlign: 'center', backgroundColor: '#FFFFFF' }}>
          <div style={{ fontSize: '15px', fontWeight: 600, color: '#334155', marginBottom: '8px' }}>
            No Matching Social Media Posts
          </div>
          <div style={{ fontSize: '13px', color: '#64748B', maxWidth: '480px', margin: '0 auto 16px auto', lineHeight: '1.5' }}>
            {searchQuery
              ? `No records matched your search query "${searchQuery}" under the current filter settings.`
              : 'No records matched the selected sentiment or topic criteria.'}
          </div>
          <button
            type="button"
            onClick={clearAllFilters}
            className="refresh-button"
            style={{ padding: '8px 16px', fontSize: '13px' }}
          >
            Clear Search & Filters
          </button>
        </div>
      )}

      {/* Feed List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {displayedRecords.map((r) => {
          const authorName = r.author?.username || r.author?.channel_title || 'Not available';
          const isReddit = r.platform === 'reddit';
          const platformColor = isReddit ? '#FF4500' : '#FF0000';
          const publishedDate = formatTimestamp(r.timestamps?.published_at);
          const title = r.content?.title_cleaned || r.content?.title_raw || 'Untitled Post';
          const body = r.content?.body_cleaned || r.content?.body_raw || 'No text content available';
          const score = typeof r.engagement?.score === 'number' ? r.engagement.score : 'Not available';
          const commentsCount = typeof r.engagement?.comments_count === 'number' ? r.engagement.comments_count : 'Not available';

          // Misinformation / Credibility derivation
          const credScore = typeof r.misinformation?.credibility_score === 'number'
            ? r.misinformation.credibility_score
            : null;
          const riskTier = r.misinformation?.risk_tier;
          const riskLabel =
            riskTier === 'low'
              ? 'Low Risk (Model Estimate)'
              : riskTier === 'moderate'
              ? 'Moderate Risk (Model Estimate)'
              : riskTier === 'high_risk'
              ? 'High Risk (Model Estimate)'
              : credScore !== null
              ? 'Assessed (Model Estimate)'
              : 'Not assessed';

          // Sentiment derivation
          const isSentimentSuccess = r.sentiment?.status === 'success' && r.sentiment?.label;
          const sentimentLabel = isSentimentSuccess ? r.sentiment.label : 'Undetermined (Weights Offline)';

          // Topic derivation
          const topicId = r.topics?.topic_id;
          const meta = topicId !== null && topicId !== undefined && topicId >= 0 ? getTopicMeta('', topicId) : null;
          const topicLabel = meta
            ? `Topic #${topicId}: ${meta.title}`
            : topicId === -1
            ? 'Topic: Outlier / Noise'
            : 'Topic: Unassigned';

          return (
            <div
              key={r.document_id}
              className="card-panel"
              style={{
                backgroundColor: '#FFFFFF',
                borderRadius: '8px',
                border: '1px solid #E2E8F0',
                padding: '20px',
                boxShadow: '0 1px 3px rgba(0, 0, 0, 0.04)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px', marginBottom: '10px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                  {/* Platform Badge */}
                  <span
                    style={{
                      display: 'inline-block',
                      padding: '2px 8px',
                      fontSize: '11px',
                      fontWeight: 700,
                      textTransform: 'uppercase',
                      borderRadius: '4px',
                      color: '#FFFFFF',
                      backgroundColor: platformColor,
                    }}
                  >
                    {r.platform}
                  </span>

                  {/* Author Name */}
                  <span style={{ fontSize: '13px', fontWeight: 600, color: '#0F172A' }}>
                    {authorName}
                  </span>
                </div>

                {/* Published Date */}
                <span style={{ fontSize: '12px', color: '#64748B' }}>
                  {publishedDate}
                </span>
              </div>

              {/* Title */}
              <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#1E293B', marginBottom: '8px', lineHeight: '1.4' }}>
                {title}
              </h3>

              <p style={{ fontSize: '13px', color: '#334155', lineHeight: '1.6', marginBottom: '14px', whiteSpace: 'pre-wrap' }}>
                {body}
              </p>

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '14px', alignItems: 'center', fontSize: '12px', color: '#64748B', marginBottom: '14px', paddingBottom: '12px', borderBottom: '1px solid #F1F5F9' }}>
                <span>Upvotes / Score: <strong style={{ color: '#334155' }}>{score}</strong></span>
                <span>Comments / Replies: <strong style={{ color: '#334155' }}>{commentsCount}</strong></span>
                {r.url && (
                  <a
                    href={r.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: '#2563EB', textDecoration: 'none', marginLeft: 'auto', fontWeight: 500 }}
                  >
                    Original Post ↗
                  </a>
                )}
              </div>

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
                <div
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '4px 10px',
                    borderRadius: '6px',
                    fontSize: '11px',
                    backgroundColor: credScore !== null && credScore >= 0.7 ? '#F0FDF4' : '#FEF3C7',
                    border: `1px solid ${credScore !== null && credScore >= 0.7 ? '#BBF7D0' : '#FDE68A'}`,
                    color: credScore !== null && credScore >= 0.7 ? '#166534' : '#92400E',
                  }}
                  title="Credibility score calculated from stylometric feature analysis."
                >
                  <span style={{ fontWeight: 600 }}>
                    {credScore !== null ? `Credibility Score: ${credScore.toFixed(2)}` : 'Credibility Score: Not assessed'}
                  </span>
                  <span>•</span>
                  <span>{riskLabel}</span>
                </div>

                <div
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '4px 10px',
                    borderRadius: '6px',
                    fontSize: '11px',
                    backgroundColor: isSentimentSuccess ? '#EFF6FF' : '#F1F5F9',
                    border: `1px solid ${isSentimentSuccess ? '#BFDBFE' : '#CBD5E1'}`,
                    color: isSentimentSuccess ? '#1E40AF' : '#475569',
                  }}
                >
                  <span style={{ fontWeight: 600 }}>Sentiment:</span>
                  <span style={{ textTransform: 'capitalize' }}>{sentimentLabel}</span>
                  {typeof r.sentiment?.score === 'number' && (
                    <span style={{ fontFamily: 'monospace' }}>({r.sentiment.score.toFixed(2)})</span>
                  )}
                </div>

                <div
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px',
                    padding: '4px 10px',
                    borderRadius: '6px',
                    fontSize: '11px',
                    backgroundColor: '#F8FAFC',
                    border: '1px solid #E2E8F0',
                    color: '#334155',
                  }}
                >
                  <span style={{ fontWeight: 600 }}>{topicLabel}</span>
                  {Array.isArray(r.topics?.keywords) && r.topics.keywords.length > 0 && (
                    <span style={{ color: '#64748B' }}>
                      ({r.topics.keywords.slice(0, 2).map((k) => k.word).join(', ')})
                    </span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

