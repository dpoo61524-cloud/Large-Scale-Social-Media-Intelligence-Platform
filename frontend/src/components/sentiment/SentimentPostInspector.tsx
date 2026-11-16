import React, { useState, useMemo } from 'react';
import type { ModernCanonicalRecord } from '../../types/dataset';
import { getTopicMeta } from '../../utils/topicHelpers';

interface SentimentPostInspectorProps {
  records: ModernCanonicalRecord[];
  initialFilter?: 'all' | 'positive' | 'neutral' | 'negative';
}

export const SentimentPostInspector: React.FC<SentimentPostInspectorProps> = ({
  records,
  initialFilter = 'all',
}) => {
  const [activeFilter, setActiveFilter] = useState<'all' | 'positive' | 'neutral' | 'negative'>(initialFilter);
  const [searchQuery, setSearchQuery] = useState('');
  const [page, setPage] = useState(1);
  const pageSize = 8;

  // Counts for tabs
  const counts = useMemo(() => {
    let pos = 0;
    let neu = 0;
    let neg = 0;
    for (const r of records) {
      const label = r.sentiment?.label;
      if (label === 'positive') pos++;
      else if (label === 'negative') neg++;
      else if (label === 'neutral') neu++;
    }
    return { all: records.length, positive: pos, neutral: neu, negative: neg };
  }, [records]);

  // Filtered list
  const filteredList = useMemo(() => {
    return records.filter((r) => {
      // 1. Sentiment filter
      if (activeFilter !== 'all') {
        if (r.sentiment?.label !== activeFilter) return false;
      }
      // 2. Search query
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const text = (
          (r.content?.title_cleaned || '') +
          ' ' +
          (r.content?.body_cleaned || '') +
          ' ' +
          (r.author?.username || '')
        ).toLowerCase();
        if (!text.includes(q)) return false;
      }
      return true;
    });
  }, [records, activeFilter, searchQuery]);

  const totalPages = Math.ceil(filteredList.length / pageSize);
  const paginatedRecords = useMemo(() => {
    const start = (page - 1) * pageSize;
    return filteredList.slice(start, start + pageSize);
  }, [filteredList, page, pageSize]);

  return (
    <div
      className="card-panel"
      style={{
        backgroundColor: '#FFFFFF',
        borderRadius: '8px',
        border: '1px solid #E2E8D0',
        padding: '20px 22px',
        boxShadow: '0 2px 6px rgba(6, 78, 59, 0.05)',
      }}
    >
      <div style={{ marginBottom: '16px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#064E3B', margin: 0 }}>
          Real-Time Sentiment Post Inspector &amp; Triage Feed
        </h3>
        <p style={{ fontSize: '12px', color: '#4B6358', margin: '4px 0 0 0' }}>
          Interactive filter drawer allowing direct inspection of posts evaluated under each sentiment class
        </p>
      </div>

      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '10px',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '16px',
        }}
      >
        {/* Sentiment Category Tabs */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          {/* All */}
          <button
            type="button"
            onClick={() => {
              setActiveFilter('all');
              setPage(1);
            }}
            style={{
              padding: '6px 14px',
              fontSize: '12px',
              fontWeight: activeFilter === 'all' ? 700 : 500,
              borderRadius: '6px',
              border: activeFilter === 'all' ? '1.5px solid #064E3B' : '1px solid #CBD5E1',
              backgroundColor: activeFilter === 'all' ? '#064E3B' : '#F8FAFC',
              color: activeFilter === 'all' ? '#FEFCE8' : '#475569',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            All Records ({counts.all})
          </button>

          {/* Positive */}
          <button
            type="button"
            onClick={() => {
              setActiveFilter('positive');
              setPage(1);
            }}
            style={{
              padding: '6px 14px',
              fontSize: '12px',
              fontWeight: activeFilter === 'positive' ? 700 : 500,
              borderRadius: '6px',
              border: activeFilter === 'positive' ? '1.5px solid #059669' : '1px solid #A7F3D0',
              backgroundColor: activeFilter === 'positive' ? '#ECFDF5' : '#FFFFFF',
              color: '#065F46',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            Positive Sentiment ({counts.positive})
          </button>

          {/* Neutral */}
          <button
            type="button"
            onClick={() => {
              setActiveFilter('neutral');
              setPage(1);
            }}
            style={{
              padding: '6px 14px',
              fontSize: '12px',
              fontWeight: activeFilter === 'neutral' ? 700 : 500,
              borderRadius: '6px',
              border: activeFilter === 'neutral' ? '1.5px solid #475569' : '1px solid #E2E8F0',
              backgroundColor: activeFilter === 'neutral' ? '#F1F5F9' : '#FFFFFF',
              color: '#334155',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            Neutral / Technical ({counts.neutral})
          </button>

          {/* Negative */}
          <button
            type="button"
            onClick={() => {
              setActiveFilter('negative');
              setPage(1);
            }}
            style={{
              padding: '6px 14px',
              fontSize: '12px',
              fontWeight: activeFilter === 'negative' ? 700 : 500,
              borderRadius: '6px',
              border: activeFilter === 'negative' ? '1.5px solid #DC2626' : '1px solid #FECACA',
              backgroundColor: activeFilter === 'negative' ? '#FEF2F2' : '#FFFFFF',
              color: '#991B1B',
              cursor: 'pointer',
              transition: 'all 0.15s ease',
            }}
          >
            Negative / Critical ({counts.negative})
          </button>
        </div>

        {/* Quick Search Input */}
        <div style={{ position: 'relative', width: '220px' }}>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setPage(1);
            }}
            placeholder="Search inspector..."
            style={{
              width: '100%',
              padding: '6px 10px',
              fontSize: '12px',
              borderRadius: '6px',
              border: '1px solid #CBD5E1',
              outline: 'none',
            }}
          />
        </div>
      </div>

      {/* Post List */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {paginatedRecords.length === 0 ? (
          <div style={{ padding: '24px', textAlign: 'center', color: '#64748B', fontSize: '13px' }}>
            No records found under the current sentiment filter.
          </div>
        ) : (
          paginatedRecords.map((r) => {
            const isReddit = r.platform === 'reddit';
            const platformColor = isReddit ? '#FF4500' : '#FF0000';
            const author = r.author?.username || r.author?.channel_title || 'Unknown';
            const sentimentLabel = r.sentiment?.label || 'undetermined';
            const score = typeof r.sentiment?.score === 'number' ? r.sentiment.score.toFixed(2) : 'N/A';
            const topicId = r.topics?.topic_id;
            const topicMeta = topicId !== null && topicId !== undefined && topicId >= 0 ? getTopicMeta('', topicId) : null;
            const snippet = r.content?.body_cleaned || r.content?.title_cleaned || 'No text snippet available.';

            const pillBg =
              sentimentLabel === 'positive' ? '#ECFDF5' : sentimentLabel === 'negative' ? '#FEF2F2' : '#F1F5F9';
            const pillBorder =
              sentimentLabel === 'positive' ? '#A7F3D0' : sentimentLabel === 'negative' ? '#FECACA' : '#CBD5E1';
            const pillColor =
              sentimentLabel === 'positive' ? '#065F46' : sentimentLabel === 'negative' ? '#991B1B' : '#475569';

            return (
              <div
                key={r.document_id}
                style={{
                  padding: '12px 14px',
                  borderRadius: '6px',
                  border: '1px solid #E2E8D0',
                  backgroundColor: '#FCFDF9',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '6px',
                }}
              >
                {/* Header row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span
                      style={{
                        padding: '1px 6px',
                        borderRadius: '3px',
                        backgroundColor: platformColor,
                        color: '#FFFFFF',
                        fontSize: '10px',
                        fontWeight: 700,
                        textTransform: 'uppercase',
                      }}
                    >
                      {r.platform}
                    </span>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: '#0F172A' }}>
                      {author}
                    </span>
                    {r.timestamps?.published_at && (
                      <span style={{ fontSize: '11px', color: '#64748B' }}>
                        • {new Date(r.timestamps.published_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                      </span>
                    )}
                  </div>

                  {/* Sentiment Confidence Badge */}
                  <div
                    style={{
                      padding: '2px 8px',
                      borderRadius: '4px',
                      backgroundColor: pillBg,
                      border: `1px solid ${pillBorder}`,
                      color: pillColor,
                      fontSize: '11px',
                      fontWeight: 700,
                      textTransform: 'capitalize',
                    }}
                  >
                    {sentimentLabel} ({score} Conf)
                  </div>
                </div>

                {/* Content snippet */}
                <p style={{ margin: 0, fontSize: '12px', color: '#334155', lineHeight: '1.5' }}>
                  {snippet.length > 200 ? snippet.slice(0, 200) + '...' : snippet}
                </p>

                {/* Footer with topic tag and external link */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '2px' }}>
                  {topicMeta ? (
                    <span style={{ fontSize: '11px', fontWeight: 600, color: '#064E3B' }}>
                      Topic #{topicId}: {topicMeta.title}
                    </span>
                  ) : (
                    <span style={{ fontSize: '11px', color: '#94A3B8' }}>Unassigned Topic</span>
                  )}
                  {r.url && (
                    <a
                      href={r.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ fontSize: '11px', color: '#2563EB', textDecoration: 'none', fontWeight: 500 }}
                    >
                      View Source ↗
                    </a>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Pagination Footer */}
      {totalPages > 1 && (
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginTop: '16px',
            fontSize: '11.5px',
            color: '#64748B',
          }}
        >
          <span>
            Showing {(page - 1) * pageSize + 1}–{Math.min(page * pageSize, filteredList.length)} of {filteredList.length} records
          </span>
          <div style={{ display: 'flex', gap: '6px' }}>
            <button
              type="button"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              style={{
                padding: '4px 10px',
                borderRadius: '4px',
                border: '1px solid #CBD5E1',
                backgroundColor: page <= 1 ? '#F1F5F9' : '#FFFFFF',
                color: page <= 1 ? '#94A3B8' : '#334155',
                cursor: page <= 1 ? 'not-allowed' : 'pointer',
              }}
            >
              Previous
            </button>
            <span style={{ padding: '4px 8px', fontWeight: 600, color: '#064E3B' }}>
              Page {page} of {totalPages}
            </span>
            <button
              type="button"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              style={{
                padding: '4px 10px',
                borderRadius: '4px',
                border: '1px solid #CBD5E1',
                backgroundColor: page >= totalPages ? '#F1F5F9' : '#FFFFFF',
                color: page >= totalPages ? '#94A3B8' : '#334155',
                cursor: page >= totalPages ? 'not-allowed' : 'pointer',
              }}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

