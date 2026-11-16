import React, { useState, useMemo } from 'react';
import type { ModernCanonicalRecord } from '../../types/dataset';

interface RiskTriageFeedProps {
  records: ModernCanonicalRecord[];
}

export const RiskTriageFeed: React.FC<RiskTriageFeedProps> = ({ records }) => {
  const [filterMode, setFilterMode] = useState<'flagged' | 'all'>('flagged');
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Extract records with credibility scores
  const assessed = useMemo(() => {
    return records.filter(
      (r) => typeof r.misinformation?.credibility_score === 'number'
    );
  }, [records]);

  // Filter flagged vs all
  const filtered = useMemo(() => {
    let list = assessed;
    if (filterMode === 'flagged') {
      list = list.filter((r) => (r.misinformation?.credibility_score || 0) < 0.7);
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter((r) => {
        const title = (r.content?.title_cleaned || r.content?.title_raw || '').toLowerCase();
        return title.includes(q);
      });
    }
    // Sort lowest credibility score first (highest risk at top)
    return [...list].sort(
      (a, b) =>
        (a.misinformation?.credibility_score || 0) -
        (b.misinformation?.credibility_score || 0)
    );
  }, [assessed, filterMode, searchQuery]);

  const flaggedCount = assessed.filter((r) => (r.misinformation?.credibility_score || 0) < 0.7).length;

  return (
    <div
      className="card-panel"
      style={{
        backgroundColor: '#FFFFFF',
        border: '1px solid #E2E8D0',
        borderRadius: '8px',
        padding: '20px 24px',
        marginBottom: '20px',
        boxShadow: '0 2px 6px rgba(6, 78, 59, 0.05)',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', marginBottom: '14px' }}>
        <div>
          <div style={{ fontSize: '15px', fontWeight: 700, color: '#064E3B', margin: 0 }}>
            Flagged Content Triage &amp; Audit Feed
          </div>
          <div style={{ fontSize: '12px', color: '#4D6B5D', marginTop: '2px' }}>
            Interactive inspection of individual posts flagged with elevated sensationalism or credibility risk
          </div>
        </div>

        {/* Filter Pills & Search */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            type="button"
            onClick={() => setFilterMode('flagged')}
            style={{
              padding: '4px 12px',
              fontSize: '11px',
              fontWeight: 700,
              borderRadius: '14px',
              border: filterMode === 'flagged' ? '1px solid #064E3B' : '1px solid #E2E8D0',
              backgroundColor: filterMode === 'flagged' ? '#064E3B' : '#FEF9C3',
              color: filterMode === 'flagged' ? '#FEFCE8' : '#064E3B',
              cursor: 'pointer',
            }}
          >
            Moderate Risk Flagged ({flaggedCount})
          </button>
          <button
            type="button"
            onClick={() => setFilterMode('all')}
            style={{
              padding: '4px 12px',
              fontSize: '11px',
              fontWeight: 700,
              borderRadius: '14px',
              border: filterMode === 'all' ? '1px solid #064E3B' : '1px solid #E2E8D0',
              backgroundColor: filterMode === 'all' ? '#064E3B' : '#FEF9C3',
              color: filterMode === 'all' ? '#FEFCE8' : '#064E3B',
              cursor: 'pointer',
            }}
          >
            All Evaluated Posts ({assessed.length})
          </button>
        </div>
      </div>

      {/* Search Input */}
      <div style={{ marginBottom: '12px' }}>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Filter posts by title or keyword..."
          style={{
            width: '100%',
            padding: '7px 12px',
            fontSize: '12px',
            borderRadius: '5px',
            border: '1px solid #E2E8D0',
            backgroundColor: '#FEFCE8',
            color: '#064E3B',
            outline: 'none',
            boxSizing: 'border-box',
          }}
        />
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', textAlign: 'left' }}>
          <thead>
            <tr style={{ backgroundColor: '#FEF9C3', borderBottom: '1px solid #E2E8D0' }}>
              <th style={{ padding: '8px 12px', fontWeight: 700, color: '#064E3B', width: '80px' }}>PLATFORM</th>
              <th style={{ padding: '8px 12px', fontWeight: 700, color: '#064E3B' }}>POST TITLE / CONTENT</th>
              <th style={{ padding: '8px 12px', fontWeight: 700, color: '#064E3B', width: '110px' }}>CREDIBILITY</th>
              <th style={{ padding: '8px 12px', fontWeight: 700, color: '#064E3B', width: '100px' }}>RISK TIER</th>
              <th style={{ padding: '8px 12px', fontWeight: 700, color: '#064E3B', width: '150px' }}>DETECTED SIGNAL</th>
            </tr>
          </thead>
          <tbody>
            {filtered.slice(0, 15).map((r) => {
              const score = r.misinformation?.credibility_score || 0;
              const isMod = score < 0.7;
              const isExpanded = expandedId === r.document_id;
              const title = r.content?.title_cleaned || r.content?.title_raw || 'Untitled Post';
              const body = r.content?.body_cleaned || r.content?.body_raw || '';

              const sigs = r.misinformation?.stylometric_signals as Record<string, number> | undefined;
              let detectedSignal = 'Organic Technical Text';
              if (sigs) {
                const allCaps = typeof sigs.all_caps_word_ratio === 'number' ? sigs.all_caps_word_ratio : 0;
                const upperRatio = typeof sigs.uppercase_char_ratio === 'number' ? sigs.uppercase_char_ratio : 0;
                const excl = typeof sigs.exclamation_intensity === 'number' ? sigs.exclamation_intensity : 0;

                if (allCaps > 0.04) {
                  detectedSignal = 'Technical Acronyms / Casing Style';
                } else if (upperRatio > 0.05) {
                  detectedSignal = 'Marketing Casing Style';
                } else if (excl > 0) {
                  detectedSignal = 'Exclamatory Urgency Marker';
                } else if (isMod) {
                  detectedSignal = 'Elevated Subjectivity Pattern';
                }
              }

              return (
                <React.Fragment key={r.document_id}>
                  <tr
                    onClick={() => setExpandedId(isExpanded ? null : r.document_id)}
                    style={{
                      borderBottom: '1px solid #E2E8D0',
                      backgroundColor: isExpanded ? '#FEF9C3' : '#FFFFFF',
                      cursor: 'pointer',
                      transition: 'background-color 0.15s ease',
                    }}
                  >
                    <td style={{ padding: '8px 12px' }}>
                      <span
                        style={{
                          fontSize: '10px',
                          fontWeight: 700,
                          color: r.platform === 'youtube' ? '#DC2626' : '#EA580C',
                          backgroundColor: r.platform === 'youtube' ? '#FEE2E2' : '#FFEDD5',
                          padding: '2px 6px',
                          borderRadius: '3px',
                        }}
                      >
                        {r.platform.toUpperCase()}
                      </span>
                    </td>
                    <td style={{ padding: '8px 12px', color: '#064E3B', fontWeight: 600 }}>
                      <div style={{ maxWidth: '440px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {title}
                      </div>
                    </td>
                    <td style={{ padding: '8px 12px', fontFamily: 'monospace', fontWeight: 700, color: '#064E3B' }}>
                      {score.toFixed(3)}
                    </td>
                    <td style={{ padding: '8px 12px' }}>
                      <span
                        style={{
                          fontSize: '10px',
                          fontWeight: 700,
                          color: isMod ? '#B45309' : '#047857',
                          backgroundColor: isMod ? '#FEF3C7' : '#DCFCE7',
                          padding: '2px 6px',
                          borderRadius: '3px',
                        }}
                      >
                        {isMod ? 'Moderate' : 'Low Risk'}
                      </span>
                    </td>
                    <td style={{ padding: '8px 12px', fontSize: '11px', color: '#4D6B5D' }}>
                      {detectedSignal}
                    </td>
                  </tr>

                  {isExpanded && (
                    <tr style={{ backgroundColor: '#FEFCE8' }}>
                      <td colSpan={5} style={{ padding: '12px 16px', borderBottom: '1px solid #CBD5B8' }}>
                        <div style={{ fontSize: '11px', color: '#4B6358', marginBottom: '4px', fontWeight: 700 }}>
                          AUTHOR: <span style={{ fontFamily: 'monospace', fontWeight: 400 }}>{r.author?.username || 'Unknown'}</span>
                        </div>
                        <div style={{ fontSize: '12.5px', color: '#064E3B', lineHeight: '1.4', marginTop: '4px' }}>
                          <strong>Full Title:</strong> {title}
                        </div>
                        {body && (
                          <div style={{ fontSize: '12px', color: '#334155', marginTop: '6px', maxHeight: '100px', overflowY: 'auto', backgroundColor: '#FFFFFF', padding: '8px', borderRadius: '4px', border: '1px solid #E2E8D0' }}>
                            {body}
                          </div>
                        )}
                        <div style={{ marginTop: '8px', fontSize: '11px', color: '#047857' }}>
                          ✓ Evaluated by LIAR-trained classifier · Veracity label: <strong>{r.misinformation?.veracity_label || 'real'}</strong>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>

        {filtered.length === 0 && (
          <div style={{ padding: '20px', textAlign: 'center', color: '#64748B', fontSize: '12px' }}>
            No records matched the current filter.
          </div>
        )}
      </div>

      <div style={{ fontSize: '11px', color: '#4D6B5D', marginTop: '10px' }}>
        Showing top {Math.min(filtered.length, 15)} of {filtered.length} posts · Click any post to inspect full body text and model signals.
      </div>
    </div>
  );
};
