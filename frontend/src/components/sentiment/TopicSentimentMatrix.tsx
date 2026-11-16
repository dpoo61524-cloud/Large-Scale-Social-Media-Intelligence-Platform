import React, { useMemo } from 'react';
import type { ModernCanonicalRecord } from '../../types/dataset';
import { getTopicMeta } from '../../utils/topicHelpers';

interface TopicSentimentMatrixProps {
  records: ModernCanonicalRecord[];
  selectedTopicId?: number | null;
  onSelectTopic?: (topicId: number | null) => void;
}

interface TopicSentimentData {
  topicId: number;
  title: string;
  themeColor: string;
  total: number;
  positive: number;
  neutral: number;
  negative: number;
  posPct: number;
  neuPct: number;
  negPct: number;
}

export const TopicSentimentMatrix: React.FC<TopicSentimentMatrixProps> = ({
  records,
  selectedTopicId,
  onSelectTopic,
}) => {
  const topicData = useMemo(() => {
    const map = new Map<number, { pos: number; neu: number; neg: number; total: number }>();

    // Initialize 0 to 4
    for (let i = 0; i <= 4; i++) {
      map.set(i, { pos: 0, neu: 0, neg: 0, total: 0 });
    }

    for (const r of records) {
      const tid = r.topics?.topic_id;
      if (tid !== null && tid !== undefined && tid >= 0 && tid <= 4) {
        const entry = map.get(tid)!;
        entry.total++;
        if (r.sentiment?.label === 'positive') entry.pos++;
        else if (r.sentiment?.label === 'negative') entry.neg++;
        else entry.neu++;
      }
    }

    const list: TopicSentimentData[] = [];
    for (let i = 0; i <= 4; i++) {
      const entry = map.get(i)!;
      const meta = getTopicMeta('', i);
      const total = entry.total || 1; // avoid / 0
      list.push({
        topicId: i,
        title: meta.title,
        themeColor: meta.themeColor,
        total: entry.total,
        positive: entry.pos,
        neutral: entry.neu,
        negative: entry.neg,
        posPct: Math.round((entry.pos / total) * 100),
        neuPct: Math.round((entry.neu / total) * 100),
        negPct: Math.round((entry.neg / total) * 100),
      });
    }

    return list;
  }, [records]);

  return (
    <div
      className="card-panel"
      style={{
        backgroundColor: '#FFFFFF',
        borderRadius: '8px',
        border: '1px solid #E2E8D0',
        padding: '20px 22px',
        boxShadow: '0 2px 6px rgba(6, 78, 59, 0.05)',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div style={{ marginBottom: '16px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#064E3B', margin: 0 }}>
          Cross-Topic Sentiment Matrix
        </h3>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', flex: 1, justifyContent: 'space-around' }}>
        {topicData.map((t) => {
          const isSelected = selectedTopicId === t.topicId;

          return (
            <div
              key={t.topicId}
              onClick={() => onSelectTopic && onSelectTopic(isSelected ? null : t.topicId)}
              style={{
                cursor: onSelectTopic ? 'pointer' : 'default',
                padding: '8px 10px',
                borderRadius: '6px',
                backgroundColor: isSelected ? '#F0FDF4' : 'transparent',
                border: isSelected ? '1px solid #86EFAC' : '1px solid transparent',
                transition: 'background-color 0.15s ease',
              }}
            >
              {/* Topic Label and Stats Header */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'baseline',
                  marginBottom: '6px',
                  fontSize: '12px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span
                    style={{
                      fontFamily: 'monospace',
                      fontWeight: 700,
                      fontSize: '11px',
                      color: t.themeColor,
                      backgroundColor: '#F1F5E8',
                      padding: '1px 6px',
                      borderRadius: '4px',
                    }}
                  >
                    #{t.topicId}
                  </span>
                  <span style={{ fontWeight: 600, color: '#1E293B' }}>{t.title}</span>
                </div>
                <div style={{ fontSize: '11px', color: '#64748B', fontFamily: 'monospace' }}>
                  {t.posPct > 0 && <span style={{ color: '#059669', fontWeight: 600 }}>{t.posPct}% Pos </span>}
                  {t.neuPct > 0 && <span style={{ color: '#64748B' }}>{t.neuPct}% Neu </span>}
                  {t.negPct > 0 && <span style={{ color: '#DC2626', fontWeight: 600 }}>{t.negPct}% Neg</span>}
                  <span style={{ color: '#94A3B8' }}> ({t.total})</span>
                </div>
              </div>

              {/* Segmented Ribbon Bar */}
              <div
                style={{
                  display: 'flex',
                  height: '10px',
                  borderRadius: '5px',
                  overflow: 'hidden',
                  backgroundColor: '#E2E8F0',
                }}
              >
                {t.posPct > 0 && (
                  <div
                    style={{
                      width: `${t.posPct}%`,
                      backgroundColor: '#059669',
                      transition: 'width 0.3s ease',
                    }}
                    title={`Positive: ${t.positive} (${t.posPct}%)`}
                  />
                )}
                {t.neuPct > 0 && (
                  <div
                    style={{
                      width: `${t.neuPct}%`,
                      backgroundColor: '#94A3B8',
                      transition: 'width 0.3s ease',
                    }}
                    title={`Neutral: ${t.neutral} (${t.neuPct}%)`}
                  />
                )}
                {t.negPct > 0 && (
                  <div
                    style={{
                      width: `${t.negPct}%`,
                      backgroundColor: '#DC2626',
                      transition: 'width 0.3s ease',
                    }}
                    title={`Negative: ${t.negative} (${t.negPct}%)`}
                  />
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Matrix Legend */}
      <div
        style={{
          display: 'flex',
          gap: '16px',
          alignItems: 'center',
          marginTop: '14px',
          paddingTop: '10px',
          borderTop: '1px solid #F1F5F9',
          fontSize: '11px',
          color: '#64748B',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '2px', backgroundColor: '#059669' }} />
          <span>Positive Sentiment</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '2px', backgroundColor: '#94A3B8' }} />
          <span>Neutral / Technical</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
          <span style={{ width: '8px', height: '8px', borderRadius: '2px', backgroundColor: '#DC2626' }} />
          <span>Negative / Critical</span>
        </div>
      </div>
    </div>
  );
};

