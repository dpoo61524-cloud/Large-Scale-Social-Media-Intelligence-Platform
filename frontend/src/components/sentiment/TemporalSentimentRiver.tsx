import React, { useState, useMemo } from 'react';
import type { ModernCanonicalRecord } from '../../types/dataset';

interface TemporalSentimentRiverProps {
  records: ModernCanonicalRecord[];
}

interface TimeBucket {
  index: number;
  label: string;
  startDate: string;
  endDate: string;
  total: number;
  positive: number;
  neutral: number;
  negative: number;
  netPolarity: number; // between -1.0 and +1.0
  posPct: number;
  negPct: number;
  neuPct: number;
}

export const TemporalSentimentRiver: React.FC<TemporalSentimentRiverProps> = ({ records }) => {
  const [hoveredBucket, setHoveredBucket] = useState<TimeBucket | null>(null);

  const validRecords = useMemo(() => {
    return records
      .filter((r) => r.timestamps?.published_at && r.sentiment?.status === 'success')
      .map((r) => ({
        ...r,
        timestampMs: new Date(r.timestamps.published_at!).getTime(),
      }))
      .filter((r) => !isNaN(r.timestampMs))
      .sort((a, b) => a.timestampMs - b.timestampMs);
  }, [records]);

  const buckets: TimeBucket[] = useMemo(() => {
    if (validRecords.length === 0) return [];

    const numBuckets = Math.min(10, Math.max(4, Math.floor(validRecords.length / 12)));
    const bucketSize = Math.ceil(validRecords.length / numBuckets);
    const result: TimeBucket[] = [];

    for (let i = 0; i < numBuckets; i++) {
      const slice = validRecords.slice(i * bucketSize, (i + 1) * bucketSize);
      if (slice.length === 0) continue;

      let pos = 0;
      let neu = 0;
      let neg = 0;

      for (const r of slice) {
        if (r.sentiment?.label === 'positive') pos++;
        else if (r.sentiment?.label === 'negative') neg++;
        else neu++;
      }

      const total = slice.length;
      const net = total > 0 ? (pos - neg) / total : 0;

      const firstDate = new Date(slice[0].timestampMs);
      const lastDate = new Date(slice[slice.length - 1].timestampMs);

      const formatShort = (d: Date) => {
        return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
      };

      const label =
        formatShort(firstDate) === formatShort(lastDate)
          ? formatShort(firstDate)
          : `${formatShort(firstDate)}–${formatShort(lastDate)}`;

      result.push({
        index: i,
        label,
        startDate: firstDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
        endDate: lastDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
        total,
        positive: pos,
        neutral: neu,
        negative: neg,
        netPolarity: parseFloat(net.toFixed(2)),
        posPct: Math.round((pos / total) * 100),
        negPct: Math.round((neg / total) * 100),
        neuPct: Math.round((neu / total) * 100),
      });
    }

    return result;
  }, [validRecords]);

  // SVG dimensions
  const svgWidth = 920;
  const svgHeight = 240;
  const paddingLeft = 55;
  const paddingRight = 45;
  const paddingTop = 35;
  const paddingBottom = 45;

  const chartWidth = svgWidth - paddingLeft - paddingRight;
  const chartHeight = svgHeight - paddingTop - paddingBottom;
  const zeroY = paddingTop + chartHeight / 2; // neutral baseline at 0.0

  if (buckets.length === 0) {
    return (
      <div className="card-panel" style={{ padding: '24px', backgroundColor: '#FFFFFF', textAlign: 'center' }}>
        <div style={{ fontSize: '13px', color: '#64748B' }}>
          No chronological sentiment data available for temporal trajectory analysis.
        </div>
      </div>
    );
  }

  const points = buckets.map((b, i) => {
    const x = paddingLeft + (i / Math.max(1, buckets.length - 1)) * chartWidth;
    // Map netPolarity (-1.0 to +1.0) to Y
    const yClamped = Math.max(-0.6, Math.min(0.6, b.netPolarity));
    const y = zeroY - (yClamped / 0.6) * (chartHeight / 2 - 10);
    // Positive volume height above baseline
    const posHeight = (b.posPct / 100) * (chartHeight / 2 - 12);
    const negHeight = (b.negPct / 100) * (chartHeight / 2 - 12);
    return { ...b, x, y, posHeight, negHeight };
  });

  const netPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');

  const posAreaPoints = points.map((p) => `${p.x.toFixed(1)},${(zeroY - p.posHeight).toFixed(1)}`);
  const posAreaPath = `M ${points[0].x.toFixed(1)},${zeroY} L ` +
    posAreaPoints.join(' L ') +
    ` L ${points[points.length - 1].x.toFixed(1)},${zeroY} Z`;

  const negAreaPoints = points.map((p) => `${p.x.toFixed(1)},${(zeroY + p.negHeight).toFixed(1)}`);
  const negAreaPath = `M ${points[0].x.toFixed(1)},${zeroY} L ` +
    negAreaPoints.join(' L ') +
    ` L ${points[points.length - 1].x.toFixed(1)},${zeroY} Z`;

  return (
    <div
      className="card-panel"
      style={{
        backgroundColor: '#FFFFFF',
        borderRadius: '8px',
        border: '1px solid #E2E8D0',
        padding: '20px 22px',
        boxShadow: '0 2px 6px rgba(6, 78, 59, 0.05)',
        marginBottom: '24px',
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: '16px' }}>
        <h3 style={{ fontSize: '15px', fontWeight: 700, color: '#064E3B', margin: 0 }}>
          Temporal Sentiment Trajectory &amp; Momentum (Over Time)
        </h3>
      </div>

      <div style={{ width: '100%', overflowX: 'auto' }}>
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          style={{ width: '100%', minWidth: '700px', height: 'auto', display: 'block' }}
        >
          <defs>
            <linearGradient id="temporalPosGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#059669" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#059669" stopOpacity="0.02" />
            </linearGradient>
            <linearGradient id="temporalNegGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#DC2626" stopOpacity="0.35" />
              <stop offset="100%" stopColor="#DC2626" stopOpacity="0.02" />
            </linearGradient>
          </defs>

          <line x1={paddingLeft} y1={paddingTop} x2={svgWidth - paddingRight} y2={paddingTop} stroke="#F1F5F9" strokeWidth="1" />
          <text x={paddingLeft - 10} y={paddingTop + 4} fontSize="10" fontFamily="monospace" fill="#94A3B8" textAnchor="end">+0.6</text>

          <line
            x1={paddingLeft}
            y1={zeroY}
            x2={svgWidth - paddingRight}
            y2={zeroY}
            stroke="#CBD5E1"
            strokeWidth="1.2"
            strokeDasharray="4 3"
          />
          <text x={paddingLeft - 10} y={zeroY + 3} fontSize="10" fontFamily="monospace" fill="#64748B" textAnchor="end">0.0</text>

          <line x1={paddingLeft} y1={paddingTop + chartHeight} x2={svgWidth - paddingRight} y2={paddingTop + chartHeight} stroke="#F1F5F9" strokeWidth="1" />
          <text x={paddingLeft - 10} y={paddingTop + chartHeight + 4} fontSize="10" fontFamily="monospace" fill="#94A3B8" textAnchor="end">-0.6</text>

          <path d={posAreaPath} fill="url(#temporalPosGrad)" />
          <path d={negAreaPath} fill="url(#temporalNegGrad)" />

          <path d={netPath} fill="none" stroke="#064E3B" strokeWidth="2.5" strokeDasharray="5 3" />

          {points.map((p, idx) => (
            <g key={idx}>
              {/* Positive Top Cap */}
              <circle cx={p.x} cy={zeroY - p.posHeight} r="3" fill="#059669" />
              {/* Negative Bottom Cap */}
              <circle cx={p.x} cy={zeroY + p.negHeight} r="3" fill="#DC2626" />

              <circle
                cx={p.x}
                cy={p.y}
                r={hoveredBucket?.index === p.index ? '6' : '4.5'}
                fill={p.netPolarity >= 0 ? '#059669' : '#DC2626'}
                stroke="#FFFFFF"
                strokeWidth="2"
                style={{ cursor: 'pointer', transition: 'r 0.15s ease' }}
                onMouseEnter={() => setHoveredBucket(p)}
                onMouseLeave={() => setHoveredBucket(null)}
              />

              {/* X-Axis Date Labels */}
              <text
                x={p.x}
                y={svgHeight - 12}
                fontSize="10"
                fontWeight="600"
                fill="#64748B"
                textAnchor="middle"
              >
                {p.label}
              </text>
            </g>
          ))}

          {/* Interactive Hover Tooltip inside SVG */}
          {hoveredBucket && (
            <g
              transform={`translate(${Math.min(
                svgWidth - 210,
                Math.max(paddingLeft, points[hoveredBucket.index].x - 85)
              )}, 8)`}
            >
              <rect
                width="175"
                height="62"
                rx="6"
                fill="#064E3B"
                filter="drop-shadow(0 2px 4px rgba(0,0,0,0.2))"
              />
              <text x="10" y="16" fontSize="10.5" fontWeight="700" fill="#FEFCE8">
                {hoveredBucket.startDate}
              </text>
              <text x="10" y="32" fontSize="10" fill="#D1FAE5">
                Vol: {hoveredBucket.total} posts | Net: {hoveredBucket.netPolarity > 0 ? `+${hoveredBucket.netPolarity}` : hoveredBucket.netPolarity}
              </text>
              <text x="10" y="48" fontSize="9.5" fill="#E2E8F0">
                Pos: {hoveredBucket.positive} ({hoveredBucket.posPct}%) • Neg: {hoveredBucket.negative} ({hoveredBucket.negPct}%)
              </text>
            </g>
          )}
        </svg>
      </div>

      {/* Chart Legend */}
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '20px',
          alignItems: 'center',
          marginTop: '12px',
          paddingTop: '12px',
          borderTop: '1px solid #F1F5F9',
          fontSize: '11.5px',
          color: '#475569',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: '#059669' }} />
          <span>Positive Sentiment Volume</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: '#DC2626' }} />
          <span>Negative Sentiment Volume</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: '18px', height: '2px', backgroundColor: '#064E3B', borderTop: '2px dashed #064E3B' }} />
          <span style={{ fontWeight: 600, color: '#064E3B' }}>Rolling Net Polarity Line</span>
        </div>
      </div>
    </div>
  );
};

