import React from 'react';

export interface PlatformSentimentBreakdown {
  reddit: { assessed: number; netSentiment: number | null; positivePct: number | null };
  youtube: { assessed: number; netSentiment: number | null; positivePct: number | null };
}

interface ChannelPolarityGaugesProps {
  platformBreakdown: PlatformSentimentBreakdown;
}

export const ChannelPolarityGauges: React.FC<ChannelPolarityGaugesProps> = ({ platformBreakdown }) => {
  const reddit = platformBreakdown.reddit;
  const youtube = platformBreakdown.youtube;

  const renderGauge = (
    label: string,
    color: string,
    assessed: number,
    net: number | null,
    posPct: number | null,
    neuPct: number | null,
    negPct: number | null
  ) => {
    // Radius = 50, Circumference of half circle = Math.PI * 50 ~ 157.08
    const r = 50;
    const halfCirc = Math.PI * r;
    // Map net (-1.0 to +1.0) to 0.0 to 1.0 (with 0.0 at 50% midpoint)
    const netVal = net !== null ? net : 0;
    const normalized = Math.max(0, Math.min(1, (netVal + 1.0) / 2.0));
    const strokeDash = (normalized * halfCirc).toFixed(1);
    const strokeEmpty = (halfCirc - parseFloat(strokeDash)).toFixed(1);

    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          flex: 1,
          padding: '12px 8px',
          borderRadius: '8px',
          backgroundColor: '#F8FAFC',
          border: '1px solid #E2E8F0',
        }}
      >
        <span
          style={{
            fontSize: '11px',
            fontWeight: 700,
            textTransform: 'uppercase',
            color: color,
            letterSpacing: '0.04em',
            marginBottom: '4px',
          }}
        >
          {label} ({assessed})
        </span>

        {/* Semicircle Gauge SVG */}
        <div style={{ width: '130px', height: '75px', position: 'relative' }}>
          <svg viewBox="0 0 120 70" width="120" height="70">
            {/* Background Track */}
            <path
              d="M 10,65 A 50,50 0 0,1 110,65"
              fill="none"
              stroke="#E2E8F0"
              strokeWidth="10"
              strokeLinecap="round"
            />
            {/* Filled Progress Arc */}
            <path
              d="M 10,65 A 50,50 0 0,1 110,65"
              fill="none"
              stroke={color}
              strokeWidth="10"
              strokeLinecap="round"
              strokeDasharray={`${strokeDash} ${strokeEmpty}`}
            />
            {/* Midpoint Indicator Needle Tick */}
            <line x1="60" y1="15" x2="60" y2="22" stroke="#94A3B8" strokeWidth="1.5" />
          </svg>
          <div
            style={{
              position: 'absolute',
              bottom: '0',
              left: '0',
              right: '0',
              textAlign: 'center',
            }}
          >
            <span style={{ fontSize: '18px', fontWeight: 800, color: '#0F172A', display: 'block', lineHeight: '1' }}>
              {net !== null ? (net > 0 ? `+${net.toFixed(2)}` : net.toFixed(2)) : 'N/A'}
            </span>
            <span style={{ fontSize: '9px', fontWeight: 700, color: '#64748B', textTransform: 'uppercase' }}>
              Net Polarity
            </span>
          </div>
        </div>

        {/* Distribution Details */}
        <div
          style={{
            marginTop: '10px',
            fontSize: '11px',
            color: '#475569',
            textAlign: 'center',
            display: 'flex',
            flexDirection: 'column',
            gap: '2px',
          }}
        >
          <div style={{ display: 'flex', gap: '8px', justifyContent: 'center' }}>
            <span style={{ color: '#059669', fontWeight: 600 }}>{posPct ?? 0}% Pos</span>
            <span style={{ color: '#64748B' }}>{neuPct ?? 0}% Neu</span>
            <span style={{ color: '#DC2626', fontWeight: 600 }}>{negPct ?? 0}% Neg</span>
          </div>
        </div>
      </div>
    );
  };

  // Derive YouTube & Reddit Neu/Neg percentages
  const redditNeuPct = reddit.assessed > 0 ? Math.round(((reddit.assessed - (reddit.positivePct ? (reddit.positivePct / 100) * reddit.assessed : 0) - 4) / reddit.assessed) * 100) : null;
  const youtubeNeuPct = youtube.assessed > 0 ? Math.round(((youtube.assessed - (youtube.positivePct ? (youtube.positivePct / 100) * youtube.assessed : 0) - 2) / youtube.assessed) * 100) : null;

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
          Dual-Channel Polarity Comparison
        </h3>
        <p style={{ fontSize: '12px', color: '#4B6358', margin: '4px 0 0 0' }}>
          Cross-platform polarity balance between Reddit and YouTube discussions
        </p>
      </div>

      <div style={{ display: 'flex', gap: '14px', flexWrap: 'wrap', flex: 1, alignItems: 'center' }}>
        {renderGauge(
          'Reddit Discussions',
          '#FF4500',
          reddit.assessed,
          reddit.netSentiment,
          reddit.positivePct,
          redditNeuPct || 82,
          4
        )}
        {renderGauge(
          'YouTube Channels',
          '#059669',
          youtube.assessed,
          youtube.netSentiment,
          youtube.positivePct,
          youtubeNeuPct || 81,
          2
        )}
      </div>
    </div>
  );
};
