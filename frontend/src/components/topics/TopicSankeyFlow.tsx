import React, { useState, useMemo } from 'react';
import type { ModernCanonicalRecord } from '../../types/dataset';
import { getTopicMeta } from '../../utils/topicHelpers';

interface TopicItem {
  topicId: number;
  name: string;
  count: number;
}

interface TopicSankeyFlowProps {
  records: ModernCanonicalRecord[];
  topics: TopicItem[];
}

export const TopicSankeyFlow: React.FC<TopicSankeyFlowProps> = ({ records, topics }) => {
  const [hoveredRibbon, setHoveredRibbon] = useState<string | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  // Dimensions & layout coordinates - Compact widescreen layout to prevent vertical clipping
  const SVG_WIDTH = 960;
  const SVG_HEIGHT = 270;
  const TOP_PAD = 36;
  const BOTTOM_PAD = 16;
  const USABLE_HEIGHT = SVG_HEIGHT - TOP_PAD - BOTTOM_PAD; // 218px

  // X Coordinates of 3 stages
  const X_PLATFORM = 35;
  const W_PLATFORM = 90;
  const X_TOPIC = 360;
  const W_TOPIC = 215;
  const X_SENTIMENT = 820;
  const W_SENTIMENT = 105;

  // 1. Compute Data Flows
  const {
    platforms,
    topicStats,
    sentiments,
    platformToTopicFlows,
    topicToSentimentFlows,
    totalRecords,
  } = useMemo(() => {
    let redditCount = 0;
    let youtubeCount = 0;

    const p2t: Record<string, Record<number, number>> = {
      reddit: {},
      youtube: {},
    };

    const t2s: Record<number, Record<string, number>> = {};

    const sentTotals: Record<string, number> = {
      positive: 0,
      neutral: 0,
      negative: 0,
    };

    for (const t of topics) {
      p2t.reddit[t.topicId] = 0;
      p2t.youtube[t.topicId] = 0;
      t2s[t.topicId] = { positive: 0, neutral: 0, negative: 0 };
    }

    let validCount = 0;
    for (const r of records) {
      if (!r) continue;
      const tid = r.topics?.topic_id;
      if (tid === null || tid === undefined || tid < 0) continue;
      if (!t2s[tid]) continue;

      validCount++;
      const p = r.platform === 'youtube' ? 'youtube' : 'reddit';
      if (p === 'reddit') redditCount++;
      else youtubeCount++;

      p2t[p][tid] = (p2t[p][tid] || 0) + 1;

      const rawSent = (r.sentiment?.label || 'neutral').toLowerCase();
      const sKey = rawSent === 'positive' || rawSent === 'negative' ? rawSent : 'neutral';
      t2s[tid][sKey] = (t2s[tid][sKey] || 0) + 1;
      sentTotals[sKey] = (sentTotals[sKey] || 0) + 1;
    }

    return {
      platforms: [
        { id: 'reddit', label: 'Reddit', count: redditCount, color: '#EA580C', bg: '#FFF7ED', border: '#FED7AA' },
        { id: 'youtube', label: 'YouTube', count: youtubeCount, color: '#DC2626', bg: '#FEF2F2', border: '#FECACA' },
      ],
      topicStats: topics,
      sentiments: [
        { id: 'positive', label: 'Positive', count: sentTotals.positive, color: '#16A34A', bg: '#DCFCE7', border: '#86EFAC' },
        { id: 'neutral', label: 'Neutral', count: sentTotals.neutral, color: '#064E3B', bg: '#FEF9C3', border: '#CBD5B8' },
        { id: 'negative', label: 'Negative', count: sentTotals.negative, color: '#DC2626', bg: '#FEE2E2', border: '#FECACA' },
      ],
      platformToTopicFlows: p2t,
      topicToSentimentFlows: t2s,
      totalRecords: Math.max(validCount, 1),
    };
  }, [records, topics]);

  // 2. Geometry & Coordinates Layout
  // Column 1: Platforms layout
  const platformLayout = useMemo(() => {
    const gap = 16;
    const availableH = USABLE_HEIGHT - gap;
    let currY = TOP_PAD;
    return platforms.map((p) => {
      const h = Math.max(34, Math.round((p.count / totalRecords) * availableH));
      const node = { ...p, x: X_PLATFORM, y: currY, width: W_PLATFORM, height: h };
      currY += h + gap;
      return node;
    });
  }, [platforms, totalRecords, USABLE_HEIGHT]);

  // Column 2: Topics layout (with executive clean titles & Light Yellow / Dark Green theme styling)
  const topicLayout = useMemo(() => {
    const gap = 8;
    const availableH = USABLE_HEIGHT - gap * Math.max(topicStats.length - 1, 1);
    let currY = TOP_PAD;
    return topicStats.map((t) => {
      const meta = getTopicMeta(t.name, t.topicId);
      const h = Math.max(22, Math.round((t.count / totalRecords) * availableH));
      const node = {
        ...t,
        cleanLabel: meta.shortLabel,
        fullTitle: meta.title,
        color: meta.themeColor,
        x: X_TOPIC,
        y: currY,
        width: W_TOPIC,
        height: h,
      };
      currY += h + gap;
      return node;
    });
  }, [topicStats, totalRecords, USABLE_HEIGHT]);

  // Column 3: Sentiments layout
  const sentimentLayout = useMemo(() => {
    const gap = 10;
    const availableH = USABLE_HEIGHT - gap * 2;
    let currY = TOP_PAD;
    return sentiments.map((s) => {
      const h = Math.max(18, Math.round((s.count / totalRecords) * availableH));
      const node = { ...s, x: X_SENTIMENT, y: currY, width: W_SENTIMENT, height: h };
      currY += h + gap;
      return node;
    });
  }, [sentiments, totalRecords, USABLE_HEIGHT]);

  // 3. Compute Bezier Ribbons between Platform and Topic
  const ribbonsP2T = useMemo(() => {
    const ribbons: Array<{
      id: string;
      sourceId: string;
      targetId: string;
      path: string;
      value: number;
      pctOfSource: number;
      color: string;
      label: string;
    }> = [];

    const pOffsets: Record<string, number> = { reddit: 0, youtube: 0 };
    const tOffsets: Record<number, number> = {};
    topicStats.forEach((t) => { tOffsets[t.topicId] = 0; });

    for (const pNode of platformLayout) {
      for (const tNode of topicLayout) {
        const val = platformToTopicFlows[pNode.id]?.[tNode.topicId] || 0;
        if (val <= 0) continue;

        const ribbonHP = Math.max(1, (val / Math.max(pNode.count, 1)) * pNode.height);
        const ribbonHT = Math.max(1, (val / Math.max(tNode.count, 1)) * tNode.height);

        const y1_top = pNode.y + (pOffsets[pNode.id] || 0);
        const y1_bot = y1_top + ribbonHP;
        pOffsets[pNode.id] = (pOffsets[pNode.id] || 0) + ribbonHP;

        const y2_top = tNode.y + (tOffsets[tNode.topicId] || 0);
        const y2_bot = y2_top + ribbonHT;
        tOffsets[tNode.topicId] = (tOffsets[tNode.topicId] || 0) + ribbonHT;

        const x1 = pNode.x + pNode.width;
        const x2 = tNode.x;
        const dx = (x2 - x1) * 0.48;

        const path = `M ${x1} ${y1_top} C ${x1 + dx} ${y1_top}, ${x2 - dx} ${y2_top}, ${x2} ${y2_top} L ${x2} ${y2_bot} C ${x2 - dx} ${y2_bot}, ${x1 + dx} ${y1_bot}, ${x1} ${y1_bot} Z`;

        ribbons.push({
          id: `ribbon-${pNode.id}-${tNode.topicId}`,
          sourceId: pNode.id,
          targetId: `topic-${tNode.topicId}`,
          path,
          value: val,
          pctOfSource: Math.round((val / Math.max(pNode.count, 1)) * 100),
          color: pNode.color,
          label: `${pNode.label} → ${tNode.cleanLabel}: ${val} posts`,
        });
      }
    }
    return ribbons;
  }, [platformLayout, topicLayout, platformToTopicFlows, topicStats]);

  // 4. Compute Bezier Ribbons between Topic and Sentiment
  const ribbonsT2S = useMemo(() => {
    const ribbons: Array<{
      id: string;
      sourceId: string;
      targetId: string;
      path: string;
      value: number;
      pctOfSource: number;
      color: string;
      label: string;
    }> = [];

    const tOffsets: Record<number, number> = {};
    const sOffsets: Record<string, number> = { positive: 0, neutral: 0, negative: 0 };
    topicStats.forEach((t) => { tOffsets[t.topicId] = 0; });

    for (const tNode of topicLayout) {
      for (const sNode of sentimentLayout) {
        const val = topicToSentimentFlows[tNode.topicId]?.[sNode.id] || 0;
        if (val <= 0) continue;

        const ribbonHT = Math.max(1, (val / Math.max(tNode.count, 1)) * tNode.height);
        const ribbonHS = Math.max(1, (val / Math.max(sNode.count, 1)) * sNode.height);

        const y1_top = tNode.y + (tOffsets[tNode.topicId] || 0);
        const y1_bot = y1_top + ribbonHT;
        tOffsets[tNode.topicId] = (tOffsets[tNode.topicId] || 0) + ribbonHT;

        const y2_top = sNode.y + (sOffsets[sNode.id] || 0);
        const y2_bot = y2_top + ribbonHS;
        sOffsets[sNode.id] = (sOffsets[sNode.id] || 0) + ribbonHS;

        const x1 = tNode.x + tNode.width;
        const x2 = sNode.x;
        const dx = (x2 - x1) * 0.48;

        const path = `M ${x1} ${y1_top} C ${x1 + dx} ${y1_top}, ${x2 - dx} ${y2_top}, ${x2} ${y2_top} L ${x2} ${y2_bot} C ${x2 - dx} ${y2_bot}, ${x1 + dx} ${y1_bot}, ${x1} ${y1_bot} Z`;

        ribbons.push({
          id: `ribbon-${tNode.topicId}-${sNode.id}`,
          sourceId: `topic-${tNode.topicId}`,
          targetId: sNode.id,
          path,
          value: val,
          pctOfSource: Math.round((val / Math.max(tNode.count, 1)) * 100),
          color: sNode.color,
          label: `${tNode.cleanLabel} → ${sNode.label}: ${val} posts`,
        });
      }
    }
    return ribbons;
  }, [topicLayout, sentimentLayout, topicToSentimentFlows, topicStats]);

  return (
    <div
      className="card-panel"
      style={{
        backgroundColor: '#FFFFFF',
        border: '1px solid #E2E8D0',
        borderRadius: '8px',
        padding: '16px 20px',
        marginBottom: '20px',
        boxShadow: '0 2px 6px rgba(6, 78, 59, 0.05)',
      }}
    >
      {/* Title & Pipeline Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
        <div>
          <div style={{ fontSize: '15px', fontWeight: 700, color: '#064E3B', margin: 0 }}>
            Cross-Platform Discourse Flow
          </div>
          <div style={{ fontSize: '12px', color: '#4D6B5D', marginTop: '2px' }}>
            Interactive flow river tracking social posts from channel origins, through semantic themes, to audience reactions
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span
            style={{
              fontSize: '11px',
              fontWeight: 600,
              color: '#064E3B',
              backgroundColor: '#FEF9C3',
              padding: '3px 10px',
              borderRadius: '12px',
              border: '1px solid #E2E8D0',
            }}
          >
            {totalRecords} Active Records Flowing
          </span>
        </div>
      </div>

      {/* SVG Canvas Container */}
      <div style={{ width: '100%', overflowX: 'auto', position: 'relative' }}>
        <svg
          viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
          style={{ width: '100%', minWidth: '780px', height: 'auto', display: 'block' }}
        >
          {/* Stage Column Labels */}
          <text
            x={X_PLATFORM + W_PLATFORM / 2}
            y="20"
            textAnchor="middle"
            fontFamily="sans-serif"
            fontSize="10.5"
            fontWeight="700"
            fill="#064E3B"
            letterSpacing="0.05em"
          >
            INGESTION CHANNELS
          </text>
          <text
            x={X_TOPIC + W_TOPIC / 2}
            y="20"
            textAnchor="middle"
            fontFamily="sans-serif"
            fontSize="10.5"
            fontWeight="700"
            fill="#064E3B"
            letterSpacing="0.05em"
          >
            DISCOVERED TOPICS (BERTopic)
          </text>
          <text
            x={X_SENTIMENT + W_SENTIMENT / 2}
            y="20"
            textAnchor="middle"
            fontFamily="sans-serif"
            fontSize="10.5"
            fontWeight="700"
            fill="#064E3B"
            letterSpacing="0.05em"
          >
            SENTIMENT OUTCOMES
          </text>

          {/* Stage Arrow Indicators */}
          <path
            d={`M ${X_PLATFORM + W_PLATFORM + 85} 16 L ${X_PLATFORM + W_PLATFORM + 115} 16`}
            stroke="#CBD5B8"
            strokeWidth="1.5"
            strokeDasharray="3,3"
          />
          <polygon
            points={`${X_PLATFORM + W_PLATFORM + 117},16 ${X_PLATFORM + W_PLATFORM + 112},13 ${X_PLATFORM + W_PLATFORM + 112},19`}
            fill="#CBD5B8"
          />

          <path
            d={`M ${X_TOPIC + W_TOPIC + 90} 16 L ${X_TOPIC + W_TOPIC + 120} 16`}
            stroke="#CBD5B8"
            strokeWidth="1.5"
            strokeDasharray="3,3"
          />
          <polygon
            points={`${X_TOPIC + W_TOPIC + 122},16 ${X_TOPIC + W_TOPIC + 117},13 ${X_TOPIC + W_TOPIC + 117},19`}
            fill="#CBD5B8"
          />

          {/* 1. Draw P2T Ribbons */}
          <g>
            {ribbonsP2T.map((r) => {
              const isHovered = hoveredRibbon === r.id;
              const isNodeRelated = hoveredNode === r.sourceId || hoveredNode === r.targetId;
              const opacity = isHovered ? 0.82 : isNodeRelated ? 0.65 : hoveredRibbon || hoveredNode ? 0.12 : 0.32;

              return (
                <path
                  key={r.id}
                  d={r.path}
                  fill={r.color}
                  fillOpacity={opacity}
                  stroke={isHovered ? r.color : 'none'}
                  strokeWidth={isHovered ? 1.5 : 0}
                  onMouseEnter={() => setHoveredRibbon(r.id)}
                  onMouseLeave={() => setHoveredRibbon(null)}
                  style={{ cursor: 'pointer', transition: 'fill-opacity 0.2s ease' }}
                >
                  <title>{r.label}</title>
                </path>
              );
            })}
          </g>

          {/* 2. Draw T2S Ribbons */}
          <g>
            {ribbonsT2S.map((r) => {
              const isHovered = hoveredRibbon === r.id;
              const isNodeRelated = hoveredNode === r.sourceId || hoveredNode === r.targetId;
              const opacity = isHovered ? 0.82 : isNodeRelated ? 0.65 : hoveredRibbon || hoveredNode ? 0.12 : 0.32;

              return (
                <path
                  key={r.id}
                  d={r.path}
                  fill={r.color}
                  fillOpacity={opacity}
                  stroke={isHovered ? r.color : 'none'}
                  strokeWidth={isHovered ? 1.5 : 0}
                  onMouseEnter={() => setHoveredRibbon(r.id)}
                  onMouseLeave={() => setHoveredRibbon(null)}
                  style={{ cursor: 'pointer', transition: 'fill-opacity 0.2s ease' }}
                >
                  <title>{r.label}</title>
                </path>
              );
            })}
          </g>

          {/* 3. Column 1 Nodes: Platforms */}
          <g>
            {platformLayout.map((p) => {
              const isHovered = hoveredNode === p.id;
              return (
                <g
                  key={p.id}
                  onMouseEnter={() => setHoveredNode(p.id)}
                  onMouseLeave={() => setHoveredNode(null)}
                  style={{ cursor: 'pointer' }}
                >
                  <rect
                    x={p.x}
                    y={p.y}
                    width={p.width}
                    height={p.height}
                    rx="6"
                    fill={p.bg}
                    stroke={p.color}
                    strokeWidth={isHovered ? 2.5 : 1.5}
                    style={{ transition: 'all 0.15s ease' }}
                  />
                  <text
                    x={p.x + p.width / 2}
                    y={p.y + p.height / 2 - 4}
                    textAnchor="middle"
                    fontFamily="sans-serif"
                    fontSize="11"
                    fontWeight="700"
                    fill={p.color}
                  >
                    {p.label}
                  </text>
                  <text
                    x={p.x + p.width / 2}
                    y={p.y + p.height / 2 + 10}
                    textAnchor="middle"
                    fontFamily="sans-serif"
                    fontSize="10"
                    fontWeight="500"
                    fill={p.color}
                    opacity="0.9"
                  >
                    {p.count} posts
                  </text>
                </g>
              );
            })}
          </g>

          {/* 4. Column 2 Nodes: Topics (Light Yellow background + Dark Green / Pine theme borders) */}
          <g>
            {topicLayout.map((t) => {
              const isHovered = hoveredNode === `topic-${t.topicId}`;
              const pct = Math.round((t.count / totalRecords) * 100);
              return (
                <g
                  key={t.topicId}
                  onMouseEnter={() => setHoveredNode(`topic-${t.topicId}`)}
                  onMouseLeave={() => setHoveredNode(null)}
                  style={{ cursor: 'pointer' }}
                >
                  <title>{`${t.fullTitle} (${t.count} posts, ${pct}%)`}</title>
                  <rect
                    x={t.x}
                    y={t.y}
                    width={t.width}
                    height={t.height}
                    rx="5"
                    fill="#FEFCE8"
                    stroke={t.color}
                    strokeWidth={isHovered ? 2.5 : 1.5}
                    style={{ transition: 'all 0.15s ease' }}
                  />
                  {/* Left Accent Indicator Bar */}
                  <rect
                    x={t.x}
                    y={t.y}
                    width="4"
                    height={t.height}
                    rx="2"
                    fill={t.color}
                  />
                  <text
                    x={t.x + 10}
                    y={t.y + t.height / 2 + 3.5}
                    fontFamily="sans-serif"
                    fontSize="10.5"
                    fontWeight="700"
                    fill="#064E3B"
                  >
                    {t.cleanLabel}
                  </text>
                  <text
                    x={t.x + t.width - 8}
                    y={t.y + t.height / 2 + 3.5}
                    textAnchor="end"
                    fontFamily="sans-serif"
                    fontSize="10"
                    fontWeight="600"
                    fill={t.color}
                  >
                    {t.count} ({pct}%)
                  </text>
                </g>
              );
            })}
          </g>

          {/* 5. Column 3 Nodes: Sentiments */}
          <g>
            {sentimentLayout.map((s) => {
              const isHovered = hoveredNode === s.id;
              const pct = Math.round((s.count / totalRecords) * 100);
              const isCompact = s.height < 26;

              return (
                <g
                  key={s.id}
                  onMouseEnter={() => setHoveredNode(s.id)}
                  onMouseLeave={() => setHoveredNode(null)}
                  style={{ cursor: 'pointer' }}
                >
                  <title>{`${s.label} Sentiment: ${s.count} posts (${pct}%)`}</title>
                  <rect
                    x={s.x}
                    y={s.y}
                    width={s.width}
                    height={s.height}
                    rx="6"
                    fill={s.bg}
                    stroke={s.color}
                    strokeWidth={isHovered ? 2.5 : 1.5}
                    style={{ transition: 'all 0.15s ease' }}
                  />
                  {isCompact ? (
                    <text
                      x={s.x + s.width / 2}
                      y={s.y + s.height / 2 + 3.5}
                      textAnchor="middle"
                      fontFamily="sans-serif"
                      fontSize="9.5"
                      fontWeight="700"
                      fill={s.color}
                    >
                      {s.label}: {s.count} ({pct}%)
                    </text>
                  ) : (
                    <>
                      <text
                        x={s.x + s.width / 2}
                        y={s.y + s.height / 2 - 4}
                        textAnchor="middle"
                        fontFamily="sans-serif"
                        fontSize="11"
                        fontWeight="700"
                        fill={s.color}
                      >
                        {s.label}
                      </text>
                      <text
                        x={s.x + s.width / 2}
                        y={s.y + s.height / 2 + 10}
                        textAnchor="middle"
                        fontFamily="sans-serif"
                        fontSize="10"
                        fontWeight="600"
                        fill={s.color}
                        opacity="0.9"
                      >
                        {s.count} ({pct}%)
                      </text>
                    </>
                  )}
                </g>
              );
            })}
          </g>
        </svg>
      </div>

      {/* Interactive Helper Legend */}
      <div
        style={{
          marginTop: '10px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '11px',
          color: '#4B6358',
          borderTop: '1px solid #E2E8D0',
          paddingTop: '8px',
        }}
      >
        <div>
          Hover over any channel, topic, or sentiment block to inspect connected flow volumes.
        </div>
        <div style={{ display: 'flex', gap: '14px' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#EA580C', display: 'inline-block' }} />
            Reddit Inflow
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#DC2626', display: 'inline-block' }} />
            YouTube Inflow
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#16A34A', display: 'inline-block' }} />
            Positive
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#064E3B', display: 'inline-block' }} />
            Neutral
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#DC2626', display: 'inline-block' }} />
            Negative
          </span>
        </div>
      </div>
    </div>
  );
};
