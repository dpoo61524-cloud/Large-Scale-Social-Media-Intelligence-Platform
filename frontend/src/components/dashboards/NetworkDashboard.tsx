import React, { useState, useMemo } from 'react';
import type { GraphAnalysisArtifact } from '../../types/dataset';
import {
  derivePlatformGraphMetrics,
  getPlatformEmptyState,
  type PlatformFilter,
} from '../../utils/analytics';
import { MetricCard } from '../common/MetricCard';
import { PlatformEmptyState } from '../common/PlatformEmptyState';
import { NetworkGraphCanvas } from '../network/NetworkGraphCanvas';

interface NetworkDashboardProps {
  graphArtifact: GraphAnalysisArtifact | null;
  platformFilter?: PlatformFilter;
  stages?: Record<string, unknown> | null;
  onResetFilter?: (filter: PlatformFilter) => void;
}

export const NetworkDashboard: React.FC<NetworkDashboardProps> = ({
  graphArtifact,
  platformFilter = 'all',
  stages,
  onResetFilter,
}) => {
  const metrics = derivePlatformGraphMetrics(graphArtifact, platformFilter);
  const isZeroEdge = metrics.totalInteractions === 0;

  // Influencer Leaderboard state
  const [infPlatform, setInfPlatform] = useState<'all' | 'reddit' | 'youtube'>('all');
  const [infSearch, setInfSearch] = useState('');
  const [showAllInfluencers, setShowAllInfluencers] = useState(false);

  // Filtered & Searched Influencer Accounts (excludes raw email/domain references for data privacy)
  const filteredInfluencers = useMemo(() => {
    return metrics.influencers.filter((inf) => {
      // Exclude raw emails and domain references to ensure privacy and avoid data leakage
      const isDomainOrEmail = /\.(com|org|net|id|io)$/i.test(inf.username) || inf.username.includes('@');
      if (isDomainOrEmail) return false;

      if (infPlatform !== 'all' && inf.platform !== infPlatform) return false;
      if (infSearch.trim() !== '') {
        const q = infSearch.toLowerCase().trim();
        const matchesUser = inf.username.toLowerCase().includes(q);
        if (!matchesUser) return false;
      }
      return true;
    });
  }, [metrics.influencers, infPlatform, infSearch]);

  const displayedInfluencers = showAllInfluencers
    ? filteredInfluencers
    : filteredInfluencers.slice(0, 10);

  // If the active platform has zero nodes, render honest empty state
  if (metrics.totalUsers === 0) {
    const emptyState = getPlatformEmptyState(platformFilter, 0, stages);
    return (
      <div className="dashboard-section">
        <div className="section-header">
          <h2 className="section-title">Network & Influencer Intelligence</h2>
        </div>
        <PlatformEmptyState info={emptyState} onResetFilter={onResetFilter} />
      </div>
    );
  }

  return (
    <div className="dashboard-section">
      <div className="section-header">
        <h2 className="section-title">Network & Influencer Intelligence</h2>
      </div>

      {/* Connectivity Metrics */}
      <div className="metrics-grid">
        <MetricCard
          label="Monitored Accounts"
          value={metrics.totalUsers}
        />
        <MetricCard
          label="Directed Interactions"
          value={metrics.totalInteractions}
        />
        <MetricCard
          label="Network Connectivity"
          value={metrics.density}
        />
        <MetricCard
          label="Community Groups"
          value={metrics.communitiesCount}
        />
      </div>

      {/* Interactive Visual Network Topology Canvas */}
      <div style={{ marginTop: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
          <div className="panel-title" style={{ margin: 0 }}>Interactive Topology Graph</div>
          <span style={{ fontSize: '12px', color: '#64748B' }}>
            {metrics.totalUsers} nodes • {metrics.totalInteractions} edges • Directed Network
          </span>
        </div>
        <NetworkGraphCanvas
          nodes={metrics.nodes}
          edges={metrics.edges}
          platformFilter={platformFilter}
        />
      </div>

      {/* Zero Edge Diagnostic Notice (if isolated) */}
      {isZeroEdge && (
        <div className="card-panel" style={{ marginTop: '24px', backgroundColor: '#FEF9C3', borderLeft: '4px solid #064E3B' }}>
          <div style={{ fontSize: '13px', fontWeight: 600, color: '#064E3B', marginBottom: '6px' }}>
            Topological State: Disconnected Component Structure ({metrics.totalUsers} Singletons, 0 Edges)
          </div>
          <div style={{ fontSize: '13px', color: '#0F2E22', lineHeight: '1.5' }}>
            The active network contains <strong>{metrics.totalUsers} nodes</strong> and <strong>0 interaction edges</strong>.
            All extracted accounts represent independent post authors without observed user-to-user mentions.
          </div>
        </div>
      )}

      {/* Top Ranked Influencer Accounts (PageRank & Centrality) */}
      {metrics.influencers.length > 0 && (
        <div className="card-panel" style={{ marginTop: '24px', padding: '20px 24px' }}>
          {/* Header & Controls */}
          <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: '16px', marginBottom: '18px' }}>
            <div>
              <div className="panel-title" style={{ margin: 0 }}>
                Top Influencer Accounts
              </div>
              <div style={{ fontSize: '12px', color: '#64748B', marginTop: '4px' }}>
                Ranked by PageRank and directed interaction centrality
              </div>
            </div>

            <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: '8px' }}>
              <button
                className={`tab-pill-btn ${infPlatform === 'all' ? 'active' : ''}`}
                onClick={() => setInfPlatform('all')}
              >
                All Accounts ({filteredInfluencers.length})
              </button>
              <button
                className={`tab-pill-btn ${infPlatform === 'reddit' ? 'active' : ''}`}
                onClick={() => setInfPlatform('reddit')}
              >
                Reddit ({filteredInfluencers.filter((i) => i.platform === 'reddit').length})
              </button>
              <button
                className={`tab-pill-btn ${infPlatform === 'youtube' ? 'active' : ''}`}
                onClick={() => setInfPlatform('youtube')}
              >
                YouTube ({filteredInfluencers.filter((i) => i.platform === 'youtube').length})
              </button>

              <input
                type="text"
                value={infSearch}
                onChange={(e) => setInfSearch(e.target.value)}
                placeholder="Filter accounts..."
                style={{
                  padding: '6px 12px',
                  fontSize: '12px',
                  borderRadius: '6px',
                  border: '1px solid var(--border-color)',
                  outline: 'none',
                  width: '160px',
                }}
              />
            </div>
          </div>

          {/* Leaderboard Table */}
          <div className="modern-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ width: '60px', textAlign: 'center' }}>Rank</th>
                  <th>Influencer Account</th>
                  <th style={{ width: '110px' }}>Platform</th>
                  <th style={{ width: '180px' }}>Influence Authority</th>
                  <th style={{ width: '130px' }}>PageRank</th>
                  <th style={{ width: '180px' }}>Directed Reach</th>
                </tr>
              </thead>
              <tbody>
                {displayedInfluencers.length === 0 ? (
                  <tr>
                    <td colSpan={6} style={{ textAlign: 'center', padding: '24px', color: '#64748B' }}>
                      No matching influencer accounts found.
                    </td>
                  </tr>
                ) : (
                  displayedInfluencers.map((inf, idx) => {
                    const rank = idx + 1;
                    const scorePct = inf.influencerScore !== null ? Math.round(inf.influencerScore * 100) : 0;
                    const isReddit = inf.platform === 'reddit';
                    const displayName = isReddit ? `u/${inf.username}` : inf.username;

                    let rankClass = 'rank-default';
                    if (rank === 1) rankClass = 'rank-1';
                    else if (rank === 2) rankClass = 'rank-2';
                    else if (rank === 3) rankClass = 'rank-3';

                    return (
                      <tr key={inf.nodeId}>
                        {/* Rank */}
                        <td style={{ textAlign: 'center' }}>
                          <span className={`rank-badge ${rankClass}`}>
                            #{rank}
                          </span>
                        </td>

                        {/* Influencer Account Name */}
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <div
                              style={{
                                width: '28px',
                                height: '28px',
                                borderRadius: '50%',
                                backgroundColor: isReddit ? '#FFF7ED' : '#FEF2F2',
                                color: isReddit ? '#EA580C' : '#DC2626',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: '12px',
                                fontWeight: 700,
                              }}
                            >
                              {inf.username.charAt(0).toUpperCase()}
                            </div>
                            <div style={{ fontWeight: 600, color: '#1E293B', fontSize: '13px' }}>{displayName}</div>
                          </div>
                        </td>

                        {/* Platform: Only font color red for YouTube, orange for Reddit. No background, no border. */}
                        <td>
                          <span
                            style={{
                              color: isReddit ? '#EA580C' : '#DC2626',
                              fontWeight: 600,
                              fontSize: '12px',
                              textTransform: 'capitalize',
                              background: 'transparent',
                              border: 'none',
                              padding: 0,
                            }}
                          >
                            {inf.platform}
                          </span>
                        </td>

                        {/* Influencer Authority Meter */}
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <div className="score-bar-bg">
                              <div className="score-bar-fill" style={{ width: `${scorePct}%` }} />
                            </div>
                            <span style={{ fontWeight: 600, fontSize: '12px', color: '#1E293B' }}>
                              {inf.influencerScore !== null ? `${scorePct}%` : 'N/A'}
                            </span>
                          </div>
                        </td>

                        {/* PageRank */}
                        <td>
                          <span style={{ fontFamily: 'monospace', fontSize: '12px', color: '#334155' }}>
                            {inf.pagerank !== null ? inf.pagerank.toFixed(4) : 'N/A'}
                          </span>
                        </td>

                        {/* Directed Reach: Only font color green for incoming. No background, no border. */}
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <span
                              style={{
                                fontSize: '12px',
                                fontWeight: 600,
                                color: '#16A34A',
                                background: 'transparent',
                                border: 'none',
                                padding: 0,
                              }}
                              title="Incoming interactions & mentions"
                            >
                              ↓ {inf.inDegree} in
                            </span>
                            <span
                              style={{
                                fontSize: '12px',
                                fontWeight: 500,
                                color: '#64748B',
                                background: 'transparent',
                                border: 'none',
                                padding: 0,
                              }}
                              title="Outgoing interactions"
                            >
                              ↑ {inf.outDegree} out
                            </span>
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          {/* Footer Toggle (Top 10 vs All) */}
          {filteredInfluencers.length > 10 && (
            <div style={{ marginTop: '14px', display: 'flex', justifyContent: 'center' }}>
              <button
                className="tab-pill-btn"
                onClick={() => setShowAllInfluencers(!showAllInfluencers)}
                style={{ fontSize: '12px', padding: '6px 16px' }}
              >
                {showAllInfluencers
                  ? 'Show Top 10 Influencers'
                  : `View All ${filteredInfluencers.length} Ranked Accounts`}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
