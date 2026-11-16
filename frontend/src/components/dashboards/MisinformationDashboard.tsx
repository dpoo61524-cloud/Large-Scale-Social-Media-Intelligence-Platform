import React from 'react';
import type { ModernCanonicalRecord, MisinformationAnalysisArtifact } from '../../types/dataset';
import {
  calculateMisinformationMetrics,
  getPlatformEmptyState,
  type PlatformFilter,
} from '../../utils/analytics';
import { MetricCard } from '../common/MetricCard';
import { PlatformEmptyState } from '../common/PlatformEmptyState';
import { CredibilityThreatDial } from '../misinformation/CredibilityThreatDial';
import { ModelBenchmarkCard } from '../misinformation/ModelBenchmarkCard';
import { RiskTriageFeed } from '../misinformation/RiskTriageFeed';

interface MisinformationDashboardProps {
  filteredRecords: ModernCanonicalRecord[];
  artifact: MisinformationAnalysisArtifact | null;
  platformFilter?: PlatformFilter;
  stages?: Record<string, unknown> | null;
  onResetFilter?: (filter: PlatformFilter) => void;
}

export const MisinformationDashboard: React.FC<MisinformationDashboardProps> = ({
  filteredRecords,
  artifact: _artifact,
  platformFilter = 'all',
  stages,
  onResetFilter,
}) => {
  const emptyState = getPlatformEmptyState(platformFilter, filteredRecords.length, stages);

  if (emptyState.isEmpty) {
    return (
      <div className="dashboard-section">
        <div className="section-header">
          <h2 className="section-title">Misinformation Risk &amp; Threat Intelligence</h2>
        </div>
        <PlatformEmptyState info={emptyState} onResetFilter={onResetFilter} />
      </div>
    );
  }

  const metrics = calculateMisinformationMetrics(filteredRecords);
  const hasAssessed = metrics.assessedCount > 0;

  return (
    <div className="dashboard-section">
      <div className="section-header">
        <h2 className="section-title">Misinformation Risk &amp; Threat Intelligence</h2>
      </div>

      {/* Overview Metrics Cards */}
      <div className="metrics-grid" style={{ marginBottom: '16px' }}>
        <MetricCard
          label="Average Credibility Score"
          value={metrics.averageCredibilityScore}
        />
        <MetricCard
          label="Low Risk Volume"
          value={metrics.riskTiers.low}
          highlight="low_risk"
        />
        <MetricCard
          label="Moderate Risk Volume"
          value={metrics.riskTiers.moderate}
          highlight="moderate_risk"
        />
        <MetricCard
          label="High Risk Volume"
          value={metrics.riskTiers.highRisk}
          highlight="high_risk"
        />
      </div>

      {/* Layer 1: Executive Credibility Threat Dial & Post Risk Horizon Spectrum */}
      <CredibilityThreatDial
        records={filteredRecords}
        averageScore={metrics.averageCredibilityScore}
        assessedCount={metrics.assessedCount}
      />

      {/* Layer 2: Nexvia Step 6 Supervised Classifier Validation & Confusion Matrix */}
      <ModelBenchmarkCard />

      {/* Content Risk Tier Breakdown Bar */}
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
        <div style={{ fontSize: '14px', fontWeight: 700, color: '#064E3B', marginBottom: '8px' }}>
          Aggregate Corpus Risk Tier Distribution
        </div>
        {hasAssessed ? (
          <div>
            <div className="distribution-bar-container">
              <div
                className="dist-segment dist-low-risk"
                style={{ width: `${Math.round((metrics.riskTiers.low / metrics.assessedCount) * 100)}%` }}
                title={`Low Risk: ${metrics.riskTiers.low}`}
              />
              <div
                className="dist-segment dist-mod-risk"
                style={{ width: `${Math.round((metrics.riskTiers.moderate / metrics.assessedCount) * 100)}%` }}
                title={`Moderate: ${metrics.riskTiers.moderate}`}
              />
              <div
                className="dist-segment dist-high-risk"
                style={{ width: `${Math.round((metrics.riskTiers.highRisk / metrics.assessedCount) * 100)}%` }}
                title={`High Risk: ${metrics.riskTiers.highRisk}`}
              />
            </div>
            <div className="distribution-legend">
              <div className="legend-item">
                <span className="legend-color dist-low-risk" />
                <span>Low Risk: <strong>{metrics.riskTiers.low} ({Math.round((metrics.riskTiers.low / metrics.assessedCount) * 100)}%)</strong></span>
              </div>
              <div className="legend-item">
                <span className="legend-color dist-mod-risk" />
                <span>Moderate Risk: <strong>{metrics.riskTiers.moderate} ({Math.round((metrics.riskTiers.moderate / metrics.assessedCount) * 100)}%)</strong></span>
              </div>
              <div className="legend-item">
                <span className="legend-color dist-high-risk" />
                <span>High Risk: <strong>{metrics.riskTiers.highRisk} (0%)</strong></span>
              </div>
            </div>
          </div>
        ) : (
          <div className="empty-panel-notice">
            No assessed records with valid credibility scores found in active selection.
          </div>
        )}
      </div>

      {/* Layer 3: Interactive Flagged Content Triage & Audit Watchlist */}
      <RiskTriageFeed records={filteredRecords} />
    </div>
  );
};
