import React from 'react';

export type MetricScopeType =
  | 'envelope'
  | 'filtered_records'
  | 'artifact'
  | 'ratio';

interface MetricCardProps {
  label: string;
  value: string | number | null;
  subtext?: string;
  scope?: MetricScopeType;
  unavailableText?: string;
  highlight?: 'normal' | 'low_risk' | 'moderate_risk' | 'high_risk';
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subtext,
  scope,
  unavailableText = 'Unavailable',
  highlight = 'normal',
}) => {
  const isAvailable = value !== null && value !== undefined;

  let scopeLabel = '';
  if (scope === 'envelope') {
    scopeLabel = 'Scope: Total Run Envelope';
  } else if (scope === 'filtered_records') {
    scopeLabel = 'Scope: Filtered Records';
  } else if (scope === 'artifact') {
    scopeLabel = 'Scope: Complete Artifact Snapshot';
  }

  let highlightClass = '';
  if (highlight === 'low_risk') highlightClass = 'metric-highlight-low';
  else if (highlight === 'moderate_risk') highlightClass = 'metric-highlight-mod';
  else if (highlight === 'high_risk') highlightClass = 'metric-highlight-high';

  return (
    <div className={`metric-card ${highlightClass}`}>
      <div className="metric-header">
        <span className="metric-label">{label}</span>
        {scopeLabel && <span className="metric-scope-badge">{scopeLabel}</span>}
      </div>
      <div className="metric-value-container">
        {isAvailable ? (
          <span className="metric-value">{value}</span>
        ) : (
          <span className="metric-value-unavailable">{unavailableText}</span>
        )}
      </div>
      {subtext && <div className="metric-subtext">{subtext}</div>}
    </div>
  );
};

