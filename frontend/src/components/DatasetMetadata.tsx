import React from 'react';
import type { DatasetSummary } from '../types/dataset';
import { StatusBadge } from './StatusBadge';

interface DatasetMetadataProps {
  dataset: DatasetSummary;
}

const STANDARD_ARTIFACTS = [
  'consolidated_intelligence.json',
  'sentiment_analysis_results.json',
  'misinformation_analysis_results.json',
  'graph_analysis_results.json',
  'topic_modeling_results.json',
];

export const DatasetMetadata: React.FC<DatasetMetadataProps> = ({ dataset }) => {
  const isModern = dataset.type === 'modern';
  const stageEntries = Object.entries(dataset.stages);

  return (
    <div className="metadata-container">
      {/* Identity Overview */}
      <div className="metadata-card identity-card">
        <div className="card-header">
          <span className="card-title">Dataset Identity</span>
          <StatusBadge status={dataset.status} />
        </div>

        <div className="identity-grid">
          <div className="identity-item">
            <span className="item-label">Identifier</span>
            <span className="item-value monospace">{dataset.id}</span>
          </div>

          <div className="identity-item">
            <span className="item-label">Type</span>
            <span className="item-value">
              {isModern ? 'Modern Execution Run (Schema-v2)' : 'Historical Baseline (Pre-Schema-v2)'}
            </span>
          </div>

          <div className="identity-item">
            <span className="item-label">Generated UTC</span>
            <span className="item-value">
              {dataset.generated_at ? dataset.generated_at : 'Not recorded'}
            </span>
          </div>

          <div className="identity-item">
            <span className="item-label">Available Artifacts</span>
            <span className="item-value">
              {dataset.available_artifacts.length} of {STANDARD_ARTIFACTS.length}
            </span>
          </div>
        </div>
      </div>

      {/* Artifact Inventory */}
      <div className="metadata-card">
        <div className="card-header">
          <span className="card-title">Artifact Status Inventory</span>
        </div>
        <div className="artifact-list">
          {STANDARD_ARTIFACTS.map((artifactName) => {
            const isAvailable = dataset.available_artifacts.includes(artifactName);
            const isTopic = artifactName === 'topic_modeling_results.json';
            const topicsFailed = isModern && isTopic && !isAvailable;

            return (
              <div key={artifactName} className="artifact-item">
                <div className="artifact-name-group">
                  <span className="artifact-filename monospace">{artifactName}</span>
                  {topicsFailed && (
                    <span className="artifact-subtext">
                      Pipeline stage failed; artifact not produced
                    </span>
                  )}
                </div>
                <span
                  className={`artifact-status-pill ${
                    isAvailable ? 'pill-available' : 'pill-unavailable'
                  }`}
                >
                  {isAvailable ? 'Available' : 'Unavailable'}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Pipeline Stage Breakdown (if available) */}
      {stageEntries.length > 0 && (
        <div className="metadata-card">
          <div className="card-header">
            <span className="card-title">Pipeline Stages Status</span>
          </div>
          <div className="stages-grid">
            {stageEntries.map(([stageName, stageStatus]) => (
              <div key={stageName} className="stage-item">
                <span className="stage-name">{stageName}</span>
                <StatusBadge status={String(stageStatus)} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

