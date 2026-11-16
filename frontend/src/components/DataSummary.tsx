import React from 'react';
import type {
  ValidatedDatasetData,
  ValidatedModernConsolidated,
  ValidatedLegacyConsolidated,
} from '../types/dataset';
import {
  formatMetricNumber,
  formatMetricDecimal,
} from '../validation/validators';

interface DataSummaryProps {
  data: ValidatedDatasetData | null;
  isLoading: boolean;
  error: Error | null;
}

export const DataSummary: React.FC<DataSummaryProps> = ({
  data,
  isLoading,
  error,
}) => {
  if (isLoading) {
    return (
      <div className="summary-status-card">
        <span className="status-message">Retrieving and validating dataset artifacts from bridge...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="summary-status-card error-card">
        <div className="error-title">Data Retrieval Error</div>
        <div className="error-message">{error.message}</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="summary-status-card">
        <span className="status-message">Select a dataset to inspect its validated analytical intelligence.</span>
      </div>
    );
  }

  const isModern = data.datasetType === 'modern';
  const consolidated = data.consolidated;

  // Extract consolidated counts safely
  let totalConsolidatedRecords: number | null = null;
  let schemaVersionDisplay = 'Historical (Pre-Schema-v2)';
  let platformCounts: Record<string, number> = {};

  if (isModern && consolidated && 'schema_version' in consolidated) {
    const modern = consolidated as ValidatedModernConsolidated;
    schemaVersionDisplay = 'Schema-v2 (2.0)';
    totalConsolidatedRecords = modern.metrics.total_records_consolidated;
    if (modern.metrics.platform_counts) {
      platformCounts = modern.metrics.platform_counts;
    }
  } else if (!isModern && consolidated && 'records' in consolidated) {
    const legacy = consolidated as ValidatedLegacyConsolidated;
    schemaVersionDisplay = 'Historical Baseline';
    totalConsolidatedRecords = legacy.records.length;
  }

  // Sentiment metrics (Source: src/sentiment/)
  const sentiment = data.sentiment;
  const sentimentEvaluated =
    sentiment?.total_records_processed ?? sentiment?.summary?.total ?? null;
  const sentimentMean = sentiment?.summary?.net_sentiment ?? null;

  // Misinformation metrics (Source: src/misinformation/)
  const misinfo = data.misinformation;
  const misinfoEvaluated = misinfo?.total_records_processed ?? null;
  const misinfoValidScores = misinfo?.total_analyzed ?? null;
  const misinfoMeanCredibility = misinfo?.average_credibility_score ?? null;

  // Graph metrics (Source: src/graph/)
  const graph = data.graph;
  const graphUsers = graph?.graph_metrics?.total_users ?? null;
  const graphInteractions = graph?.graph_metrics?.total_interactions ?? null;

  // Topic metrics (Source: src/topics/)
  const topic = data.topic;
  const topicCount = topic?.topic_info?.length ?? null;

  return (
    <div className="data-summary-section">
      <div className="section-header">
        <h2 className="section-title">Validated Data Summary</h2>
        <span className="section-subtitle monospace">
          Active Dataset: {data.datasetId}
        </span>
      </div>

      <div className="metrics-grid">
        {/* Consolidated Intelligence KPI */}
        <div className="metric-card">
          <div className="metric-header">Consolidated Intelligence</div>
          <div className="metric-value">
            {formatMetricNumber(totalConsolidatedRecords)}
          </div>
          <div className="metric-subtext">
            Total records consolidated ({schemaVersionDisplay})
          </div>
          {Object.keys(platformCounts).length > 0 && (
            <div className="platform-breakdown">
              {Object.entries(platformCounts).map(([platform, count]) => (
                <span key={platform} className="platform-tag">
                  {platform}: {count}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Sentiment Analysis KPI */}
        <div className="metric-card">
          <div className="metric-header">Sentiment Analysis</div>
          {sentiment ? (
            <>
              <div className="metric-value">
                {formatMetricDecimal(sentimentMean, 3)}
              </div>
              <div className="metric-subtext">
                Mean sentiment score ({formatMetricNumber(sentimentEvaluated)} records evaluated)
              </div>
            </>
          ) : (
            <div className="metric-unavailable">
              <span className="unavailable-title">Unavailable</span>
              <span className="unavailable-subtext">Artifact not available in dataset</span>
            </div>
          )}
        </div>

        {/* Misinformation Analysis KPI */}
        <div className="metric-card">
          <div className="metric-header">Misinformation & Credibility</div>
          {misinfo ? (
            <>
              <div className="metric-value">
                {formatMetricDecimal(misinfoMeanCredibility, 3)}
              </div>
              <div className="metric-subtext">
                Mean credibility index
              </div>
              <div className="denominator-detail">
                Valid scores: {formatMetricNumber(misinfoValidScores)} / Evaluated:{' '}
                {formatMetricNumber(misinfoEvaluated)}
              </div>
            </>
          ) : (
            <div className="metric-unavailable">
              <span className="unavailable-title">Unavailable</span>
              <span className="unavailable-subtext">Artifact not available in dataset</span>
            </div>
          )}
        </div>

        {/* Graph Network Analysis KPI */}
        <div className="metric-card">
          <div className="metric-header">Graph Network Analysis</div>
          {graph ? (
            <>
              <div className="metric-value">
                {formatMetricNumber(graphUsers)}
              </div>
              <div className="metric-subtext">
                Total network entities ({formatMetricNumber(graphInteractions)} interactions)
              </div>
            </>
          ) : (
            <div className="metric-unavailable">
              <span className="unavailable-title">Unavailable</span>
              <span className="unavailable-subtext">Artifact not available in dataset</span>
            </div>
          )}
        </div>

        {/* Topic Modeling KPI */}
        <div className="metric-card">
          <div className="metric-header">Topic Modeling</div>
          {topic ? (
            <>
              <div className="metric-value">
                {formatMetricNumber(topicCount)}
              </div>
              <div className="metric-subtext">
                Extracted topics (Historical Baseline)
              </div>
            </>
          ) : (
            <div className="metric-unavailable">
              <span className="unavailable-title">Unavailable</span>
              <span className="unavailable-subtext">
                {data.topicUnavailableReason || 'Stage failed or artifact omitted; no fallback to legacy data'}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
