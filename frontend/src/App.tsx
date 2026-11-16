import React, { useState, useEffect, useRef, useCallback } from 'react';
import type {
  DatasetSummary,
  ValidatedDatasetData,
  ModernCanonicalRecord,
  ValidatedModernConsolidated,
} from './types/dataset';
import {
  fetchDatasets,
  fetchDatasetDetails,
  fetchArtifact,
} from './api/client';
import { DatasetRequestCoordinator } from './api/datasetLoader';
import { OverviewDashboard } from './components/dashboards/OverviewDashboard';
import { SocialFeedExplorer } from './components/dashboards/SocialFeedExplorer';
import { SentimentDashboard } from './components/dashboards/SentimentDashboard';
import { TopicDashboard } from './components/dashboards/TopicDashboard';
import { MisinformationDashboard } from './components/dashboards/MisinformationDashboard';
import { NetworkDashboard } from './components/dashboards/NetworkDashboard';
import { DataSummary } from './components/DataSummary';

export type DashboardTab =
  | 'overview'
  | 'feed'
  | 'sentiment'
  | 'topics'
  | 'misinformation'
  | 'network';

const VALID_TABS: DashboardTab[] = ['overview', 'feed', 'sentiment', 'topics', 'misinformation', 'network'];

function getInitialTab(): DashboardTab {
  if (typeof window === 'undefined') return 'overview';
  const hash = window.location.hash.replace('#', '') as DashboardTab;
  if (VALID_TABS.includes(hash)) return hash;
  const params = new URLSearchParams(window.location.search);
  const tabParam = params.get('tab') as DashboardTab;
  if (VALID_TABS.includes(tabParam)) return tabParam;
  return 'overview';
}

export const App: React.FC = () => {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [datasetData, setDatasetData] = useState<ValidatedDatasetData | null>(null);

  const [_isLoadingDatasets, setIsLoadingDatasets] = useState<boolean>(true);
  const [isLoadingData, setIsLoadingData] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);

  // Active analytical dashboard tab
  const [activeTab, setActiveTabState] = useState<DashboardTab>(getInitialTab);

  const setActiveTab = useCallback((tab: DashboardTab) => {
    setActiveTabState(tab);
    if (typeof window !== 'undefined') {
      window.location.hash = tab;
    }
  }, []);

  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#', '') as DashboardTab;
      if (VALID_TABS.includes(hash)) {
        setActiveTabState(hash);
      }
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  const coordinatorRef = useRef<DatasetRequestCoordinator>(new DatasetRequestCoordinator());

  // Discover datasets from bridge
  const loadDatasets = useCallback(async (selectLatest = false) => {
    setIsLoadingDatasets(true);
    setError(null);

    try {
      const response = await fetchDatasets();
      setDatasets(response.datasets);

      // Auto-select first modern dataset if available, otherwise first dataset
      if (response.datasets.length > 0) {
        setSelectedDatasetId((prev) => {
          if (!selectLatest && prev && response.datasets.some((d) => d.id === prev)) {
            return prev;
          }
          const modern = response.datasets.find((d) => d.type === 'modern');
          return modern ? modern.id : response.datasets[0].id;
        });
      } else {
        setSelectedDatasetId(null);
        setDatasetData(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      setDatasets([]);
      setSelectedDatasetId(null);
      setDatasetData(null);
    } finally {
      setIsLoadingDatasets(false);
    }
  }, []);

  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  // Load and validate artifacts for the currently selected dataset
  useEffect(() => {
    if (!selectedDatasetId) {
      setDatasetData(null);
      return;
    }

    setIsLoadingData(true);
    setError(null);

    const loadData = async () => {
      try {
        const summary = await fetchDatasetDetails(selectedDatasetId);
        const data = await coordinatorRef.current.switchDataset(summary, fetchArtifact);
        if (data) {
          setDatasetData(data);
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          return;
        }
        setError(err instanceof Error ? err : new Error(String(err)));
        setDatasetData(null);
      } finally {
        setIsLoadingData(false);
      }
    };

    loadData();

    const coordinator = coordinatorRef.current;
    return () => {
      coordinator.abortActive();
    };
  }, [selectedDatasetId]);

  const selectedDataset = datasets.find((d) => d.id === selectedDatasetId) || null;

  // Extract modern records when available
  const isModern =
    datasetData?.datasetType === 'modern' &&
    datasetData.consolidated !== null &&
    'schema_version' in datasetData.consolidated &&
    datasetData.consolidated.schema_version === '2.0';

  const modernPayload = isModern
    ? (datasetData.consolidated as ValidatedModernConsolidated)
    : null;

  const modernRecords: ModernCanonicalRecord[] = modernPayload?.records || [];

  return (
    <div className="app-container">
      {/* 1. Executive Dark Green Header Banner */}
      <header
        style={{
          backgroundColor: '#064E3B',
          borderRadius: '8px',
          padding: '16px 24px',
          marginBottom: '20px',
          boxShadow: '0 2px 8px rgba(6, 78, 59, 0.15)',
        }}
      >
        <h1 style={{ fontSize: '20px', fontWeight: 700, color: '#FEFCE8', margin: 0, letterSpacing: '-0.01em' }}>
          Social Media Intelligence Platform
        </h1>
      </header>

      {/* Loading & Error States */}
      {isLoadingData && (
        <div className="summary-status-card">
          <p className="status-message">Loading intelligence dataset and analytical models...</p>
        </div>
      )}

      {error && !isLoadingData && (
        <div className="error-card">
          <div className="error-title">Data Retrieval Error</div>
          <div className="error-message">{error.message}</div>
        </div>
      )}

      {/* Modern Analytical Dashboards */}
      {!isLoadingData && !error && datasetData && isModern && (
        <div>
          {/* Primary Analytical Navigation */}
          <div className="dashboard-nav" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === 'overview'}
              className={`nav-tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
              onClick={() => setActiveTab('overview')}
            >
              Executive Overview
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === 'feed'}
              className={`nav-tab-btn ${activeTab === 'feed' ? 'active' : ''}`}
              onClick={() => setActiveTab('feed')}
            >
              Comment Feed ({modernRecords.length})
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === 'sentiment'}
              className={`nav-tab-btn ${activeTab === 'sentiment' ? 'active' : ''}`}
              onClick={() => setActiveTab('sentiment')}
            >
              Sentiment Intelligence
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === 'topics'}
              className={`nav-tab-btn ${activeTab === 'topics' ? 'active' : ''}`}
              onClick={() => setActiveTab('topics')}
            >
              Topic Intelligence
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === 'misinformation'}
              className={`nav-tab-btn ${activeTab === 'misinformation' ? 'active' : ''}`}
              onClick={() => setActiveTab('misinformation')}
            >
              Misinformation Risk
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === 'network'}
              className={`nav-tab-btn ${activeTab === 'network' ? 'active' : ''}`}
              onClick={() => setActiveTab('network')}
            >
              Network Intelligence
            </button>
          </div>

          {/* Active Scoped Dashboard Views */}
          {activeTab === 'overview' && (
            <OverviewDashboard
              consolidated={datasetData.consolidated}
              sentimentArtifact={datasetData.sentiment}
              topicArtifact={datasetData.topic}
              graphArtifact={datasetData.graph}
              misinformationArtifact={datasetData.misinformation}
              filteredRecords={modernRecords}
              summary={selectedDataset}
              platformFilter="all"
              onNavigateToTab={(tab) => setActiveTab(tab)}
              onNavigateToFeed={() => setActiveTab('feed')}
            />
          )}

          {activeTab === 'feed' && (
            <SocialFeedExplorer
              records={modernRecords}
              platformFilter="all"
              stages={selectedDataset?.stages}
            />
          )}

          {activeTab === 'sentiment' && (
            <SentimentDashboard
              filteredRecords={modernRecords}
              artifact={datasetData.sentiment}
              platformFilter="all"
              stages={selectedDataset?.stages}
            />
          )}

          {activeTab === 'topics' && (
            <TopicDashboard
              topicArtifact={datasetData.topic}
              filteredRecords={modernRecords}
              summary={selectedDataset}
              platformFilter="all"
            />
          )}

          {activeTab === 'misinformation' && (
            <MisinformationDashboard
              filteredRecords={modernRecords}
              artifact={datasetData.misinformation}
              platformFilter="all"
              stages={selectedDataset?.stages}
            />
          )}

          {activeTab === 'network' && (
            <NetworkDashboard
              graphArtifact={datasetData.graph}
              platformFilter="all"
              stages={selectedDataset?.stages}
            />
          )}
        </div>
      )}

      {/* Legacy Baseline Viewer (Strictly Isolated) */}
      {!isLoadingData && !error && datasetData && !isModern && (
        <div>
          <div className="card-panel" style={{ marginBottom: '20px', backgroundColor: '#F8FAFC' }}>
            <div className="panel-title">Historical Baseline Dataset</div>
            <div className="text-muted" style={{ fontSize: '13px' }}>
              This legacy dataset precedes the Schema-v2 modern analytical pipeline.
              Raw consolidated baseline records are displayed below without Schema-v2 analytical derivations.
            </div>
          </div>
          <DataSummary
            data={datasetData}
            isLoading={false}
            error={null}
          />
        </div>
      )}
    </div>
  );
};

export default App;
