/**
 * Data contracts and TypeScript type definitions for Social Media Intelligence Platform.
 * Authoritative source: src/pipeline/consolidator.py, src/sentiment/, src/misinformation/, src/graph/, src/topics/
 * Enforces strict separation between modern execution runs (Schema-v2) and legacy historical baseline.
 */

export type DatasetType = 'modern' | 'legacy';
export type PlatformType = 'reddit' | 'youtube';
export type ConsolidationStatus = 'complete' | 'degraded' | 'partial' | 'unprocessed';
export type IngestionSourceStatus = 'loaded' | 'empty' | 'missing' | 'corrupted';

export type StageExecutionStatus =
  | 'success'
  | 'failed'
  | 'partial'
  | 'skipped'
  | 'empty'
  | 'available'
  | 'unavailable'
  | string;

export interface DatasetSummary {
  id: string;
  type: DatasetType;
  label: string;
  generated_at: string | null;
  status: string;
  stages: Record<string, StageExecutionStatus>;
  available_artifacts: string[];
}

export interface DatasetListResponse {
  datasets: DatasetSummary[];
  total_datasets: number;
}

// Modern Consolidated Intelligence (Schema-v2)
// Authoritative Source: src/pipeline/consolidator.py

export type SentimentStatus =
  | 'success'
  | 'fallback'
  | 'empty_text'
  | 'error'
  | 'unavailable'
  | 'corrupted'
  | 'missing_status';

export type SentimentLabel = 'positive' | 'neutral' | 'negative';

export type TopicStatus =
  | 'assigned'
  | 'outlier'
  | 'fallback'
  | 'empty_text'
  | 'error'
  | 'unavailable'
  | 'corrupted'
  | 'missing_status';

export type MisinformationStatus =
  | 'success'
  | 'fallback'
  | 'empty_text'
  | 'error'
  | 'unavailable'
  | 'corrupted'
  | 'missing_status';

export type MisinformationVeracityLabel = 'real' | 'fake';
export type MisinformationRiskTier = 'low' | 'moderate' | 'high_risk';

export type AuthorGraphStatus =
  | 'ranked_influencer'
  | 'not_in_top_influencers'
  | 'anonymous'
  | 'unavailable'
  | 'corrupted';

// Subfield 1: Timestamps
export interface RecordTimestamps {
  published_at: string | null;
  collected_at: string | null;
  processed_at: string | null;
}

// Subfield 2: Author
export interface RecordAuthor {
  username: string;
  channel_id: string | null;
  channel_title: string | null;
  graph_status: AuthorGraphStatus;
  influencer_score: number | null;
  pagerank: number | null;
  community_id: number | null;
  in_degree: number | null;
  out_degree: number | null;
}

// Subfield 3: Content
export interface RecordContent {
  title_raw: string;
  body_raw: string;
  title_cleaned: string;
  body_cleaned: string;
  detected_language: string;
  language_code: string;
  preprocessing_status: string;
}

// Subfield 4: Engagement
export interface RecordEngagement {
  score: number | null;
  views_count: number | null;
  likes_count: number | null;
  comments_count: number | null;
}

// Subfield 5: Platform Metadata
export interface RecordRedditMetadata {
  subreddit: string | null;
  score: number | null;
}

export interface RecordYoutubeMetadata {
  channel_id: string | null;
  channel_title: string | null;
}

export interface RecordMetadata {
  reddit: RecordRedditMetadata | null;
  youtube: RecordYoutubeMetadata | null;
}

// Subfield 6: Sentiment
export interface RecordSentiment {
  label: SentimentLabel | null;
  score: number | null; // finite float in [-1.0, 1.0] when status == 'success', null otherwise
  status: SentimentStatus;
}

// Subfield 7: Topics
export interface TopicKeywordItem {
  word: string;
  weight: number | null;
}

export interface RecordTopics {
  topic_id: number | null; // non-negative int if assigned, -1 if outlier, null otherwise
  probability: number | null; // finite float in [0.0, 1.0] or null
  keywords: TopicKeywordItem[];
  status: TopicStatus;
}

// Subfield 8: Misinformation
export interface RecordMisinformation {
  veracity_label: MisinformationVeracityLabel | null;
  credibility_score: number | null; // finite float in [0.0, 1.0] when status == 'success', null otherwise
  risk_tier: MisinformationRiskTier | null;
  status: MisinformationStatus;
  stylometric_signals: Record<string, unknown> | null;
}

// Subfield 9: Consolidation
export interface RecordConsolidation {
  status: ConsolidationStatus;
  is_complete: boolean; // strict boolean
  sources_joined: string[];
}

// Canonical Record Root (13 exact canonical root fields)
export interface ModernCanonicalRecord {
  document_id: string; // strictly matches `${platform}_${native_id}`
  platform: PlatformType;
  native_id: string;
  url: string | null;
  timestamps: RecordTimestamps;
  author: RecordAuthor;
  content: RecordContent;
  engagement: RecordEngagement;
  metadata: RecordMetadata;
  sentiment: RecordSentiment;
  topics: RecordTopics;
  misinformation: RecordMisinformation;
  consolidation: RecordConsolidation;
}

// Top-Level Envelope Provenance
export interface ProvenanceDetails {
  reddit_input_file: string;
  reddit_status: IngestionSourceStatus;
  reddit_records_count: number;
  youtube_input_file: string;
  youtube_status: IngestionSourceStatus;
  youtube_records_count: number;
  [key: string]: unknown;
}

// Top-Level Envelope Metrics
export interface ModernConsolidatedMetrics {
  total_records_consolidated: number;
  platform_counts: {
    reddit: number;
    youtube: number;
  };
  complete_records: number;
  degraded_records: number;
  partial_records: number;
  unprocessed_records: number;
  duplicates_deduplicated: number;
  rejected_records_count: number;
  [key: string]: unknown;
}

// Validated Modern Consolidated Envelope (Schema-v2)
export interface ValidatedModernConsolidated {
  schema_version: '2.0';
  pipeline_version: string;
  status: 'success';
  consolidation_status: ConsolidationStatus;
  generated_at: string;
  provenance: ProvenanceDetails;
  metrics: ModernConsolidatedMetrics;
  audit_warnings: unknown[];
  rejected_records: unknown[];
  records: ModernCanonicalRecord[];
}

// Legacy Consolidated Intelligence (Pre-Schema-v2)

export interface ValidatedLegacyConsolidated {
  pipeline_version?: string;
  status?: string;
  records: unknown[];
  [key: string]: unknown;
}

// Analytical Artifacts Data Contracts
// Authoritative Sources: src/sentiment/, src/misinformation/, src/graph/, src/topics/

export interface SentimentWindowSummary {
  total: number;
  positive: number;
  neutral: number;
  negative: number;
  undetermined: number;
  positive_pct: number;
  neutral_pct: number;
  negative_pct: number;
  undetermined_pct: number;
  net_sentiment: number; // in [-1.0, 1.0]
}

export interface SentimentAnalysisArtifact {
  status: string;
  generated_at: string;
  model_name?: string;
  total_records_processed: number;
  summary: SentimentWindowSummary;
  reddit_summary?: SentimentWindowSummary;
  youtube_summary?: SentimentWindowSummary;
  temporal_aggregations?: Record<string, unknown>;
  reddit_records?: unknown[];
  youtube_records?: unknown[];
  [key: string]: unknown;
}

export interface MisinformationAnalysisArtifact {
  schema_version?: string;
  status: string;
  generated_at: string;
  total_records_processed: number;
  total_analyzed: number;
  average_credibility_score: number; // in [0.0, 1.0]
  flagged_high_risk_count: number;
  risk_tier_distribution?: Record<string, number>;
  veracity_distribution?: Record<string, number>;
  platform_breakdown?: Record<string, unknown>;
  predictions?: unknown[];
  reddit_records?: unknown[];
  youtube_records?: unknown[];
  [key: string]: unknown;
}

export interface GraphMetrics {
  total_users: number;
  total_interactions: number;
  density: number; // in [0.0, 1.0]
  communities_count: number;
  is_directed: boolean;
}

export interface GraphAnalysisArtifact {
  schema_version?: string;
  status: string;
  generated_at: string;
  nodes?: unknown[];
  edges?: unknown[];
  graph_metrics: GraphMetrics;
  community_distributions?: Record<string, unknown>;
  top_influencers?: unknown[];
  [key: string]: unknown;
}

export interface TopicInfoItem {
  topic_id: number;
  count?: number;
  name?: string;
  keywords?: unknown[];
  [key: string]: unknown;
}

export interface TopicModelingArtifact {
  status: string;
  generated_at: string;
  embedding_model?: string;
  total_documents_modeled: number;
  outlier_count?: number;
  outlier_pct?: number;
  topic_info: TopicInfoItem[];
  similarity_matrix?: unknown;
  trajectories?: unknown;
  cross_platform_divergence?: unknown;
  temporal_trends?: unknown;
  reddit_records?: unknown[];
  youtube_records?: unknown[];
  [key: string]: unknown;
}

// UI State & Validated Dataset Data Model

export interface ArtifactAvailability {
  name: string;
  isAvailable: boolean;
  statusText: string;
  reason?: string;
}

export interface ValidatedDatasetData {
  datasetId: string;
  datasetType: DatasetType;
  consolidated: ValidatedModernConsolidated | ValidatedLegacyConsolidated | null;
  sentiment: SentimentAnalysisArtifact | null;
  misinformation: MisinformationAnalysisArtifact | null;
  graph: GraphAnalysisArtifact | null;
  topic: TopicModelingArtifact | null;
  topicUnavailableReason?: string;
  retrievedAt: string;
}

export interface BridgeApiError {
  error: string;
  message: string;
  status: number;
}
