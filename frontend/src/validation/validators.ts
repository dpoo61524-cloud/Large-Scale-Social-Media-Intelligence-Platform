/**
 * Runtime data validation functions for Social Media Intelligence Platform.
 * Authoritative Source: src/pipeline/consolidator.py, src/sentiment/, src/misinformation/, src/graph/, src/topics/
 * Enforces strict boundary verification and dataset isolation.
 */

import type {
  DatasetListResponse,
  DatasetSummary,
  DatasetType,
  ModernCanonicalRecord,
  ModernConsolidatedMetrics,
  PlatformType,
  ProvenanceDetails,
  RecordAuthor,
  AuthorGraphStatus,
  RecordConsolidation,
  RecordContent,
  RecordEngagement,
  RecordMetadata,
  RecordMisinformation,
  RecordSentiment,
  RecordTimestamps,
  RecordTopics,
  SentimentAnalysisArtifact,
  SentimentLabel,
  SentimentStatus,
  SentimentWindowSummary,
  TopicKeywordItem,
  TopicStatus,
  MisinformationStatus,
  MisinformationVeracityLabel,
  MisinformationRiskTier,
  ConsolidationStatus,
  IngestionSourceStatus,
  ValidatedModernConsolidated,
  ValidatedLegacyConsolidated,
  MisinformationAnalysisArtifact,
  GraphAnalysisArtifact,
  GraphMetrics,
  TopicModelingArtifact,
  TopicInfoItem,
} from '../types/dataset';

export class ValidationError extends Error {
  public readonly code: string;
  constructor(code: string, message: string) {
    super(`[${code}] ${message}`);
    this.name = 'ValidationError';
    this.code = code;
  }
}

// Primitive and Invariant Type Helpers

// Strict ISO-8601 UTC timestamp pattern matching Python datetime.isoformat() output (+00:00 with 0 or exactly 6 fractional digits)
// Uses (?![\s\S])$ as an absolute end-of-string assertion (JavaScript \z equivalent) preventing matching before any terminal line terminator (\n, \r)
const STRICT_UTC_ISO_PATTERN =
  /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{6})?\+00:00(?![\s\S])$/;

export function isValidIsoTimestamp(val: unknown): val is string {
  if (typeof val !== 'string') return false;
  // Absolute length check: exactly 25 chars (without fractional digits) or 32 chars (with exactly 6 fractional digits)
  if (val.length !== 25 && val.length !== 32) return false;
  const match = STRICT_UTC_ISO_PATTERN.exec(val);
  if (!match || match[0] !== val) return false;

  const year = parseInt(match[1], 10);
  const month = parseInt(match[2], 10);
  const day = parseInt(match[3], 10);
  const hour = parseInt(match[4], 10);
  const min = parseInt(match[5], 10);
  const sec = parseInt(match[6], 10);

  // Reject year zero and out-of-range time values
  if (year <= 0) return false;
  if (month < 1 || month > 12) return false;
  if (hour < 0 || hour > 23) return false;
  if (min < 0 || min > 59) return false;
  if (sec < 0 || sec > 59) return false;

  // Gregorian calendar days per month with leap-year rule
  // Avoids JavaScript's Date.UTC() year 0-99 mapping quirk
  const isLeapYear = (year % 4 === 0 && year % 100 !== 0) || (year % 400 === 0);
  let maxDays = 31;
  if (month === 2) {
    maxDays = isLeapYear ? 29 : 28;
  } else if (month === 4 || month === 6 || month === 9 || month === 11) {
    maxDays = 30;
  }

  if (day < 1 || day > maxDays) return false;

  return true;
}

export function isStrictBool(val: unknown): val is boolean {
  return typeof val === 'boolean';
}

export function isStrictInt(val: unknown): val is number {
  return typeof val === 'number' && Number.isInteger(val) && !Number.isNaN(val);
}

export function isFiniteFloat(val: unknown): val is number {
  return typeof val === 'number' && Number.isFinite(val) && !Number.isNaN(val);
}

export function isNonEmptyString(val: unknown): val is string {
  return typeof val === 'string' && val.trim().length > 0;
}

export function safeFiniteNumber(
  val: unknown,
  options?: { min?: number; max?: number }
): number | null {
  if (typeof val !== 'number' || !Number.isFinite(val) || Number.isNaN(val)) {
    return null;
  }
  if (options?.min !== undefined && val < options.min) {
    return null;
  }
  if (options?.max !== undefined && val > options.max) {
    return null;
  }
  return val;
}

export function formatMetricNumber(
  val: number | null | undefined,
  fallback = 'Unavailable'
): string {
  if (val === null || val === undefined || !Number.isFinite(val) || Number.isNaN(val)) {
    return fallback;
  }
  return val.toLocaleString();
}

export function formatMetricDecimal(
  val: number | null | undefined,
  digits = 3,
  fallback = 'Unavailable'
): string {
  if (val === null || val === undefined || !Number.isFinite(val) || Number.isNaN(val)) {
    return fallback;
  }
  return val.toFixed(digits);
}

// Dataset Discovery Validation

export function validateDatasetListResponse(raw: unknown): DatasetListResponse {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new ValidationError('malformed_response', 'Dataset list response must be a JSON object.');
  }

  const obj = raw as Record<string, unknown>;
  if (!Array.isArray(obj.datasets)) {
    throw new ValidationError('missing_datasets_array', 'Response missing datasets array.');
  }

  const validatedDatasets: DatasetSummary[] = obj.datasets.map((item, idx) => {
    try {
      return validateDatasetSummary(item);
    } catch (err) {
      throw new ValidationError(
        'invalid_dataset_entry',
        `Dataset at index ${idx} failed validation: ${err instanceof Error ? err.message : String(err)}`
      );
    }
  });

  const totalCount = safeFiniteNumber(obj.total_datasets, { min: 0 }) ?? validatedDatasets.length;

  return {
    datasets: validatedDatasets,
    total_datasets: totalCount,
  };
}

export function validateDatasetSummary(raw: unknown): DatasetSummary {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new ValidationError('invalid_summary', 'Dataset summary must be an object.');
  }

  const obj = raw as Record<string, unknown>;

  if (!isNonEmptyString(obj.id)) {
    throw new ValidationError('missing_dataset_id', 'Dataset must contain a valid non-empty id.');
  }

  if (obj.type !== 'modern' && obj.type !== 'legacy') {
    throw new ValidationError('invalid_dataset_type', `Unknown dataset type: ${String(obj.type)}`);
  }

  const datasetType: DatasetType = obj.type;
  const label = isNonEmptyString(obj.label) ? obj.label : obj.id;
  const generated_at = typeof obj.generated_at === 'string' ? obj.generated_at : null;
  const status = typeof obj.status === 'string' ? obj.status : 'unknown';

  const stages: Record<string, string> = {};
  if (obj.stages && typeof obj.stages === 'object' && !Array.isArray(obj.stages)) {
    for (const [k, v] of Object.entries(obj.stages as Record<string, unknown>)) {
      if (typeof v === 'string') {
        stages[k] = v;
      }
    }
  }

  const available_artifacts: string[] = [];
  if (Array.isArray(obj.available_artifacts)) {
    for (const item of obj.available_artifacts) {
      if (typeof item === 'string') {
        available_artifacts.push(item);
      }
    }
  }

  return {
    id: obj.id,
    type: datasetType,
    label,
    generated_at,
    status,
    stages,
    available_artifacts,
  };
}

// Authoritative Schema-v2 Canonical Record Constants
// Source: src/pipeline/consolidator.py

const ALLOWED_PLATFORMS = new Set<string>(['reddit', 'youtube']);
const ALLOWED_CONSOLIDATION_STATUSES = new Set<string>([
  'complete',
  'degraded',
  'partial',
  'unprocessed',
]);
const ALLOWED_SOURCE_STATUSES = new Set<string>(['loaded', 'empty', 'missing', 'corrupted']);

const CANONICAL_RECORD_ROOT_FIELDS = new Set<string>([
  'document_id',
  'platform',
  'native_id',
  'url',
  'timestamps',
  'author',
  'content',
  'engagement',
  'metadata',
  'sentiment',
  'topics',
  'misinformation',
  'consolidation',
]);

const ALLOWED_SENTIMENT_STATUSES = new Set<string>([
  'success',
  'fallback',
  'empty_text',
  'error',
  'unavailable',
  'corrupted',
  'missing_status',
]);

const ALLOWED_SENTIMENT_LABELS = new Set<string>(['positive', 'neutral', 'negative']);

const ALLOWED_TOPIC_STATUSES = new Set<string>([
  'assigned',
  'outlier',
  'fallback',
  'empty_text',
  'error',
  'unavailable',
  'corrupted',
  'missing_status',
]);

const ALLOWED_MISINFO_STATUSES = new Set<string>([
  'success',
  'fallback',
  'empty_text',
  'error',
  'unavailable',
  'corrupted',
  'missing_status',
]);

const ALLOWED_VERACITY_LABELS = new Set<string>(['real', 'fake']);
const ALLOWED_RISK_TIERS = new Set<string>(['low', 'moderate', 'high_risk']);

// Deep Canonical Record Validator
// Source: src/pipeline/consolidator.py: ConsolidatedArtifactValidator._validate_records

export function validateCanonicalRecord(
  raw: unknown,
  index: number,
  seenDocIds: Set<string>
): ModernCanonicalRecord {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new ValidationError('invalid_record_type', `Record at index ${index} must be an object.`);
  }

  const rec = raw as Record<string, unknown>;
  const recKeys = Object.keys(rec);

  // 1. Exact 13 canonical root fields check (No missing fields, no extra fields)
  for (const field of CANONICAL_RECORD_ROOT_FIELDS) {
    if (!(field in rec)) {
      throw new ValidationError(
        'missing_canonical_field',
        `Record at index ${index} missing canonical field '${field}'.`
      );
    }
  }

  for (const key of recKeys) {
    if (!CANONICAL_RECORD_ROOT_FIELDS.has(key)) {
      throw new ValidationError(
        'forbidden_extra_field',
        `Record at index ${index} contains forbidden extra field '${key}' (envelope leakage prevention).`
      );
    }
  }

  // 2. Platform validation
  const platform = rec.platform;
  if (typeof platform !== 'string' || !ALLOWED_PLATFORMS.has(platform)) {
    throw new ValidationError(
      'invalid_platform',
      `Record at index ${index} has invalid platform '${String(platform)}'. Expected 'reddit' or 'youtube'.`
    );
  }
  const typedPlatform = platform as PlatformType;

  // 3. Native ID validation
  if (!isNonEmptyString(rec.native_id)) {
    throw new ValidationError(
      'missing_native_id',
      `Record at index ${index} has missing or empty 'native_id'.`
    );
  }
  const nativeId = rec.native_id;

  // 4. Document ID canonical construction and uniqueness
  const expectedDocId = `${typedPlatform}_${nativeId}`;
  if (rec.document_id !== expectedDocId) {
    throw new ValidationError(
      'invalid_document_id',
      `Record at index ${index} document_id '${String(rec.document_id)}' does not match expected '${expectedDocId}'.`
    );
  }
  const docId = rec.document_id as string;

  if (seenDocIds.has(docId)) {
    throw new ValidationError(
      'duplicate_document_id',
      `Duplicate document_id '${docId}' detected at index ${index}.`
    );
  }
  seenDocIds.add(docId);

  // 5. URL (string or null)
  if (rec.url !== null && typeof rec.url !== 'string') {
    throw new ValidationError('invalid_url', `Record '${docId}' field 'url' must be string or null.`);
  }
  const url = rec.url as string | null;

  // 6. Sub-dictionaries validation
  for (const subfield of [
    'timestamps',
    'author',
    'content',
    'engagement',
    'metadata',
    'sentiment',
    'topics',
    'misinformation',
    'consolidation',
  ]) {
    const val = rec[subfield];
    if (!val || typeof val !== 'object' || Array.isArray(val)) {
      throw new ValidationError(
        'invalid_subfield_dict',
        `Record '${docId}' field '${subfield}' must be a dictionary.`
      );
    }
  }

  // Timestamps (exact 3 keys, strict ISO-8601 UTC validation for non-null values)
  const tsObj = rec.timestamps as Record<string, unknown>;
  for (const f of ['published_at', 'collected_at', 'processed_at']) {
    if (!(f in tsObj)) {
      throw new ValidationError(
        'missing_timestamp_field',
        `Record '${docId}' timestamps missing '${f}'.`
      );
    }
    const val = tsObj[f];
    if (val !== null && !isValidIsoTimestamp(val)) {
      throw new ValidationError(
        'invalid_timestamp_format',
        `Record '${docId}' timestamps.${f} must be valid ISO-8601 UTC string or null, got ${String(val)}.`
      );
    }
  }
  const timestamps: RecordTimestamps = {
    published_at: tsObj.published_at as string | null,
    collected_at: tsObj.collected_at as string | null,
    processed_at: tsObj.processed_at as string | null,
  };

  // Author (exact 9 keys, strict platform rules, graph_status enums, numeric bounds)
  const authObj = rec.author as Record<string, unknown>;
  const ALLOWED_GRAPH_STATUSES = new Set<string>([
    'ranked_influencer',
    'not_in_top_influencers',
    'anonymous',
    'unavailable',
    'corrupted',
  ]);

  for (const f of [
    'username',
    'channel_id',
    'channel_title',
    'graph_status',
    'influencer_score',
    'pagerank',
    'community_id',
    'in_degree',
    'out_degree',
  ]) {
    if (!(f in authObj)) {
      throw new ValidationError('missing_author_field', `Record '${docId}' author missing '${f}'.`);
    }
  }

  if (typeof authObj.username !== 'string') {
    throw new ValidationError('invalid_author_username', `Record '${docId}' author.username must be a string.`);
  }

  const graphStatus = authObj.graph_status;
  if (typeof graphStatus !== 'string' || !ALLOWED_GRAPH_STATUSES.has(graphStatus)) {
    throw new ValidationError(
      'invalid_author_graph_status',
      `Record '${docId}' author.graph_status '${String(graphStatus)}' invalid.`
    );
  }

  if (typedPlatform === 'reddit') {
    if (authObj.channel_id !== null) {
      throw new ValidationError(
        'invalid_author_platform_invariant',
        `Record '${docId}' author.channel_id must be null for Reddit records, got ${String(authObj.channel_id)}.`
      );
    }
    if (authObj.channel_title !== null) {
      throw new ValidationError(
        'invalid_author_platform_invariant',
        `Record '${docId}' author.channel_title must be null for Reddit records, got ${String(authObj.channel_title)}.`
      );
    }
  } else {
    if (authObj.channel_id !== null && typeof authObj.channel_id !== 'string') {
      throw new ValidationError(
        'invalid_author_channel_id',
        `Record '${docId}' author.channel_id must be string or null, got ${typeof authObj.channel_id}.`
      );
    }
    if (authObj.channel_title !== null && typeof authObj.channel_title !== 'string') {
      throw new ValidationError(
        'invalid_author_channel_title',
        `Record '${docId}' author.channel_title must be string or null, got ${typeof authObj.channel_title}.`
      );
    }
  }

  if (authObj.influencer_score !== null && !isFiniteFloat(authObj.influencer_score)) {
    throw new ValidationError(
      'invalid_author_metric',
      `Record '${docId}' author.influencer_score must be finite float or null, got ${typeof authObj.influencer_score}.`
    );
  }
  if (authObj.pagerank !== null && !isFiniteFloat(authObj.pagerank)) {
    throw new ValidationError(
      'invalid_author_metric',
      `Record '${docId}' author.pagerank must be finite float or null, got ${typeof authObj.pagerank}.`
    );
  }
  if (authObj.community_id !== null && !isStrictInt(authObj.community_id)) {
    throw new ValidationError(
      'invalid_author_metric',
      `Record '${docId}' author.community_id must be integer or null, got ${typeof authObj.community_id}.`
    );
  }
  if (authObj.in_degree !== null && (!isStrictInt(authObj.in_degree) || authObj.in_degree < 0)) {
    throw new ValidationError(
      'invalid_author_metric',
      `Record '${docId}' author.in_degree must be non-negative integer or null, got ${String(authObj.in_degree)}.`
    );
  }
  if (authObj.out_degree !== null && (!isStrictInt(authObj.out_degree) || authObj.out_degree < 0)) {
    throw new ValidationError(
      'invalid_author_metric',
      `Record '${docId}' author.out_degree must be non-negative integer or null, got ${String(authObj.out_degree)}.`
    );
  }

  const author: RecordAuthor = {
    username: authObj.username,
    channel_id: authObj.channel_id as string | null,
    channel_title: authObj.channel_title as string | null,
    graph_status: graphStatus as AuthorGraphStatus,
    influencer_score: authObj.influencer_score as number | null,
    pagerank: authObj.pagerank as number | null,
    community_id: authObj.community_id as number | null,
    in_degree: authObj.in_degree as number | null,
    out_degree: authObj.out_degree as number | null,
  };

  // Content (strict validation: all 7 required properties must be strings; no silent defaulting)
  const contObj = rec.content as Record<string, unknown>;
  const REQUIRED_CONTENT_FIELDS = [
    'title_raw',
    'body_raw',
    'title_cleaned',
    'body_cleaned',
    'detected_language',
    'language_code',
    'preprocessing_status',
  ];
  for (const f of REQUIRED_CONTENT_FIELDS) {
    if (!(f in contObj)) {
      throw new ValidationError(
        'missing_content_property',
        `Record '${docId}' content missing required property '${f}'.`
      );
    }
    if (typeof contObj[f] !== 'string') {
      throw new ValidationError(
        'invalid_content_property',
        `Record '${docId}' content.${f} must be a string, got ${typeof contObj[f]}.`
      );
    }
  }

  const content: RecordContent = {
    title_raw: contObj.title_raw as string,
    body_raw: contObj.body_raw as string,
    title_cleaned: contObj.title_cleaned as string,
    body_cleaned: contObj.body_cleaned as string,
    detected_language: contObj.detected_language as string,
    language_code: contObj.language_code as string,
    preprocessing_status: contObj.preprocessing_status as string,
  };

  // Engagement (strict platform consistency invariants; no silent null coercion)
  const engObj = rec.engagement as Record<string, unknown>;
  for (const f of ['score', 'views_count', 'likes_count', 'comments_count']) {
    if (!(f in engObj)) {
      throw new ValidationError(
        'missing_engagement_field',
        `Record '${docId}' engagement missing '${f}'.`
      );
    }
  }

  if (typedPlatform === 'reddit') {
    if (engObj.views_count !== null) {
      throw new ValidationError(
        'invalid_engagement_invariant',
        `Record '${docId}' engagement.views_count must be null for Reddit, got ${String(engObj.views_count)}.`
      );
    }
    if (engObj.likes_count !== null) {
      throw new ValidationError(
        'invalid_engagement_invariant',
        `Record '${docId}' engagement.likes_count must be null for Reddit, got ${String(engObj.likes_count)}.`
      );
    }
    if (engObj.score !== null && !isFiniteFloat(engObj.score)) {
      throw new ValidationError(
        'invalid_engagement_metric',
        `Record '${docId}' engagement.score must be finite number or null, got ${typeof engObj.score}.`
      );
    }
    if (engObj.comments_count !== null && (!isStrictInt(engObj.comments_count) || engObj.comments_count < 0)) {
      throw new ValidationError(
        'invalid_engagement_metric',
        `Record '${docId}' engagement.comments_count must be non-negative integer or null, got ${String(engObj.comments_count)}.`
      );
    }
  } else {
    // YouTube
    if (engObj.score !== null) {
      throw new ValidationError(
        'invalid_engagement_invariant',
        `Record '${docId}' engagement.score must be null for YouTube, got ${String(engObj.score)}.`
      );
    }
    if (engObj.views_count !== null && (!isStrictInt(engObj.views_count) || engObj.views_count < 0)) {
      throw new ValidationError(
        'invalid_engagement_metric',
        `Record '${docId}' engagement.views_count must be non-negative integer or null, got ${String(engObj.views_count)}.`
      );
    }
    if (engObj.likes_count !== null && (!isStrictInt(engObj.likes_count) || engObj.likes_count < 0)) {
      throw new ValidationError(
        'invalid_engagement_metric',
        `Record '${docId}' engagement.likes_count must be non-negative integer or null, got ${String(engObj.likes_count)}.`
      );
    }
    if (engObj.comments_count !== null && (!isStrictInt(engObj.comments_count) || engObj.comments_count < 0)) {
      throw new ValidationError(
        'invalid_engagement_metric',
        `Record '${docId}' engagement.comments_count must be non-negative integer or null, got ${String(engObj.comments_count)}.`
      );
    }
  }

  const engagement: RecordEngagement = {
    score: engObj.score as number | null,
    views_count: engObj.views_count as number | null,
    likes_count: engObj.likes_count as number | null,
    comments_count: engObj.comments_count as number | null,
  };

  // Metadata (strict platform consistency invariants)
  const metaObj = rec.metadata as Record<string, unknown>;
  for (const f of ['reddit', 'youtube']) {
    if (!(f in metaObj)) {
      throw new ValidationError(
        'missing_metadata_field',
        `Record '${docId}' metadata missing '${f}'.`
      );
    }
  }

  let redditMeta: RecordMetadata['reddit'] = null;
  let youtubeMeta: RecordMetadata['youtube'] = null;

  if (typedPlatform === 'reddit') {
    if (metaObj.youtube !== null) {
      throw new ValidationError(
        'invalid_metadata_invariant',
        `Record '${docId}' metadata.youtube must be null for Reddit records, got ${String(metaObj.youtube)}.`
      );
    }
    if (!metaObj.reddit || typeof metaObj.reddit !== 'object' || Array.isArray(metaObj.reddit)) {
      throw new ValidationError(
        'invalid_metadata_object',
        `Record '${docId}' metadata.reddit must be an object for Reddit records.`
      );
    }
    const rMeta = metaObj.reddit as Record<string, unknown>;
    if (!('subreddit' in rMeta) || !('score' in rMeta)) {
      throw new ValidationError(
        'missing_reddit_metadata_properties',
        `Record '${docId}' metadata.reddit missing 'subreddit' or 'score'.`
      );
    }
    if (rMeta.subreddit !== null && typeof rMeta.subreddit !== 'string') {
      throw new ValidationError(
        'invalid_reddit_metadata_type',
        `Record '${docId}' metadata.reddit.subreddit must be string or null, got ${typeof rMeta.subreddit}.`
      );
    }
    if (rMeta.score !== null && !isFiniteFloat(rMeta.score)) {
      throw new ValidationError(
        'invalid_reddit_metadata_type',
        `Record '${docId}' metadata.reddit.score must be finite number or null, got ${typeof rMeta.score}.`
      );
    }
    redditMeta = {
      subreddit: rMeta.subreddit as string | null,
      score: rMeta.score as number | null,
    };
  } else {
    // YouTube
    if (metaObj.reddit !== null) {
      throw new ValidationError(
        'invalid_metadata_invariant',
        `Record '${docId}' metadata.reddit must be null for YouTube records, got ${String(metaObj.reddit)}.`
      );
    }
    if (!metaObj.youtube || typeof metaObj.youtube !== 'object' || Array.isArray(metaObj.youtube)) {
      throw new ValidationError(
        'invalid_metadata_object',
        `Record '${docId}' metadata.youtube must be an object for YouTube records.`
      );
    }
    const yMeta = metaObj.youtube as Record<string, unknown>;
    if (!('channel_id' in yMeta) || !('channel_title' in yMeta)) {
      throw new ValidationError(
        'missing_youtube_metadata_properties',
        `Record '${docId}' metadata.youtube missing 'channel_id' or 'channel_title'.`
      );
    }
    if (yMeta.channel_id !== null && typeof yMeta.channel_id !== 'string') {
      throw new ValidationError(
        'invalid_youtube_metadata_type',
        `Record '${docId}' metadata.youtube.channel_id must be string or null, got ${typeof yMeta.channel_id}.`
      );
    }
    if (yMeta.channel_title !== null && typeof yMeta.channel_title !== 'string') {
      throw new ValidationError(
        'invalid_youtube_metadata_type',
        `Record '${docId}' metadata.youtube.channel_title must be string or null, got ${typeof yMeta.channel_title}.`
      );
    }
    youtubeMeta = {
      channel_id: yMeta.channel_id as string | null,
      channel_title: yMeta.channel_title as string | null,
    };
  }

  const metadata: RecordMetadata = {
    reddit: redditMeta,
    youtube: youtubeMeta,
  };

  // Sentiment (with nullability and score bounds [-1.0, 1.0])
  const sentObj = rec.sentiment as Record<string, unknown>;
  const sentStatus = sentObj.status;
  if (typeof sentStatus !== 'string' || !ALLOWED_SENTIMENT_STATUSES.has(sentStatus)) {
    throw new ValidationError(
      'invalid_sentiment_status',
      `Record '${docId}' sentiment.status '${String(sentStatus)}' is invalid.`
    );
  }

  let sentLabel: SentimentLabel | null = null;
  let sentScore: number | null = null;

  if (sentStatus === 'success') {
    if (typeof sentObj.label !== 'string' || !ALLOWED_SENTIMENT_LABELS.has(sentObj.label)) {
      throw new ValidationError(
        'invalid_sentiment_label',
        `Record '${docId}' sentiment.label '${String(sentObj.label)}' invalid for success status.`
      );
    }
    sentLabel = sentObj.label as SentimentLabel;

    if (!isFiniteFloat(sentObj.score) || sentObj.score < -1.0 || sentObj.score > 1.0) {
      throw new ValidationError(
        'invalid_sentiment_score',
        `Record '${docId}' sentiment.score must be finite float in [-1.0, 1.0] for success status, got ${String(sentObj.score)}.`
      );
    }
    sentScore = sentObj.score;
  } else {
    if (sentObj.label !== null && sentObj.label !== undefined) {
      throw new ValidationError(
        'invalid_sentiment_nullability',
        `Record '${docId}' sentiment.label must be null when status is '${sentStatus}', got ${String(sentObj.label)}.`
      );
    }
    if (sentObj.score !== null && sentObj.score !== undefined) {
      throw new ValidationError(
        'invalid_sentiment_nullability',
        `Record '${docId}' sentiment.score must be null when status is '${sentStatus}', got ${String(sentObj.score)}.`
      );
    }
  }

  const sentiment: RecordSentiment = {
    status: sentStatus as SentimentStatus,
    label: sentLabel,
    score: sentScore,
  };

  // Topics (with status rules, outlier rules, probability in [0.0, 1.0], and keywords)
  const topObj = rec.topics as Record<string, unknown>;
  const topStatus = topObj.status;
  if (typeof topStatus !== 'string' || !ALLOWED_TOPIC_STATUSES.has(topStatus)) {
    throw new ValidationError(
      'invalid_topic_status',
      `Record '${docId}' topics.status '${String(topStatus)}' is invalid.`
    );
  }

  if (!Array.isArray(topObj.keywords)) {
    throw new ValidationError('invalid_topic_keywords', `Record '${docId}' topics.keywords must be a list.`);
  }

  let topId: number | null = null;
  let topProb: number | null = null;
  const keywords: TopicKeywordItem[] = [];

  if (topStatus === 'assigned') {
    if (!isStrictInt(topObj.topic_id) || topObj.topic_id < 0) {
      throw new ValidationError(
        'invalid_topic_id',
        `Record '${docId}' topics.topic_id must be non-negative integer for assigned status, got ${String(topObj.topic_id)}.`
      );
    }
    topId = topObj.topic_id;

    if (topObj.probability !== null && topObj.probability !== undefined) {
      if (!isFiniteFloat(topObj.probability) || topObj.probability < 0.0 || topObj.probability > 1.0) {
        throw new ValidationError(
          'invalid_topic_probability',
          `Record '${docId}' topics.probability must be finite float in [0.0, 1.0], got ${String(topObj.probability)}.`
        );
      }
      topProb = topObj.probability;
    }

    for (const [kidx, kw] of (topObj.keywords as unknown[]).entries()) {
      if (!kw || typeof kw !== 'object' || Array.isArray(kw)) {
        throw new ValidationError('invalid_keyword_entry', `Record '${docId}' topics.keywords[${kidx}] must be a dict.`);
      }
      const kwObj = kw as Record<string, unknown>;
      if (!isNonEmptyString(kwObj.word)) {
        throw new ValidationError('invalid_keyword_word', `Record '${docId}' topics.keywords[${kidx}] missing valid 'word'.`);
      }
      const weight = isFiniteFloat(kwObj.weight) ? kwObj.weight : null;
      keywords.push({ word: kwObj.word, weight });
    }
  } else if (topStatus === 'outlier') {
    if (topObj.topic_id !== -1) {
      throw new ValidationError(
        'invalid_topic_outlier_id',
        `Record '${docId}' topics.topic_id must equal -1 for outlier status, got ${String(topObj.topic_id)}.`
      );
    }
    topId = -1;

    if (topObj.probability !== null && topObj.probability !== undefined) {
      if (!isFiniteFloat(topObj.probability) || topObj.probability < 0.0 || topObj.probability > 1.0) {
        throw new ValidationError(
          'invalid_topic_probability',
          `Record '${docId}' topics.probability must be finite float in [0.0, 1.0], got ${String(topObj.probability)}.`
        );
      }
      topProb = topObj.probability;
    }

    if (topObj.keywords.length !== 0) {
      throw new ValidationError(
        'invalid_topic_outlier_keywords',
        `Record '${docId}' topics.keywords must be empty list for outlier status, got ${topObj.keywords.length} items.`
      );
    }
  } else {
    if (topObj.topic_id !== null && topObj.topic_id !== undefined) {
      throw new ValidationError(
        'invalid_topic_nullability',
        `Record '${docId}' topics.topic_id must be null when status is '${topStatus}', got ${String(topObj.topic_id)}.`
      );
    }
    if (topObj.probability !== null && topObj.probability !== undefined) {
      throw new ValidationError(
        'invalid_topic_nullability',
        `Record '${docId}' topics.probability must be null when status is '${topStatus}', got ${String(topObj.probability)}.`
      );
    }
    if (topObj.keywords.length !== 0) {
      throw new ValidationError(
        'invalid_topic_nullability',
        `Record '${docId}' topics.keywords must be empty list when status is '${topStatus}', got ${topObj.keywords.length} items.`
      );
    }
  }

  const topics: RecordTopics = {
    status: topStatus as TopicStatus,
    topic_id: topId,
    probability: topProb,
    keywords,
  };

  // Misinformation (with status rules, credibility in [0.0, 1.0], veracity, risk tier)
  const misObj = rec.misinformation as Record<string, unknown>;
  const misStatus = misObj.status;
  if (typeof misStatus !== 'string' || !ALLOWED_MISINFO_STATUSES.has(misStatus)) {
    throw new ValidationError(
      'invalid_misinfo_status',
      `Record '${docId}' misinformation.status '${String(misStatus)}' is invalid.`
    );
  }

  let veracityLabel: MisinformationVeracityLabel | null = null;
  let credScore: number | null = null;
  let riskTier: MisinformationRiskTier | null = null;
  let stylometrics: Record<string, unknown> | null = null;

  if (misStatus === 'success') {
    if (typeof misObj.veracity_label !== 'string' || !ALLOWED_VERACITY_LABELS.has(misObj.veracity_label)) {
      throw new ValidationError(
        'invalid_misinfo_veracity',
        `Record '${docId}' misinformation.veracity_label '${String(misObj.veracity_label)}' invalid for success status.`
      );
    }
    veracityLabel = misObj.veracity_label as MisinformationVeracityLabel;

    if (typeof misObj.risk_tier !== 'string' || !ALLOWED_RISK_TIERS.has(misObj.risk_tier)) {
      throw new ValidationError(
        'invalid_misinfo_risk_tier',
        `Record '${docId}' misinformation.risk_tier '${String(misObj.risk_tier)}' invalid for success status.`
      );
    }
    riskTier = misObj.risk_tier as MisinformationRiskTier;

    if (!isFiniteFloat(misObj.credibility_score) || misObj.credibility_score < 0.0 || misObj.credibility_score > 1.0) {
      throw new ValidationError(
        'invalid_misinfo_credibility_score',
        `Record '${docId}' misinformation.credibility_score must be finite float in [0.0, 1.0], got ${String(misObj.credibility_score)}.`
      );
    }
    credScore = misObj.credibility_score;

    if (misObj.stylometric_signals !== null && misObj.stylometric_signals !== undefined) {
      if (typeof misObj.stylometric_signals !== 'object' || Array.isArray(misObj.stylometric_signals)) {
        throw new ValidationError(
          'invalid_misinfo_signals',
          `Record '${docId}' misinformation.stylometric_signals must be a dictionary or null.`
        );
      }
      const rawSig = misObj.stylometric_signals as Record<string, unknown>;
      for (const [sKey, sVal] of Object.entries(rawSig)) {
        if (typeof sKey !== 'string') {
          throw new ValidationError(
            'invalid_stylometric_key',
            `Record '${docId}' stylometric_signals key must be string.`
          );
        }
        if (
          sVal !== null &&
          typeof sVal !== 'boolean' &&
          typeof sVal !== 'number' &&
          typeof sVal !== 'string' &&
          !Array.isArray(sVal)
        ) {
          throw new ValidationError(
            'invalid_stylometric_value',
            `Record '${docId}' stylometric_signals['${sKey}'] must be a primitive value, array, or null.`
          );
        }
      }
      stylometrics = rawSig;
    }
  } else {
    if (misObj.veracity_label !== null && misObj.veracity_label !== undefined) {
      throw new ValidationError(
        'invalid_misinfo_nullability',
        `Record '${docId}' misinformation.veracity_label must be null when status is '${misStatus}'.`
      );
    }
    if (misObj.risk_tier !== null && misObj.risk_tier !== undefined) {
      throw new ValidationError(
        'invalid_misinfo_nullability',
        `Record '${docId}' misinformation.risk_tier must be null when status is '${misStatus}'.`
      );
    }
    if (misObj.credibility_score !== null && misObj.credibility_score !== undefined) {
      throw new ValidationError(
        'invalid_misinfo_nullability',
        `Record '${docId}' misinformation.credibility_score must be null when status is '${misStatus}'.`
      );
    }
    if (misObj.stylometric_signals !== null && misObj.stylometric_signals !== undefined) {
      throw new ValidationError(
        'invalid_misinfo_nullability',
        `Record '${docId}' misinformation.stylometric_signals must be null when status is '${misStatus}'.`
      );
    }
  }

  const misinformation: RecordMisinformation = {
    status: misStatus as MisinformationStatus,
    veracity_label: veracityLabel,
    credibility_score: credScore,
    risk_tier: riskTier,
    stylometric_signals: stylometrics,
  };

  // Consolidation (status, strict is_complete boolean, sources_joined list)
  const consObj = rec.consolidation as Record<string, unknown>;
  const consStatus = consObj.status;
  if (typeof consStatus !== 'string' || !ALLOWED_CONSOLIDATION_STATUSES.has(consStatus)) {
    throw new ValidationError(
      'invalid_consolidation_status',
      `Record '${docId}' consolidation.status '${String(consStatus)}' invalid.`
    );
  }

  if (!isStrictBool(consObj.is_complete)) {
    throw new ValidationError(
      'invalid_consolidation_is_complete',
      `Record '${docId}' consolidation.is_complete must be a strict boolean, got ${String(consObj.is_complete)}.`
    );
  }

  if (!Array.isArray(consObj.sources_joined)) {
    throw new ValidationError(
      'invalid_sources_joined',
      `Record '${docId}' consolidation.sources_joined must be a list.`
    );
  }

  const sourcesJoined: string[] = [];
  for (const src of consObj.sources_joined) {
    if (typeof src === 'string') {
      sourcesJoined.push(src);
    }
  }

  const consolidation: RecordConsolidation = {
    status: consStatus as ConsolidationStatus,
    is_complete: consObj.is_complete,
    sources_joined: sourcesJoined,
  };

  return {
    document_id: docId,
    platform: typedPlatform,
    native_id: nativeId,
    url,
    timestamps,
    author,
    content,
    engagement,
    metadata,
    sentiment,
    topics,
    misinformation,
    consolidation,
  };
}

// Modern Consolidated Intelligence Envelope Validator (Schema-v2)
// Source: src/pipeline/consolidator.py

export function validateModernConsolidated(raw: unknown): ValidatedModernConsolidated {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new ValidationError('malformed_consolidated', 'Consolidated intelligence must be an object.');
  }

  const obj = raw as Record<string, unknown>;

  // 1. Schema version validation (strict '2.0')
  if (obj.schema_version !== '2.0') {
    throw new ValidationError(
      'schema_version_mismatch',
      `Expected Schema-v2 artifact (schema_version='2.0'), found '${String(obj.schema_version)}'.`
    );
  }

  // 2. Status validation (strict 'success')
  if (obj.status !== 'success') {
    throw new ValidationError(
      'invalid_status',
      `Invalid artifact status '${String(obj.status)}'; expected 'success'.`
    );
  }

  // 3. Pipeline version validation
  if (!isNonEmptyString(obj.pipeline_version)) {
    throw new ValidationError('missing_pipeline_version', 'Missing or invalid non-empty pipeline_version.');
  }

  // 4. Consolidation status validation
  if (
    typeof obj.consolidation_status !== 'string' ||
    !ALLOWED_CONSOLIDATION_STATUSES.has(obj.consolidation_status)
  ) {
    throw new ValidationError(
      'invalid_consolidation_status',
      `Invalid 'consolidation_status' '${String(obj.consolidation_status)}'.`
    );
  }

  // 5. Generated_at ISO timestamp validation
  if (!isNonEmptyString(obj.generated_at) || !isValidIsoTimestamp(obj.generated_at)) {
    throw new ValidationError('missing_generated_at', 'Missing or invalid ISO-8601 generated_at timestamp.');
  }

  // 6. Audit warnings & rejected records collections
  if (!Array.isArray(obj.audit_warnings)) {
    throw new ValidationError('missing_audit_warnings', "'audit_warnings' must be an array.");
  }
  if (!Array.isArray(obj.rejected_records)) {
    throw new ValidationError('missing_rejected_records', "'rejected_records' must be an array.");
  }

  // 7. Provenance Validation
  if (!obj.provenance || typeof obj.provenance !== 'object' || Array.isArray(obj.provenance)) {
    throw new ValidationError('missing_provenance', "'provenance' must be a dictionary.");
  }
  const prov = obj.provenance as Record<string, unknown>;

  for (const plat of ['reddit', 'youtube']) {
    const inFile = prov[`${plat}_input_file`];
    if (!isNonEmptyString(inFile)) {
      throw new ValidationError(
        'invalid_provenance_file',
        `provenance['${plat}_input_file'] must be a non-empty string.`
      );
    }

    const pStatus = prov[`${plat}_status`];
    if (typeof pStatus !== 'string' || !ALLOWED_SOURCE_STATUSES.has(pStatus)) {
      throw new ValidationError(
        'invalid_provenance_status',
        `provenance['${plat}_status'] '${String(pStatus)}' invalid.`
      );
    }

    const count = prov[`${plat}_records_count`];
    if (!isStrictInt(count) || count < 0) {
      throw new ValidationError(
        'invalid_provenance_count',
        `provenance['${plat}_records_count'] must be non-negative integer, got ${String(count)}.`
      );
    }

    if ((pStatus === 'missing' || pStatus === 'empty' || pStatus === 'corrupted') && count !== 0) {
      throw new ValidationError(
        'invalid_provenance_invariant',
        `provenance['${plat}_records_count'] must be 0 when status is '${pStatus}', got ${count}.`
      );
    }
  }

  const provenance: ProvenanceDetails = {
    reddit_input_file: String(prov.reddit_input_file),
    reddit_status: prov.reddit_status as IngestionSourceStatus,
    reddit_records_count: prov.reddit_records_count as number,
    youtube_input_file: String(prov.youtube_input_file),
    youtube_status: prov.youtube_status as IngestionSourceStatus,
    youtube_records_count: prov.youtube_records_count as number,
    ...prov,
  };

  // 8. Records collection validation
  if (!Array.isArray(obj.records)) {
    throw new ValidationError('missing_records', 'Consolidated artifact records must be an array.');
  }

  const seenDocIds = new Set<string>();
  const records: ModernCanonicalRecord[] = obj.records.map((r, idx) =>
    validateCanonicalRecord(r, idx, seenDocIds)
  );

  // 9. Metrics Validation & Sum Invariants
  if (!obj.metrics || typeof obj.metrics !== 'object' || Array.isArray(obj.metrics)) {
    throw new ValidationError('missing_metrics', 'Consolidated artifact missing metrics object.');
  }

  const rawMetrics = obj.metrics as Record<string, unknown>;

  const totalCount = rawMetrics.total_records_consolidated;
  if (!isStrictInt(totalCount) || totalCount < 0) {
    throw new ValidationError(
      'invalid_total_records',
      `metrics['total_records_consolidated'] must be non-negative integer, got ${String(totalCount)}.`
    );
  }

  if (totalCount !== records.length) {
    throw new ValidationError(
      'metric_records_mismatch',
      `metrics['total_records_consolidated'] (${totalCount}) does not match len(records) (${records.length}).`
    );
  }

  // Platform counts
  if (!rawMetrics.platform_counts || typeof rawMetrics.platform_counts !== 'object') {
    throw new ValidationError('missing_platform_counts', "metrics['platform_counts'] must be an object.");
  }
  const rawPlatCounts = rawMetrics.platform_counts as Record<string, unknown>;
  const redditCount = rawPlatCounts.reddit;
  const youtubeCount = rawPlatCounts.youtube;

  if (!isStrictInt(redditCount) || redditCount < 0) {
    throw new ValidationError(
      'invalid_platform_count',
      `metrics['platform_counts']['reddit'] must be non-negative integer, got ${String(redditCount)}.`
    );
  }
  if (!isStrictInt(youtubeCount) || youtubeCount < 0) {
    throw new ValidationError(
      'invalid_platform_count',
      `metrics['platform_counts']['youtube'] must be non-negative integer, got ${String(youtubeCount)}.`
    );
  }

  const actualReddit = records.filter((r) => r.platform === 'reddit').length;
  const actualYoutube = records.filter((r) => r.platform === 'youtube').length;

  if (redditCount !== actualReddit) {
    throw new ValidationError(
      'platform_count_mismatch',
      `metrics['platform_counts']['reddit'] (${redditCount}) != actual (${actualReddit}).`
    );
  }
  if (youtubeCount !== actualYoutube) {
    throw new ValidationError(
      'platform_count_mismatch',
      `metrics['platform_counts']['youtube'] (${youtubeCount}) != actual (${actualYoutube}).`
    );
  }

  if (redditCount + youtubeCount !== totalCount) {
    throw new ValidationError(
      'platform_sum_mismatch',
      `Sum of platform counts (${redditCount + youtubeCount}) != total (${totalCount}).`
    );
  }

  // Status counts
  for (const stField of [
    'complete_records',
    'degraded_records',
    'partial_records',
    'unprocessed_records',
    'duplicates_deduplicated',
    'rejected_records_count',
  ]) {
    const val = rawMetrics[stField];
    if (!isStrictInt(val) || val < 0) {
      throw new ValidationError(
        'invalid_metric_field',
        `metrics['${stField}'] must be non-negative integer, got ${String(val)}.`
      );
    }
  }

  const completeCnt = rawMetrics.complete_records as number;
  const degradedCnt = rawMetrics.degraded_records as number;
  const partialCnt = rawMetrics.partial_records as number;
  const unprocessedCnt = rawMetrics.unprocessed_records as number;
  const rejectedCnt = rawMetrics.rejected_records_count as number;

  if (rejectedCnt !== obj.rejected_records.length) {
    throw new ValidationError(
      'rejected_metric_mismatch',
      `metrics['rejected_records_count'] (${rejectedCnt}) != len(rejected_records) (${obj.rejected_records.length}).`
    );
  }

  const statusSum = completeCnt + degradedCnt + partialCnt + unprocessedCnt;
  if (statusSum !== totalCount) {
    throw new ValidationError(
      'status_sum_mismatch',
      `Sum of record status counts (${statusSum}) != total_records_consolidated (${totalCount}).`
    );
  }

  const actualComplete = records.filter((r) => r.consolidation.status === 'complete').length;
  const actualDegraded = records.filter((r) => r.consolidation.status === 'degraded').length;
  const actualPartial = records.filter((r) => r.consolidation.status === 'partial').length;
  const actualUnprocessed = records.filter((r) => r.consolidation.status === 'unprocessed').length;

  if (completeCnt !== actualComplete || degradedCnt !== actualDegraded || partialCnt !== actualPartial || unprocessedCnt !== actualUnprocessed) {
    throw new ValidationError(
      'record_status_count_mismatch',
      'Record consolidation status distribution does not match metric counters.'
    );
  }

  const metrics: ModernConsolidatedMetrics = {
    total_records_consolidated: totalCount,
    platform_counts: {
      reddit: redditCount,
      youtube: youtubeCount,
    },
    complete_records: completeCnt,
    degraded_records: degradedCnt,
    partial_records: partialCnt,
    unprocessed_records: unprocessedCnt,
    duplicates_deduplicated: rawMetrics.duplicates_deduplicated as number,
    rejected_records_count: rejectedCnt,
    ...rawMetrics,
  };

  return {
    schema_version: '2.0',
    pipeline_version: obj.pipeline_version as string,
    status: 'success',
    consolidation_status: obj.consolidation_status as ConsolidationStatus,
    generated_at: obj.generated_at as string,
    provenance,
    metrics,
    audit_warnings: obj.audit_warnings,
    rejected_records: obj.rejected_records,
    records,
  };
}

// Legacy Consolidated Intelligence Validator

export function validateLegacyConsolidated(raw: unknown): ValidatedLegacyConsolidated {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new ValidationError('malformed_legacy', 'Legacy consolidated data must be an object.');
  }

  const obj = raw as Record<string, unknown>;

  // Reject modern data attempting to enter the legacy parser
  if (obj.schema_version === '2.0') {
    throw new ValidationError(
      'modern_data_in_legacy_parser',
      "Modern Schema-v2 data cannot be parsed by legacy consolidated validator (schema_version='2.0' detected)."
    );
  }

  const records = Array.isArray(obj.records) ? obj.records : [];
  const pipeline_version = typeof obj.pipeline_version === 'string' ? obj.pipeline_version : undefined;
  const status = typeof obj.status === 'string' ? obj.status : undefined;

  return {
    pipeline_version,
    status,
    records,
    ...obj,
  };
}

// Real Structural Validators for Analytical Artifacts
// Sources: src/sentiment/, src/misinformation/, src/graph/, src/topics/

export function validateSentimentArtifact(raw: unknown): SentimentAnalysisArtifact {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new ValidationError('malformed_sentiment', 'Sentiment artifact must be an object.');
  }

  const obj = raw as Record<string, unknown>;

  if (!isNonEmptyString(obj.status)) {
    throw new ValidationError('missing_sentiment_status', "Sentiment artifact missing 'status'.");
  }
  if (!isNonEmptyString(obj.generated_at)) {
    throw new ValidationError('missing_sentiment_generated_at', "Sentiment artifact missing 'generated_at'.");
  }

  if (!isStrictInt(obj.total_records_processed) || obj.total_records_processed < 0) {
    throw new ValidationError(
      'invalid_sentiment_total',
      `'total_records_processed' must be non-negative integer, got ${String(obj.total_records_processed)}.`
    );
  }

  if (!obj.summary || typeof obj.summary !== 'object' || Array.isArray(obj.summary)) {
    throw new ValidationError('missing_sentiment_summary', "Sentiment artifact missing 'summary' object.");
  }
  const sum = obj.summary as Record<string, unknown>;

  for (const f of ['total', 'positive', 'neutral', 'negative', 'undetermined']) {
    if (!isStrictInt(sum[f]) || (sum[f] as number) < 0) {
      throw new ValidationError(
        'invalid_sentiment_summary_metric',
        `sentiment summary['${f}'] must be non-negative integer.`
      );
    }
  }

  const netSentiment = sum.net_sentiment;
  if (!isFiniteFloat(netSentiment) || netSentiment < -1.0 || netSentiment > 1.0) {
    throw new ValidationError(
      'invalid_net_sentiment',
      `sentiment summary['net_sentiment'] must be finite float in [-1.0, 1.0], got ${String(netSentiment)}.`
    );
  }

  const summary: SentimentWindowSummary = {
    total: sum.total as number,
    positive: sum.positive as number,
    neutral: sum.neutral as number,
    negative: sum.negative as number,
    undetermined: sum.undetermined as number,
    positive_pct: isFiniteFloat(sum.positive_pct) ? sum.positive_pct : 0,
    neutral_pct: isFiniteFloat(sum.neutral_pct) ? sum.neutral_pct : 0,
    negative_pct: isFiniteFloat(sum.negative_pct) ? sum.negative_pct : 0,
    undetermined_pct: isFiniteFloat(sum.undetermined_pct) ? sum.undetermined_pct : 0,
    net_sentiment: netSentiment,
  };

  return {
    status: obj.status,
    generated_at: obj.generated_at,
    total_records_processed: obj.total_records_processed,
    summary,
    ...obj,
  };
}

export function validateMisinformationArtifact(raw: unknown): MisinformationAnalysisArtifact {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new ValidationError('malformed_misinformation', 'Misinformation artifact must be an object.');
  }

  const obj = raw as Record<string, unknown>;

  if (!isNonEmptyString(obj.status)) {
    throw new ValidationError('missing_misinfo_status', "Misinformation artifact missing 'status'.");
  }
  if (!isNonEmptyString(obj.generated_at)) {
    throw new ValidationError('missing_misinfo_generated_at', "Misinformation artifact missing 'generated_at'.");
  }

  if (!isStrictInt(obj.total_records_processed) || obj.total_records_processed < 0) {
    throw new ValidationError(
      'invalid_misinfo_total_processed',
      `'total_records_processed' must be non-negative integer, got ${String(obj.total_records_processed)}.`
    );
  }

  if (!isStrictInt(obj.total_analyzed) || obj.total_analyzed < 0) {
    throw new ValidationError(
      'invalid_misinfo_total_analyzed',
      `'total_analyzed' must be non-negative integer, got ${String(obj.total_analyzed)}.`
    );
  }

  if (obj.total_analyzed > obj.total_records_processed) {
    throw new ValidationError(
      'misinfo_analyzed_exceeds_total',
      `'total_analyzed' (${obj.total_analyzed}) cannot exceed 'total_records_processed' (${obj.total_records_processed}).`
    );
  }

  if (
    !isFiniteFloat(obj.average_credibility_score) ||
    obj.average_credibility_score < 0.0 ||
    obj.average_credibility_score > 1.0
  ) {
    throw new ValidationError(
      'invalid_average_credibility_score',
      `'average_credibility_score' must be finite float in [0.0, 1.0], got ${String(obj.average_credibility_score)}.`
    );
  }

  if (!isStrictInt(obj.flagged_high_risk_count) || obj.flagged_high_risk_count < 0) {
    throw new ValidationError(
      'invalid_flagged_high_risk_count',
      `'flagged_high_risk_count' must be non-negative integer, got ${String(obj.flagged_high_risk_count)}.`
    );
  }

  return {
    status: obj.status,
    generated_at: obj.generated_at,
    total_records_processed: obj.total_records_processed,
    total_analyzed: obj.total_analyzed,
    average_credibility_score: obj.average_credibility_score,
    flagged_high_risk_count: obj.flagged_high_risk_count,
    ...obj,
  };
}

export function validateGraphArtifact(raw: unknown): GraphAnalysisArtifact {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new ValidationError('malformed_graph', 'Graph artifact must be an object.');
  }

  const obj = raw as Record<string, unknown>;

  if (!isNonEmptyString(obj.status)) {
    throw new ValidationError('missing_graph_status', "Graph artifact missing 'status'.");
  }
  if (!isNonEmptyString(obj.generated_at)) {
    throw new ValidationError('missing_graph_generated_at', "Graph artifact missing 'generated_at'.");
  }

  if (!obj.graph_metrics || typeof obj.graph_metrics !== 'object' || Array.isArray(obj.graph_metrics)) {
    throw new ValidationError('missing_graph_metrics', "Graph artifact missing 'graph_metrics' object.");
  }
  const gm = obj.graph_metrics as Record<string, unknown>;

  if (!isStrictInt(gm.total_users) || gm.total_users < 0) {
    throw new ValidationError('invalid_graph_total_users', "graph_metrics['total_users'] must be non-negative integer.");
  }
  if (!isStrictInt(gm.total_interactions) || gm.total_interactions < 0) {
    throw new ValidationError(
      'invalid_graph_total_interactions',
      "graph_metrics['total_interactions'] must be non-negative integer."
    );
  }
  if (!isFiniteFloat(gm.density) || gm.density < 0.0 || gm.density > 1.0) {
    throw new ValidationError(
      'invalid_graph_density',
      `graph_metrics['density'] must be finite float in [0.0, 1.0], got ${String(gm.density)}.`
    );
  }
  if (!isStrictInt(gm.communities_count) || gm.communities_count < 0) {
    throw new ValidationError(
      'invalid_graph_communities_count',
      "graph_metrics['communities_count'] must be non-negative integer."
    );
  }
  if (gm.is_directed !== true) {
    throw new ValidationError('invalid_graph_is_directed', "graph_metrics['is_directed'] must strictly equal true.");
  }

  const graph_metrics: GraphMetrics = {
    total_users: gm.total_users,
    total_interactions: gm.total_interactions,
    density: gm.density,
    communities_count: gm.communities_count,
    is_directed: true,
  };

  return {
    status: obj.status,
    generated_at: obj.generated_at,
    graph_metrics,
    ...obj,
  };
}

export function validateTopicArtifact(raw: unknown): TopicModelingArtifact {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new ValidationError('malformed_topic', 'Topic artifact must be an object.');
  }

  const obj = raw as Record<string, unknown>;

  if (!isNonEmptyString(obj.status)) {
    throw new ValidationError('missing_topic_status', "Topic artifact missing 'status'.");
  }
  if (!isNonEmptyString(obj.generated_at)) {
    throw new ValidationError('missing_topic_generated_at', "Topic artifact missing 'generated_at'.");
  }

  if (!isStrictInt(obj.total_documents_modeled) || obj.total_documents_modeled < 0) {
    throw new ValidationError(
      'invalid_topic_total_modeled',
      `'total_documents_modeled' must be non-negative integer, got ${String(obj.total_documents_modeled)}.`
    );
  }

  if (!Array.isArray(obj.topic_info)) {
    throw new ValidationError('missing_topic_info', "Topic artifact missing 'topic_info' array.");
  }

  const topic_info: TopicInfoItem[] = [];
  for (const [idx, item] of (obj.topic_info as unknown[]).entries()) {
    if (!item || typeof item !== 'object' || Array.isArray(item)) {
      throw new ValidationError('invalid_topic_item', `topic_info at index ${idx} must be an object.`);
    }
    const tObj = item as Record<string, unknown>;
    if (!isStrictInt(tObj.topic_id)) {
      throw new ValidationError('invalid_topic_id', `topic_info[${idx}] missing valid integer topic_id.`);
    }
    topic_info.push({
      topic_id: tObj.topic_id,
      count: isStrictInt(tObj.count) ? tObj.count : undefined,
      name: typeof tObj.name === 'string' ? tObj.name : undefined,
      keywords: Array.isArray(tObj.keywords) ? tObj.keywords : undefined,
      ...tObj,
    });
  }

  return {
    status: obj.status,
    generated_at: obj.generated_at,
    total_documents_modeled: obj.total_documents_modeled,
    topic_info,
    ...obj,
  };
}
