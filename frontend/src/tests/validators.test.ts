/**
 * Comprehensive regression unit tests for Schema-v2 validation,
 * analytical artifact validation, dataset isolation, and asynchronous safety.
 * Runs under Node.js native test runner: node --experimental-strip-types --test
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  safeFiniteNumber,
  formatMetricNumber,
  formatMetricDecimal,
  validateDatasetListResponse,
  validateDatasetSummary,
  validateModernConsolidated,
  validateLegacyConsolidated,
  validateSentimentArtifact,
  validateMisinformationArtifact,
  validateGraphArtifact,
  validateTopicArtifact,
  isValidIsoTimestamp,
  ValidationError,
} from '../validation/validators.ts';

import {
  loadDatasetArtifacts,
  DatasetRequestCoordinator,
} from '../api/datasetLoader.ts';

import type {
  DatasetSummary,
} from '../types/dataset.ts';

// Test Helpers: Authoritative Schema-v2 Fixture Generators

function createValidRecord(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    document_id: 'reddit_post101',
    platform: 'reddit',
    native_id: 'post101',
    url: 'https://reddit.com/r/test/comments/post101',
    timestamps: {
      published_at: '2026-11-10T12:00:00+00:00',
      collected_at: '2026-11-10T12:05:00+00:00',
      processed_at: '2026-11-10T12:10:00+00:00',
    },
    author: {
      username: 'analyst_alpha',
      channel_id: null,
      channel_title: null,
      graph_status: 'ranked_influencer',
      influencer_score: 0.85,
      pagerank: 0.012,
      community_id: 1,
      in_degree: 5,
      out_degree: 2,
    },
    content: {
      title_raw: 'Raw title text',
      body_raw: 'Raw body text',
      title_cleaned: 'Clean title text',
      body_cleaned: 'Clean body text',
      detected_language: 'English',
      language_code: 'en',
      preprocessing_status: 'success',
    },
    engagement: {
      score: 150,
      views_count: null,
      likes_count: null,
      comments_count: 25,
    },
    metadata: {
      reddit: { subreddit: 'technology', score: 150 },
      youtube: null,
    },
    sentiment: {
      status: 'success',
      label: 'positive',
      score: 0.75,
    },
    topics: {
      status: 'assigned',
      topic_id: 2,
      probability: 0.88,
      keywords: [{ word: 'artificial', weight: 0.9 }],
    },
    misinformation: {
      status: 'success',
      veracity_label: 'real',
      credibility_score: 0.92,
      risk_tier: 'low',
      stylometric_signals: { caps_ratio: 0.05 },
    },
    consolidation: {
      status: 'complete',
      is_complete: true,
      sources_joined: ['preprocessing', 'sentiment', 'topics', 'graph', 'misinformation'],
    },
    ...overrides,
  };
}

function createValidModernPayload(records: Record<string, unknown>[] = [createValidRecord()]): Record<string, unknown> {
  const redditCount = records.filter((r) => r.platform === 'reddit').length;
  const youtubeCount = records.filter((r) => r.platform === 'youtube').length;
  const completeCount = records.filter((r) => (r.consolidation as Record<string, unknown>)?.status === 'complete').length;
  const degradedCount = records.filter((r) => (r.consolidation as Record<string, unknown>)?.status === 'degraded').length;
  const partialCount = records.filter((r) => (r.consolidation as Record<string, unknown>)?.status === 'partial').length;
  const unprocessedCount = records.filter((r) => (r.consolidation as Record<string, unknown>)?.status === 'unprocessed').length;

  return {
    schema_version: '2.0',
    pipeline_version: '1.0.0',
    status: 'success',
    consolidation_status: 'complete',
    generated_at: '2026-11-10T12:15:00+00:00',
    provenance: {
      reddit_input_file: 'data/processed/reddit_posts_processed.json',
      reddit_status: 'loaded',
      reddit_records_count: redditCount,
      youtube_input_file: 'data/processed/youtube_videos_processed.json',
      youtube_status: youtubeCount > 0 ? 'loaded' : 'empty',
      youtube_records_count: youtubeCount,
    },
    metrics: {
      total_records_consolidated: records.length,
      platform_counts: {
        reddit: redditCount,
        youtube: youtubeCount,
      },
      complete_records: completeCount,
      degraded_records: degradedCount,
      partial_records: partialCount,
      unprocessed_records: unprocessedCount,
      duplicates_deduplicated: 0,
      rejected_records_count: 0,
    },
    audit_warnings: [],
    rejected_records: [],
    records,
  };
}

// Primitive & Helper Tests

test('safeFiniteNumber: preserves finite numbers and respects bounds', () => {
  assert.equal(safeFiniteNumber(0), 0);
  assert.equal(safeFiniteNumber(42), 42);
  assert.equal(safeFiniteNumber(-10), -10);
  assert.equal(safeFiniteNumber(3.1415), 3.1415);

  assert.equal(safeFiniteNumber(null), null);
  assert.equal(safeFiniteNumber(undefined), null);
  assert.equal(safeFiniteNumber(NaN), null);
  assert.equal(safeFiniteNumber(Infinity), null);
  assert.equal(safeFiniteNumber(-Infinity), null);
  assert.equal(safeFiniteNumber('42'), null);

  assert.equal(safeFiniteNumber(10, { min: 0, max: 100 }), 10);
  assert.equal(safeFiniteNumber(-1, { min: 0 }), null);
  assert.equal(safeFiniteNumber(101, { max: 100 }), null);
});

test('formatMetricNumber and formatMetricDecimal: preserve missing values', () => {
  assert.equal(formatMetricNumber(12500), '12,500');
  assert.equal(formatMetricNumber(0), '0');
  assert.equal(formatMetricNumber(null), 'Unavailable');
  assert.equal(formatMetricNumber(undefined), 'Unavailable');
  assert.equal(formatMetricNumber(NaN), 'Unavailable');

  assert.equal(formatMetricDecimal(0.8523, 2), '0.85');
  assert.equal(formatMetricDecimal(null), 'Unavailable');
  assert.equal(formatMetricDecimal(undefined, 3, 'N/A'), 'N/A');
});

// Dataset Discovery Validation

test('validateDatasetListResponse: validates modern and legacy dataset entries', () => {
  const samplePayload = {
    datasets: [
      {
        id: 'run_20261111_084915_66d8c041',
        type: 'modern',
        label: 'Modern Run 20261111_084915',
        generated_at: '2026-11-11T08:49:15.690607+00:00',
        status: 'failed',
        stages: { sentiment: 'partial', topics: 'failed' },
        available_artifacts: ['consolidated_intelligence.json', 'sentiment_analysis_results.json'],
      },
      {
        id: 'legacy_historical_processed',
        type: 'legacy',
        label: 'Historical Baseline',
        generated_at: null,
        status: 'legacy',
        stages: { legacy_baseline: 'available' },
        available_artifacts: ['consolidated_intelligence.json', 'topic_modeling_results.json'],
      },
    ],
    total_datasets: 2,
  };

  const validated = validateDatasetListResponse(samplePayload);
  assert.equal(validated.datasets.length, 2);
  assert.equal(validated.total_datasets, 2);
  assert.equal(validated.datasets[0].type, 'modern');
  assert.equal(validated.datasets[1].type, 'legacy');
});

test('validateDatasetListResponse: rejects malformed payload structures', () => {
  assert.throws(() => validateDatasetListResponse(null), ValidationError);
  assert.throws(() => validateDatasetListResponse({}), ValidationError);
  assert.throws(() => validateDatasetListResponse({ datasets: 'not-an-array' }), ValidationError);
  assert.throws(
    () =>
      validateDatasetListResponse({
        datasets: [{ id: 'bad_type', type: 'unknown_type' }],
      }),
    ValidationError
  );
});

test('validateDatasetSummary: enforces required identity and classification fields', () => {
  const valid = validateDatasetSummary({
    id: 'run_20261110_134958_2674e2d2',
    type: 'modern',
    label: 'Modern Run',
    status: 'success',
    stages: { reddit_ingestion: 'success' },
    available_artifacts: ['consolidated_intelligence.json'],
  });

  assert.equal(valid.id, 'run_20261110_134958_2674e2d2');
  assert.equal(valid.type, 'modern');

  assert.throws(() => validateDatasetSummary({ type: 'modern' }), /missing_dataset_id/);
  assert.throws(
    () => validateDatasetSummary({ id: 'test', type: 'unsupported' }),
    /invalid_dataset_type/
  );
});

// Schema-v2 Envelope Validation

test('validateModernConsolidated: enforces Schema-v2 envelope strictly', () => {
  const valid = createValidModernPayload();
  const result = validateModernConsolidated(valid);
  assert.equal(result.schema_version, '2.0');
  assert.equal(result.status, 'success');
  assert.equal(result.metrics.total_records_consolidated, 1);
  assert.equal(result.records.length, 1);
  assert.equal(result.records[0].document_id, 'reddit_post101');
});

test('validateModernConsolidated: rejects invalid schema_version or non-success status', () => {
  const invalidVersion = createValidModernPayload();
  invalidVersion.schema_version = '1.0';
  assert.throws(() => validateModernConsolidated(invalidVersion), /schema_version_mismatch/);

  const invalidStatus = createValidModernPayload();
  invalidStatus.status = 'failed';
  assert.throws(() => validateModernConsolidated(invalidStatus), /invalid_status/);
});

test('validateModernConsolidated: rejects envelope metrics mismatches', () => {
  // Total count mismatch with actual records length
  const totalMismatch = createValidModernPayload();
  (totalMismatch.metrics as Record<string, unknown>).total_records_consolidated = 999;
  assert.throws(() => validateModernConsolidated(totalMismatch), /metric_records_mismatch/);

  // Platform sum mismatch
  const platMismatch = createValidModernPayload();
  ((platMismatch.metrics as Record<string, unknown>).platform_counts as Record<string, unknown>).reddit = 0;
  assert.throws(() => validateModernConsolidated(platMismatch), /platform_count_mismatch/);

  // Status sum mismatch
  const statusMismatch = createValidModernPayload();
  (statusMismatch.metrics as Record<string, unknown>).complete_records = 0;
  assert.throws(() => validateModernConsolidated(statusMismatch), /status_sum_mismatch/);
});

test('validateModernConsolidated: enforces provenance consistency rules', () => {
  const provMismatch = createValidModernPayload();
  (provMismatch.provenance as Record<string, unknown>).reddit_status = 'missing';
  (provMismatch.provenance as Record<string, unknown>).reddit_records_count = 10; // must be 0 for missing
  assert.throws(() => validateModernConsolidated(provMismatch), /invalid_provenance_invariant/);
});

// Deep Canonical Record Validation

test('canonical record: rejects flat string author or content (requires nested dictionaries)', () => {
  const flatAuthor = createValidModernPayload([
    createValidRecord({ author: 'analyst_alpha_string' }),
  ]);
  assert.throws(() => validateModernConsolidated(flatAuthor), /invalid_subfield_dict/);

  const flatContent = createValidModernPayload([
    createValidRecord({ content: 'raw content string' }),
  ]);
  assert.throws(() => validateModernConsolidated(flatContent), /invalid_subfield_dict/);
});

test('canonical record: rejects missing canonical fields or forbidden extra fields', () => {
  // Missing root field 'timestamps'
  const recordMissingTs = createValidRecord();
  delete recordMissingTs.timestamps;
  assert.throws(
    () => validateModernConsolidated(createValidModernPayload([recordMissingTs])),
    /missing_canonical_field/
  );

  // Forbidden extra field (envelope leakage prevention)
  const recordExtraField = createValidRecord({ leaked_envelope_key: 'malicious' });
  assert.throws(
    () => validateModernConsolidated(createValidModernPayload([recordExtraField])),
    /forbidden_extra_field/
  );
});

test('canonical record: enforces document_id canonical format and uniqueness', () => {
  // Mismatched document_id
  const mismatchedDocId = createValidModernPayload([
    createValidRecord({ document_id: 'reddit_mismatch123', native_id: 'real123' }),
  ]);
  assert.throws(() => validateModernConsolidated(mismatchedDocId), /invalid_document_id/);

  // Duplicate document_id in records
  const rec1 = createValidRecord({ document_id: 'reddit_dup1', native_id: 'dup1' });
  const rec2 = createValidRecord({ document_id: 'reddit_dup1', native_id: 'dup1' });
  assert.throws(
    () => validateModernConsolidated(createValidModernPayload([rec1, rec2])),
    /duplicate_document_id/
  );
});

test('canonical record: enforces sentiment score bounds and nullability', () => {
  // Score out of bounds: 1.5 > 1.0
  const outOfBoundsHigh = createValidModernPayload([
    createValidRecord({
      sentiment: { status: 'success', label: 'positive', score: 1.5 },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(outOfBoundsHigh), /invalid_sentiment_score/);

  // Score out of bounds: -1.2 < -1.0
  const outOfBoundsLow = createValidModernPayload([
    createValidRecord({
      sentiment: { status: 'success', label: 'negative', score: -1.2 },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(outOfBoundsLow), /invalid_sentiment_score/);

  // Non-success status with non-null score or label
  const nonSuccessWithScore = createValidModernPayload([
    createValidRecord({
      sentiment: { status: 'error', label: 'positive', score: 0.5 },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(nonSuccessWithScore), /invalid_sentiment_nullability/);
});

test('canonical record: enforces misinformation credibility bounds and nullability', () => {
  // Credibility score < 0.0 (e.g. -0.2)
  const credLow = createValidModernPayload([
    createValidRecord({
      misinformation: {
        status: 'success',
        veracity_label: 'fake',
        credibility_score: -0.2,
        risk_tier: 'high_risk',
        stylometric_signals: null,
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(credLow), /invalid_misinfo_credibility_score/);

  // Credibility score > 1.0 (e.g. 1.2)
  const credHigh = createValidModernPayload([
    createValidRecord({
      misinformation: {
        status: 'success',
        veracity_label: 'real',
        credibility_score: 1.2,
        risk_tier: 'low',
        stylometric_signals: null,
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(credHigh), /invalid_misinfo_credibility_score/);

  // Non-success status with non-null label
  const nonSuccessMisinfo = createValidModernPayload([
    createValidRecord({
      misinformation: {
        status: 'error',
        veracity_label: 'real',
        credibility_score: null,
        risk_tier: null,
        stylometric_signals: null,
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(nonSuccessMisinfo), /invalid_misinfo_nullability/);
});

test('canonical record: enforces topic probability bounds and outlier conventions', () => {
  // Probability out of bounds: 1.4 > 1.0
  const probHigh = createValidModernPayload([
    createValidRecord({
      topics: {
        status: 'assigned',
        topic_id: 1,
        probability: 1.4,
        keywords: [{ word: 'tech', weight: 0.5 }],
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(probHigh), /invalid_topic_probability/);

  // Outlier with invalid topic_id (must be -1)
  const badOutlierId = createValidModernPayload([
    createValidRecord({
      topics: {
        status: 'outlier',
        topic_id: 5,
        probability: null,
        keywords: [],
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(badOutlierId), /invalid_topic_outlier_id/);

  // Outlier with non-empty keywords (must be empty list)
  const badOutlierKw = createValidModernPayload([
    createValidRecord({
      topics: {
        status: 'outlier',
        topic_id: -1,
        probability: null,
        keywords: [{ word: 'forbidden', weight: 0.1 }],
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(badOutlierKw), /invalid_topic_outlier_keywords/);
});

test('canonical record: enforces strict boolean for consolidation.is_complete', () => {
  // Non-boolean is_complete (e.g. integer 1 or string 'true')
  const intComplete = createValidModernPayload([
    createValidRecord({
      consolidation: {
        status: 'complete',
        is_complete: 1,
        sources_joined: ['preprocessing'],
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(intComplete), /invalid_consolidation_is_complete/);
});

test('canonical record: rejects malformed or non-string content properties without silent defaulting', () => {
  const contentProps = [
    'title_raw',
    'body_raw',
    'title_cleaned',
    'body_cleaned',
    'detected_language',
    'language_code',
    'preprocessing_status',
  ];

  // Test missing each of the 7 properties
  for (const prop of contentProps) {
    const rec = createValidRecord();
    delete (rec.content as Record<string, unknown>)[prop];
    assert.throws(
      () => validateModernConsolidated(createValidModernPayload([rec])),
      /missing_content_property/
    );
  }

  // Test non-string for each of the 7 properties
  for (const prop of contentProps) {
    const rec = createValidRecord();
    (rec.content as Record<string, unknown>)[prop] = 12345;
    assert.throws(
      () => validateModernConsolidated(createValidModernPayload([rec])),
      /invalid_content_property/
    );
  }
});

test('canonical record: enforces strict ISO-8601 UTC timestamp format', () => {
  // Malformed date string (not an ISO timestamp)
  const badDateRecord = createValidModernPayload([
    createValidRecord({
      timestamps: {
        published_at: 'not-a-valid-date-string',
        collected_at: '2026-11-10T12:05:00+00:00',
        processed_at: '2026-11-10T12:10:00+00:00',
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(badDateRecord), /invalid_timestamp_format/);

  // Missing published_at key
  const missingTsRecord = createValidRecord();
  delete (missingTsRecord.timestamps as Record<string, unknown>).published_at;
  assert.throws(
    () => validateModernConsolidated(createValidModernPayload([missingTsRecord])),
    /missing_timestamp_field/
  );

  // Impossible calendar date (Feb 30)
  const impossibleDateRecord = createValidModernPayload([
    createValidRecord({
      timestamps: {
        published_at: '2026-02-30T12:00:00+00:00',
        collected_at: '2026-11-10T12:05:00+00:00',
        processed_at: '2026-11-10T12:10:00+00:00',
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(impossibleDateRecord), /invalid_timestamp_format/);

  // Non-+00:00 representation (Z suffix rejected)
  const zDateRecord = createValidModernPayload([
    createValidRecord({
      timestamps: {
        published_at: '2026-11-10T12:00:00Z',
        collected_at: '2026-11-10T12:05:00+00:00',
        processed_at: '2026-11-10T12:10:00+00:00',
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(zDateRecord), /invalid_timestamp_format/);

  // Timezone offset (+05:30 rejected)
  const offsetDateRecord = createValidModernPayload([
    createValidRecord({
      timestamps: {
        published_at: '2026-11-10T12:00:00+05:30',
        collected_at: '2026-11-10T12:05:00+00:00',
        processed_at: '2026-11-10T12:10:00+00:00',
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(offsetDateRecord), /invalid_timestamp_format/);

  // Whitespace-padded timestamp rejected without silent trimming
  const leadingSpaceRecord = createValidModernPayload([
    createValidRecord({
      timestamps: {
        published_at: ' 2026-11-10T12:00:00+00:00',
        collected_at: '2026-11-10T12:05:00+00:00',
        processed_at: '2026-11-10T12:10:00+00:00',
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(leadingSpaceRecord), /invalid_timestamp_format/);

  const trailingSpaceRecord = createValidModernPayload([
    createValidRecord({
      timestamps: {
        published_at: '2026-11-10T12:00:00+00:00 ',
        collected_at: '2026-11-10T12:05:00+00:00',
        processed_at: '2026-11-10T12:10:00+00:00',
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(trailingSpaceRecord), /invalid_timestamp_format/);

  // Trailing line terminators (\n, \r, \r\n, \t) rejected
  const newlineRecord = createValidModernPayload([
    createValidRecord({
      timestamps: {
        published_at: '2026-11-10T12:00:00+00:00\n',
        collected_at: '2026-11-10T12:05:00+00:00',
        processed_at: '2026-11-10T12:10:00+00:00',
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(newlineRecord), /invalid_timestamp_format/);

  const crlfRecord = createValidModernPayload([
    createValidRecord({
      timestamps: {
        published_at: '2026-11-10T12:00:00+00:00\r\n',
        collected_at: '2026-11-10T12:05:00+00:00',
        processed_at: '2026-11-10T12:10:00+00:00',
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(crlfRecord), /invalid_timestamp_format/);

  // Seven fractional digits rejected (canonical Python format expects 0 or exactly 6)
  const sevenDigitsRecord = createValidModernPayload([
    createValidRecord({
      timestamps: {
        published_at: '2026-11-10T12:00:00.1234567+00:00',
        collected_at: '2026-11-10T12:05:00+00:00',
        processed_at: '2026-11-10T12:10:00+00:00',
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(sevenDigitsRecord), /invalid_timestamp_format/);

  // Valid exactly 6 fractional seconds UTC accepted
  const validFractional = createValidModernPayload([
    createValidRecord({
      timestamps: {
        published_at: '2026-11-10T12:00:00.123456+00:00',
        collected_at: '2026-11-10T12:05:00+00:00',
        processed_at: '2026-11-10T12:10:00+00:00',
      },
    }),
  ]);
  const validated = validateModernConsolidated(validFractional);
  assert.equal(validated.records[0].timestamps.published_at, '2026-11-10T12:00:00.123456+00:00');
});

test('isValidIsoTimestamp: enforces strict UTC format, calendar validity, leap years, and rejects year zero', () => {
  // Impossible calendar dates
  assert.equal(isValidIsoTimestamp('2026-02-30T12:00:00+00:00'), false, 'Feb 30 impossible');
  assert.equal(isValidIsoTimestamp('2026-04-31T12:00:00+00:00'), false, 'Apr 31 impossible');
  assert.equal(isValidIsoTimestamp('2026-02-29T12:00:00+00:00'), false, 'Feb 29 non-leap 2026');
  assert.equal(isValidIsoTimestamp('1900-02-29T12:00:00+00:00'), false, 'Feb 29 century non-leap 1900');

  // Valid leap year dates
  assert.equal(isValidIsoTimestamp('2024-02-29T12:00:00+00:00'), true, 'Feb 29 leap 2024');
  assert.equal(isValidIsoTimestamp('2000-02-29T12:00:00+00:00'), true, 'Feb 29 400-year leap 2000');

  // Year zero rejection
  assert.equal(isValidIsoTimestamp('0000-01-01T00:00:00+00:00'), false, 'Year zero rejected');

  // Low year without JS Date.UTC 0-99 quirk
  assert.equal(isValidIsoTimestamp('0050-01-01T00:00:00+00:00'), true, 'Year 50 valid');

  // Out of bounds components
  assert.equal(isValidIsoTimestamp('2026-13-01T12:00:00+00:00'), false, 'Month 13 rejected');
  assert.equal(isValidIsoTimestamp('2026-11-32T12:00:00+00:00'), false, 'Day 32 rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T24:00:00+00:00'), false, 'Hour 24 rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:60:00+00:00'), false, 'Minute 60 rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:60+00:00'), false, 'Second 60 rejected');

  // Strict +00:00 required (Z and offsets rejected)
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00Z'), false, 'Z suffix rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00+05:30'), false, 'Offset +05:30 rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00-08:00'), false, 'Offset -08:00 rejected');

  // Strict whitespace and line-terminator rejection (no silent trimming; absolute end-of-string)
  assert.equal(isValidIsoTimestamp(' 2026-11-10T12:00:00+00:00'), false, 'Leading whitespace rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00+00:00 '), false, 'Trailing whitespace rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10 12:00:00+00:00'), false, 'Space separator instead of T rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00+00:00\n'), false, 'Trailing newline \\n rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00+00:00\r'), false, 'Trailing carriage return \\r rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00+00:00\r\n'), false, 'Trailing CRLF \\r\\n rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00+00:00\t'), false, 'Trailing tab \\t rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00.123456+00:00\n'), false, 'Fractional with \\n rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00.123456+00:00\r'), false, 'Fractional with \\r rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00.123456+00:00\r\n'), false, 'Fractional with \\r\\n rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00.123456+00:00\t'), false, 'Fractional with \\t rejected');

  // Strict fractional-second digits (0 or exactly 6 digits)
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00+00:00'), true, 'Standard 0 fractional digits accepted');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00.123456+00:00'), true, 'Exactly 6 fractional digits accepted');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00.1+00:00'), false, '1 fractional digit rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00.123+00:00'), false, '3 fractional digits rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00.12345+00:00'), false, '5 fractional digits rejected');
  assert.equal(isValidIsoTimestamp('2026-11-10T12:00:00.1234567+00:00'), false, '7 fractional digits rejected');

  assert.equal(isValidIsoTimestamp('not-a-timestamp'), false, 'Non-timestamp rejected');
  assert.equal(isValidIsoTimestamp(null), false, 'null rejected');
  assert.equal(isValidIsoTimestamp(12345), false, 'number rejected');
});

test('canonical record: enforces platform author invariants and metric types', () => {
  // Reddit record with non-null YouTube channel_id
  const redditWithChannel = createValidModernPayload([
    createValidRecord({
      platform: 'reddit',
      author: {
        username: 'reddit_user',
        channel_id: 'UC123456789', // forbidden on Reddit
        channel_title: null,
        graph_status: 'ranked_influencer',
        influencer_score: 0.5,
        pagerank: 0.01,
        community_id: 1,
        in_degree: 2,
        out_degree: 3,
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(redditWithChannel), /invalid_author_platform_invariant/);

  // Author with invalid graph_status enum
  const badGraphStatus = createValidModernPayload([
    createValidRecord({
      author: {
        username: 'reddit_user',
        channel_id: null,
        channel_title: null,
        graph_status: 'unsupported_graph_status',
        influencer_score: null,
        pagerank: null,
        community_id: null,
        in_degree: null,
        out_degree: null,
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(badGraphStatus), /invalid_author_graph_status/);
});

test('canonical record: enforces platform engagement consistency invariants', () => {
  // Reddit record with non-null views_count (forbidden for Reddit)
  const redditWithViews = createValidModernPayload([
    createValidRecord({
      platform: 'reddit',
      engagement: {
        score: 100,
        views_count: 5000, // forbidden on Reddit
        likes_count: null,
        comments_count: 10,
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(redditWithViews), /invalid_engagement_invariant/);

  // YouTube record with non-null score (forbidden for YouTube)
  const youtubeWithScore = createValidModernPayload([
    createValidRecord({
      platform: 'youtube',
      native_id: 'vid123',
      document_id: 'youtube_vid123',
      author: {
        username: 'channel_user',
        channel_id: 'UCabc',
        channel_title: 'My Channel',
        graph_status: 'ranked_influencer',
        influencer_score: 0.8,
        pagerank: 0.05,
        community_id: 2,
        in_degree: 10,
        out_degree: 5,
      },
      engagement: {
        score: 42, // forbidden on YouTube
        views_count: 10000,
        likes_count: 500,
        comments_count: 50,
      },
      metadata: {
        reddit: null,
        youtube: { channel_id: 'UCabc', channel_title: 'My Channel' },
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(youtubeWithScore), /invalid_engagement_invariant/);
});

test('canonical record: enforces platform metadata mutual exclusivity and properties', () => {
  // Reddit record with non-null YouTube metadata
  const redditWithYtMeta = createValidModernPayload([
    createValidRecord({
      platform: 'reddit',
      metadata: {
        reddit: { subreddit: 'news', score: 100 },
        youtube: { channel_id: 'UCxyz', channel_title: 'Title' }, // forbidden on Reddit
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(redditWithYtMeta), /invalid_metadata_invariant/);

  // Reddit record missing subreddit in metadata.reddit
  const badRedditMeta = createValidModernPayload([
    createValidRecord({
      platform: 'reddit',
      metadata: {
        reddit: { score: 100 }, // missing subreddit
        youtube: null,
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(badRedditMeta), /missing_reddit_metadata_properties/);
});

test('canonical record: validates stylometric_signals structure when present', () => {
  // Stylometric signals with non-primitive value (e.g. nested object)
  const badSignals = createValidModernPayload([
    createValidRecord({
      misinformation: {
        status: 'success',
        veracity_label: 'real',
        credibility_score: 0.9,
        risk_tier: 'low',
        stylometric_signals: { invalid_nested: { deep: 'nested' } },
      },
    }),
  ]);
  assert.throws(() => validateModernConsolidated(badSignals), /invalid_stylometric_value/);
});

// Real Structural Validators for Analytical Artifacts

test('validateSentimentArtifact: validates structure and score range, rejects malformed', () => {
  const valid = {
    status: 'success',
    generated_at: '2026-11-10T12:00:00+00:00',
    model_name: 'test_model',
    total_records_processed: 100,
    summary: {
      total: 100,
      positive: 60,
      neutral: 20,
      negative: 20,
      undetermined: 0,
      positive_pct: 60,
      neutral_pct: 20,
      negative_pct: 20,
      undetermined_pct: 0,
      net_sentiment: 0.4,
    },
  };
  const result = validateSentimentArtifact(valid);
  assert.equal(result.status, 'success');
  assert.equal(result.summary.net_sentiment, 0.4);

  // Out of bounds net_sentiment (e.g. 1.5)
  const invalidNet = { ...valid, summary: { ...valid.summary, net_sentiment: 1.5 } };
  assert.throws(() => validateSentimentArtifact(invalidNet), /invalid_net_sentiment/);
});

test('validateMisinformationArtifact: validates structure and bounds, rejects invalid', () => {
  const valid = {
    status: 'success',
    generated_at: '2026-11-10T12:00:00+00:00',
    total_records_processed: 50,
    total_analyzed: 45,
    average_credibility_score: 0.78,
    flagged_high_risk_count: 5,
  };
  const result = validateMisinformationArtifact(valid);
  assert.equal(result.average_credibility_score, 0.78);

  // Out of bounds credibility (> 1.0)
  assert.throws(
    () => validateMisinformationArtifact({ ...valid, average_credibility_score: 1.25 }),
    /invalid_average_credibility_score/
  );

  // Total analyzed exceeding total processed
  assert.throws(
    () => validateMisinformationArtifact({ ...valid, total_analyzed: 100 }),
    /misinfo_analyzed_exceeds_total/
  );
});

test('validateGraphArtifact: validates structure and metrics, rejects invalid density/direction', () => {
  const valid = {
    status: 'success',
    generated_at: '2026-11-10T12:00:00+00:00',
    graph_metrics: {
      total_users: 200,
      total_interactions: 450,
      density: 0.022,
      communities_count: 4,
      is_directed: true,
    },
  };
  const result = validateGraphArtifact(valid);
  assert.equal(result.graph_metrics.total_users, 200);

  // Density out of bounds (> 1.0)
  assert.throws(
    () =>
      validateGraphArtifact({
        ...valid,
        graph_metrics: { ...valid.graph_metrics, density: 1.5 },
      }),
    /invalid_graph_density/
  );

  // is_directed not true
  assert.throws(
    () =>
      validateGraphArtifact({
        ...valid,
        graph_metrics: { ...valid.graph_metrics, is_directed: false },
      }),
    /invalid_graph_is_directed/
  );
});

test('validateTopicArtifact: validates topic info array and models count', () => {
  const valid = {
    status: 'success',
    generated_at: '2026-11-10T12:00:00+00:00',
    total_documents_modeled: 80,
    topic_info: [
      { topic_id: 0, count: 50, name: 'AI Models' },
      { topic_id: 1, count: 30, name: 'Hardware' },
    ],
  };
  const result = validateTopicArtifact(valid);
  assert.equal(result.total_documents_modeled, 80);
  assert.equal(result.topic_info.length, 2);

  // Missing topic_info
  assert.throws(
    () =>
      validateTopicArtifact({
        status: 'success',
        generated_at: '2026-11-10T12:00:00+00:00',
        total_documents_modeled: 10,
      }),
    /missing_topic_info/
  );
});

// Dataset Isolation & Boundary Enforcement

test('dataset isolation: modern data cannot enter legacy parser', () => {
  const modernData = createValidModernPayload();
  assert.throws(() => validateLegacyConsolidated(modernData), /modern_data_in_legacy_parser/);
});

test('dataset isolation: legacy data cannot enter modern parser', () => {
  const legacyData = {
    pipeline_version: '1.0.0',
    status: 'success',
    records: [{ id: 'legacy_rec_1' }],
  };
  // Missing schema_version 2.0
  assert.throws(() => validateModernConsolidated(legacyData), /schema_version_mismatch/);
});

test('dataset isolation: missing modern topics never requests legacy topics', async () => {
  const modernSummary: DatasetSummary = {
    id: 'run_modern_without_topics',
    type: 'modern',
    label: 'Modern Run',
    generated_at: '2026-11-10T12:00:00+00:00',
    status: 'failed',
    stages: { topics: 'failed' },
    available_artifacts: ['consolidated_intelligence.json'], // topic artifact absent
  };

  const requestedArtifacts: string[] = [];
  const mockFetch = async (_datasetId: string, artifactName: string) => {
    requestedArtifacts.push(artifactName);
    if (artifactName === 'consolidated_intelligence.json') {
      return createValidModernPayload();
    }
    return {};
  };

  const loaded = await loadDatasetArtifacts(modernSummary, mockFetch);

  // Confirm topic_modeling_results.json was never requested
  assert.equal(requestedArtifacts.includes('topic_modeling_results.json'), false);
  assert.equal(loaded.topic, null);
  assert.match(loaded.topicUnavailableReason || '', /No legacy fallback permitted/);
});

test('dataset isolation: failed artifact request remains null without fallback', async () => {
  const modernSummary: DatasetSummary = {
    id: 'run_modern_partial',
    type: 'modern',
    label: 'Modern Run',
    generated_at: '2026-11-10T12:00:00+00:00',
    status: 'partial',
    stages: { sentiment: 'failed' },
    available_artifacts: ['consolidated_intelligence.json', 'sentiment_analysis_results.json'],
  };

  const mockFetch = async (_datasetId: string, artifactName: string) => {
    if (artifactName === 'consolidated_intelligence.json') {
      return createValidModernPayload();
    }
    if (artifactName === 'sentiment_analysis_results.json') {
      throw new Error('HTTP 500 Internal Error from bridge');
    }
    return {};
  };

  const loaded = await loadDatasetArtifacts(modernSummary, mockFetch);
  // Sentiment remains null; consolidated is still loaded
  assert.notEqual(loaded.consolidated, null);
  assert.equal(loaded.sentiment, null);
});

// Asynchronous Request Safety & Race-Condition Cancellation Tests

test('DatasetRequestCoordinator: switching datasets aborts in-flight request', async () => {
  const coordinator = new DatasetRequestCoordinator();

  const dataset1: DatasetSummary = {
    id: 'dataset_slow_1',
    type: 'modern',
    label: 'Dataset 1',
    generated_at: null,
    status: 'success',
    stages: {},
    available_artifacts: ['consolidated_intelligence.json'],
  };

  const dataset2: DatasetSummary = {
    id: 'dataset_fast_2',
    type: 'modern',
    label: 'Dataset 2',
    generated_at: null,
    status: 'success',
    stages: {},
    available_artifacts: ['consolidated_intelligence.json'],
  };

  let dataset1Aborted = false;

  // Slow fetcher for dataset 1 that listens to abort signal
  const slowFetcher = async (_id: string, _artifact: string, signal?: AbortSignal) => {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        resolve(createValidModernPayload());
      }, 100);

      signal?.addEventListener('abort', () => {
        clearTimeout(timer);
        dataset1Aborted = true;
        reject(new DOMException('Aborted', 'AbortError'));
      });
    });
  };

  const fastFetcher = async (_id: string, _artifact: string) => {
    return createValidModernPayload([createValidRecord({ document_id: 'reddit_fast2', native_id: 'fast2' })]);
  };

  // Launch request 1 (slow)
  const req1Promise = coordinator.switchDataset(dataset1, slowFetcher);

  // Immediately launch request 2 (fast), which aborts request 1
  const req2Promise = coordinator.switchDataset(dataset2, fastFetcher);

  const [res1, res2] = await Promise.all([req1Promise, req2Promise]);

  // Request 1 must have been aborted and resolved to null (discarded)
  assert.equal(dataset1Aborted, true);
  assert.equal(res1, null);

  // Request 2 must have succeeded with dataset2 data
  assert.notEqual(res2, null);
  assert.equal(res2?.datasetId, 'dataset_fast_2');
  assert.equal(res2?.consolidated?.records.length, 1);
});

test('DatasetRequestCoordinator: stale slow response cannot overwrite newer selection', async () => {
  const coordinator = new DatasetRequestCoordinator();

  const dataset1: DatasetSummary = {
    id: 'stale_dataset_1',
    type: 'modern',
    label: 'Stale Dataset 1',
    generated_at: null,
    status: 'success',
    stages: {},
    available_artifacts: ['consolidated_intelligence.json'],
  };

  const dataset2: DatasetSummary = {
    id: 'active_dataset_2',
    type: 'modern',
    label: 'Active Dataset 2',
    generated_at: null,
    status: 'success',
    stages: {},
    available_artifacts: ['consolidated_intelligence.json'],
  };

  // Fetcher 1 does not check signal and resolves after 50ms (simulating delayed uncooperative network)
  const delayedFetcher = async () => {
    await new Promise((r) => setTimeout(r, 50));
    return createValidModernPayload();
  };

  const immediateFetcher = async () => {
    return createValidModernPayload();
  };

  const req1Promise = coordinator.switchDataset(dataset1, delayedFetcher);
  const req2Promise = coordinator.switchDataset(dataset2, immediateFetcher);

  const [res1, res2] = await Promise.all([req1Promise, req2Promise]);

  // res1 was superseded by req2, so coordinator must return null for res1
  assert.equal(res1, null);
  assert.notEqual(res2, null);
  assert.equal(res2?.datasetId, 'active_dataset_2');
});
