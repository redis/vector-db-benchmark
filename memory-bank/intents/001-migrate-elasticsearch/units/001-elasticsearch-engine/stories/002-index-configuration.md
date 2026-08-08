---
id: 002-index-configuration
unit: 001-elasticsearch-engine
intent: 001-migrate-elasticsearch
status: complete
priority: must
created: 2026-02-27T00:00:00.000Z
assigned_bolt: 001-elasticsearch-engine
implemented: true
---

# Story: 002-index-configuration

## User Story

**As a** benchmark operator
**I want** the engine to create an Elasticsearch dense vector index with HNSW configuration
**So that** vectors can be indexed and searched with the specified parameters

## Acceptance Criteria

- [ ] **Given** a dataset with L2 distance, **When** configuring, **Then** index is created with `similarity: "l2_norm"`
- [ ] **Given** a dataset with COSINE distance, **When** configuring, **Then** index is created with `similarity: "cosine"`
- [ ] **Given** a dataset with DOT distance, **When** configuring, **Then** an incompatibility error is returned
- [ ] **Given** a dataset with vector_size > 2048, **When** configuring, **Then** an incompatibility error is returned
- [ ] **Given** collection_params `{"index_options": {"m": 32, "ef_construction": 256}}`, **When** configuring, **Then** HNSW index uses those values
- [ ] **Given** a dataset with schema fields (int, geo types), **When** configuring, **Then** field mappings are created (int->long, geo->geo_point)
- [ ] **Given** an existing index, **When** configuring, **Then** old index is deleted first

## Technical Notes

- Index settings: `number_of_shards: 1, number_of_replicas: 0, refresh_interval: "10s"`
  - SUPERSEDED by #240: `refresh_interval` is now `-1` (no periodic refresh during
    ingest), matching `opensearch.rs` and upstream's
    `v0/engine/clients/elasticsearch/configure.py`. `upload()` issues an explicit
    `_refresh` before the search phase. Do not restore `"10s"`.
- Vector field: `type: "dense_vector", dims: vector_size, index: true`
- Source excludes vector field: `_source.excludes: ["vector"]`
- Clean (delete) should catch NotFoundError silently
- Reference: `v0/engine/clients/elasticsearch/configure.py`

## Dependencies

### Requires
- 001-connection-and-config

### Enables
- 003-bulk-upload

## Edge Cases

| Scenario | Expected Behavior |
|----------|-------------------|
| Index already exists | Delete and recreate |
| Delete non-existent index | Silently succeed |
| Unknown field type in schema | Pass through as-is |

## Out of Scope

- Custom analyzers
- Multiple index support
