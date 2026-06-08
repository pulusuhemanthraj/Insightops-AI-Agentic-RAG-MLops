# API Documentation

Base URL: `http://localhost:8000`

## GET /health
Returns service health.

## GET /metrics/kpis
Returns total events, average latency, average conversion rate, and average failure rate.

## GET /tickets/summary
Returns ticket counts by severity and issue type.

## GET /logs/anomalies
Returns detected anomalous logs after running anomaly detection.

## POST /ask
Request:

```json
{
  "question": "Why did checkout errors increase?",
  "agentic": true
}
```

Returns routed agent output or RAG answer with citations.

## POST /predict/severity
Request:

```json
{
  "priority_score": 88,
  "resolution_hours": 12.5,
  "text_length": 120,
  "issue_type_encoded": 2,
  "customer_segment_encoded": 1,
  "status_encoded": 3
}
```
