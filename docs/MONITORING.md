# EDAPHNET Observability & Model Monitoring

## 1. Structured JSON Logging

Every inference request logs a structured JSON entry with correlation context:
```json
{
  "timestamp": "2026-10-08T20:30:15.120Z",
  "request_id": "902d338f-5192-4f76-8809-77c8e9860b29",
  "endpoint": "/api/v1/analyze",
  "method": "POST",
  "status_code": 200,
  "latency_ms": 138.4,
  "soil_type": "Red Soil",
  "soil_confidence": 0.87,
  "disease_detected": "Early Blight",
  "disease_confidence": 0.92,
  "ood_detected": false,
  "reliability_level": "HIGH"
}
```

---

## 2. Telemetry Metrics

Tracked operational metrics:
- **Request Volume**: Total count of API calls grouped by status and endpoint.
- **Latency Percentiles**: $p_{50}$, $p_{95}$, $p_{99}$ latency metrics per route.
- **Abstention Rate**: Ratio of requests flagged as `ABSTAIN` due to OOD or rejected image quality.
- **Feature Distribution Drift**: Kolmogorov-Smirnov test tracking shifts in environmental features ($N, P, K, pH, \text{Rainfall}$).

---

## 3. Thresholds & Automated Alerting Rules

| Metric | Warning Threshold | Critical Alert | Action Item |
|:---|:---|:---|:---|
| **OOD Rate** | $> 12\%$ over 1 hour | $> 25\%$ | Investigate new camera sensors, seasonality, or novel pathogen strain |
| **Abstention Rate** | $> 8\%$ | $> 18\%$ | Audit image quality gate parameters |
| **$p_{95}$ Latency** | $> 500\text{ms}$ | $> 1500\text{ms}$ | Scale API worker replicas / Enable batching |
| **Top-1 Confidence** | Mean drops below $0.70$ | Mean drops below $0.55$ | Model performance drift; trigger retrain |
