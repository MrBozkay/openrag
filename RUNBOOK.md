# OpenRAG Operations Runbook

## Contact

- **Slack**: #openrag-ops
- **Email**: ops@openrag.dev

## Common Operations

### Restart Service
```bash
kubectl rollout restart deployment/openrag-deployment -n openrag
```

### Scale Service
```bash
kubectl scale deployment/openrag-deployment --replicas=5 -n openrag
```

### View Logs
```bash
kubectl logs -f deployment/openrag -n openrag
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| High latency | Check vector store, scale pods |
| 503 errors | Check Qdrant health |
| Memory issues | Increase pod memory limits |

## Health Checks

```bash
curl http://localhost:8000/health
```

## Backup

```bash
kubectl exec -n openrag qdrant-0 -- qdrant-cli backup --storage /qdrant/storage
```
