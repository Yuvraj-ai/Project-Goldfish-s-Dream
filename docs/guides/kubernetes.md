# Kubernetes Deployment Guide

Deploy the Open Deep Research API server on Kubernetes.

---

## Quick Start

```bash
# Create namespace
kubectl create namespace odr

# Apply manifests (order matters)
kubectl apply -n odr -f configmap.yaml
kubectl apply -n odr -f secret.yaml
kubectl apply -n odr -f pvc.yaml
kubectl apply -n odr -f deployment.yaml
kubectl apply -n odr -f service.yaml

# Optional: ingress for external access
kubectl apply -n odr -f ingress.yaml

# Optional: autoscaling
kubectl apply -n odr -f hpa.yaml

# Verify
kubectl -n odr get pods
kubectl -n odr get svc
```

The API is available at `http://odr-api:8000` within the cluster. Metrics at `http://odr-api:8000/metrics`.

---

## Manifests

### 1. ConfigMap

Create `configmap.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: odr-config
  labels:
    app.kubernetes.io/name: open-deep-research
data:
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  API_DB_PATH: "/data/research.db"
  ENABLE_REST_API: "true"
  MAX_TOTAL_TOKENS: "500000"
  MAX_COST_USD: "5.0"
  PYTHONPATH: "/app/src"
```

### 2. Secret

Create `secret.yaml`:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: odr-secret
  labels:
    app.kubernetes.io/name: open-deep-research
type: Opaque
stringData:
  API_KEY: "sk-your-api-key-here"
```

Replace `sk-your-api-key-here` with your actual API key. For production, use an external secrets manager (e.g., AWS Secrets Manager, HashiCorp Vault) and reference the secret via a mutating webhook or CSI driver.

### 3. PersistentVolumeClaim

Create `pvc.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: odr-data
  labels:
    app.kubernetes.io/name: open-deep-research
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
  # storageClassName: standard  # uncomment if your cluster requires it
```

The SQLite database configured via `API_DB_PATH=/data/research.db` is stored on this volume.

### 4. Deployment

Create `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: odr-api
  labels:
    app.kubernetes.io/name: open-deep-research
    app.kubernetes.io/component: api
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: open-deep-research
      app.kubernetes.io/component: api
  template:
    metadata:
      labels:
        app.kubernetes.io/name: open-deep-research
        app.kubernetes.io/component: api
    spec:
      securityContext:
        runAsUser: 1001
        runAsGroup: 1001
        fsGroup: 1001
      containers:
        - name: api
          image: open-deep-research:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
              protocol: TCP
          envFrom:
            - configMapRef:
                name: odr-config
            - secretRef:
                name: odr-secret
          env:
            - name: LOG_LEVEL
              value: "info"
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "2"
              memory: "2Gi"
          livenessProbe:
            httpGet:
              path: /metrics
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /metrics
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 2
          volumeMounts:
            - name: data
              mountPath: /data
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: odr-data
```

Notes:
- The container image must be built first (see [Docker guide](./docker.md)).
- Both liveness and readiness probes use `GET /metrics` which returns 200 when the API is healthy.
- Resource limits reflect typical usage — adjust based on load testing. The 1 MB body-size limit (enforced by middleware in `main.py:44-57`) caps memory per request.
- The container runs as non-root user 1001 (matching the Docker image convention).

### 5. Service

Create `service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: odr-api
  labels:
    app.kubernetes.io/name: open-deep-research
    app.kubernetes.io/component: api
spec:
  type: ClusterIP
  ports:
    - name: http
      port: 8000
      targetPort: 8000
      protocol: TCP
  selector:
    app.kubernetes.io/name: open-deep-research
    app.kubernetes.io/component: api
```

---

## Optional: Ingress

Create `ingress.yaml` for external access:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: odr-api
  labels:
    app.kubernetes.io/name: open-deep-research
    app.kubernetes.io/component: api
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "1m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  ingressClassName: nginx
  rules:
    - host: research.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: odr-api
                port:
                  number: 8000
  # tls:
  #   - hosts:
  #       - research.example.com
  #     secretName: odr-tls
```

The `proxy-body-size` annotation enforces the 1 MB request limit at the ingress layer, matching the middleware in `main.py:44-57`. Adjust the host and TLS configuration for your domain.

---

## Optional: HorizontalPodAutoscaler

Create `hpa.yaml` for CPU-based scaling:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: odr-api
  labels:
    app.kubernetes.io/name: open-deep-research
    app.kubernetes.io/component: api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: odr-api
  minReplicas: 1
  maxReplicas: 5
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

HPA requires a metrics server installed in the cluster. For memory-based or custom metrics, extend the `metrics` array.

---

## Applying All Manifests

Save each manifest as a separate file in a `k8s/` directory and apply:

```bash
kubectl apply -n odr -f k8s/
```

Or combine into a single file with `---` delimiters:

```bash
kubectl apply -n odr -f - <<'EOF'
# ... all YAML above ...
EOF
```

---

## Verify Deployment

```bash
# Check pods
kubectl -n odr get pods

# Check logs
kubectl -n odr logs deployment/odr-api

# Port-forward to test locally
kubectl -n odr port-forward svc/odr-api 8000:8000

# Test health endpoint
curl http://localhost:8000/metrics
```

---

## Upgrading

1. Build a new image and push to your registry.
2. Update the deployment:

```bash
kubectl -n odr set image deployment/odr-api api=open-deep-research:new-tag
kubectl -n odr rollout status deployment/odr-api
```

For ConfigMap/Secret changes, pods must be restarted (either via `kubectl rollout restart` or a tool like Reloader):

```bash
kubectl -n odr rollout restart deployment/odr-api
```

---

## Cleanup

```bash
kubectl delete namespace odr
```

**Warning:** Deleting the namespace removes the PVC and all persisted data.

---

## Notes

- The API server only starts when `ENABLE_REST_API=true` (gated by `ApiConfig.enable_rest_api`).
- Request body size is limited to **1 MB** (enforced by middleware in `main.py:44-57`).
- JSON-structured logging is emitted to stdout — collect with your cluster log aggregator (EFK, PLG, Datadog, etc.).
- SQLite is not designed for concurrent writes — keep `replicas: 1` unless you switch to a networked database.
- All manifests use `app.kubernetes.io/name` and `app.kubernetes.io/component` labels following Kubernetes recommended labels.
