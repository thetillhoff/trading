# Compute Architecture & Performance Optimization

**Status**: Single-machine orchestration implemented (Feb 2026). This document outlines future scaling architecture.

---

## Table of Contents

1. [Current State](#current-state)
2. [Next Optimizations](#next-optimizations)
3. [Distributed Architecture](#distributed-architecture)
4. [Technology Stack](#technology-stack)
5. [Data Flow & Caching Strategy](#data-flow--caching-strategy)
6. [Storage Architecture](#storage-architecture)
7. [Orchestration: Custom Controller vs Argo Workflows](#orchestration-custom-controller-vs-argo-workflows)
8. [Bootstrap & Restore Mechanism](#bootstrap--restore-mechanism)
9. [Image Build & Distribution](#image-build--distribution)
10. [Migration Path](#migration-path)

---

## Current State

### Implemented (Feb 2026)

- ✅ **TaskGraph**: DAG-based orchestration with topological scheduling
- ✅ **ProcessPoolExecutor**: Parallel execution across CPU cores
- ✅ **Local file-based cache**: Content-addressable caching (`~/.cache/trading/`)
- ✅ **Checkpoint/resume**: JSON-based state persistence
- ✅ **Vectorization**: scipy/pandas optimizations (90s → 48s per instrument)
- ✅ **Conditional computation**: Skip unused indicators based on weights

### Current Bottleneck

**Signal detection**: 60-833s per task (massive variance)
- Task granularity too coarse
- Queue starvation (fast tasks finish, workers idle waiting for slow ones)
- Only ~10% CPU utilization during signal phase

---

## Next Optimizations

### 1. Task Subdivision (High Priority)

**Problem:** Signal tasks too coarse-grained (60-833s) → queue starvation

**Solution:** Split by date ranges, on monthly granularity

```python
# Instead of:
run_signals_task(instrument="sp500", date_range=(2000, 2020))

# Split into:
for year_batch in [(2000,2005), (2005,2010), (2010,2015), (2015,2020)]:
    run_signals_task(instrument="sp500", date_range=year_batch)
```

**Expected gain:** 2-3x from better load balancing, 95% CPU utilization

### 2. Ray for Better Scheduling (Medium Priority)

**Why Ray:**

**Why Ray:**
- Scales from laptop to cluster
- Better task scheduling for variance (60-833s tasks)
- Built-in fault tolerance
- Dashboard for monitoring

**Use case:** Better scheduling for 83 signal tasks with high variance

**Expected gain:** 20-30% from smarter scheduling

### 3. Dask (Alternative to Ray)

**Pros:**
- Excellent Pandas/NumPy integration
- Dynamic task graphs
- Good for data-parallel workloads

**Expected gain:** Similar to Ray

---

## Distributed Architecture

### When to Consider Distribution

**Migrate when:**
1. Hypothesis suites take >6 hours on single machine
2. Need to test 10+ instruments × 100+ configs regularly
3. Budget allows for cloud compute costs

### Architecture Components

```
CLI (local) 
  → Orchestrator Pod (k8s)
    → Data Prep Job (k8s) → writes to persistent storage
    → Indicator Jobs (k8s, parallel) → read data, write indicators
    → Signal Jobs (k8s, parallel) → read data + indicators, write signals
    → Simulation Jobs (k8s, parallel) → read signals, write results
    → Aggregator Job (k8s) → read all results, generate reports
  → Download results (local)
```

**1. Orchestrator Service**
- Runs as Kubernetes Deployment (1 replica)
- Builds TaskGraph from CLI args
- Submits Kubernetes Jobs for each task
- Monitors job completion and schedules dependent tasks

**2. Task Workers**
- Kubernetes Jobs (one per task)
- Ephemeral pods that run a single task
- Read inputs from persistent storage
- Write outputs to persistent storage
- Exit after task completion

**3. Persistent Storage**
- Kubernetes PersistentVolumes (ReadWriteMany)
- Stores: raw data cache, indicator cache, signal results, simulation outputs
- All tasks read/write through persistent storage

**4. Result Aggregator**
- Kubernetes Job triggered after all tasks complete
- Generates comparison charts, CSVs, analysis reports

**5. Centralized Cache (Redis)**
- Shared cache for indicator results across all workers
- Cache key: `<instrument>:<indicator_type>:<params_hash>`
- Dramatically reduces computation time in grid searches

---

## Technology Stack

### Compute: Kubernetes

**Why K8s:**
- Industry-standard orchestration
- Horizontal scaling (10s-100s of workers)
- Elastic scaling (spin up workers on demand)
- Fault tolerance (automatic job rescheduling)
- Built-in retry mechanisms and event tracking
- Local and cloud parity (kind for local, GKE/EKS/AKS for cloud)

### Job Execution: Kubernetes Jobs

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: signals-sp500-config1
spec:
  backoffLimit: 3  # Retry up to 3 times on failure
  template:
    spec:
      containers:
      - name: worker
        image: trading-worker:latest
        env:
        - name: TASK_TYPE
          value: "signals"
        - name: INSTRUMENT
          value: "sp500"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        volumeMounts:
        - name: shared-storage
          mountPath: /mnt/shared
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
      volumes:
      - name: shared-storage
        persistentVolumeClaim:
          claimName: trading-pvc
      restartPolicy: Never
```

**Each task = separate Job manifest**
- ✅ Can use different languages per task type (Python, Rust, etc.)
- ✅ Resource limits per task (CPU, memory)
- ✅ Isolated failures (one job failure doesn't crash others)
- ✅ K8s handles retries automatically (no custom DLQ needed)

### Observability: OpenTelemetry + Prometheus + Grafana

**Metrics: OpenTelemetry → Prometheus**

```python
from opentelemetry import metrics

meter = metrics.get_meter(__name__)
task_duration = meter.create_histogram("task.duration", unit="s")

@task_duration.time()
def run_signals_task(...):
    ...
```

**Visualization: Grafana**
- Dashboards for task duration, queue depth, resource usage
- Alerts for failed jobs
- Integration with Prometheus

**Logs: Structured JSON → Loki**

```python
import structlog

log = structlog.get_logger()
log.info("task.started", task_id=task.id, instrument=instrument)
```

---

## Data Flow & Caching Strategy

### Local Caching (Current)

**File-based cache:**
- Location: `~/.cache/trading/orchestration/`
- Cache key: SHA256(instrument + indicator_params + config_content)
- Problem: Not shared across workers, 0% cache hit rate in distributed setup

### Centralized Caching (Future - Redis)

**Why Redis:**
- In-memory cache, <1ms latency
- Shared across all workers
- Automatic eviction policies (LRU, TTL)
- Supports persistence (RDB/AOF) for restore after restart

**Cache Structure:**

```python
import redis
import pickle
import hashlib

class DistributedCache:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
    
    def _cache_key(self, instrument: str, indicator_type: str, params: dict) -> str:
        """Generate cache key: <instrument>:<indicator>:<params_hash>"""
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
        return f"{instrument}:{indicator_type}:{params_hash}"
    
    def get_indicators(self, instrument: str, indicator_type: str, params: dict):
        """Retrieve cached indicator results."""
        key = self._cache_key(instrument, indicator_type, params)
        data = self.redis.get(key)
        if data:
            return pickle.loads(data)
        return None
    
    def set_indicators(self, instrument: str, indicator_type: str, params: dict, 
                      results: pd.DataFrame, ttl: int = 86400):
        """Cache indicator results with TTL (default 24 hours)."""
        key = self._cache_key(instrument, indicator_type, params)
        self.redis.setex(key, ttl, pickle.dumps(results))
```

**Usage in Worker:**

```python
def run_indicators_task(instrument: str, params: dict):
    cache = DistributedCache(os.environ['REDIS_URL'])
    
    # Try cache first
    cached = cache.get_indicators(instrument, 'rsi', params)
    if cached is not None:
        log.info("cache.hit", instrument=instrument, indicator="rsi")
        return cached
    
    # Compute
    log.info("cache.miss", instrument=instrument, indicator="rsi")
    results = calculate_rsi(data, params['period'])
    
    # Store in cache
    cache.set_indicators(instrument, 'rsi', params, results)
    
    return results
```

**Expected Impact:**
- Grid search with 100 configs × 10 instruments
- Without cache: 1000 indicator calculations
- With cache: ~10 indicator calculations (one per instrument), 990 cache hits
- **Speedup: 100x for indicator phase**

---

## Storage Architecture

### Kubernetes PersistentVolumes (Recommended)

**Storage Class:**
- Local (kind): `hostPath` or `local-path-provisioner`
- Cloud (GKE): `pd-ssd` (Google Cloud SSD Persistent Disk)
- Cloud (EKS): `gp3` (AWS EBS General Purpose SSD)
- Cloud (AKS): `managed-premium` (Azure Premium SSD)

**PersistentVolume Claim:**

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: trading-pvc
spec:
  accessModes:
    - ReadWriteMany  # Multiple pods can mount
  storageClassName: pd-ssd  # or gp3, managed-premium
  resources:
    requests:
      storage: 100Gi
```

**Storage Layout:**

```
/mnt/shared/
  data/
    raw/
      sp500.csv
      djia.csv
    processed/
      sp500_daily.parquet
      sp500_weekly.parquet
  cache/
    indicators/
      <instrument>_<indicator>_<hash>.pkl
  results/
    <run_id>/
      <config>/
        <instrument>_signals.pkl
        <instrument>_results.csv
  configs/
    baseline.yaml
    grid_search/
      config1.yaml
      config2.yaml
```

### Redis Deployment (for Caching)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        args:
          - --maxmemory 8gb
          - --maxmemory-policy allkeys-lru
          - --save 900 1  # Persist to disk every 15 min if ≥1 key changed
        volumeMounts:
        - name: redis-storage
          mountPath: /data
      volumes:
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
spec:
  ports:
  - port: 6379
  selector:
    app: redis
```

**Redis PVC for Persistence:**

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: redis-pvc
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: pd-ssd
  resources:
    requests:
      storage: 20Gi
```

---

## Orchestration: Custom Controller vs Argo Workflows

### Option A: Custom Python Controller

**Architecture:**

```python
from kubernetes import client, config

class K8sOrchestrator:
    def __init__(self, namespace="trading"):
        config.load_incluster_config()
        self.batch_api = client.BatchV1Api()
        self.namespace = namespace
    
    def submit_task(self, task: TaskNode) -> str:
        """Create Kubernetes Job for task."""
        job_manifest = self._build_job_manifest(task)
        self.batch_api.create_namespaced_job(self.namespace, job_manifest)
        return job_manifest.metadata.name
    
    def _build_job_manifest(self, task: TaskNode):
        """Convert TaskNode to K8s Job YAML."""
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": f"task-{task.id}"},
            "spec": {
                "backoffLimit": 3,
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "worker",
                            "image": "trading-worker:latest",
                            "env": [
                                {"name": "TASK_TYPE", "value": task.task_type},
                                {"name": "TASK_PAYLOAD", "value": json.dumps(task.payload)},
                                {"name": "REDIS_URL", "value": "redis://redis-service:6379"},
                            ],
                            "volumeMounts": [{"name": "shared", "mountPath": "/mnt/shared"}],
                        }],
                        "volumes": [{"name": "shared", "persistentVolumeClaim": {"claimName": "trading-pvc"}}],
                        "restartPolicy": "Never",
                    }
                }
            }
        }
    
    def watch_job_completion(self, job_name: str) -> bool:
        """Poll job status until completion or failure."""
        w = watch.Watch()
        for event in w.stream(self.batch_api.list_namespaced_job, self.namespace):
            job = event['object']
            if job.metadata.name == job_name:
                if job.status.succeeded:
                    return True
                if job.status.failed:
                    return False
    
    def execute_graph(self, graph: TaskGraph):
        """
        Execute TaskGraph on K8s:
        1. Topological sort
        2. Submit jobs for level 0 (no deps)
        3. Wait for level to complete
        4. Submit jobs for level 1 (deps satisfied), etc.
        """
        levels = graph.get_topological_levels()
        for level in levels:
            job_names = []
            for task_id in level:
                task = graph.get_task(task_id)
                job_name = self.submit_task(task)
                job_names.append(job_name)
            
            # Wait for all jobs in level to complete
            for job_name in job_names:
                success = self.watch_job_completion(job_name)
                if not success:
                    raise RuntimeError(f"Job {job_name} failed")
```

**Interaction with Orchestrator:**

**Option 1: REST API on Orchestrator**

```python
# Orchestrator exposes REST API
from fastapi import FastAPI
app = FastAPI()

@app.post("/submit")
def submit_job(config_dir: str):
    graph = build_grid_search_task_graph(config_dir)
    orchestrator.execute_graph(graph)
    return {"status": "submitted", "run_id": graph.id}

@app.get("/status/{run_id}")
def get_status(run_id: str):
    return orchestrator.get_run_status(run_id)
```

**From local:**
```bash
curl -X POST http://orchestrator-service:8080/submit \
  -d '{"config_dir": "/mnt/shared/configs/grid_search/"}'
```

**Pros:**
- Full control over scheduling logic
- Tight integration with existing TaskGraph code
- No new dependencies
- Lightweight

**Cons:**
- Must implement:
  - Job monitoring and retry logic
  - Progress tracking
  - Result aggregation
  - Web UI (optional)

---

### Option B: Argo Workflows

**What is Argo Workflows?**

Argo Workflows is a Kubernetes-native workflow engine for orchestrating parallel jobs. It's widely used in ML/data pipelines and CI/CD.

**Key Features:**
- DAG-based workflow definition (like your TaskGraph)
- Built-in retry, timeout, cron scheduling
- Web UI for monitoring
- Artifact passing between steps
- Event-driven triggers

**Architecture:**

Argo Workflows consists of:
1. **Workflow Controller**: K8s controller that watches Workflow CRDs
2. **Argo Server**: Web UI and REST API
3. **Workflow CRD**: Custom resource defining the DAG

**Example Workflow:**

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: grid-search-20260207
spec:
  entrypoint: main
  arguments:
    parameters:
    - name: config-dir
      value: "/mnt/shared/configs/grid_search/"
  
  templates:
  # Main DAG
  - name: main
    dag:
      tasks:
      # Level 0: Data prep
      - name: data-prep
        template: data-task
      
      # Level 1: Indicators (parallel, after data)
      - name: indicators-sp500
        dependencies: [data-prep]
        template: indicator-task
        arguments:
          parameters:
          - name: instrument
            value: "sp500"
      
      - name: indicators-djia
        dependencies: [data-prep]
        template: indicator-task
        arguments:
          parameters:
          - name: instrument
            value: "djia"
      
      # Level 2: Signals (parallel, after indicators)
      - name: signals-sp500-config1
        dependencies: [indicators-sp500]
        template: signal-task
        arguments:
          parameters:
          - name: instrument
            value: "sp500"
          - name: config
            value: "config1.yaml"
      
      # Level 3: Aggregation (after all signals)
      - name: aggregate
        dependencies: [signals-sp500-config1, ...]
        template: aggregate-task
  
  # Task templates
  - name: data-task
    container:
      image: trading-worker:latest
      command: [python, worker.py]
      env:
      - name: TASK_TYPE
        value: "data"
      volumeMounts:
      - name: shared
        mountPath: /mnt/shared
  
  - name: indicator-task
    inputs:
      parameters:
      - name: instrument
    container:
      image: trading-worker:latest
      command: [python, worker.py]
      env:
      - name: TASK_TYPE
        value: "indicators"
      - name: INSTRUMENT
        value: "{{inputs.parameters.instrument}}"
      - name: REDIS_URL
        value: "redis://redis-service:6379"
      volumeMounts:
      - name: shared
        mountPath: /mnt/shared
  
  volumeClaimTemplates:
  - metadata:
      name: shared
    spec:
      accessModes: [ReadWriteMany]
      storageClassName: pd-ssd
      resources:
        requests:
          storage: 100Gi
```

**Submitting Workflow:**

```bash
# From local machine
argo submit workflow.yaml -n trading

# Or via REST API
curl -X POST http://argo-server:2746/api/v1/workflows/trading \
  -H "Content-Type: application/json" \
  -d @workflow.json
```

**Monitoring:**

```bash
# CLI
argo list -n trading
argo get grid-search-20260207 -n trading
argo logs grid-search-20260207 -n trading

# Web UI
open http://argo-server:2746
```

**Programmatic Generation:**

Instead of hand-writing YAML, generate from your TaskGraph:

```python
def taskgraph_to_argo_workflow(graph: TaskGraph) -> dict:
    """Convert TaskGraph to Argo Workflow manifest."""
    tasks = []
    for level in graph.get_topological_levels():
        for task_id in level:
            task_node = graph.get_task(task_id)
            dependencies = [dep for dep in task_node.dependencies]
            
            tasks.append({
                "name": task_id,
                "dependencies": dependencies,
                "template": f"{task_node.task_type}-task",
                "arguments": {
                    "parameters": [
                        {"name": k, "value": v} 
                        for k, v in task_node.payload.items()
                    ]
                }
            })
    
    return {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "Workflow",
        "metadata": {"name": f"grid-search-{graph.id}"},
        "spec": {
            "entrypoint": "main",
            "templates": [
                {"name": "main", "dag": {"tasks": tasks}},
                # ... task templates ...
            ]
        }
    }
```

**Pros:**
- Production-ready DAG execution engine
- Built-in retry, timeout, parallelism
- Excellent Web UI for monitoring
- Widely adopted (CNCF project)
- Event-driven (can trigger on K8s events, webhooks, cron)
- Artifact management (pass data between tasks)

**Cons:**
- Learning curve (YAML can be verbose)
- Additional dependency (Argo controller + server)
- Less control over scheduling logic than custom controller

---

### Recommendation: Hybrid Approach

**Phase 1 (MVP):** Custom Python Controller
- Reuse existing TaskGraph code
- Lightweight, no new dependencies
- Expose simple REST API for job submission

**Phase 2 (Production):** Migrate to Argo Workflows
- Generate Argo Workflows from TaskGraph
- Leverage Argo's monitoring, retry, and event-driven features
- Use Argo Web UI for observability

**Interaction Model (both approaches):**

```python
# Local machine: Submit via REST API
import requests

response = requests.post("http://orchestrator:8080/submit", json={
    "config_dir": "/configs/grid_search/",
    "instruments": ["sp500", "djia"],
    "run_id": "20260207-test"
})

run_id = response.json()["run_id"]

# Poll status
while True:
    status = requests.get(f"http://orchestrator:8080/status/{run_id}").json()
    if status["state"] in ["completed", "failed"]:
        break
    time.sleep(10)

# Download results
results = requests.get(f"http://orchestrator:8080/results/{run_id}").json()
```

---

## Bootstrap & Restore Mechanism

### Goal

Enable rapid infrastructure spin-up → validate hypothesis → tear down, with persistent cache and results.

### Architecture

**Persistent Storage (survives infrastructure deletion):**
- Cloud object storage: AWS S3, GCP GCS, Azure Blob
- Contains: Redis snapshots, result archives, indicator cache

**Ephemeral Compute (deleted after use):**
- K8s cluster with worker nodes
- Redis pod (loads snapshot from S3 on startup)
- Orchestrator pod

### Workflow

**1. Bootstrap (Spin Up)**

```bash
# From local machine
./scripts/bootstrap.sh

# What it does:
# 1. Create K8s cluster (kind locally, or GKE/EKS in cloud)
# 2. Deploy PersistentVolume
# 3. Deploy Redis, restore snapshot from S3
# 4. Deploy orchestrator
# 5. Sync configs to /mnt/shared/configs/
```

**bootstrap.sh:**

```bash
#!/bin/bash
set -e

CLUSTER_NAME="trading-dev"
S3_BUCKET="trading-cache-backup"

# 1. Create cluster
if [ "$ENV" = "local" ]; then
    kind create cluster --name $CLUSTER_NAME
else
    # Cloud: GKE example
    gcloud container clusters create $CLUSTER_NAME \
      --zone us-central1-a \
      --num-nodes 3 \
      --machine-type n1-standard-4 \
      --enable-autoscaling --min-nodes 1 --max-nodes 10
fi

# 2. Deploy storage
kubectl apply -f k8s/storage.yaml

# 3. Restore Redis snapshot from S3
aws s3 cp s3://$S3_BUCKET/redis-dump.rdb /tmp/redis-dump.rdb
kubectl cp /tmp/redis-dump.rdb redis-0:/data/dump.rdb

# 4. Deploy Redis
kubectl apply -f k8s/redis.yaml

# 5. Deploy orchestrator
kubectl apply -f k8s/orchestrator.yaml

# 6. Wait for pods to be ready
kubectl wait --for=condition=ready pod -l app=redis --timeout=300s
kubectl wait --for=condition=ready pod -l app=orchestrator --timeout=300s

# 7. Sync configs
kubectl cp configs/ orchestrator-pod:/mnt/shared/configs/

echo "✅ Cluster ready. Submit jobs via:"
echo "  kubectl exec orchestrator-pod -- python submit.py --config /mnt/shared/configs/baseline.yaml"
```

**2. Run Hypothesis**

```bash
# Submit grid search
kubectl exec orchestrator-pod -- python submit.py \
  --config-dir /mnt/shared/configs/grid_search/ \
  --run-id h7-mtf-certainty

# Monitor
argo get h7-mtf-certainty -n trading --watch
```

**3. Backup Results & Cache**

```bash
# After grid search completes
./scripts/backup.sh h7-mtf-certainty

# What it does:
# 1. Backup Redis snapshot to S3
# 2. Archive results to S3
# 3. Update manifest (what's in S3)
```

**backup.sh:**

```bash
#!/bin/bash
set -e

RUN_ID=$1
S3_BUCKET="trading-cache-backup"

# 1. Trigger Redis save
kubectl exec redis-0 -- redis-cli SAVE

# 2. Copy Redis snapshot to S3
kubectl cp redis-0:/data/dump.rdb /tmp/redis-dump.rdb
aws s3 cp /tmp/redis-dump.rdb s3://$S3_BUCKET/redis-dump.rdb

# 3. Archive results
kubectl exec orchestrator-pod -- tar -czf /tmp/results-$RUN_ID.tar.gz /mnt/shared/results/$RUN_ID/
kubectl cp orchestrator-pod:/tmp/results-$RUN_ID.tar.gz /tmp/
aws s3 cp /tmp/results-$RUN_ID.tar.gz s3://$S3_BUCKET/results/

# 4. Update manifest
echo "$RUN_ID: $(date)" >> manifest.txt
aws s3 cp manifest.txt s3://$S3_BUCKET/manifest.txt

echo "✅ Backed up to s3://$S3_BUCKET/"
```

**4. Teardown**

```bash
# Delete cluster
./scripts/teardown.sh

# What it does:
# 1. Ensure backup completed
# 2. Delete K8s cluster
# 3. Cache and results persist in S3
```

**teardown.sh:**

```bash
#!/bin/bash
set -e

CLUSTER_NAME="trading-dev"

# Confirm backup exists
aws s3 ls s3://$S3_BUCKET/redis-dump.rdb || { echo "❌ No backup found!"; exit 1; }

# Delete cluster
if [ "$ENV" = "local" ]; then
    kind delete cluster --name $CLUSTER_NAME
else
    gcloud container clusters delete $CLUSTER_NAME --zone us-central1-a --quiet
fi

echo "✅ Cluster deleted. Cache persists in S3."
```

**5. Next Hypothesis (Restore)**

```bash
# Bootstrap again (restores cache from S3)
./scripts/bootstrap.sh

# Redis now has all previously cached indicators
# New grid search will hit cache immediately
```

### Storage Cost Optimization

**AWS S3 Lifecycle Policy:**

```json
{
  "Rules": [
    {
      "Id": "Archive old results",
      "Status": "Enabled",
      "Prefix": "results/",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ]
    }
  ]
}
```

**Redis Snapshot Compression:**

```bash
# Compress Redis snapshot before upload
gzip /tmp/redis-dump.rdb
aws s3 cp /tmp/redis-dump.rdb.gz s3://$S3_BUCKET/redis-dump.rdb.gz

# On restore
aws s3 cp s3://$S3_BUCKET/redis-dump.rdb.gz /tmp/
gunzip /tmp/redis-dump.rdb.gz
```

**Estimated Costs (AWS example):**
- S3 storage: ~$0.023/GB/month (Standard)
- 20 GB Redis snapshot: $0.46/month
- 100 GB results archive: $2.30/month
- **Total:** ~$3/month to persist cache between hypothesis tests

---

## Image Build & Distribution

### Local Development

**Build images:**

```bash
# Orchestrator
docker build -t trading-orchestrator:latest orchestrator/

# Worker
docker build -t trading-worker:latest worker/

# Load into kind (local K8s)
kind load docker-image trading-orchestrator:latest --name trading
kind load docker-image trading-worker:latest --name trading
```

**Image locations:**
- Local Docker daemon
- Loaded into kind cluster

### Cloud Setup

**Build & push to registry:**

```bash
# Set registry (GCR example)
REGISTRY="gcr.io/my-project"

# Build
docker build -t $REGISTRY/trading-orchestrator:latest orchestrator/
docker build -t $REGISTRY/trading-worker:latest worker/

# Push
docker push $REGISTRY/trading-orchestrator:latest
docker push $REGISTRY/trading-worker:latest
```

**In K8s manifests:**

```yaml
spec:
  containers:
  - name: worker
    image: gcr.io/my-project/trading-worker:latest
    imagePullPolicy: Always  # Cloud: always pull latest
```

**Image locations:**
- GCR (Google Container Registry)
- ECR (AWS Elastic Container Registry)
- ACR (Azure Container Registry)
- Docker Hub

### CI/CD (Future)

**GitHub Actions example:**

```yaml
name: Build and Push Images
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build orchestrator
      run: docker build -t ${{ secrets.REGISTRY }}/trading-orchestrator:${{ github.sha }} orchestrator/
    
    - name: Build worker
      run: docker build -t ${{ secrets.REGISTRY }}/trading-worker:${{ github.sha }} worker/
    
    - name: Push images
      run: |
        echo ${{ secrets.REGISTRY_PASSWORD }} | docker login -u ${{ secrets.REGISTRY_USER }} --password-stdin
        docker push ${{ secrets.REGISTRY }}/trading-orchestrator:${{ github.sha }}
        docker push ${{ secrets.REGISTRY }}/trading-worker:${{ github.sha }}
```

---

## Migration Path

### Current: Phase 1 Complete ✅

- TaskGraph + ProcessPoolExecutor
- Vectorization (50% speedup)
- Local file-based cache
- Checkpoint/resume

### Phase 2: Single-Machine Optimization (1-2 weeks)

1. **Task subdivision** for better load balancing
2. **Ray or Dask** for smarter scheduling
3. Target: 95% CPU utilization
4. **Expected gain:** 2-3x overall speedup

### Phase 3: Centralized Caching (1 week)

1. Deploy Redis for shared indicator cache
2. Update workers to use Redis
3. **Expected gain:** 10-100x for hypothesis suites (indicator reuse)

### Phase 4: Kubernetes Migration (2-3 weeks)

**Only when:**
- Hypothesis suites take >6 hours after Phase 2-3
- Regular testing of 10+ instruments × 100+ configs
- Budget for cloud compute

**Steps:**
1. Containerize (orchestrator + worker images)
2. Deploy to kind (local testing)
3. Implement custom controller with REST API
4. Deploy to cloud (GKE/EKS/AKS)
5. Set up observability (Prometheus + Grafana)
6. (Optional) Migrate to Argo Workflows

---

## References

### Current Implementation
- TaskGraph: `core/orchestration/task_graph.py`
- Executor: `core/orchestration/executor.py`
- Cache: `core/orchestration/cache.py`
- Checkpoint: `core/orchestration/checkpoint.py`

### Future Technologies
- **Ray**: https://docs.ray.io/
- **Dask**: https://docs.dask.org/
- **Redis**: https://redis.io/
- **Argo Workflows**: https://argoproj.github.io/workflows
- **Kubernetes**: https://kubernetes.io/

---

## Summary

**Current:** Single-machine with vectorization (50% speedup)

**Next steps:**
1. Task subdivision + Ray/Dask → 2-3x gain
2. Redis centralized cache → 10-100x gain for hypothesis suites
3. K8s only when single-machine exhausted

**Key principle:** Optimize locally first. Distribute only when scale demands it.

**Bootstrap model (future):** Rapid spin-up → test hypothesis → tear down, with persistent cache in S3.
