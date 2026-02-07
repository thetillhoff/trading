# GPU Acceleration for Trading Strategy Backtesting

**Status**: Design phase - Not yet implemented

This document outlines GPU acceleration strategies for compute-intensive components of the trading backtesting system.

---

## Table of Contents

1. [Use Cases for GPU Computation](#use-cases-for-gpu-computation)
2. [Technology Options](#technology-options)
3. [Platform-Specific Considerations](#platform-specific-considerations)
4. [Implementation Strategy](#implementation-strategy)
5. [Expected Performance Gains](#expected-performance-gains)
6. [Migration Path](#migration-path)

---

## Use Cases for GPU Computation

### Suitable for GPU Acceleration

**1. Technical Indicators (High Priority)**
- EMA, RSI, MACD calculations (vectorized operations on large arrays)
- Rolling window computations (moving averages, standard deviations)
- Expected speedup: 5-10x for bulk indicator calculation

**2. Elliott Wave Extrema Detection**
- Peak/valley finding on large price arrays
- Convolution operations for pattern matching
- Expected speedup: 3-5x

**3. Multi-Timeframe Analysis (MTF)**
- **Yes, suitable for GPU**: MTF involves:
  - Resampling daily data to weekly (downsampling operation)
  - Computing EMA on resampled data (rolling window operation)
  - Broadcasting weekly trend back to daily timeframe (upsampling)
- All these operations are parallelizable and array-based
- Expected speedup: 2-4x for MTF confirmation computation

**4. Monte Carlo Simulations** (if implemented in future)
- Run 1000s of simulations in parallel
- Perfect for GPU parallelism
- Expected speedup: 50-100x

### Not Suitable for GPU

- Elliott Wave pattern matching (logic-heavy, branching)
- Configuration parsing, file I/O
- Result aggregation, report generation
- Signal generation logic (too much branching)

---

## Technology Options

### Option A: PyTorch (Recommended)

```python
import torch

# Automatic device selection with CPU fallback
device = torch.device("cuda" if torch.cuda.is_available() 
                     else "mps" if torch.backends.mps.is_available()
                     else "cpu")

def calculate_ema(prices: pd.Series, period: int):
    """GPU-accelerated EMA with automatic CPU fallback."""
    prices_tensor = torch.tensor(prices.values, device=device, dtype=torch.float32)
    
    alpha = 2 / (period + 1)
    ema = torch.zeros_like(prices_tensor)
    ema[0] = prices_tensor[0]
    
    for i in range(1, len(prices_tensor)):
        ema[i] = alpha * prices_tensor[i] + (1 - alpha) * ema[i - 1]
    
    return ema.cpu().numpy()
```

**Pros:**
- Works on NVIDIA (CUDA), AMD (ROCm), and Apple Silicon (MPS)
- Automatic CPU fallback
- Mature ecosystem
- Good Docker support

**Cons:**
- Heavier dependency
- Some learning curve

**Verdict**: ✅ Best choice for cross-platform GPU support

### Option B: CuPy (NVIDIA-only)

```python
import cupy as cp

# Drop-in NumPy replacement
arr = cp.array([1, 2, 3])
result = cp.sum(arr)
```

**Pros:**
- Drop-in NumPy replacement
- Minimal code changes

**Cons:**
- NVIDIA GPUs only (CUDA required)
- No fallback mechanism

**Verdict**: ⚠️ Only if standardizing on NVIDIA hardware

### Option C: Numba + CUDA

```python
from numba import cuda

@cuda.jit
def gpu_function(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] = arr[idx] * 2
```

**Pros:**
- Fine-grained control
- Compiles Python to GPU kernels

**Cons:**
- Requires writing GPU kernels
- NVIDIA only
- High complexity

**Verdict**: ⚠️ Only for performance-critical hotspots

---

## Platform-Specific Considerations

### NVIDIA GPUs (CUDA)

**Docker:**
```dockerfile
FROM nvidia/cuda:12.2-runtime-ubuntu22.04
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu122
```

**Run:**
```bash
docker run --gpus all trading-worker:latest
```

**Cloud:** AWS (p3/p4), GCP (A100), Azure (NC-series)

### AMD GPUs (ROCm)

**Docker:**
```dockerfile
FROM rocm/pytorch:rocm5.7_ubuntu22.04_py3.10_pytorch_2.0.1
```

**Run:**
```bash
docker run --device=/dev/kfd --device=/dev/dri trading-worker:latest
```

**Cloud:** AWS (g4ad), Azure (NVv4)

### Apple Silicon (M1/M2/M3) - MPS

**PyTorch with MPS:**
```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

**Limitations:**
- ❌ Docker on macOS doesn't support GPU passthrough
- ✅ Run natively (not in Docker) for GPU acceleration
- ⚠️ MPS has limited operator support compared to CUDA

**Recommendation for macOS:**
- Use MPS for local development/testing
- Use cloud GPUs (AWS/GCP) for production workloads

### CPU-Only Fallback

Always maintain CPU fallback for:
- Machines without GPU
- Development on non-GPU machines
- CI/CD pipelines

```python
def calculate_indicators(data: pd.DataFrame, device: str = "auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert to tensor
    tensor = torch.tensor(data.values, device=device)
    
    # Compute
    result = compute_on_device(tensor)
    
    # Return to CPU/NumPy
    return result.cpu().numpy()
```

---

## Implementation Strategy

### Phase 1: Single Indicator Prototype

**Goal:** Validate GPU speedup for one indicator (RSI)

```python
# core/indicators/gpu_technical.py
import torch

class GPUTechnicalIndicators:
    def __init__(self, device="auto"):
        self.device = self._get_device(device)
    
    def _get_device(self, device):
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        return "cpu"
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """GPU-accelerated RSI calculation."""
        prices_tensor = torch.tensor(prices.values, device=self.device, dtype=torch.float32)
        
        # Calculate price changes
        deltas = prices_tensor[1:] - prices_tensor[:-1]
        
        # Separate gains and losses
        gains = torch.clamp(deltas, min=0)
        losses = torch.clamp(-deltas, min=0)
        
        # Calculate rolling average
        avg_gains = self._rolling_mean(gains, period)
        avg_losses = self._rolling_mean(losses, period)
        
        # RSI calculation
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return pd.Series(rsi.cpu().numpy(), index=prices.index[period:])
```

**Benchmark:**
```python
# Compare CPU vs GPU
import time

# CPU
start = time.time()
rsi_cpu = calculate_rsi_cpu(prices)
cpu_time = time.time() - start

# GPU
start = time.time()
rsi_gpu = calculate_rsi_gpu(prices)
gpu_time = time.time() - start

print(f"CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s, Speedup: {cpu_time/gpu_time:.1f}x")
```

**Decision criteria:**
- If speedup >2x on representative dataset → proceed to Phase 2
- If speedup <2x → GPU not worth complexity, abort

### Phase 2: Batch Indicator Computation

**Goal:** Compute all indicators for multiple instruments in parallel

```python
def calculate_all_indicators_batch(instruments: List[str], 
                                   data_dict: Dict[str, pd.DataFrame]) -> Dict:
    """
    Compute indicators for all instruments on GPU in parallel.
    
    GPU can process 10-100 instruments simultaneously.
    """
    gpu_indicators = GPUTechnicalIndicators(device="auto")
    
    results = {}
    for instrument in instruments:
        data = data_dict[instrument]
        results[instrument] = {
            'rsi': gpu_indicators.calculate_rsi(data['Close']),
            'ema_short': gpu_indicators.calculate_ema(data['Close'], 20),
            'ema_long': gpu_indicators.calculate_ema(data['Close'], 50),
            'macd': gpu_indicators.calculate_macd(data['Close']),
        }
    
    return results
```

### Phase 3: MTF on GPU

**Goal:** Accelerate multi-timeframe confirmation

```python
def compute_mtf_on_gpu(daily_data: pd.DataFrame, weekly_ema_period: int = 8):
    """
    GPU-accelerated multi-timeframe analysis.
    
    Steps:
    1. Resample daily to weekly (downsampling)
    2. Compute weekly EMA on GPU
    3. Broadcast weekly trend back to daily timeframe
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert to tensor
    daily_close = torch.tensor(daily_data['Close'].values, device=device)
    
    # Resample to weekly (group by week)
    weekly_indices = daily_data.resample('W').last().index
    weekly_close = daily_data['Close'].resample('W').last()
    weekly_tensor = torch.tensor(weekly_close.values, device=device)
    
    # Compute weekly EMA on GPU
    weekly_ema = compute_ema_gpu(weekly_tensor, weekly_ema_period)
    
    # Broadcast back to daily timeframe
    daily_ema = torch.repeat_interleave(weekly_ema, repeats=5)[:len(daily_close)]
    
    # MTF confirmation: daily close vs weekly EMA
    mtf_bullish = daily_close >= daily_ema
    
    return mtf_bullish.cpu().numpy()
```

### Phase 4: Integration with Orchestration

**Goal:** Seamlessly use GPU workers in K8s

```yaml
# k8s/worker-gpu.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: indicators-gpu-sp500
spec:
  template:
    spec:
      containers:
      - name: worker
        image: trading-worker:gpu
        resources:
          limits:
            nvidia.com/gpu: 1  # Request 1 GPU
        env:
        - name: TASK_TYPE
          value: "indicators"
        - name: DEVICE
          value: "cuda"
```

**Orchestrator logic:**
```python
def submit_indicator_task(instrument: str, use_gpu: bool = False):
    """Submit indicator calculation task, optionally using GPU worker."""
    if use_gpu:
        job_template = "worker-gpu.yaml"
        image = "trading-worker:gpu"
    else:
        job_template = "worker-cpu.yaml"
        image = "trading-worker:cpu"
    
    # Submit job
    ...
```

---

## Expected Performance Gains

### Realistic Estimates (based on typical GPU speedups for array operations)

| Component | Current (CPU) | With GPU | Speedup | Priority |
|-----------|---------------|----------|---------|----------|
| RSI calculation | 0.1s/instrument | 0.02s | 5x | High |
| EMA calculation | 0.05s/instrument | 0.01s | 5x | High |
| MACD calculation | 0.15s/instrument | 0.03s | 5x | High |
| Elliott Wave extrema | 1-2s/instrument | 0.3-0.6s | 3-4x | Medium |
| MTF confirmation | 0.5s/instrument | 0.15s | 3x | Medium |
| **Total per instrument** | **~48s** | **~30s** | **~1.6x** | - |

**Note:** Total speedup is less than individual components because:
- Not all computation is GPU-accelerated (signal logic remains CPU)
- Data transfer overhead (CPU ↔ GPU)
- Only worth it when processing many instruments in parallel

**Break-even point:** GPU acceleration becomes worthwhile when processing:
- 10+ instruments simultaneously (batch processing)
- 100+ configs in grid search (reuse GPU for all indicator calculations)

---

## Migration Path

### Step 1: Prototype & Benchmark (1-2 days)

1. Implement GPU RSI calculation
2. Benchmark on representative dataset (DJIA 2008-2012)
3. Measure speedup and data transfer overhead
4. **Decision point:** Proceed only if speedup >2x

### Step 2: Expand to All Indicators (1 week)

1. Implement GPU EMA, MACD, ADX
2. Batch processing for multiple instruments
3. Add comprehensive tests (GPU vs CPU correctness)
4. Profile memory usage (GPU RAM limits)

### Step 3: MTF Acceleration (3 days)

1. Implement GPU MTF resampling and EMA
2. Benchmark MTF speedup
3. Validate correctness against CPU version

### Step 4: Docker Images (2 days)

1. Create `trading-worker:gpu` image with CUDA
2. Create `trading-worker:rocm` image for AMD
3. Test on local GPU and cloud (AWS p3)

### Step 5: K8s Integration (1 week)

1. Deploy GPU node pool in K8s cluster
2. Create GPU job templates
3. Update orchestrator to use GPU workers for indicator tasks
4. Monitor GPU utilization and costs

### Step 6: Optimization (ongoing)

1. Profile GPU kernel usage
2. Minimize CPU ↔ GPU data transfers
3. Use GPU for multiple indicators in single kernel
4. Add auto-scaling for GPU nodes

---

## Cost-Benefit Analysis

### GPU Cloud Costs (AWS example, 2026 pricing estimates)

**p3.2xlarge (1x V100 GPU):**
- On-demand: ~$3.00/hour
- Spot: ~$0.90/hour
- 8 vCPUs, 61 GB RAM, 16 GB GPU RAM

**c5.4xlarge (CPU-only, equivalent compute):**
- On-demand: ~$0.68/hour
- Spot: ~$0.20/hour
- 16 vCPUs, 32 GB RAM

### Break-even Calculation

**Scenario:** Grid search with 100 configs × 10 instruments = 1000 indicator calculations

**CPU-only (c5.4xlarge):**
- Time: 1000 instruments × 48s = 48,000s = 13.3 hours
- Cost: 13.3 hours × $0.20 = **$2.66**

**GPU (p3.2xlarge):**
- Time: 1000 instruments × 30s = 30,000s = 8.3 hours
- Cost: 8.3 hours × $0.90 = **$7.47**

**Verdict:** GPU is **2.8x more expensive** for this workload. Only worthwhile if:
- Time is critical (need results in 8 hours instead of 13)
- Running very large grid searches (>1000 configs)
- Using reserved instances (lower GPU cost)

---

## Recommendations

### Phase 1: Don't Rush to GPU

Current CPU optimizations (vectorization, conditional computation) achieved 50% speedup. Focus on:
1. Task subdivision for better load balancing
2. Ray/Dask for better scheduling
3. Horizontal scaling (more CPU nodes) before vertical (GPU)

### Phase 2: Prototype GPU for Specific Cases

Only implement GPU acceleration when:
1. Processing 100+ instruments regularly
2. Running hypothesis suites with 1000+ configs
3. Time-to-results is critical
4. Budget allows for GPU costs

### Phase 3: Hybrid Approach

**Best strategy:**
- Use CPU workers for signal detection, simulation (logic-heavy)
- Use GPU workers for indicator calculation (array-heavy)
- Orchestrator intelligently routes tasks to appropriate workers

---

## References

- **PyTorch CUDA**: https://pytorch.org/docs/stable/cuda.html
- **PyTorch MPS**: https://pytorch.org/docs/stable/notes/mps.html
- **NVIDIA Docker**: https://github.com/NVIDIA/nvidia-docker
- **K8s GPU Operator**: https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/
- **ROCm**: https://rocmdocs.amd.com/

---

## Open Questions

1. **Data transfer overhead**: How much time is spent copying data CPU ↔ GPU?
   - Profile actual transfer times for typical DataFrames
   - May negate speedup for small datasets

2. **GPU memory limits**: How many instruments fit in GPU RAM simultaneously?
   - V100: 16 GB
   - A100: 40 GB or 80 GB
   - May need batching

3. **Mixed precision**: Can we use fp16 instead of fp32 for indicators?
   - 2x memory savings
   - 2x speedup on Tensor Cores
   - Need to validate accuracy impact

4. **Custom CUDA kernels**: Worth writing custom kernels for specific operations?
   - PyTorch may not have optimal implementations
   - Numba allows JIT compilation of Python to CUDA
   - High development cost, only for proven bottlenecks
