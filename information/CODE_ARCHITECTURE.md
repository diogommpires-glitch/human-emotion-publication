# Human Emotion - Code Architecture

## Overview

Human Emotion is a full-stack web application with:
- **Backend**: Python HTTP server with PyTorch spiking neural network simulation
- **Frontend**: Vanilla JavaScript web interface with Canvas visualizations
- **Simulation**: CUDA/GPU-accelerated adaptive exponential integrate-and-fire (AdEx) network

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  index.html │  │   app.js    │  │  styles.css │        │
│  └─────────────┘  └──────┬──────┘  └─────────────┘        │
└──────────────────────────┼────────────────────────────────┘
                           │ HTTP POST /api/simulate
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    Python Server                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                   server.py                           │  │
│  │  • HTTP endpoints (GET /api/info, POST /api/simulate)│  │
│  │  • Request validation                                  │  │
│  │  • Activity tracking & idle shutdown                   │  │
│  └──────────────────────┬────────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼────────────────────────────────┐  │
│  │                    model.py                            │  │
│  │  • Request parsing                                     │  │
│  │  • Memory fitting (GPU)                                │  │
│  │  • Reference run for mood                              │  │
│  │  • Response construction                               │  │
│  └──────────────────────┬────────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼────────────────────────────────┐  │
│  │                  simulator.py                          │  │
│  │  • Topology construction (CSR matrices)                │  │
│  │  • Main simulation loop (6000 steps × 1ms)             │  │
│  │  • Neuromodulator dynamics                            │  │
│  │  • Recording & output                                 │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ↓
                    ┌──────────────┐
                    │    GPU/CPU   │
                    │   PyTorch    │
                    └──────────────┘
```

---

## 2. File Structure

```
Human-Emotion/
├── brain_model/
│   ├── model.py          # Request handling, emotion readouts
│   ├── simulator.py       # Spiking network simulation
│   └── network.json      # Network configuration
├── web/
│   ├── index.html        # Web interface
│   ├── app.js            # Client-side JavaScript
│   └── styles.css        # Styling
├── tests/
│   └── check_scenarios.py # Validation tests
├── information/           # Documentation (generated)
│   ├── NEUROSCIENCE_ARCHITECTURE.md
│   ├── BIOPHYSICAL_MODELS.md
│   ├── CODE_ARCHITECTURE.md
│   ├── VALIDATION_REPORT.md
│   ├── ENHANCEMENT_ROADMAP.md
│   └── QUICK_START.md
├── server.py             # HTTP server
├── benchmark.py          # Performance benchmarking
├── requirements.txt      # Dependencies (PyTorch)
└── README.txt           # Project documentation
```

---

## 3. simulator.py - Core Simulation

### 3.1 CSR Connectivity Matrix Construction

**Key insight**: No global sort - fills CSR arrays directly by computing nnz per row first.

```python
def _csr(parts, n, device, generator, jitter, index_dtype):
    # Step 1: Accumulate nnz per row
    starts = torch.zeros(n + 1, dtype=int64)
    for _, dst_start, dst_count, _, _, k, _ in parts:
        starts[dst_start] += k
        starts[dst_start + dst_count] -= k
    
    # Step 2: Prefix sum → row pointers
    per_row = torch.cumsum(starts[:-1], 0)
    crow = torch.zeros(n + 1, dtype=int64)
    crow[1:] = torch.cumsum(per_row, 0)
    
    # Step 3: Allocate exact-size arrays
    nnz = int(crow[-1].item())
    col = torch.empty(nnz, dtype=index_dtype)
    val = torch.empty(nnz, dtype=float32)
    
    # Step 4: Fill per block
    fill = crow[:-1].clone()  # cursor per row
    for _, dst_start, dst_count, src_start, src_count, k, weight in parts:
        rows = slice(dst_start, dst_start + dst_count)
        positions = (fill[rows, None] + torch.arange(k, device=device)[None, :]).reshape(-1)
        src = torch.randint(src_count, (dst_count, k), ...) + src_start
        
        # Remove self-contacts
        own = torch.arange(dst_start, dst_start + dst_count, device=device)[:, None]
        src = torch.where(src == own, src_start + (src - src_start + 1) % src_count, src)
        
        col[positions] = src
        val[positions] = weight * (1 + jitter * torch.rand(nnz) - jitter/2)
        fill[rows] += k
```

**Performance**: 
- ~13M synapses for 2.1M cells
- ~500ms build time on RTX 5090
- int32 indices halve memory vs int64

### 3.2 Cell Parameters

**Per-cell parameter tensors** (14 named parameters, all float32):
```python
params = {
    'v_rest_mV', 'v_reset_mV', 'v_threshold_mV', 'delta_t_mV',
    'v_peak_mV', 'tau_m_ms', 'a', 'b_mV', 'tau_w_ms',
    'refractory_ms', 'threshold_jitter_mV', 
    'release_u', 'recovery_ms', 'facilitation_ms'
}
```

**Assignment**: Two slices per region:
- Projection cells: use region's `projection_cell_type`
- Interneurons: always `fast_spiking`

**Threshold jitter**:
```python
params['v_threshold_mV'] += params.pop('threshold_jitter_mV') * torch.randn(n)
```

### 3.3 Main Simulation Loop (6000 iterations × 1ms)

```python
def run(cfg, topology, inputs, seed, duration_s=6.0, record=True):
    # State variables (all device tensors)
    v = v_rest + 4 * (rand - 0.5)      # Membrane potential
    w = zeros(n)                          # Adaptation current
    g_ampa, g_nmda, g_gaba_a, g_gaba_b  # Conductances
    refractory = zeros(n, int16)          # Refractory countdown
    u = release_u.clone()                 # Facilitation variable
    x = ones(n)                          # Depression variable
    release = zeros(n)                    # Synaptic release
    ring = zeros((delay, n))             # Delay buffer
    
    for step in range(steps):
        # 1. Decay conductances
        g_ampa *= exp(-dt_ms / 5)        # AMPA decay
        g_nmda *= exp(-dt_ms / 100)     # NMDA decay
        g_gaba_a *= exp(-dt_ms / 8)     # GABA-A decay
        g_gaba_b *= exp(-dt_ms / 150)   # GABA-B decay
        
        # 2. Poisson external drive
        g_ampa += poisson(lam_stim if stimulus_on else lam_base)
        
        # 3. Neuromodulator update
        source_rates = rates[mod.source, 0]
        target = K * density * (source_rates / resting_hz)
        chem = target + (chem - target) * exp(-dt_ms / clearance_ms)
        occ = chem / (chem + K)
        d_occ = occ - occ_rest
        
        i_mod = excitability_scale * (excitability * d_occ).sum(1) + tonic_mV
        adapt_gain = clamp(1 + gain_scale * (adaptation * d_occ).sum(1), 0.2, 3.0)
        nmda_gain = clamp(1 + gain_scale * (nmda * d_occ).sum(1), 0.2, 3.0)
        
        # 4. Synaptic transmission
        ring[(step - 1) % delay] = release
        delayed = ring[(step - delay) % delay]
        incoming_e = spmm(local_glu, release) + spmm(long_glu, delayed)
        incoming_i = spmm(local_gaba, release) + spmm(long_gaba, delayed)
        g_ampa += incoming_e
        g_nmda += incoming_e * nmda_ratio * nmda_gain
        g_gaba_a += incoming_i
        g_gaba_b += incoming_i * gaba_b_ratio
        
        # 5. Membrane dynamics (AdEx)
        mg_block = 1 / (1 + 0.28 * exp(-0.062 * v))
        i_glu = (g_ampa + g_nmda * mg_block) * (E_glu - v)
        i_gaba = g_gaba_a * (E_gaba_a - v) + g_gaba_b * (E_gaba_b - v)
        upswing = delta_t * exp(clamp((v - v_th) / delta_t, max=10))
        dv = dt_over_tau * (v_rest - v + upswing - w + i_glu + i_gaba + i_mod)
        
        v_new = where(refractory == 0, v + dv, v_reset)
        spike = v_new >= v_peak
        v = where(spike, v_reset, v_new)
        w += w_rate * (a * (v - v_rest) - w)
        w += spike * b * adapt_gain
        refractory = where(spike, ref_steps, clamp(refractory - 1, min=0))
        
        # 6. Tsodyks-Markram plasticity
        x += (1 - x) * rec_rate
        u_spike = release_u + facilitating * (u + release_u * (1 - u) - release_u)
        released = spike * u_spike * x
        x -= released
        u = where(spike, u_spike, u)
        release = released / release_u
        
        # 7. Firing rates
        counts.zero_().index_add_(0, bin_of_cell, spike.float())
        rates += rate_alpha * (counts / (bin_sizes * dt) - rates)
        
        # 8. Recording
        if step % record_every == 0:
            rec_counts[step // record_every] = counts
            rec_occ[step // record_every] = occ
```

### 3.4 Neuromodulator System

```python
def _modulator_tables(cfg, topology, inputs):
    # Density matrix: (regions × modulators)
    density = zeros(R, num_mods)
    excitability = zeros(R, num_mods)
    adaptation = zeros(R, num_mods)
    nmda = zeros(R, num_mods)
    
    for m, mod in enumerate(mods):
        for region_id, value in mod['targets'].items():
            density[index[region_id], m] = value
        for region_id, effects in mod['effects'].items():
            excitability[index[region_id], m] = effects.get('excitability', 0)
            adaptation[index[region_id], m] = effects.get('adaptation', 0)
            nmda[index[region_id], m] = effects.get('nmda', 0)
    
    # Tonic factors (cortisol, adenosine, ghrelin)
    tonic = zeros(R)
    for factor in cfg['tonic_factors']:
        change = inputs[factor['input']] - defaults[factor['input']]
        for region_id, effect in factor['effects'].items():
            tonic[index[region_id]] += excitability_scale * effect * change
    
    return {
        'density': density,
        'excitability': excitability,
        'adaptation': adaptation,
        'nmda': nmda,
        'source': source_indices,
        'resting_hz': resting_firing,
        'clearance_ms': clearance_tau,
        'tonic_mV': tonic,
    }
```

---

## 4. model.py - Request Handling

### 4.1 Request Parsing

```python
def parse_request(payload, device):
    # Validate inputs
    inputs = {key: _unit(raw.get(key, default), key) 
              for key, default in DEFAULT_INPUTS.items()}
    
    # Apply person context
    if person := payload.get('person'):
        inputs['social_touch'] += 0.6 * warmth
        inputs['threat_cue'] += 0.6 * coldness
        inputs['novelty'] *= 1 - 0.5 * closeness
    
    # Memory fitting (GPU)
    neurons = _fit_to_memory(request['neurons'], device)
    
    return {"inputs", "neurons", "seed", "person", "event"}
```

### 4.2 Memory Fitting

```python
def _fit_to_memory(neurons, device):
    if device.type != 'cuda':
        return neurons
    
    sizes = sorted(s for s in _device_sizes(device)[0] if s <= neurons)
    free = torch.cuda.mem_get_info(device)[0]
    
    for size in reversed(sizes):
        if simulator.estimated_bytes(NETWORK, size) <= 0.9 * free:
            return size
    
    raise SimulationError("GPU has too little memory")
```

### 4.3 Emotion Readout Construction

```python
class _Activity:
    def rate(self, region):
        # log2 change in projection-cell firing
        ratio = (self.a['projection_hz'][i] + 0.5) / (self.b['projection_hz'][i] + 0.5)
        return max(-2.0, min(3.0, math.log2(ratio)))
    
    def chem(self, name, region):
        # 8x scaling: 0.1 occupancy ≈ doubling of firing
        return 8.0 * (self.a['occupancy'][i][m] - self.b['occupancy'][i][m])

def _components(act):
    return {
        'threat': 0.35*act.rate('cea') + 0.25*act.rate('pag') + ...,
        'anxiety': 0.50*act.rate('bnst') + 0.30*act.rate('dacc') + ...,
        # ... 10 components
    }

def _emotions(e, phasic, mood):
    logits = {
        'Fear-like': 1.2*threat + 0.6*arousal - 0.5*control,
        'Joy-like': 1.3*reward + 0.3*arousal - 0.6*threat - 0.5*loss,
        # ... 10 emotions
    }
    # Softmax ranking
    peak = max(logits.values())
    exps = {k: exp((v - peak) / 0.35) for k, v in logits.items()}
    total = sum(exps.values())
    ranked = sorted([{'label': k, 'score': 100*v/total} for k,v in exps.items()],
                    key=lambda x: x['score'], reverse=True)
    return ranked[:3], logits
```

---

## 5. server.py - HTTP Interface

### 5.1 Endpoints

```
GET  /__ping              → Signature check, keepalive
GET  /api/info             → Hardware info, network metadata
GET  /api/network          → network.json download
POST /api/simulate         → Run simulation
```

### 5.2 Security

```python
def _host_allowed(self):
    """Reject DNS rebinding attacks"""
    if self.headers.get('Host') in (f'{HOST}:{port}', f'localhost:{port}'):
        return True
    self._json({'error': 'Requests must be addressed to this computer'}, 403)
    return False
```

**Security layers**:
- DNS rebinding protection (Host header check)
- CORS denied (no cross-site POSTs)
- 64KB request body limit
- JSON-only content type

### 5.3 Idle Shutdown

```python
class Activity:
    started = time.time()
    last = None
    running = 0  # Simulations in progress
    
# Shutdown conditions:
# - No page opened + >180s → shutdown
# - No activity + >600s + no simulations → shutdown
# - Never shutdown while simulation running
```

---

## 6. Frontend (app.js)

### 6.1 State Management

```javascript
// Module-level state (no framework)
let memory = loadMemory();           // {people: [], episodes: []}
let latest = null;                  // Most recent simulation result
let pendingPhoto = null;            // New photo pending save
let modelInfo = null;               // /api/info response

// Flow:
// 1. load() → loadMemory() + renderMemory() + loadModelInfo()
// 2. run → POST /api/simulate → renderResult()
// 3. save → persistMemory() → localStorage
```

### 6.2 Visualization

**Spike Raster**:
```javascript
function drawSpikeRaster(result) {
    const layout = result.spikes.layout;
    const band = box.height / layout.length;
    
    // Binary search for cell → region mapping
    const regionOf = cell => {
        // offsets[] = cumulative cell counts per region
        let lo = 0, hi = offsets.length - 1;
        while (lo < hi) {
            const mid = (lo + hi + 1) >> 1;
            if (offsets[mid] <= cell) lo = mid;
            else hi = mid - 1;
        }
        return lo;
    };
    
    // Draw spikes as 1.2×1px rectangles
    for (const [timeMs, cell] of events) {
        const index = regionOf(cell);
        ctx.fillStyle = gaba ? '#bb624c' : '#536f5c';
        ctx.fillRect(x, y, 1.2, 1);
    }
}
```

**Chemistry Timecourse**:
```javascript
function drawChemistryChart(result) {
    // Line chart with auto-scaled Y-axis
    // Colors from CHEMICAL_COLORS map
    for (const [name, series] of Object.entries(series)) {
        ctx.beginPath();
        ctx.strokeStyle = CHEMICAL_COLORS[name];
        series.forEach((value, i) => {
            const x = ..., y = ...;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    }
}
```

### 6.3 Canvas Scaling (HiDPI)

```javascript
function canvasContext(canvas, fallbackHeight) {
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(canvas.clientWidth * ratio);
    canvas.height = Math.round(canvas.clientHeight * ratio);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { ctx, width: canvas.clientWidth, height: canvas.clientHeight };
}
```

---

## 7. Performance Characteristics

### 7.1 Memory Budget (per cell)

| Component | Bytes/Cell | Notes |
|-----------|-----------|-------|
| Named params (14 floats) | ~240 | v_rest, tau_m, etc. |
| Runtime state | ~260 | v, w, g_ampa, g_nmda, g_gaba, etc. |
| Delay buffer | ~20 | 5 × float32 |
| **Total** | **~520** | |

### 7.2 CSR Matrix Memory

```
Synapse count ≈ 13M for 2.1M cells
Per-synapse: int32 (4B) + float32 (4B) = 8B
Overhead: ~0.5B per synapse
Total: ~13M × 8.5B ≈ 110 MB
```

### 7.3 VRAM Requirements

| Cells | Synapses | VRAM |
|-------|----------|------|
| 16,384 | ~1.2M | ~1.25 GB |
| 65,536 | ~5M | ~1.5 GB |
| 262,144 | ~20M | ~2.8 GB |
| 1,048,576 | ~80M | ~8.0 GB |
| 2,097,152 | ~106M | ~12.0 GB |

### 7.4 Simulation Speed (RTX 5090, 2.1M cells)

- Build time: ~500ms
- Step time: ~8-10ms
- Full 6s run: ~50-60 seconds
- Real-time ratio: ~8-9x slower than realtime

---

## 8. Data Flow

```
User Input (sliders)
       ↓
network.json (inputs)
       ↓
simulator.run() - 6000 × 1ms steps
       ↓
Spike trains per region
       ↓
Regional firing rates (Hz)
       ↓
Receptor occupancy (% modulator)
       ↓
model._components() - circuit signatures
       ↓
model._emotions() - emotion labels
       ↓
JSON response
       ↓
app.js - Canvas visualizations
```

---

## 9. Testing

### 9.1 check_scenarios.py

```python
SCENARIOS = [
    ("nothing happens", {}, {"Little change"}),
    ("threat cue", {"threat_cue": 1.0}, {"Fear-like"}),
    ("uncertain and stressed", {"uncertainty": 1.0, "stress": 0.8}, {"Anxiety-like"}),
    ("reward cue", {"reward_cue": 1.0}, {"Joy-like"}),
    ("loss or rejection", {"loss_cue": 1.0}, {"Sadness-like"}),
    # ...
]
```

### 9.2 benchmark.py

```bash
# Run benchmark
python benchmark.py

# Output:
# cells      synapses    build   ms/step  full-run   peak-GiB  cell-Hz
# 16,384       1.2M       0.1s      0.1      0.6s       0.5       2.3
# 65,536       5M         0.3s      0.3      1.8s       0.9       2.1
# ...
```

---

## 10. Key Algorithms

### 10.1 Region Cell Distribution

```python
def region_sizes(cfg, n):
    raw = [n * region['relative_size'] / total for region in regions]
    sizes = [max(minimum, int(value)) for value in raw]
    
    # Round-off: distribute remainder by largest fractional part
    order = sorted(range(len(raw)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    while sum(sizes) != n:
        i = order[index % len(order)]
        if sum(sizes) < n:
            sizes[i] += 1
        elif sizes[i] > minimum:
            sizes[i] -= 1
        index += 1
```

### 10.2 LFP Proxy

```python
# Mazzoni et al. (2015)
LFP = |I_AMPA|(t - 6ms) - 1.65 × |I_GABA|(t)
```

### 10.3 Spectrum Analysis

```python
# DFT over 1-45 Hz
for freq in range(1, 46):
    omega = 2π * freq / sample_rate
    power = sum(sample * cos(omega * i) for i, sample in enumerate(samples))²
           + sum(sample * sin(omega * i) for i, sample in enumerate(samples))²
```

---

*Document generated: 2026-09-28*
*Part of Human Emotion - Comprehensive Analysis*
