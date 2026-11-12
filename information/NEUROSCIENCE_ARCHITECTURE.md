# Human Emotion - Comprehensive Neuroscience Architecture

## Overview

Human Emotion simulates a reduced spiking neural network of emotion circuits with **21 brain regions**, **~71 projections**, **5 neuromodulatory systems**, and **adaptive exponential integrate-and-fire (AdEx) neurons**. This document provides the complete circuit architecture.

---

## 1. Sensory Entry Points

### 1.1 Sensory Thalamus
**Role**: First relay for all sensory modalities (visual, auditory, somatosensory).

**Subregions** (future expansion):
- **LGN**: Lateral geniculate nucleus - visual thalamus
- **MGN**: Medial geniculate nucleus - auditory thalamus  
- **VPM**: Ventral posteromedial nucleus - somatosensory thalamus
- **PO**: Posterior medial nucleus - polymodal association
- **MD**: Mediodorsal nucleus - prefrontal relay

**Projections**:
- `sensory_thalamus → sensory_cortex` (weight 0.6)
- `sensory_thalamus → amygdala_bla` (weight 0.35) - **"Low Road"**
- `sensory_thalamus → insula` (weight 0.2)
- `sensory_thalamus → amygdala_bla_reward` (weight 0.2)

**Low Road to Amygdala**: Direct thalamo-amygdala projection bypassing cortex. Latency ~5-20ms. Fast, coarse threat detection.

### 1.2 Sensory Cortex
**Role**: Detailed perceptual analysis; sends slower "high road" to amygdala.

**Subregions** (future expansion):
- **Layer 1 (L1)**: Dendritic tufts, VIP/SST targeting
- **Layer 2/3 (L2/3)**: Association pyramidal cells, horizontal connections
- **Layer 4 (L4)**: Feedforward input from thalamus
- **Layer 5 (L5)**: Output to striatum, brainstem
- **Layer 6 (L6)**: Feedback to thalamus

**Projections**:
- `sensory_cortex → amygdala_bla` (weight 0.45) - **"High Road"**
- `sensory_cortex → amygdala_bla_reward` (weight 0.3)
- `sensory_cortex → hippocampus` (weight 0.35)
- `sensory_cortex → dACC` (weight 0.15)
- `sensory_cortex → sensory_thalamus` (weight 0.25) - corticothalamic feedback

**High Road to Amygdala**: Cortico-cortical route via sensory cortex. Latency ~50-100ms. Detailed, contextual threat evaluation.

---

## 2. Amygdala Complex

### 2.1 Basolateral Amygdala (BLA)

#### Threat Cells (`amygdala_bla`)
**Role**: Detects cues learned to predict danger; drives central amygdala output.

**Cell Type**: regular_spiking (glutamatergic projection neurons)
- v_rest=-65mV, tau_m=20ms
- Moderate adaptation (a=0.13, b=2.5mV)

**Input Convergence**:
- Thalamic "low road" (fast, coarse) - weight 0.35
- Cortical "high road" (slow, detailed) - weight 0.45
- Insular interoceptive input - weight 0.25
- Hippocampal context - weight 0.15

**Output Projections**:
- `amygdala_bla → amygdala_cea` (weight 0.5) - threat output
- `amygdala_bla → bnst` (weight 0.3) - sustained anxiety
- `amygdala_bla → nucleus_accumbens` (weight 0.1)
- `amygdala_bla → vmpfc` (weight 0.1)
- `amygdala_bla → dACC` (weight 0.2)
- `amygdala_bla → hippocampus` (weight 0.1)
- `amygdala_bla → amygdala_bla_reward` (weight -0.2) - **valence competition**

**Subnuclei** (future expansion):
- **Lateral Amygdala (LA)**: Sensory input processing, CS-US conjunction
- **Basal Amygdala (B)**: Context integration, plasticity site

#### Reward Cells (`amygdala_bla_reward`)
**Role**: Detects cues learned to predict reward; drives accumbens/VTA reward circuit.

**Cell Type**: regular_spiking (glutamatergic)
- Same parameters as threat cells

**Input Convergence**:
- Thalamic input - weight 0.2
- Cortical input - weight 0.3
- Hippocampal context - weight 0.1

**Output Projections**:
- `amygdala_bla_reward → nucleus_accumbens` (weight 0.45) - **positive valence**
- `amygdala_bla_reward → vmpfc` (weight 0.15) - value computation
- `amygdala_bla_reward → amygdala_bla` (weight -0.2) - **valence competition**

### 2.2 Central Amygdala (CeA)
**Role**: GABAergic output nucleus; triggers defensive, autonomic, and arousal responses.

**Cell Type**: regular_spiking (GABAergic projection neurons)
- v_rest=-65mV, projection_fraction=0.9 (mostly projection cells)

**Disinhibition Mechanism** (key anatomical feature):
The CeA does NOT directly excite downstream targets. Instead:
1. CeA fires → inhibits interneurons in target regions
2. Target regions release from inhibition → disinhibition
3. This enables rapid, coordinated responses

**Output Projections (all targeting interneurons)**:
- `amygdala_cea → pag` (weight 0.9, target: interneurons) - **disinhibits PAG**
- `amygdala_cea → hypothalamus` (weight 0.8, target: interneurons) - **disinhibits HPA**
- `amygdala_cea → locus_coeruleus` (weight 0.8, target: interneurons) - **disinhibits LC**
- `amygdala_cea → motor_brainstem` (weight 0.6, target: interneurons) - **disinhibits brainstem**
- `amygdala_cea → basal_forebrain` (weight 0.15, target: interneurons)

**Subdivisions** (future expansion):
- **CeL (lateral)**: Receives BLA input, processes valence
- **CeM (medial)**: Output to brainstem autonomic centers

**Intercalated Cells (ITC)** (future expansion):
- Pure GABA regions gating BLA→CeA transmission
- ITC→BLA (inhibition) - stops threat detection
- ITC→CeA (inhibition) - stops threat expression
- Different ITC clusters for approach vs avoidance

---

## 3. Bed Nucleus of the Stria Terminalis (BNST)

**Role**: Sustained, uncertain-threat responses (anxiety-like).

**Cell Type**: regular_spiking (GABAergic)
- projection_fraction=0.9
- local_in_degree projection_from_interneuron=4 (sparse inhibition)

**Input Convergence**:
- `amygdala_bla → bnst` (weight 0.3) - threat input
- `hippocampus → bnst` (weight 0.15) - contextual modulation
- `dACC → bnst` (uncertainty input via control slider)

**Key Distinction from CeA**:
- CeA: Phasic threat response (discrete cue, rapid onset/offset)
- BNST: Sustained threat response (uncertainty, prolonged, slow onset)

**Output Projections**:
- `bnst → hypothalamus` (weight 0.35, target: interneurons) - sustained stress
- `bnst → pag` (weight 0.2, target: interneurons) - sustained defensive posture

**Clinical Relevance**: BNST hyperactivity in generalized anxiety disorder, PTSD.

---

## 4. Prefrontal Regulation Circuit

### 4.1 Ventromedial Prefrontal Cortex (vmPFC)
**Role**: Value computation, safety signaling, regulation of amygdala.

**Cell Type**: regular_spiking (glutamatergic)

**Input Convergence**:
- `hippocampus → vmpfc` (weight 0.35) - context-dependent value
- `amygdala_bla → vmpfc` (weight 0.1) - threat valuation
- `amygdala_bla_reward → vmpfc` (weight 0.15) - reward valuation
- `dACC → vmpfc` (weight 0.1) - conflict processing

**Safety Signaling** (key function):
- `vmpfc → amygdala_cea` (weight -0.4) - **inhibits threat output** (via ITC cells)
- `vmpfc → amygdala_bla` (weight -0.2) - reduces threat detection
- `vmpfc → hypothalamus` (weight -0.15) - reduces stress axis
- `vmpfc → pag` (weight -0.15) - reduces defensive behavior

**Positive Affect**:
- `vmpfc → raphe` (weight 0.15) - serotonin enhancement
- `vmpfc → vta` (weight 0.1) - dopamine enhancement
- `vmpfc → oxytocin_pvn` (weight 0.1) - social bonding

### 4.2 Dorsal Anterior Cingulate Cortex (dACC)
**Role**: Conflict monitoring, effort computation, social pain, uncertainty.

**Cell Type**: regular_spiking (glutamatergic)

**Input Convergence**:
- `sensory_cortex → dACC` (weight 0.15) - sensory conflict
- `insula → dACC` (weight 0.3) - interoceptive conflict
- `amygdala_bla → dACC` (weight 0.2) - emotional conflict

**Output Projections**:
- `dACC → dlpfc` (weight 0.3) - recruits cognitive control
- `dACC → amygdala_bla` (weight 0.15) - emotional salience
- `dACC → locus_coeruleus` (weight 0.25) - arousal during conflict
- `dACC → pag` (weight 0.1) - autonomic response
- `dACC → vmpfc` (weight 0.1) - value computation

### 4.3 Dorsolateral Prefrontal Cortex (dLPFC)
**Role**: Executive control, cognitive reappraisal, working memory.

**Cell Type**: regular_spiking (glutamatergic)

**Top-Down Regulation**:
- `dlpfc → vmpfc` (weight 0.25) - **reappraisal route**: enhance safety signaling
- `dlpfc → sensory_cortex` (weight 0.15) - top-down attention
- `dlpfc → nucleus_accumbens` (weight 0.1) - goal-directed motivation

**Inhibition of ACC**:
- `dlpfc → dACC` (weight -0.2) - **top-down inhibition**: suppress conflict signal when successfully regulated

**Cognitive Control Under Stress**:
- Chronic stress: dLPFC function impaired (cortisol effect)
- Acute stress: NE optimizes dLPFC for task performance (Yerkes-Dodson)

---

## 5. Hippocampus

**Role**: Context encoding, memory, novelty detection, spatial navigation.

**Cell Type**: regular_spiking (glutamatergic)

**Subregions** (future expansion):
- **CA1**: Output to subiculum, cortex
- **CA3**: Recurrent collaterals, theta generation, pattern completion
- **Dentate Gyrus**: Pattern separation, new neuron birth (adult neurogenesis)
- **Entorhinal Cortex**: Gateway to/from cortex

**Theta Oscillations** (4-12 Hz):
- CA3 recurrent collaterals generate theta
- HCN channels create theta resonance
- Theta phase organizes memory encoding

**Input Convergence**:
- `sensory_cortex → hippocampus` (weight 0.35) - sensory context
- `amygdala_bla → hippocampus` (weight 0.1) - emotional tagging
- `amygdala_bla_reward → hippocampus` (weight 0.1)

**Output Projections**:
- `hippocampus → vmpfc` (weight 0.35) - context-dependent valuation
- `hippocampus → nucleus_accumbens` (weight 0.3) - memory-guided reward
- `hippocampus → amygdala_bla` (weight 0.15) - context-gated fear
- `hippocampus → amygdala_bla_reward` (weight 0.1)
- `hippocampus → bnst` (weight 0.15) - contextual anxiety
- `hippocampus → vta` (weight 0.1) - novelty reward
- `hippocampus → hypothalamus` (weight -0.1) - **ventral hippocampus inhibits stress axis**

**Novelty Detection**:
- Dentate granule cells: pattern separation
- CA3: pattern completion
- Novel = unexpected pattern → activates exploration

---

## 6. Nucleus Accumbens (NAc)

**Role**: Reward, motivation, reinforcement learning, goal-directed behavior.

**Cell Type**: medium_spiny (GABAergic MSNs)

**Special Properties**:
- Hyperpolarized rest (v_rest=-72mV) - requires strong depolarization to fire
- Strong facilitation (facilitation_ms=350ms) - sustained input builds firing
- Slow membrane (tau_w=200ms) - integrates over hundreds of ms

**Core vs Shell** (future expansion):
- **NAc_core**: Motor actions, connects to motor cortex
- **NAc_shell**: Limbic, connects to limbic areas, feeding behavior

**D1 vs D2 Pathways** (future expansion):
- **D1-MSNs (direct pathway)**: "Go" signal, approach behavior
- **D2-MSNs (indirect pathway)**: "NoGo" signal, avoidance behavior

**Input Convergence**:
- `amygdala_bla_reward → nucleus_accumbens` (weight 0.45) - **positive valence**
- `hippocampus → nucleus_accumbens` (weight 0.3) - memory-guided reward
- `vmpfc → nucleus_accumbens` (weight 0.25) - goal selection

**Output (Disinhibition)**:
- `nucleus_accumbens → vta` (weight 0.3, target: interneurons) - **disinhibits VTA DA cells**
- `nucleus_accumbens → hypothalamus` (weight 0.2) - feeding/motivation

**Cholinergic Interneurons (CINs)** (future expansion):
- Tonic firing at 2-5 Hz
- Pause responses to salient events
- Encode prediction error via pauses

---

## 7. Lateral Habenula

**Role**: "Disappointment signal," negative reward prediction error, loss processing.

**Cell Type**: regular_spiking (glutamatergic)

**Key Discovery** (Matsumoto & Hikosaka, 2007):
- Lateral habenula fires when reward is worse than expected
- Signals "negative RPE" to VTA and raphe
- Major target for depression and addiction

**Input**:
- `sensory_cortex → lateral_habenula` (weight 0.8) - loss cue input
- `hypothalamus → lateral_habenula` (weight 0.15) - energy state

**Output (Inhibition)**:
- `lateral_habenula → vta` (weight -0.5) - **strongly inhibits VTA DA cells** (via RMTg)
- `lateral_habenula → raphe` (weight -0.35) - suppresses serotonin

**Circuit Logic**:
- Reward: VTA active, habenula silent, DA high
- Reward Omission: VTA should be silent, habenula fires, DA suppressed
- Loss: habenula fires strongly, DA/5-HT suppressed

---

## 8. Hypothalamus

**Role**: Stress hormones (HPA axis), autonomic control, hunger, thirst, temperature.

**Cell Type**: regular_spiking (glutamatergic)

**Subregions** (future expansion):
- **Paraventricular (PVN)**: Oxytocin, vasopressin, CRH
- **Arcuate (ARC)**: NPY/AgRP, POMC (feeding)
- **Lateral (LH)**: Orexin/hypocretin (arousal, feeding)
- **Ventromedial (VMH)**: Satiety

**CRH/HPA Axis**:
- Stress → hypothalamus releases CRH
- CRH → pituitary → cortisol release
- Cortisol feedback: inhibits hypothalamus, hippocampus; facilitates amygdala

**Input Convergence**:
- `insula → hypothalamus` (weight 0.2) - interoceptive state
- `motor_brainstem → hypothalamus` (weight 0.35) - autonomic feedback
- `hypothalamus → lateral_habenula` (weight 0.15) - energy state

**Output Projections**:
- `hypothalamus → pag` (weight 0.3) - autonomic control
- `hypothalamus → locus_coeruleus` (weight 0.2) - arousal
- `hypothalamus → motor_brainstem` (weight 0.35) - autonomic output
- `hypothalamus → vta` (weight 0.15) - orexin/feeding drive
- `hypothalamus → lateral_habenula` (weight 0.15)
- `hypothalamus → basal_forebrain` (weight 0.1)

---

## 9. Periaqueductal Gray (PAG)

**Role**: Defensive behavior (freeze, flight, fight), endogenous pain control, vocalization.

**Cell Type**: regular_spiking (glutamatergic)

**Anatomical Columns** (future expansion):
- **dPAG**: Fight/escape responses (active coping)
- **lPAG**: Flight responses
- **vlPAG**: Freeze responses, fear conditioning
- **dmPAG**: Panic/defensive threat

**Input Convergence**:
- `amygdala_cea → pag` (disinhibition, weight 0.9)
- `bnst → pag` (disinhibition, weight 0.2)
- `hypothalamus → pag` (weight 0.3)
- `pain → pag` (weight 0.6) - direct pain input

**Output Projections**:
- `pag → motor_brainstem` (weight 0.45) - autonomic/motor output
- `pag → locus_coeruleus` (weight 0.15) - arousal
- `pag → hypothalamus` (weight 0.1) - feedback
- `pag → raphe` (weight 0.1) - pain modulation

**Endogenous Analgesia**:
- vlPAG → raphe magnus → spinal cord dorsal horn
- Activates on opioid release → reduces pain transmission

---

## 10. Neuromodulator Source Nuclei

### 10.1 Ventral Tegmental Area (VTA)
**Neurotransmitter**: Dopamine
**Cell Type**: pacemaker (depolarized rest, slow membrane)

**Properties**:
- v_rest=-52mV (near threshold - spontaneous firing)
- Background_scale=0.9 (high baseline activity)
- Dopamine clearance: 400ms

**Phasic vs Tonic Firing**:
- Tonic: ~2-5 Hz background firing
- Phasic burst: ~20 Hz, 100ms burst on unexpected reward
- Phasic pause: silence for ~200ms on no-reward

**Projection Targets** (dopamine effects):
| Region | Density | Excitability | NMDA | Function |
|--------|---------|--------------|------|----------|
| nucleus_accumbens | 1.0 | +0.35 | +0.40 | Reward learning |
| amygdala_bla_reward | 0.5 | +0.20 | - | Reinforcement |
| amygdala_bla | 0.4 | +0.15 | - | Threat modulation |
| vmpfc | 0.35 | +0.10 | +0.30 | Value computation |
| dACC | 0.3 | +0.10 | - | Effort valuation |
| dlpfc | 0.3 | +0.05 | +0.40 | Working memory |
| hippocampus | 0.2 | +0.10 | - | Memory |
| bnst | 0.3 | +0.10 | - | Anxiety |
| vta (autoreceptor) | 0.5 | -0.50 | - | Negative feedback |

### 10.2 Locus Coeruleus (LC)
**Neurotransmitter**: Noradrenaline
**Cell Type**: pacemaker

**Properties**:
- v_rest=-52mV, background_scale=0.6
- NE clearance: 1000ms

**LC-NE System** (Aston-Jones & Cohen, 2005):
- **Tonic mode**: Low LC firing, focused attention
- **Phasic mode**: Burst on salient events, enhanced signal-to-noise
- **LC Tune**: LC activity optimizes cortical circuits (Yerkes-Dodson)

**Projection Targets**:
| Region | Density | Excitability | Adaptation | Function |
|--------|---------|--------------|-----------|----------|
| amygdala_bla | 0.9 | +0.30 | -0.30 | Enhanced threat detection |
| sensory_thalamus | 0.7 | +0.20 | -0.30 | Sensory gain |
| sensory_cortex | 0.7 | +0.15 | -0.30 | Cortical gain |
| bnst | 0.8 | +0.30 | - | Sustained anxiety |
| hippocampus | 0.6 | +0.10 | - | Memory enhancement |
| vmpfc | 0.5 | -0.10 | - | Safety reduced |
| dlpfc | 0.5 | -0.15 | - | Cognitive load |
| locus_coeruleus | 0.6 | -0.50 | - | Autoinhibition |

### 10.3 Dorsal Raphe (DRN)
**Neurotransmitter**: Serotonin (5-HT)
**Cell Type**: pacemaker

**Properties**:
- v_rest=-52mV, background_scale=0.6
- 5-HT clearance: 2000ms (slowest - most sustained effects)

**5-HT Functions**:
- Mood regulation (SSRIs target this)
- Impulse control
- Pain modulation
- Sleep/wake
- Appetite

**Projection Targets**:
| Region | Density | Excitability | Function |
|--------|---------|--------------|----------|
| amygdala_bla | 0.8 | -0.25 | Mood regulation |
| amygdala_cea | 0.6 | -0.20 | Fear expression gating |
| pag | 0.7 | -0.35 | Pain modulation |
| hippocampus | 0.7 | -0.10 | Memory modulation |
| vmpfc | 0.8 | +0.10 | Positive affect |
| raphe (autoreceptor) | 0.6 | -0.50 | Negative feedback |

### 10.4 Basal Forebrain (BF)
**Neurotransmitter**: Acetylcholine
**Cell Type**: pacemaker

**Properties**:
- v_rest=-52mV, background_scale=0.75
- ACh clearance: 200ms (fastest - phasic attention)

**ACh Functions**:
- Attention to novel/salient stimuli
- Memory encoding (enhances LTP)
- Cortical plasticity
- Arousal

**Projection Targets**:
| Region | Density | Excitability | Adaptation | Function |
|--------|---------|--------------|-----------|----------|
| sensory_cortex | 1.0 | +0.20 | -0.50 | Novelty processing |
| hippocampus | 0.8 | +0.15 | -0.40 | Memory encoding |
| amygdala_bla | 0.8 | +0.20 | -0.30 | Fear learning |
| amygdala_bla_reward | 0.6 | +0.15 | -0.30 | Reward learning |
| dACC | 0.6 | +0.10 | -0.30 | Conflict attention |
| dlpfc | 0.7 | +0.10 | -0.40 | Attention |

### 10.5 Oxytocin Cells (PVN/SON)
**Neurotransmitter**: Oxytocin
**Cell Type**: pacemaker

**Properties**:
- v_rest=-52mV, background_scale=0.6
- OT clearance: 8000ms (very slow)
- Fast output: false (acts only through chemical, not synaptic)

**Social Functions**:
- Social bonding (pair bonds, maternal behavior)
- Trust enhancement
- Anxiety reduction
- Pain modulation
-feeding behavior

**Projection Targets**:
| Region | Density | Excitability | Function |
|--------|---------|--------------|----------|
| amygdala_cea | 1.0 | -0.45 | **Anxiolytic** |
| amygdala_bla | 0.4 | -0.10 | Social threat detection |
| bnst | 0.5 | -0.20 | Sustained anxiety reduction |
| nucleus_accumbens | 0.6 | +0.15 | Social reward |
| vta | 0.4 | +0.15 | Positive affect |
| pag | 0.4 | -0.15 | Defensive behavior reduction |
| hypothalamus | 0.5 | -0.15 | Stress axis suppression |
| raphe | 0.4 | +0.15 | Mood enhancement |

---

## 11. Interoceptive Circuit

### 11.1 Insula
**Role**: Interoceptive cortex - processes bodily signals (heart, breathing, gut, pain).

**Cell Type**: regular_spiking (glutamatergic)

**Subdivisions**:
- **Anterior insula**: Salience detection, subjective feeling
- **Posterior insula**: Primary interoceptive cortex, homeostatic state
- **Mid insula**: Emotional awareness, gustatory

**Input Convergence**:
- `sensory_thalamus → insula` (weight 0.2) - sensory signals
- `motor_brainstem → insula` (weight 0.12) - **visceral feedback loop**

**Output Projections**:
- `insula → amygdala_bla` (weight 0.25) - bodily threat
- `insula → amygdala_cea` (weight 0.15) - emotional feeling
- `insula → dACC` (weight 0.3) - conflict monitoring
- `insula → hypothalamus` (weight 0.2) - autonomic regulation
- `insula → oxytocin_pvn` (weight 0.15) - social touch processing

### 11.2 Motor Brainstem
**Role**: Autonomic and motor output - heart rate, breathing, startle, posture.

**Cell Type**: regular_spiking (glutamatergic)

**Nuclei** (future expansion):
- **NTS**: Nucleus tractus solitarius - visceral afferents
- **NA**: Nucleus ambiguus - parasympathetic
- **RVLM**: Rostral ventrolateral medulla - sympathetic
- **PB**: Parabrachial nucleus - pain, taste, visceral
- **PAG**: Already modeled separately

**Output Projections**:
- `motor_brainstem → insula` (weight 0.12) - **closes visceromotor loop**
- `motor_brainstem → locus_coeruleus` (weight 0.2) - visceral arousal

---

## 12. Functional Circuit Summaries

### 12.1 Threat/Fear Circuit
```
Thalamic Low Road:    Thalamus → amygdala_bla (5-20ms)
Cortical High Road:   Thalamus → Cortex → amygdala_bla (50-100ms)
                       ↓
                    amygdala_bla → amygdala_cea (pooled threat signal)
                       ↓
                    amygdala_cea (GABA) → [PAG, Hypothalamus, LC] interneurons
                       ↓ (disinhibition)
                    PAG: Freeze/Fight/Flight
                    Hypothalamus: HPA axis activation
                    LC: Arousal enhancement
```

### 12.2 Reward Circuit
```
Reward Cue:           Cortex → amygdala_bla_reward
                       ↓
                    amygdala_bla_reward → nucleus_accumbens
                       ↓
                    nucleus_accumbens (GABA) → VTA interneurons
                       ↓ (disinhibition)
                    VTA DA neurons fire → Dopamine release
                       ↓
                    NAc: Reinforcement learning
                    vmPFC: Reward valuation
                    Hippocampus: Context-reward association
```

### 12.3 Loss/Depression Circuit
```
Loss Signal:         Cortex → lateral_habenula
                       ↓
                    lateral_habenula (glutamate) → RMTg GABA cells
                       ↓
                    RMTg → VTA DA neurons (inhibition)
                    RMTg → Dorsal Raphe (inhibition)
                       ↓
                    VTA: DA suppression
                    Raphe: 5-HT suppression
                       ↓
                    Reduced motivation, negative mood
```

### 12.4 Anxiety/Sustained Threat Circuit
```
Uncertainty:         dACC + Hippocampus → BNST
                       ↓
                    BNST (GABA) → [Hypothalamus, PAG] interneurons
                       ↓ (disinhibition)
                    Sustained stress axis activation
                    Sustained defensive posture
                    No discrete offset - persists until safety signal
```

### 12.5 Social Bonding Circuit
```
Social Touch:        Sensory → Insula → Oxytocin PVN
                       ↓
                    Oxytocin release throughout brain
                       ↓
                    Amygdala CeA: OT inhibits output → reduced fear
                    NAc: OT enhances reward → social pleasure
                    PAG: OT reduces defensive → social engagement
```

---

## 13. Interneuron Subtypes (Future Expansion)

### 13.1 PV+ Basket Cells
- **Location**: Perisomatic region of pyramidal cells
- **Properties**: Fast-spiking, no adaptation, gap junctions
- **Function**: Gamma oscillations (30-80 Hz), feedforward/feedback inhibition
- **In emotion model**: Regulate BLA and cortical firing precision

### 13.2 SST+ Martinotti Cells
- **Location**: Dendritic tufts of pyramidal cells
- **Properties**: Low probability release, facilitation, dendrite-targeting
- **Function**: Input integration control, plasticity gating
- **In emotion model**: Regulate sensory input strength to amygdala

### 13.3 VIP+ Disinhibitory Cells
- **Location**: Layer 1, near other interneurons
- **Properties**: Strong facilitation, inhibit SST cells
- **Function**: "Gate openers" - disinhibit cortex during attention
- **In emotion model**: Enhanced processing during reward/novelty

---

## 14. Oscillatory Mechanisms (Future Expansion)

### 14.1 Theta (4-12 Hz) - Hippocampus
- CA3 recurrent collaterals generate theta via HCN channels
- Theta phase organizes memory encoding (trough = new info)

### 14.2 Gamma (30-80 Hz) - Cortex
- PING: Pyramidal → PV → Pyramidal loop
- VIP→SST→Pyramidal disinhibition during attention

### 14.3 Beta (13-30 Hz) - Motor/ACC
- Increased during sustained posture and conflict monitoring
- dACC beta: cognitive control demands

### 14.4 Ripple (150-250 Hz) - Hippocampus
- Occur during theta trough (memory consolidation windows)
- CA3 bursts with PV cell assistance

---

## 15. Projection Weight Summary

### Anatomically Constrained Weights
| From | To | Weight | Type | Note |
|------|-----|--------|------|------|
| thalamus | sensory_cortex | 0.6 | Glu | Feedforward |
| thalamus | amygdala_bla | 0.35 | Glu | Low road |
| cortex | amygdala_bla | 0.45 | Glu | High road |
| cortex | hippocampus | 0.35 | Glu | Context |
| amygdala_bla | amygdala_cea | 0.5 | Glu | Threat output |
| amygdala_cea | PAG | 0.9 | GABA | Disinhibition |
| amygdala_cea | Hypothal | 0.8 | GABA | Disinhibition |
| amygdala_cea | LC | 0.8 | GABA | Disinhibition |
| amygdala_bla_reward | NAc | 0.45 | Glu | Reward |
| NAc | VTA | 0.3 | GABA | Disinhibition |
| habenula | VTA | -0.5 | Glu | RMTg pathway |
| habenula | Raphe | -0.35 | Glu | Inhibition |
| vmPFC | amygdala_cea | -0.4 | Glu | Safety (via ITC) |
| vmPFC | Hypothal | -0.15 | Glu | Stress reduction |
| dACC | dLPFC | 0.3 | Glu | Control |
| dLPFC | dACC | -0.2 | Glu | Regulation |
| Hippocampus | vmPFC | 0.35 | Glu | Context-value |
| Hippocampus | NAc | 0.3 | Glu | Memory-reward |

---

*Document generated: 2026-09-28*
*Part of Human Emotion - Comprehensive Analysis*
