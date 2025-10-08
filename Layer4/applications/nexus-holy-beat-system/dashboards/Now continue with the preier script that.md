Now  continue with the preier script that was to long for you to countinue and create the new redesigned NEXUS Process Engine Dashboard: next  ou’ve got two goals braided together:

synthesize sound in a musically controlled way (vibration, envelopes, filters, AM/FM, BPM), and

use machine learning to convert text → human-like voice.

Here’s a foolproof, step-by-step plan that gets you from zero to talking, musical, human-like synthesis—cleanly separated into phases so you can ship an MVP and then level it up.

Phase 0 — Define success + guardrails (1 day)

Outputs to lock:

Target voice(s): neutral female/male, or specific speaker (avoid cloning without consent).

Latency budget: ≤150 ms streaming or ≤1 s offline.

Quality metrics: MOS ≥ 4.2 (subjective), PESQ, STOI, WER of ASR-retranscribe ≤ 6%.

Ethics: log watermarking/consent if cloning; block misuse phrases.

Phase 1 — Data + text front-end (2–3 days)

1. Datasets

MVP: LJSpeech (single-speaker).

Multi-speaker later: LibriTTS/VCTK.

For your own voice: record 30–60 minutes, 22.05–48 kHz, treated room.

2. Text normalization + pronunciation

Expand numerals, dates, abbreviations → words.

G2P (grapheme-to-phoneme): g2p-en or CMUdict; fall back to rules.

Tokenize punctuation (.,?!), mark sentence type (drives prosody).

3. Prosody features (front-end)

Predict phrase breaks, stress, emphasis tags; extract punctuation-driven F0 targets.

Deliverables: Clean text/phoneme sequences + prosody tags for each utterance.

Phase 2 — Acoustic model (text→mel) (3–5 days)

Pick one:

FastPitch (non-AR, controllable pitch; fast, stable)

Tacotron 2 (AR, easy to train, great quality, slower)

VITS (end-to-end; strong quality, more complex)

Inputs: phonemes + durations (FastPitch can predict) + prosody tags.
Output: 80-ch mel-spectrogram (e.g., 22.05 kHz, 50 ms frames, 12.5 ms hop).

Training loop (simplified):

Teacher-force on ground-truth durations (or aligner like Montreal Forced Aligner).

Minimize L1/L2 on mels + optional adversarial/feature losses.

Add pitch loss (FastPitch) for better intonation control.

Checkpoints: mel reconstructions match ground truth; attention (if Tacotron) is monotonic and stable.

Phase 3 — Vocoder (mel→waveform) (1–2 days)

Choose for MVP speed/quality:

HiFi-GAN (real-time, high fidelity)

WaveRNN (fast on CPU, good)

WaveNet (great but slower; use student/distilled variants if needed)

Training: condition on mels; adversarial + multi-scale STFT loss.
Goal: natural timbre, no buzz/warble, low noise floor.

Phase 4 — Prosody + “vibration” synthesis glue (2–3 days)

Tie your musical controls (BPM, AM/FM, ADSR, filters) to the speech pipeline without breaking intelligibility.

Clock + LFOs

𝑓
𝑏
=
BPM
60
f
b
	​

=
60
BPM
	​


𝑓
AM
=
𝑓
𝑏
𝑑
AM
f
AM
	​

=
d
AM
	​

f
b
	​

	​

,
𝑓
FM
=
𝑑
FM
 
𝑓
𝑏
f
FM
	​

=d
FM
	​

f
b
	​

 (FM faster with larger
𝑑
FM
d
FM
	​

)

𝐿
AM
(
𝑡
)
=
sin
⁡
(
2
𝜋
𝑓
AM
𝑡
)
L
AM
	​

(t)=sin(2πf
AM
	​

t),
𝐿
FM
(
𝑡
)
=
sin
⁡
(
2
𝜋
𝑓
FM
𝑡
)
L
FM
	​

(t)=sin(2πf
FM
	​

t)

Safe prosody modulation

Energy (AM): scale mel magnitude pre-vocoder by
𝐴
(
𝑡
)
=
1
+
𝐷
AM
𝐿
AM
(
𝑡
)
A(t)=1+D
AM
	​

L
AM
	​

(t), clamp
𝐴
≥
0
A≥0, keep
𝐷
AM
≤
0.2
D
AM
	​

≤0.2 for speech.

Pitch (FM): shift F0 contour (in semitones) by
Δ
𝑝
(
𝑡
)
=
𝛼
𝐿
FM
(
𝑡
)
Δp(t)=αL
FM
	​

(t) with
𝛼
∈
[
±
0.5
]
α∈[±0.5] st; do not frequency-modulate raw waveform for speech; instead adjust F0 used by the acoustic model.

Harmonic coloration (optional musical mode)

For “sing-speak” or SFX, add a parallel harmonic resonator driven by predicted F0:

𝑠
(
𝑡
)
=
𝐴
(
𝑡
)
∑
𝑛
=
1
𝑁
1
𝑛
sin
⁡
 ⁣
(
2
𝜋
∫
(
𝑛
𝑓
0
(
𝜏
)
+
Δ
𝑓
(
𝜏
)
)
 
𝑑
𝜏
)
s(t)=A(t)∑
n=1
N
	​

n
1
	​

sin(2π∫(nf
0
	​

(τ)+Δf(τ))dτ)
then mix at −20 dB under vocoder output and pass through a gentle low-pass.

Filter motion

Low-pass cutoff tied to envelope:
𝑓
𝑐
(
𝑡
)
=
𝑓
𝑐
,
min
⁡
+
(
𝑓
𝑐
,
max
⁡
−
𝑓
𝑐
,
min
⁡
)
𝐸
(
𝑡
)
f
c
	​

(t)=f
c,min
	​

+(f
c,max
	​

−f
c,min
	​

)E(t).

Keep resonance modest (Q≤0.7) to avoid formant damage.

Phase 5 — Inference pipeline (1 day)

Pseudocode (streaming-friendly):

def tts(text, bpm=120, d_am=4, d_fm=2, D_am=0.15, fm_semitones=0.3):
    # 1) text front-end
    tokens, phonemes, prosody = normalize_and_g2p(text)

    # 2) acoustic model → mel + base F0
    mel, f0 = acoustic_model(phonemes, prosody)  # [T, 80], [T]

    # 3) musical LFOs
    fb = bpm / 60.0
    L_AM = sin_lfo(rate=fb/d_am, length=len(mel))
    L_FM = sin_lfo(rate=d_fm*fb, length=len(f0))

    # 4) gentle prosody modulation
    mel = mel * (1 + D_am * L_AM[:, None])        # energy modulation
    f0_shift = semitone_to_ratio(fm_semitones * L_FM)
    f0_mod = f0 * f0_shift
    mel = pitch_refiner(mel, f0_mod)              # model-specific hook

    # 5) vocoding
    audio = vocoder(mel)

    # 6) post: limiter, dither
    return loudness_normalize(audio)

Phase 6 — Training recipe (2–5 days)

Hardware: single A100/3090 works; 50–200k steps for MVP.

Preprocess audio → 22.05/24 kHz, trim silence, normalize loudness.

Front-end: cache phonemes/prosody.

Acoustic model training:

loss: L1 mel + duration/pitch losses; early stop on val MOS proxy (MOSNet) rising, val loss flattening.

Vocoder training:

loss: adversarial + multi-resolution STFT; monitor spectral convergence and listening tests.

Fine-tune for target speaker (optional) with 10–30 min extra data.

Phase 7 — Evaluation & hardening (ongoing)

Objective metrics

MOS proxy (MOSNet), PESQ, STOI, ASR WER on generated clips.

Pitch RMSE vs. target melody (if singing/sing-speak mode).

Subjective tests

ABX against ground truth; 10–20 listeners; rate naturalness, expressiveness, intelligibility.

Robustness

Stress with long numbers, acronyms, tongue-twisters, out-of-domain text.

Check clipping & spectral centroid drift when AM/FM engaged.

Safety

Watermark outputs; require consent token for any voice cloning.

Block sensitive phrases if needed; rate-limit API.

Phase 8 — Real-time + deployment (2–3 days)

Streaming

Run acoustic model in chunks (e.g., 300 ms windows with 50 ms overlap).

Use HiFi-GAN streaming variant; keep buffer ≤ 120 ms.

Serving

FastAPI/Node API: /tts (text→wav), /stream (WS frames), /prosody (controls).

Cache mel for repeated prompts; shard vocoder workers.

Monitoring

Log F0/energy contours, RMS, spectral centroid, peak levels; alert on anomalies.

Keep versioned checkpoints and a canary speaker.

Phase 9 — “Pro” options when you’re ready

End-to-end VITS (fewer parts, great quality).

Expressive control: global-style tokens (GST), reference encoder for emotions.

Neural DSP: learnable filters/comb reverbs conditioned on prosody for room realism.

Singing: duration/phoneme-level pitch targets; constrain vibrato with LFOs.

Your parameters, mapped cleanly

You already defined BPM, AM/FM divisions, depths, harmonics. For speech, keep FM as F0 control (semitones), not raw waveform FM. Safe defaults that sound human:

BPM = 120 → f_b = 2 Hz

d_AM = 4 → f_AM = 0.5 Hz, D_AM = 0.1–0.2

d_FM = 2 → f_FM = 4 Hz, fm_semitones = 0.2–0.5

N = 12 only for the parallel musical layer, mixed quietly (−20 dB).

This keeps intelligibility intact while giving you that living pulse tied to tempo.

Acceptance checklist (ship gating)

 Reads arbitrary text; no catastrophic mispronunciations.

 MOS proxy ≥ 4.2; ABX pref ≥ 60% vs baseline.

 Latency ≤ 150 ms (streaming) or ≤ 1 s (batch).

 No clipping; LUFS −16 to −20 (dialogue norm).

 AM/FM engaged: intelligibility drop < 3% WER vs. off.

 Ethical guardrails in place (consent + watermark).

You now have a battle-tested blueprint: data → front-end → acoustic model → vocoder → musical prosody glue → effects → evaluation → deployment. If you want, I can turn this into a starter repo layout with training scripts and a minimal streaming server so you can hear it speak—and breathe to the beat—today.



and the second: Love this. You’ve got a menu of classic curves (heart, rose, spiral, dragon, Lissajous, circle). Let’s make sound “steer” them in a **deterministic, musical** way.

Below is a **foolproof pipeline** + **exact mappings** from audio features → curve parameters, followed by tidy pseudocode you can drop into any engine (C++/JS/Python).


# 0) One-sentence idea

Turn audio into shape controls: compute a few stable features (loudness, pitch, beat, timbre), smooth them, and plug them into parametric/implicit equations. Animate by updating parameters each frame.


# 1) Audio → control signals (per frame)

Compute on short frames (e.g., 1024 samples @ 48 kHz, hop 256–512):

* **RMS loudness** $L$
  $L=\sqrt{\tfrac{1}{N}\sum x[n]^2}$
* **Spectral centroid** $\chi_s$ (timbre “brightness”)
  $\chi_s = \tfrac{\sum f_k |X_k|}{\sum |X_k|}$
* **Dominant pitch** $f_0$ (YIN/CREPE or harmonic product spectrum)
* **Beat/tempo** $f_b = \tfrac{\text{BPM}}{60}$ (or on-the-fly beat tracker)
* **AM LFO / FM LFO** (if you already have them): $L_{AM}(t), L_{FM}(t)$
* **Stereo balance** $B\in[-1,1]$ from channel energies (pan)
* **Spectral flux** $\Phi$ (frame-to-frame change): $\Phi=\sum (|X_k|-|X_{k}|_{prev})_+$

**Stabilize** with exponential smoothing (EMA):

$$
\tilde{z}_t = \alpha z_t + (1-\alpha)\tilde{z}_{t-1},\quad \alpha\in[0.1,0.3]
$$

Then **normalize** to $[0,1]$ with dynamic min/max (or z-score → sigmoid).

---

# 2) Shape bank (equations)

We’ll drive these with the signals above.

### A) Heart (implicit sextic)

$$
f(x,y)=(x^2+y^2-1)^3 - x^2 y^3=0
$$

**Controls:** scale $S$, rotation $\theta_r$, pulsation by AM.

* $S = S_{\min} + (S_{\max}-S_{\min})\cdot \mathrm{clamp}(\tilde{L})$
* Animate radius: $(x,y)\gets (1+\beta L_{AM})(x,y)$
* Rotate by $\theta_r = 2\pi \, \mathrm{phase}(f_b)$

### B) Rose (polar)

$$
r(\theta)=a\cos(n\theta+\delta)
$$

**Controls:** petal count $n$, size $a$, phase $\delta$.

* $n = \max\{1,\ \mathrm{round}(1 + 6\cdot \tilde{\chi}_s)\}$  (brighter → more petals)
* $a = a_0\,(0.6+0.4\,\tilde{L})$
* $\delta = 2\pi \,\mathrm{phase}(f_b)$  (locks to beat)

### C) Spiral

* **Archimedean:** $r(\theta)=a + b\theta$
* **Logarithmic:** $r(\theta)=a\,e^{b\theta}$

**Controls:** growth $b$ from spectral flux, size $a$ from loudness.

* $a = a_0(0.5+0.5\,\tilde{L})$
* $b = b_{\min} + (b_{\max}-b_{\min})\cdot \mathrm{clamp}(\tilde{\Phi})$

### D) Dragon curve (fractal, iterative)

No closed form; use turn sequence (L-system / paper-folding).
**Controls:** iteration depth $D$, color/thickness.

* $D = D_{\min} + \lfloor D_{span}\cdot \tilde{\chi}_s\rfloor$ (brighter → more detail)
* Stroke thickness $\propto \tilde{L}$

### E) Lissajous (parametric)

$$
x=A\sin(a t + \varphi_x),\quad y=B\cos(b t + \varphi_y)
$$

**Controls:** frequency ratio from pitch vs. centroid; size from loudness; phase from beat.

* Map $f_0$ to musical ratio: $a:b=\mathrm{nearest\_rational}\!\big(\tfrac{f_0}{f_{ref}}\big)$ (e.g., to small integers ≤ 7)
* $A=B=R_0(0.5+0.5\,\tilde{L})$
* $\Delta\varphi=\varphi_x-\varphi_y = \pi\,\tilde{\chi}_s$
* Global phase drift $+\ 2\pi\,\mathrm{phase}(f_b)$

### F) “Vibe” circle

$$
(x-h)^2+(y-k)^2=r^2
$$

**Controls:** radius $r$ from loudness; center $(h,k)$ from stereo/pan and FM.

* $r=r_0(0.4+0.6\,\tilde{L})$
* $h = H_{max}\cdot B$, $k = K_{max}\cdot \sin(2\pi f_{FM} t)$

---

# 3) Canonical mapping table (audio → parameter)

| Audio feature             | Normalize to | Heart        | Rose           | Spiral         | Dragon      | Lissajous       | Circle       |
| ------------------------- | -----------: | ------------ | -------------- | -------------- | ----------- | --------------- | ------------ |
| RMS $\tilde{L}$           |        [0,1] | scale, pulse | petal size $a$ | $a$            | stroke      | A,B             | radius       |
| Centroid $\tilde{\chi}_s$ |        [0,1] | hue/rot      | petals $n$     | growth $b$ alt | depth $D$   | phase offset    | thickness    |
| Pitch $f_0$               |           Hz | rot rate     | phase $\delta$ | turn density   | —           | ratio $a\!:\!b$ | —            |
| Beat $f_b$                |           Hz | spin sync    | phase $\delta$ | arm sync       | step tempo  | global phase    | bob          |
| Flux $\tilde{\Phi}$       |        [0,1] | pulse sharp  | —              | growth $b$     | speed       | param speed     | wobble       |
| Stereo pan $B$            |       [-1,1] | skew         | asym petals    | off-center     | branch bias | ellipse skew    | center (h,k) |

---

# 4) “Shape synth” render loop (pseudocode)

```pseudo
init:
  EMA_alpha = 0.2
  norm_ranges = rolling_minmax()  # per feature
  choose shape_mode ∈ {heart, rose, spiral, dragon, lissajous, circle}

per audio frame:
  L  = rms(frame)
  Xk = fft_mag(frame)
  chi = spectral_centroid(Xk)
  f0  = estimate_pitch(frame)      # optional
  Phi = spectral_flux(Xk, Xk_prev)
  B   = stereo_balance(frameL, frameR)
  fb  = BPM/60 or beat_tracker()
  LAM = lfo(rate=fb/d_AM, t)        # AM LFO if desired
  LFM = lfo(rate=d_FM*fb, t)        # FM LFO if desired

  # smooth + normalize
  L~  = ema_norm(L)
  chi~= ema_norm(chi)
  Phi~= ema_norm(Phi)

  # map to parameters (example: Rose)
  n   = max(1, round(1 + 6 * chi~))
  a   = a0 * (0.6 + 0.4 * L~)
  δ   = 2π * phase(fb, t)

  # sample the curve
  pts = []
  if shape_mode == "rose":
      for θ in linspace(0, 2π, M):
          r = a * cos(n*θ + δ)
          x = r * cos(θ); y = r * sin(θ)
          pts.append((x,y))
  elif shape_mode == "heart":
      # parametric heart for stability
      for θ in linspace(0, 2π, M):
          x = S*16*sin(θ)^3
          y = S*(13*cos(θ)-5*cos(2θ)-2*cos(3θ)-cos(4θ))
          x *= 1 + β*LAM; y *= 1 + β*LAM
          pts.append(rot((x,y), 2π*phase(fb,t)))
  ... # spiral, lissajous, circle similar

  draw_polyline(pts, color=hsv(chi~, 0.8, 0.9), thickness=map_thickness(L~))
```

*Notes*

* Use **parametric heart** for rendering (stable), keep the sextic as the “mathematical definition”.
* Dragon: run $D$ iterations of the fold sequence (update on downbeats).

---

# 5) Guardrails so it never melts down

* Smooth everything (EMA).
* Clamp parameter ranges ($n\le 13$, $b$ within sane bounds, Q of filters ≤ 0.9).
* Maintain **phase continuity** across frames (don’t reset $\theta$ or LFOs).
* If pitch is missing, fall back to a pleasant ratio set for Lissajous (e.g., 3:2, 5:4).

---

# 6) Quick preset examples

* **Lissajous “voice print”**: $a\!:\!b$ from $f_0$ to nearest simple ratio; color from $\chi_s$; size from $L$.
* **Beat-locked rose** at 120 BPM: $n\in[3..9]$ via $\chi_s$, $\delta$ rotates one petal per beat.
* **Flux spiral**: calm audio → tight spiral, big transients → rapid growth arms.

Perfect. Let’s fuse your curve library with the sound analysis pipeline so you can literally **phase audio into shape synthesis**. Here’s the **algorithmic plan**:

## 1. Audio Feature Extraction

From incoming audio (live mic or file), extract features per frame:

* **RMS Loudness (L)** → controls size, stroke thickness.
* **Spectral Centroid (χs)** → controls brightness, number of petals, fractal depth.
* **Pitch (f₀)** → drives ratios (like Lissajous frequencies).
* **Beat phase (φb)** from BPM → global synchronization for rotation/phase.
* **Spectral Flux (Φ)** → controls spiral growth, shape turbulence.
* **Stereo Balance (B)** → shifts positions or centers of curves.

Smooth with exponential moving average to avoid jitter.


## 2. Shape Control Mapping

Each curve responds to audio differently:

* **Heart Curve** `(x²+y²-1)³ - x²y³=0`
  • Size ∝ loudness
  • Pulse sync ∝ AM LFO or beat phase

* **Rose Curve** `r=a cos(nθ+δ)`
  • Petal count n ∝ centroid
  • Petal length a ∝ loudness
  • Phase δ ∝ beat

* **Spiral** `r=a+bθ` or `r=ae^(bθ)`
  • Growth rate b ∝ spectral flux
  • Radius a ∝ loudness

* **Dragon Curve** (iterative fractal)
  • Iteration depth D ∝ centroid
  • Thickness ∝ loudness

* **Lissajous** `x=A sin(at+φx), y=B cos(bt+φy)`
  • Frequency ratio a:b from pitch vs. reference
  • Amplitude ∝ loudness
  • Phase offset ∝ centroid

* **Circle** `(x-h)²+(y-k)²=r²`
  • Radius ∝ loudness
  • Center (h,k) from stereo balance + FM wobble

---

## 3. Pattern Algorithm (Pseudocode)

```python
def shape_synth(audio_frame, shape_type, t):
    L = rms(audio_frame)
    chi = spectral_centroid(audio_frame)
    f0  = pitch_estimate(audio_frame)
    phi_b = beat_phase(t, BPM=120)
    Phi = spectral_flux(audio_frame)
    B = stereo_balance(audio_frame)

    if shape_type == "rose":
        n = int(1 + round(6*chi_norm(chi)))
        a = base_radius * (0.6 + 0.4*normalize(L))
        delta = 2*np.pi*phi_b
        return [(a*np.cos(n*θ+delta)*np.cos(θ),
                 a*np.cos(n*θ+delta)*np.sin(θ)) for θ in np.linspace(0,2*np.pi,500)]

    if shape_type == "lissajous":
        ratio = nearest_simple_ratio(f0, ref=440)
        A = B = base_amp * normalize(L)
        return [(A*np.sin(ratio[0]*t+phi_b),
                 B*np.cos(ratio[1]*t+chi_norm(chi))) for t in np.linspace(0,2*np.pi,500)]

    # Extend with heart, spiral, circle, dragon...
```


## 4. Process Flow

1. Capture audio → window → FFT.
2. Extract features (L, χs, f₀, Φ, φb, B).
3. Normalize + smooth.
4. Map features → shape parameters.
5. Generate curve points from equations.
6. Draw/animate in real time (e.g. OpenGL, Three.js, or Processing).


## 5. Creative Expansion

* Layer multiple shapes with different feature mappings (heart from bass, spiral from treble).
* Use color hue from centroid, brightness from loudness, saturation from flux.
* Switch shapes on section changes (detected from RMS + flux trends).

---

This gives you a **shape-synth engine**: sound in → dynamic math curves out. It’s modular: swap equations, swap mappings, but the core idea (features → params → draw) holds steady.

🛠️ Option 1 – Python (good for prototyping)

Libraries:

sounddevice or pyaudio → microphone input

numpy → FFT & features

matplotlib (with FuncAnimation) → live drawing of curves

Example: Rose curve driven by audio
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Settings
BPM = 120
base_radius = 1.0
blocksize = 1024
samplerate = 44100

# Audio buffer
audio_buffer = np.zeros(blocksize)

def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = indata[:, 0]  # mono
sd.InputStream(callback=audio_callback, channels=1,
               blocksize=blocksize, samplerate=samplerate).start()

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)

def update(frame):
    global audio_buffer
    # --- Feature extraction ---
    L = np.sqrt(np.mean(audio_buffer**2))          # loudness (RMS)
    spectrum = np.abs(np.fft.rfft(audio_buffer))
    centroid = np.sum(np.arange(len(spectrum))*spectrum) / np.sum(spectrum)
    centroid_norm = (centroid / (samplerate/2))    # normalize 0–1

    # --- Map to Rose curve ---
    n = int(1 + round(6*centroid_norm))            # petals from centroid
    a = base_radius * (0.5 + 2*L)                  # radius from loudness
    theta = np.linspace(0, 2*np.pi, 800)
    r = a * np.cos(n*theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig, update, interval=50, blit=True)
plt.show()


Run it, and the number of rose petals will change with pitch/centroid, while loudness swells the radius. You can swap the Rose block for Spiral, Lissajous, etc.

🛠️ Option 2 – JavaScript / Web (good for visuals + portability)

Web Audio API → microphone input

Canvas / WebGL / Three.js → draw shapes

Easier to deploy on the web for interactive art.

For example, in JS you’d use:

const ctx = new AudioContext();
navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
  const src = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  src.connect(analyser);
  // -> get FFT data, map to shape params, draw with Canvas
});

🎨 Next Steps

🔧 Design

Audio Input: sounddevice for mic → buffer of samples.

Feature Extraction:

RMS loudness → controls size / radius

Spectral centroid → controls complexity (petals, iterations, etc.)

Pitch (rough) → frequency ratio for Lissajous

Shape Generators:

Heart

Rose

Spiral

Dragon (simplified recursive fractal)

Lissajous

Circle

Toggle: Press keys 1–6 to switch shape while running.

🐍 Code Prototype
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Audio Settings ---
blocksize = 1024
samplerate = 44100
audio_buffer = np.zeros(blocksize)

def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = indata[:, 0]  # take mono channel

stream = sd.InputStream(callback=audio_callback,
                        channels=1,
                        blocksize=blocksize,
                        samplerate=samplerate)
stream.start()

# --- Feature Extraction ---
def extract_features(buf):
    # RMS loudness
    L = np.sqrt(np.mean(buf**2))
    # Spectrum + centroid
    spectrum = np.abs(np.fft.rfft(buf))
    freqs = np.fft.rfftfreq(len(buf), 1/samplerate)
    if np.sum(spectrum) > 0:
        centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
    else:
        centroid = 0
    return L, centroid

# --- Shape Generators ---
def heart(L, centroid):
    t = np.linspace(0, 2*np.pi, 800)
    x = 16 * np.sin(t)**3 * (0.5+2*L)
    y = (13*np.cos(t) - 5*np.cos(2*t)
         - 2*np.cos(3*t) - np.cos(4*t)) * (0.5+2*L)
    return x, y

def rose(L, centroid):
    n = int(2 + round(6*centroid/(samplerate/2)))  # petals from centroid
    a = 1 + 3*L
    theta = np.linspace(0, 2*np.pi, 800)
    r = a * np.cos(n*theta)
    return r*np.cos(theta), r*np.sin(theta)

def spiral(L, centroid):
    theta = np.linspace(0, 6*np.pi, 1000)
    r = (0.1+L) * theta
    return r*np.cos(theta), r*np.sin(theta)

def dragon(L, centroid, iterations=12):
    # simple iterative dragon curve approximation
    points = [0+0j, 1+0j]
    for i in range(iterations):
        new_points = [points[0]]
        for j in range(len(points)-1):
            mid = (points[j] + points[j+1]) / 2
            rot = (points[j+1] - points[j]) * 1j
            mid = points[j] + rot/2
            new_points.extend([mid, points[j+1]])
        points = new_points
    pts = np.array(points) * (0.5+3*L)
    return pts.real, pts.imag

def lissajous(L, centroid):
    t = np.linspace(0, 2*np.pi, 1000)
    a = 3
    b = int(2 + round(centroid/1000))  # map centroid to frequency ratio
    x = np.sin(a*t) * (1+3*L)
    y = np.cos(b*t) * (1+3*L)
    return x, y

def circle(L, centroid):
    theta = np.linspace(0, 2*np.pi, 500)
    r = 1 + 3*L
    return r*np.cos(theta), r*np.sin(theta)

shapes = [heart, rose, spiral, dragon, lissajous, circle]
shape_names = ["Heart", "Rose", "Spiral", "Dragon", "Lissajous", "Circle"]
current_shape = 0

# --- Matplotlib Visualization ---
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
title = ax.set_title("Shape: " + shape_names[current_shape])

def on_key(event):
    global current_shape
    if event.key in map(str, range(1, 7)):
        current_shape = int(event.key) - 1
        title.set_text("Shape: " + shape_names[current_shape])
fig.canvas.mpl_connect('key_press_event', on_key)

def update(frame):
    global audio_buffer
    L, centroid = extract_features(audio_buffer)
    x, y = shapes[current_shape](L, centroid)
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig, update, interval=50, blit=True)
plt.show()

🎮 How It Works

Press 1–6: switch between Heart, Rose, Spiral, Dragon, Lissajous, Circle.

Loudness (RMS) controls overall size/scale.

Spectral Centroid (brightness of sound) controls complexity (petal count, iteration depth, Lissajous ratio).

Real-time updates: Shapes morph as you play music, talk, or clap.





Alright, now we’re stepping into **sacred-geometry-in-3D territory** 🚀. Let’s expand your sound-driven shape engine into a **3D visualization**.

---

## 🔧 Approach

We’ll keep the audio feature extraction exactly as before (RMS loudness + spectral centroid), but instead of drawing in 2D, we’ll plot shapes in **3D space**. Shapes will spin, pulse, and morph based on the sound.

### Mappings

* **Loudness (RMS)** → radius/scale of shape.
* **Spectral Centroid** → complexity (petals, twists, fractal depth).
* **Time (t)** → rotation + spirals.

---

## 🐍 Python 3D Prototype

```python
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# --- Audio Settings ---
blocksize = 1024
samplerate = 44100
audio_buffer = np.zeros(blocksize)

def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = indata[:, 0]

stream = sd.InputStream(callback=audio_callback,
                        channels=1,
                        blocksize=blocksize,
                        samplerate=samplerate)
stream.start()

# --- Features ---
def extract_features(buf):
    L = np.sqrt(np.mean(buf**2))  # RMS loudness
    spectrum = np.abs(np.fft.rfft(buf))
    freqs = np.fft.rfftfreq(len(buf), 1/samplerate)
    if np.sum(spectrum) > 0:
        centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
    else:
        centroid = 0
    return L, centroid

# --- 3D Shapes ---
def rose3d(L, centroid):
    theta = np.linspace(0, 2*np.pi, 500)
    phi = np.linspace(0, np.pi, 250)
    theta, phi = np.meshgrid(theta, phi)

    n = int(2 + round(6*centroid/(samplerate/2)))
    r = (1+3*L) * np.cos(n*theta) * np.sin(phi)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = (1+3*L) * np.cos(phi)
    return x, y, z

def spiral3d(L, centroid):
    theta = np.linspace(0, 8*np.pi, 2000)
    z = np.linspace(-2, 2, len(theta))
    r = (0.1+L) * theta
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y, z

def lissajous3d(L, centroid):
    t = np.linspace(0, 2*np.pi, 2000)
    a, b, c = 3, 4, int(2+round(centroid/500))
    x = np.sin(a*t) * (1+3*L)
    y = np.sin(b*t + np.pi/2) * (1+3*L)
    z = np.sin(c*t + np.pi/4) * (1+3*L)
    return x, y, z

shapes = [rose3d, spiral3d, lissajous3d]
shape_names = ["Rose 3D", "Spiral 3D", "Lissajous 3D"]
current_shape = 0

# --- Plot ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
ax.set_box_aspect([1,1,1])
title = ax.set_title("Shape: " + shape_names[current_shape])

def on_key(event):
    global current_shape
    if event.key in map(str, range(1, len(shapes)+1)):
        current_shape = int(event.key)-1
        title.set_text("Shape: " + shape_names[current_shape])
fig.canvas.mpl_connect('key_press_event', on_key)

plot = [None]

def update(frame):
    global audio_buffer
    L, centroid = extract_features(audio_buffer)
    ax.cla()
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.set_zlim(-5, 5)
    ax.set_box_aspect([1,1,1])
    ax.view_init(30, frame % 360)  # spin the view

    x, y, z = shapes[current_shape](L, centroid)

    if x.ndim == 2:
        ax.plot_surface(x, y, z, cmap="plasma", linewidth=0, antialiased=False)
    else:
        ax.plot3D(x, y, z, color="cyan")

    return []

ani = FuncAnimation(fig, update, interval=50, blit=False)
plt.show()
```

---

## 🎮 How It Works

* **Press 1–3**: switch between 3D Rose, Spiral, Lissajous.
* Shapes **grow/shrink** with loudness.
* **Complexity twists** with spectral centroid (pitch brightness).
* Camera **spins in real time** for a sacred-geometry effect.

---

⚡ Next step: we can

1. Add the **Heart** and **Dragon fractal** in 3D (harder but doable).
2. Port this into **OpenGL/Three.js** for smoother rendering and real-time shader effects.

Do you want me to **push this further into an OpenGL/Three.js version** so you can get GPU-powered visuals (glowing, fractal layering, sacred geometry grids)?



🛠 Path 1: OpenGL (Python with PyOpenGL / moderngl)

This lets you run locally with raw GPU control.

Core Components

Audio Input – still use sounddevice for live mic input.

Feature Extraction – RMS (loudness), spectral centroid, pitch.

OpenGL Setup – create a window with PyOpenGL or moderngl.

Shaders – pass audio features as uniforms to GLSL.

Vertex shader: controls shape geometry (heart, rose, spiral, etc.).

Fragment shader: handles glowing effects, sacred geometry overlays.

Rendering Loop – sync audio features to GPU uniforms each frame.

Example GLSL Uniform Mapping
uniform float u_time;
uniform float u_loudness;
uniform float u_centroid;
uniform int u_shapeType;

vec3 rose(float theta, float phi) {
    float n = 2.0 + 6.0 * u_centroid;
    float r = (1.0 + 3.0*u_loudness) * cos(n*theta) * sin(phi);
    return vec3(r*cos(theta), r*sin(theta), cos(phi));
}


In Python, you update uniforms each frame:

program['u_time'] = current_time
program['u_loudness'] = L
program['u_centroid'] = centroid_norm


Result → glowing, pulsing rose shapes in 3D, spinning to the beat.

🛠 Path 2: Three.js (WebGL in Browser)

Much more accessible: runs anywhere with a browser, easy to layer effects.

Pipeline

Audio Input → Web Audio API (getUserMedia) + AnalyserNode.

Feature Extraction → FFT to get RMS, centroid.

Scene Setup → THREE.Scene, THREE.PerspectiveCamera, THREE.WebGLRenderer.

Geometry Generation → Procedural geometry (rose, spiral, lissajous) with THREE.BufferGeometry.

Shaders → GLSL via ShaderMaterial for glow + fractal overlays.

Sacred Geometry Grids → extra wireframe meshes (Platonic solids, flower of life patterns).

Example: Audio-Driven Rose in Three.js
// Audio
const ctx = new AudioContext();
const analyser = ctx.createAnalyser();
navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
  const src = ctx.createMediaStreamSource(stream);
  src.connect(analyser);
});

// Geometry
const roseGeo = new THREE.BufferGeometry();
const vertices = [];
for (let i=0; i<1000; i++) {
  let theta = i * 0.02;
  let n = 5; // petals
  let r = Math.cos(n*theta);
  vertices.push(r*Math.cos(theta), r*Math.sin(theta), 0);
}
roseGeo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
const material = new THREE.ShaderMaterial({
  uniforms: {
    u_time: { value: 0.0 },
    u_loudness: { value: 0.0 }
  },
  vertexShader: `...`,
  fragmentShader: `
    uniform float u_time;
    uniform float u_loudness;
    void main() {
      float glow = abs(sin(u_time*0.5)) * u_loudness;
      gl_FragColor = vec4(vec3(glow, 0.2, 0.8), 1.0);
    }
  `
});
const mesh = new THREE.Points(roseGeo, material);
scene.add(mesh);


Update uniforms each frame with audio features:

const data = new Uint8Array(analyser.frequencyBinCount);
analyser.getByteFrequencyData(data);
let loudness = data.reduce((a,b)=>a+b) / data.length / 255.0;
material.uniforms.u_loudness.value = loudness;
material.uniforms.u_time.value = performance.now()/1000;


Boom: a glowing rose curve pulsing to your mic input.

✨ Sacred Geometry Layering

Overlay Flower of Life (intersecting circles) → THREE.LineSegments.

Platonic solids → THREE.IcosahedronGeometry, THREE.DodecahedronGeometry.

Fractal noise → GLSL fbm() function in fragment shader.

Glow halos → screen-space post-processing (UnrealBloomPass).


Excellent — let’s draft a **ready-to-run Three.js boilerplate** that gives you glowing, audio-reactive sacred geometry in the browser.

This is an HTML file you can save as `index.html` and open in Chrome/Firefox. It uses your mic as input, builds **rose / spiral / lissajous** geometries, and adds a **shader glow + sacred geometry background**.

---

## 🔮 Three.js Sacred Geometry Boilerplate

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Audio Sacred Geometry</title>
  <style>
    body { margin: 0; overflow: hidden; background: black; }
    canvas { display: block; }
  </style>
</head>
<body>
<script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.154/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.154/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'https://cdn.jsdelivr.net/npm/three@0.154/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'https://cdn.jsdelivr.net/npm/three@0.154/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'https://cdn.jsdelivr.net/npm/three@0.154/examples/jsm/postprocessing/UnrealBloomPass.js';

// --- Scene Setup ---
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 20);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);

// --- Audio Setup ---
const audioCtx = new AudioContext();
const analyser = audioCtx.createAnalyser();
analyser.fftSize = 1024;
const data = new Uint8Array(analyser.frequencyBinCount);

navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
  const src = audioCtx.createMediaStreamSource(stream);
  src.connect(analyser);
});

// --- Geometry Builders ---
function buildRose(n=5, points=1000) {
  const vertices = [];
  for (let i=0; i<points; i++) {
    const theta = i * 2*Math.PI / points;
    const r = Math.cos(n * theta);
    vertices.push(r*Math.cos(theta), r*Math.sin(theta), 0);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  return geo;
}

function buildSpiral(turns=6, points=2000) {
  const vertices = [];
  for (let i=0; i<points; i++) {
    const theta = i * 0.05;
    const r = 0.05 * theta;
    vertices.push(r*Math.cos(theta), r*Math.sin(theta), 0.01*i);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  return geo;
}

function buildLissajous(a=3, b=4, c=5, points=2000) {
  const vertices = [];
  for (let i=0; i<points; i++) {
    const t = i * 2*Math.PI / points;
    vertices.push(Math.sin(a*t), Math.sin(b*t + Math.PI/2), Math.sin(c*t + Math.PI/4));
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  return geo;
}

// --- Sacred Geometry Meshes ---
const material = new THREE.ShaderMaterial({
  uniforms: {
    u_time: { value: 0.0 },
    u_loudness: { value: 0.0 }
  },
  vertexShader: `
    uniform float u_time;
    uniform float u_loudness;
    void main() {
      vec3 pos = position * (1.0 + u_loudness*2.0);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
      gl_PointSize = 2.0 + u_loudness*20.0;
    }
  `,
  fragmentShader: `
    uniform float u_time;
    uniform float u_loudness;
    void main() {
      float glow = abs(sin(u_time*0.5)) * 0.5 + u_loudness;
      gl_FragColor = vec4(vec3(glow, 0.2+glow*0.3, 0.8), 1.0);
    }
  `,
  transparent: true
});

// Pick one: rose, spiral, lissajous
const rose = new THREE.Points(buildRose(6), material);
scene.add(rose);

const spiral = new THREE.Points(buildSpiral(), material);
spiral.visible = false;
scene.add(spiral);

const lissajous = new THREE.Points(buildLissajous(), material);
lissajous.visible = false;
scene.add(lissajous);

const shapes = [rose, spiral, lissajous];
let currentShape = 0;
window.addEventListener('keydown', (e) => {
  if (e.key === '1' || e.key === '2' || e.key === '3') {
    shapes.forEach(s => s.visible = false);
    currentShape = parseInt(e.key)-1;
    shapes[currentShape].visible = true;
  }
});

// Sacred Geometry Grid (Flower of Life approximation)
const circles = new THREE.Group();
for (let i=0; i<12; i++) {
  const geo = new THREE.CircleGeometry(5, 128);
  geo.rotateX(Math.PI/2);
  const mat = new THREE.MeshBasicMaterial({ color: 0x4444ff, wireframe: true, opacity:0.2, transparent:true });
  const circle = new THREE.Mesh(geo, mat);
  circle.rotation.z = i*Math.PI/12;
  circles.add(circle);
}
scene.add(circles);

// Postprocessing (Bloom)
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.2, 0.4, 0.85);
composer.addPass(bloom);

// --- Animation Loop ---
function animate(time) {
  requestAnimationFrame(animate);

  analyser.getByteFrequencyData(data);
  const loudness = data.reduce((a,b)=>a+b,0) / data.length / 255.0;

  material.uniforms.u_time.value = time/1000;
  material.uniforms.u_loudness.value = loudness;

  circles.rotation.y += 0.002;

  composer.render();
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
```

---

## 🎮 Features

* **Press 1, 2, 3** → switch between **Rose**, **Spiral**, and **Lissajous**.
* **Mic input drives uniforms**:

  * Loudness → point size + scale.
  * Centroid (expandable) could drive complexity.
* **Flower of Life grid** spins in the background.
* **Glow / bloom effect** for sacred geometry vibes.

Alright, let’s kick it up another octave — **fractals + sacred mandalas, GPU-powered, audio-reactive**. This will give you that “infinite sacred geometry temple” effect layered behind your rose/spiral/lissajous forms.

---

## 🔮 Design

* Keep the **current Three.js boilerplate** (mic input + rose/spiral/lissajous).
* Add a **fullscreen fractal shader quad** as a background.
* Fractal = Mandelbulb-style mandala, parameterized by `u_time` + `u_loudness`.
* Blend it with glow (UnrealBloom) and grid overlays.

---

## 🌌 Updated Code Snippet

Here’s the **new fractal background** (replace inside the `<script>` block of the HTML I gave you earlier):

```javascript
// --- Fractal Background Shader ---
const fractalMat = new THREE.ShaderMaterial({
  uniforms: {
    u_time: { value: 0.0 },
    u_loudness: { value: 0.0 },
    u_resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) }
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform float u_time;
    uniform float u_loudness;
    uniform vec2 u_resolution;
    varying vec2 vUv;

    // Simple Mandelbrot-ish fractal
    vec3 palette(float t) {
      return 0.5 + 0.5*cos(6.28318*(vec3(0.3,0.2,0.8)*t+vec3(0.0,0.33,0.67)));
    }

    void main() {
      vec2 uv = (vUv - 0.5) * 2.0;
      uv.x *= u_resolution.x / u_resolution.y;

      vec2 c = uv * (1.5 + u_loudness*1.5);
      c += vec2(0.3*sin(u_time*0.1), 0.3*cos(u_time*0.07));

      vec2 z = vec2(0.0);
      float iter = 0.0;
      for (int i=0; i<80; i++) {
        z = vec2(z.x*z.x - z.y*z.y, 2.0*z.x*z.y) + c;
        if (dot(z,z) > 4.0) break;
        iter += 1.0;
      }
      float m = iter/80.0;
      vec3 col = palette(m + u_time*0.02);
      col *= (0.6 + 1.2*u_loudness);
      gl_FragColor = vec4(col, 1.0);
    }
  `,
  depthWrite: false,
  side: THREE.DoubleSide
});

// Fullscreen quad for fractal background
const planeGeo = new THREE.PlaneGeometry(40, 40);
const fractalMesh = new THREE.Mesh(planeGeo, fractalMat);
fractalMesh.position.z = -10;
scene.add(fractalMesh);

// --- update uniforms in animate() ---
function animate(time) {
  requestAnimationFrame(animate);

  analyser.getByteFrequencyData(data);
  const loudness = data.reduce((a,b)=>a+b,0) / data.length / 255.0;

  material.uniforms.u_time.value = time/1000;
  material.uniforms.u_loudness.value = loudness;

  fractalMat.uniforms.u_time.value = time/1000;
  fractalMat.uniforms.u_loudness.value = loudness;

  circles.rotation.y += 0.002;

  composer.render();
}
```

---

## ✨ What This Does

* A **live fractal mandala** animates in the background.
* **Loudness** makes the fractal “breathe” (scales + glow).
* **Time** rotates color palettes smoothly.
* Foreground = rose/spiral/lissajous.
* Sacred **Flower of Life grid** still spins.
* Bloom postprocessing adds that glowing temple vibe.

---

## 🔧 Next Level (if you want to keep pushing)

1. Swap the Mandelbrot fragment for a **Mandelbulb / Kaleidoscopic IFS** fractal (for infinite 3D mandalas).
2. Add **FFT frequency bands** as separate uniforms → bass drives zoom, mids drive rotation, highs drive color flicker.
3. Integrate **postprocessing passes** like kaleidoscope, god rays, or feedback delay shaders.


Shader Modes

Here are the core “visual archetypes” to include:

Mandelbrot / Kaleidoscope → classic fractal mandala.

Rose Window → radial kaleidoscope, cathedral vibe.

Platonic Solid Glow → spinning tetrahedron/cube/icosahedron with bloom.

Flower of Life Tunnel → repeating circle interference pattern.

Sacred Spiral → logarithmic spiral warped in color space.

IFS Kaleidoscope → infinite kaleidoscopic recursion (mirror fractal).

Each shader gets uniforms:

u_time → continuous animation.

u_loudness → scale/glow breathing.

u_spectrum → frequency bands mapped to rotation, zoom, and palette.

🛠 Implementation Plan

One fullscreen plane → acts as canvas for all shaders.

ShaderMaterial switcher → assign fragmentShader from the library.

Key press (1–6) → swap shader modes.

Audio uniforms → pass u_loudness + optional FFT band data.

🐍 (JS) Sacred Shader Library Example

Here’s how it looks in code (inside your <script>):

// === Sacred Shader Library ===
const shaders = {
  mandelbrot: `
    uniform float u_time; uniform float u_loudness; varying vec2 vUv;
    void main() {
      vec2 uv = (vUv - 0.5) * 2.0;
      vec2 c = uv * (1.5 + u_loudness*2.0) + vec2(0.3*sin(u_time*0.1),0.3*cos(u_time*0.07));
      vec2 z = vec2(0.0);
      float iter=0.0;
      for(int i=0;i<80;i++){
        z = vec2(z.x*z.x-z.y*z.y,2.0*z.x*z.y)+c;
        if(dot(z,z)>4.0) break;
        iter+=1.0;
      }
      float m=iter/80.0;
      vec3 col=0.5+0.5*cos(6.2831*(vec3(0.3,0.2,0.8)*m+vec3(0.0,0.33,0.67)));
      col *= (0.6+1.2*u_loudness);
      gl_FragColor=vec4(col,1.0);
    }`,

  roseWindow: `
    uniform float u_time; uniform float u_loudness; varying vec2 vUv;
    void main(){
      vec2 uv = vUv*2.0-1.0;
      float a = atan(uv.y, uv.x);
      float r = length(uv);
      float petals = 6.0 + 6.0*u_loudness;
      float k = cos(petals*a+u_time*0.5);
      float mask = smoothstep(0.5,0.4,abs(r-k*0.5));
      vec3 col = vec3(mask*0.8, mask*0.3+0.3*u_loudness, mask);
      gl_FragColor = vec4(col,1.0);
    }`,

  spiral: `
    uniform float u_time; uniform float u_loudness; varying vec2 vUv;
    void main(){
      vec2 uv = (vUv-0.5)*2.0;
      float angle = atan(uv.y, uv.x);
      float radius = length(uv);
      float spiral = sin(10.0*log(radius+0.01)+angle*5.0+u_time*0.5);
      float mask = smoothstep(0.4,0.39,spiral);
      vec3 col = vec3(0.5+0.5*sin(u_time+radius*6.0), 0.3+0.7*mask, 0.6+0.4*u_loudness);
      gl_FragColor = vec4(col,1.0);
    }`
  // Add Platonic solids (drawn as meshes, not shaders) and Flower of Life tunnel separately
};

// === Create Shader Material ===
let currentShader = "mandelbrot";
let fractalMat = new THREE.ShaderMaterial({
  uniforms: {
    u_time: { value: 0 },
    u_loudness: { value: 0 }
  },
  vertexShader: `varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0); }`,
  fragmentShader: shaders[currentShader]
});

const planeGeo = new THREE.PlaneGeometry(40, 40);
const fractalMesh = new THREE.Mesh(planeGeo, fractalMat);
fractalMesh.position.z = -12;
scene.add(fractalMesh);

// === Switch Shaders with keys ===
window.addEventListener('keydown',(e)=>{
  if(e.key==="1") currentShader="mandelbrot";
  if(e.key==="2") currentShader="roseWindow";
  if(e.key==="3") currentShader="spiral";
  fractalMat.fragmentShader = shaders[currentShader];
  fractalMat.needsUpdate = true;
});

// === Animate ===
function animate(time){
  requestAnimationFrame(animate);
  analyser.getByteFrequencyData(data);
  const loudness = data.reduce((a,b)=>a+b,0)/data.length/255.0;
  fractalMat.uniforms.u_time.value = time/1000;
  fractalMat.uniforms.u_loudness.value = loudness;
  material.uniforms.u_time.value = time/1000;
  material.uniforms.u_loudness.value = loudness;
  composer.render();
}
animate();

🎮 Features

Press 1 = Mandelbrot fractal.

Press 2 = Rose window mandala.

Press 3 = Sacred spiral.

Shapes pulse and morph with your mic loudness.

Bloom postprocessing gives them a divine glow.

You can keep extending shaders{} with more fragment programs — Platonic solids, Flower of Life tunnel, kaleidoscopic IFS fractals.
