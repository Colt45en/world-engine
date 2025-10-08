Perfect — let’s rewrite this in a way that makes the **atomic breakdown** and the **reconstruction** very explicit. Think of it as two halves of the same figure-8:

---

# 🔽 Atomic Breakdown (Downward Funnel)

We start with a **word** or a **math expression**. The system **tears it apart** into morphemes/operators, then converts each into atomic transforms (Stationary Units).

---

### Example:

Input: `"rebuild"`
Math analog: `2 + 3`

---

### English Breakdown

* Word: **rebuild**
* Prefix: `re-` → repetition / restore
* Root: `build` → construct

Atomic morphemes:

```json
[
  { "type": "prefix", "text": "re", "effect": "scale<1, α=0.95" },
  { "type": "root", "text": "build", "effect": "scale>1, δℓ=+1" }
]
```

---

### Math Breakdown

* Expression: `2 + 3`
* Atom 1: `2` (constant)
* Atom 2: `3` (constant)
* Operator: `+` (combine)

Atomic math ops:

```json
[
  { "type": "number", "value": 2 },
  { "type": "number", "value": 3 },
  { "type": "operator", "op": "+", "effect": "combine" }
]
```

---

# 🔼 Reconstruction (Upward Spiral)

Now we **rebuild meaning** by recomposing the atoms with the Button/Morpheme transforms.

---

### English Reconstruction

* Apply `re-` matrix: compress slightly, bias shift.
* Apply `build` matrix: expand and increase level.

Reconstructed SU:

```json
{
  "word": "rebuild",
  "M": [[1.15,0,0],[0,1.15,0],[0,0,1.05]],
  "b": [0.05,0.05,0],
  "level": 1,
  "kappa": 0.95
}
```

---

### Math Reconstruction

* Apply operator `+`:

  ```
  result = 2 + 3 = 5
  ```
* Combine SUs for 2 and 3 → new SU with aggregated state.

Reconstructed SU:

```json
{
  "expression": "2+3",
  "value": 5,
  "M": [[1,0],[0,1]],
  "b": [5],
  "level": 1
}
```

---

# 🔁 Figure-8 Connection

Both streams — English and Math — followed the **same path**:

1. **Break into atoms** (morphemes/operators).
2. **Map each atom to a linear transform** (M, b, C).
3. **Recombine upward** into a **higher-level Stationary Unit** with new level and kappa.

---

## Visualization

```
Downward Funnel        Upward Spiral
----------------       ----------------
"rebuild"             [RB] Button apply
   re-                 restore matrix
   build               construct matrix
----------------       ----------------
"2+3"                 sum operator
   2                   constant SU
   3                   constant SU
----------------       ----------------
Atoms extracted       Meaning recomposed
```

---

✨ Difference Highlight:

* **Atomic breakdown** = destructive: takes apart into minimal units (`re`, `build`, `+`, `2`, `3`).
* **Reconstruction** = generative: applies transformations and recombines into a meaningful composite (`rebuild`, `5`).

---

Would you like me to **add this as a debug mode into your `MetaFractalEngine`** so you can literally call something like `engine.debug("rebuild")` and it will print both:

* atomic breakdown
* reconstruction trace (step by step)?
Great — let’s add a **debug mode** into your `MetaFractalEngine` so you can see both sides:

1. **Downward Funnel** → atomic breakdown (morphemes/operators → StationaryUnit atoms).
2. **Upward Spiral** → reconstruction trace (how each Button/Morpheme transform recomposes meaning).

Here’s the patch:

```js
/************************************
 * Debug Trace: Breakdown + Rebuild
 ************************************/
class MetaFractalEngine {
  // ... keep everything else the same ...

  debug(input) {
    const atomized = this._down(input);

    console.log("🔽 Atomic Breakdown");
    atomized.forEach(({ token, morph }) => {
      console.log(`Word: ${token}`);
      console.log(" Prefixes:", morph.prefixes);
      console.log(" Root:", morph.root);
      console.log(" Suffixes:", morph.suffixes);
      console.log(" Morphemes:", morph.morphemes.map(m => `${m.type}:${m.text}`).join(", "));
    });

    const candidates = this._up(atomized);

    console.log("\n🔼 Reconstruction Trace");
    candidates.forEach(c => {
      console.log(`Candidate ${c.id}`);
      console.log(` Button: ${c.btn}`);
      console.log(` SU.x: [${c.su.x.map(v => v.toFixed(3)).join(", ")}]`);
      console.log(` Level: ${c.su.level}, κ: ${c.su.kappa.toFixed(3)}`);
      console.log(` Score Parts:`, c.scoreParts);
    });

    return { atomized, candidates };
  }
}
```

---

### Usage

```js
import { runPipeline, inspectLayers, explain } from './meta-fractal-pipeline.js';

// Call debug mode directly:
__engine.debug("rebuild");

// Example output:
🔽 Atomic Breakdown
Word: rebuild
 Prefixes: [ 're' ]
 Root: build
 Suffixes: []
 Morphemes: prefix:re, root:build

🔼 Reconstruction Trace
Candidate cand_0_rebuild
 Button: RB
 SU.x: [1.200, 1.200, 1.050]
 Level: 1, κ: 0.950
 Score Parts: { morphComplexity: 1, buttonBoost: 0.2, kappa: 0.475 }
```

---

That way you’ll always see **both views** (atoms vs recomposition) side-by-side.
Got it ✅

We can add visuals **without touching your core algorithm** by keeping the **debug trace separate**. That means your `MetaFractalEngine` logic stays intact (pure math + linguistics), and we only attach a *visualization layer* on top.

Here’s how:

---

### 1. Extend Debug Output (Non-invasive)

Keep the debug method as is, but emit structured JSON:

```js
debug(input) {
  const atomized = this._down(input);
  const candidates = this._up(atomized);

  return {
    downward: atomized.map(({ token, morph }) => ({
      token,
      prefixes: morph.prefixes,
      root: morph.root,
      suffixes: morph.suffixes,
      morphemes: morph.morphemes
    })),
    upward: candidates.map(c => ({
      id: c.id,
      token: c.token,
      btn: c.btn,
      su: { x: c.su.x, level: c.su.level, kappa: c.su.kappa },
      score: c.score,
      scoreParts: c.scoreParts
    }))
  };
}
```

This keeps your pipeline **pure** but gives us structured data for rendering.

---

### 2. Canvas Visual Layer

Now add a standalone renderer (safe to remove anytime):

```js
export function renderDebug(ctx, debugData) {
  const { downward, upward } = debugData;

  // Funnel = downward atoms
  ctx.fillStyle = "cyan";
  downward.forEach((d, i) => {
    ctx.fillText(`${d.token} → [${d.morphemes.map(m => m.text).join("+")}]`, 20, 40 + i * 20);
  });

  // Spiral = upward candidates
  ctx.fillStyle = "magenta";
  upward.forEach((u, i) => {
    ctx.fillText(`↑ ${u.token} via ${u.btn} (κ=${u.su.kappa.toFixed(2)})`, 300, 40 + i * 20);
  });
}
```

---

### 3. Hook into Your Canvas

```html
<canvas id="metaCanvas" width="800" height="400"></canvas>
<script type="module">
  import { __engine, renderDebug } from './meta-fractal-pipeline.js';

  const ctx = document.getElementById('metaCanvas').getContext('2d');
  const dbg = __engine.debug("rebuild");
  renderDebug(ctx, dbg);
</script>
```

---

✅ **Result**

* Core math/linguistics untouched.
* Debug returns a **structured atomic → reconstruction trace**.
* Visual renderer is **optional** and modular.
* You get console + canvas at the same time.

---

Perfect ⚡️
Let’s upgrade the visualization so the **atomic breakdown funnels down** → then rebuilds in a **spiral (figure-8)** as it reconstructs.

---

## 🔹 Step 1: Keep Engine Untouched

We don’t touch your `MetaFractalEngine` math at all. We only add a **non-invasive debug view**:

```js
// meta-fractal-pipeline.js
export function getDebugTrace(input) {
  const atoms = __engine._down(input);
  const cands = __engine._up(atoms);
  return { atoms, cands };
}
```

---

## 🔹 Step 2: Animated Funnel → Spiral Renderer

This sits outside your engine, attached only if you want visuals:

```js
export function animateFunnelSpiral(ctx, debugData, t) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.save();
  ctx.translate(ctx.canvas.width/2, ctx.canvas.height/2);

  // Funnel: draw atoms flowing downward
  debugData.atoms.forEach((a, i) => {
    const y = i * 30 - 100;
    const x = Math.sin(t/500 + i) * 80;
    ctx.fillStyle = "cyan";
    ctx.fillText(`${a.token}`, x, y);
  });

  // Spiral (figure-8): draw reconstruction upward
  debugData.cands.forEach((c, i) => {
    const angle = t/800 + i;
    const radius = 40 + i * 10;
    const x = Math.sin(angle*2) * radius;
    const y = Math.sin(angle) * radius;
    ctx.fillStyle = "magenta";
    ctx.fillText(`${c.token}:${c.btn}`, x, y);
  });

  ctx.restore();
}
```

---

## 🔹 Step 3: Hook to Canvas Loop

```html
<canvas id="metaCanvas" width="800" height="500"></canvas>
<script type="module">
  import { getDebugTrace, animateFunnelSpiral } from './meta-fractal-pipeline.js';

  const ctx = document.getElementById('metaCanvas').getContext('2d');
  const trace = getDebugTrace("rebuild multi-scale");

  function loop(t) {
    animateFunnelSpiral(ctx, trace, t);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
</script>
```

---

## ✅ Effect

* Atoms flow **downwards in a funnel**.
* Reconstructed candidates swirl upward in a **spiral / figure-8 path**.
* Completely modular → **engine stays pure math**, visuals only listen.

---

Alright 🔥 let’s **sync the funnel–spiral animation with AM/FM phase modulation** so the **visuals dance with sound**. This way your pseudo-math/English engine connects directly to audio synthesis.

---

## 🔹 Step 1: Add Master Oscillator

We make a phaser with AM + FM modulators, locked to tempo:

```js
class MasterPhaser {
  constructor(audioCtx, bpm=120, dAM=4, dFM=2, depthAM=0.2, depthFM=6) {
    this.ctx = audioCtx;
    this.f0 = bpm / 60; // base pulse (2 Hz @ 120 BPM)

    // Carrier
    this.carrier = this.ctx.createOscillator();
    this.carrier.type = "sine";
    this.carrier.frequency.value = 220; // audible base

    // Gain for AM
    this.amGain = this.ctx.createGain();
    this.amGain.gain.value = depthAM;

    // AM modulator
    this.amOsc = this.ctx.createOscillator();
    this.amOsc.type = "sine";
    this.amOsc.frequency.value = this.f0 / dAM; // 0.5 Hz tremolo
    this.amOsc.connect(this.amGain).connect(this.carrier.frequency);

    // FM modulator
    this.fmOsc = this.ctx.createOscillator();
    this.fmOsc.type = "sine";
    this.fmOsc.frequency.value = this.f0 * dFM; // 4 Hz vibrato
    this.fmGain = this.ctx.createGain();
    this.fmGain.gain.value = depthFM;
    this.fmOsc.connect(this.fmGain).connect(this.carrier.frequency);

    // Master gain
    this.output = this.ctx.createGain();
    this.output.gain.value = 0.2;
    this.carrier.connect(this.output);
  }

  start() {
    this.carrier.start();
    this.amOsc.start();
    this.fmOsc.start();
  }
}
```

---

## 🔹 Step 2: Bind to Visual Funnel–Spiral

Update the spiral with **phase offsets** from the oscillators:

```js
export function animateFunnelSpiral(ctx, debugData, t, audioPhase) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.save();
  ctx.translate(ctx.canvas.width/2, ctx.canvas.height/2);

  // Funnel atoms (AM influence = vertical pulsation)
  debugData.atoms.forEach((a, i) => {
    const y = i * 30 - 100 + Math.sin(audioPhase.am) * 20;
    const x = Math.sin(t/500 + i) * 80;
    ctx.fillStyle = "cyan";
    ctx.fillText(`${a.token}`, x, y);
  });

  // Spiral candidates (FM influence = radial wobble)
  debugData.cands.forEach((c, i) => {
    const angle = t/800 + i;
    const baseR = 40 + i * 10;
    const radius = baseR + Math.sin(audioPhase.fm) * 10;
    const x = Math.sin(angle*2) * radius;
    const y = Math.sin(angle) * radius;
    ctx.fillStyle = "magenta";
    ctx.fillText(`${c.token}:${c.btn}`, x, y);
  });

  ctx.restore();
}
```

---

## 🔹 Step 3: Glue Audio ↔ Visual

```html
<canvas id="metaCanvas" width="800" height="500"></canvas>
<script type="module">
  import { getDebugTrace, animateFunnelSpiral, MasterPhaser } from './meta-fractal-pipeline.js';

  const ctx = document.getElementById('metaCanvas').getContext('2d');
  const trace = getDebugTrace("rebuild multi-scale");

  // Audio setup
  const audioCtx = new AudioContext();
  const phaser = new MasterPhaser(audioCtx);
  phaser.output.connect(audioCtx.destination);
  phaser.start();

  function loop(t) {
    const phase = {
      am: phaser.amOsc.frequency.value * t/1000,
      fm: phaser.fmOsc.frequency.value * t/1000
    };
    animateFunnelSpiral(ctx, trace, t, phase);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
</script>
```

---

## ✅ What Happens

* **Atoms (down funnel):** bounce with AM tremolo.
* **Candidates (spiral up):** swirl with FM vibrato.
* **Audio carrier:** produces actual sound while visuals mirror modulation.
* Everything is tempo-locked (`BPM → f0 → AM/FM modulators`).

---

Perfect 🔥 let’s wire **morphemes ↔ sound modulation**, so the *English morphology literally drives the AM/FM sound synthesis*.

---

## 🔹 Step 1: Extend Morphemes with Audio Effects

We already have `effects` in your `Morpheme` class. Let’s add AM/FM/audio hooks:

```js
class Morpheme {
  constructor(symbol, M, b, effects={}) {
    this.symbol = symbol;
    this.M = M;
    this.b = b;
    this.effects = effects;
    // new audio hooks
    this.audio = effects.audio || { amDepth: 0, fmDepth: 0, carrierShift: 0 };
  }
}
```

---

## 🔹 Step 2: Morpheme Registry with Audio Influence

Example:

* `re` (restore) = slow AM tremolo.
* `multi` = more harmonics → FM depth boost.
* `ing` = ongoing action → FM speed increase.

```js
function createMorphemeRegistry(dim=3){
  const R=new Map();

  R.set('re', new Morpheme('re',
    [[0.95,0,0],[0,1.05,0],[0,0,1]], [0,0,0],
    { deltaLevel:-1, description:'again/restore',
      audio:{ amDepth:0.3, fmDepth:0, carrierShift:0 }
    }));

  R.set('multi', new Morpheme('multi',
    [[1.4,0,0],[0,1.4,0],[0,0,1.15]], [0,0,0.05],
    { deltaLevel:1, description:'many/scale up',
      audio:{ amDepth:0, fmDepth:8, carrierShift:100 }
    }));

  R.set('ing', new Morpheme('ing',
    [[1,0.15,0],[0,1.15,0],[0,0,1]], [0,0,0],
    { description:'ongoing',
      audio:{ amDepth:0.1, fmDepth:4, carrierShift:0 }
    }));

  return R;
}
```

---

## 🔹 Step 3: MasterPhaser with Morph Hooks

Update modulation live when a morpheme is applied:

```js
class MasterPhaser {
  constructor(audioCtx, bpm=120) {
    this.ctx = audioCtx;
    this.f0 = bpm / 60;

    this.carrier = this.ctx.createOscillator();
    this.carrier.type = "sine";
    this.carrier.frequency.value = 220;

    this.amGain = this.ctx.createGain();
    this.amGain.gain.value = 0.2;

    this.amOsc = this.ctx.createOscillator();
    this.amOsc.type = "sine";
    this.amOsc.frequency.value = this.f0/4;
    this.amOsc.connect(this.amGain).connect(this.carrier.frequency);

    this.fmOsc = this.ctx.createOscillator();
    this.fmOsc.type = "sine";
    this.fmOsc.frequency.value = this.f0*2;
    this.fmGain = this.ctx.createGain();
    this.fmGain.gain.value = 6;
    this.fmOsc.connect(this.fmGain).connect(this.carrier.frequency);

    this.output = this.ctx.createGain();
    this.output.gain.value = 0.3;
    this.carrier.connect(this.output);
  }

  start(){ this.carrier.start(); this.amOsc.start(); this.fmOsc.start(); }

  applyMorpheme(m) {
    if(!m.audio) return;
    if(m.audio.amDepth !== undefined) this.amGain.gain.value = m.audio.amDepth;
    if(m.audio.fmDepth !== undefined) this.fmGain.gain.value = m.audio.fmDepth;
    if(m.audio.carrierShift !== undefined) this.carrier.frequency.value += m.audio.carrierShift;
  }
}
```

---

## 🔹 Step 4: Bind Engine ↔ Audio

When a morpheme is chosen in `_up()`, tell the phaser:

```js
class MetaFractalEngine {
  constructor(dim=3, phaser=null){
    this.dim=dim;
    this.su=new StationaryUnit(dim);
    this.buttons=createStandardButtons(dim);
    this.morpho=createLLEMorphologyV2();
    this.phaser=phaser; // hook audio engine
  }

  _up(atomized){
    const candidates=[];
    for(const item of atomized){
      const { token, morph }=item;
      let chosen='RB';
      if(morph.suffixes.includes('ize')) chosen='CV';
      else if(morph.prefixes.includes('multi')) chosen='UP';
      else if(morph.root==='move') chosen='MV';

      const btn=this.buttons.get(chosen);
      const suNext=btn?btn.apply(this.su):this.su.copy();

      // 🔹 if button maps to morpheme, push into phaser
      if(this.phaser){
        morph.morphemes.forEach(m=>this.phaser.applyMorpheme(m));
      }

      candidates.push({id:`cand_${token}`, token, morph, btn:chosen, su:suNext});
    }
    return candidates;
  }
}
```

---

## 🔹 Step 5: Visual + Audio Together

* **Visual funnel/spiral** = shows atomization/reconstruction.
* **Sound modulation** = mirrors morphemes applied.

Example:

* Type `rebuild multi-ing`

  * Visual: funnel → spiral with “re”, “build”, “multi”, “ing”.
  * Audio: tremolo (`re`), pitch shift (`multi`), FM wobble (`ing`).

---

Alright, here’s the next evolution. We’ll let your **tokens (morphemes)** not only affect AM/FM audio but also **emit visual tokens** in Phaser (JS game framework) so that sound, language, and shape are synchronized. This gives you the **“master phaser” + tempo-locked modulators + visual token stream** you asked for.

---

## 🔹 MasterPhaser with Tempo-Locked Modulators

* Two modulators:

  * AM LFO: locked at `f₀/d_AM`
  * FM LFO: locked at `f₀*d_FM`
* Carrier oscillator that morphs per morpheme.
* Optional filter: low-pass / peak.

```js
class MasterPhaser {
  constructor(audioCtx, bpm=120, dAM=4, dFM=2) {
    this.ctx = audioCtx;
    this.f0 = bpm/60;   // base pulse in Hz

    // Carrier (voice base)
    this.carrier = this.ctx.createOscillator();
    this.carrier.type = "sine";
    this.carrier.frequency.value = 220; // base voice pitch

    // AM modulation
    this.amGain = this.ctx.createGain();
    this.amGain.gain.value = 0.0;
    this.amOsc = this.ctx.createOscillator();
    this.amOsc.type = "sine";
    this.amOsc.frequency.value = this.f0/dAM;
    this.amOsc.connect(this.amGain).connect(this.carrier.frequency);

    // FM modulation
    this.fmGain = this.ctx.createGain();
    this.fmGain.gain.value = 0.0;
    this.fmOsc = this.ctx.createOscillator();
    this.fmOsc.type = "sine";
    this.fmOsc.frequency.value = this.f0*dFM;
    this.fmOsc.connect(this.fmGain).connect(this.carrier.frequency);

    // Optional filter (resonant lowpass)
    this.filter = this.ctx.createBiquadFilter();
    this.filter.type = "lowpass";
    this.filter.frequency.value = 800;
    this.filter.Q.value = 1;

    this.output = this.ctx.createGain();
    this.output.gain.value = 0.3;

    this.carrier.connect(this.filter).connect(this.output);
  }

  start() {
    this.carrier.start();
    this.amOsc.start();
    this.fmOsc.start();
  }

  // Apply morpheme audio shaping
  applyMorpheme(m) {
    if(!m.audio) return;
    if(m.audio.amDepth !== undefined) this.amGain.gain.value = m.audio.amDepth;
    if(m.audio.fmDepth !== undefined) this.fmGain.gain.value = m.audio.fmDepth;
    if(m.audio.carrierShift !== undefined) this.carrier.frequency.value += m.audio.carrierShift;
    if(m.audio.filterCut !== undefined) this.filter.frequency.value = m.audio.filterCut;
  }
}
```

---

## 🔹 Phaser Token Emission

Each morpheme creates a token in Phaser. Tokens drift/rotate in a **figure-8 path** around the spiral, mirroring the AM/FM phases.

```js
class TokenEmitter extends Phaser.Scene {
  constructor() {
    super("TokenEmitter");
    this.tokens = [];
  }

  preload(){ this.load.image('token','/assets/circle.png'); }

  create(){
    this.group = this.add.group();
  }

  emitMorpheme(morpheme){
    const token = this.add.sprite(400, 300, 'token');
    token.morpheme = morpheme.symbol;
    token.amDepth = morpheme.audio?.amDepth || 0;
    token.fmDepth = morpheme.audio?.fmDepth || 0;
    token.t = 0;
    this.group.add(token);
  }

  update(time, delta){
    this.group.children.iterate(tok=>{
      tok.t += delta*0.002;
      // figure-8 path (Lissajous)
      tok.x = 400 + Math.sin(tok.t*tok.amDepth+1) * 100;
      tok.y = 300 + Math.sin(tok.t*tok.fmDepth+2) * 60;
      tok.alpha = 0.5 + 0.5*Math.sin(tok.t);
    });
  }
}
```

---

## 🔹 Engine ↔ Audio ↔ Visual

When `_up()` chooses morphemes, you now:

```js
if(this.phaser){
  morph.morphemes.forEach(m=>{
    this.phaser.applyMorpheme(m);   // sound modulation
    if(window.phaserScene) window.phaserScene.emitMorpheme(m); // visual token
  });
}
```

---

## 🔹 What You Get

* **Language atoms (morphemes)** trigger:

  * AM/FM depth & phase shifts (speech-like wobble).
  * Filter sweeps (resonance = emotional tone).
  * Carrier pitch shifts (intonation).
* **Visual tokens** orbit in figure-8 loops that match those modulations.
* **Scaling**: Add morphemes → the modulation stack & tokens scale up seamlessly.

---
 let’s merge it directly into your `meta-fractal-pipeline.js` so that every pipeline run can trigger **sound + visuals**.

Here’s what I’ll do:

1. **Extend Morphemes with audio mappings**

   * Each prefix/suffix/root gets an `audio` profile:

     * `amDepth` (AM modulation depth)
     * `fmDepth` (FM modulation depth)
     * `carrierShift` (intonation)
     * `filterCut` (tone/brightness)

2. **Add `MasterPhaser` (WebAudio)** inside the file.

3. **Add `TokenEmitter` stub** that plugs into Phaser if it exists on the page.

4. **Modify `MetaFractalEngine._up()`** so that whenever a morpheme is applied, it also pushes into `MasterPhaser` + `TokenEmitter`.

---

### 🔹 Code Injection for `meta-fractal-pipeline.js`

```js
/************************************
 * A) Audio Engine: MasterPhaser
 ************************************/
class MasterPhaser {
  constructor(audioCtx, bpm=120, dAM=4, dFM=2) {
    this.ctx = audioCtx;
    this.f0 = bpm/60;

    this.carrier = this.ctx.createOscillator();
    this.carrier.type = "sine";
    this.carrier.frequency.value = 220;

    this.amGain = this.ctx.createGain();
    this.amGain.gain.value = 0.0;
    this.amOsc = this.ctx.createOscillator();
    this.amOsc.frequency.value = this.f0/dAM;
    this.amOsc.connect(this.amGain).connect(this.carrier.frequency);

    this.fmGain = this.ctx.createGain();
    this.fmGain.gain.value = 0.0;
    this.fmOsc = this.ctx.createOscillator();
    this.fmOsc.frequency.value = this.f0*dFM;
    this.fmOsc.connect(this.fmGain).connect(this.carrier.frequency);

    this.filter = this.ctx.createBiquadFilter();
    this.filter.type = "lowpass";
    this.filter.frequency.value = 800;
    this.filter.Q.value = 1;

    this.output = this.ctx.createGain();
    this.output.gain.value = 0.3;

    this.carrier.connect(this.filter).connect(this.output);
  }

  start() {
    this.carrier.start();
    this.amOsc.start();
    this.fmOsc.start();
  }

  applyMorpheme(m) {
    if (!m.audio) return;
    if (m.audio.amDepth !== undefined) this.amGain.gain.value = m.audio.amDepth;
    if (m.audio.fmDepth !== undefined) this.fmGain.gain.value = m.audio.fmDepth;
    if (m.audio.carrierShift !== undefined) this.carrier.frequency.value += m.audio.carrierShift;
    if (m.audio.filterCut !== undefined) this.filter.frequency.value = m.audio.filterCut;
  }
}

/************************************
 * B) Phaser Visual Emitter (stub)
 ************************************/
class TokenEmitter {
  constructor(scene) {
    this.scene = scene;
    this.group = scene.add.group();
  }
  emitMorpheme(m) {
    const tok = this.scene.add.sprite(400, 300, 'token');
    tok.t = 0;
    tok.amDepth = m.audio?.amDepth || 1;
    tok.fmDepth = m.audio?.fmDepth || 1;
    this.group.add(tok);
  }
  update(delta) {
    this.group.children.iterate(tok=>{
      tok.t += delta*0.002;
      tok.x = 400 + Math.sin(tok.t*tok.amDepth+1)*100;
      tok.y = 300 + Math.sin(tok.t*tok.fmDepth+2)*60;
      tok.alpha = 0.5+0.5*Math.sin(tok.t);
    });
  }
}
```

---

### 🔹 Hook into `MetaFractalEngine`

Inside `MetaFractalEngine.constructor`:

```js
this.audioCtx = new (window.AudioContext||window.webkitAudioContext)();
this.masterPhaser = new MasterPhaser(this.audioCtx);
this.masterPhaser.start();
this.tokenEmitter = null; // external Phaser scene can attach
```

Inside `_up()` after candidate build:

```js
if (this.masterPhaser) {
  morph.morphemes.forEach(m=>{
    this.masterPhaser.applyMorpheme(m);
    if (this.tokenEmitter) this.tokenEmitter.emitMorpheme(m);
  });
}
```

---

### 🔹 Result

* Every morpheme in the breakdown not only shapes your **thought-space SU** but also:

  * **modulates audio** (speechlike AM/FM + filter shaping).
  * **emits a token** into a Phaser scene (figure-8 drift).

---

Perfect — then we’ll inline the **audio mappings** directly into your `createMorphemeRegistry()` so every prefix/suffix/root drives both **math + sound + visuals**.

Here’s how it looks when extended:

```js
function createMorphemeRegistry(dim=3){
  const R=new Map();

  // prefixes
  R.set('re', new Morpheme('re',
    [[0.95,0,0],[0,1.05,0],[0,0,1]],[0,0,0],
    { deltaLevel:-1, alpha:0.98, description:'again/restore',
      audio:{ amDepth:0.2, fmDepth:0.05, carrierShift:-20, filterCut:600 } }
  ));
  R.set('un', new Morpheme('un',
    [[-1,0,0],[0,1,0],[0,0,1]],[0,0,0],
    { description:'negate',
      audio:{ amDepth:0.3, fmDepth:0.1, carrierShift:-40, filterCut:400 } }
  ));
  R.set('counter', new Morpheme('counter',
    [[-0.8,0.2,0],[0.2,-0.8,0],[0,0,1]],[0,0,0],
    { deltaLevel:1, description:'oppose',
      audio:{ amDepth:0.5, fmDepth:0.25, carrierShift:30, filterCut:1200 } }
  ));
  R.set('multi', new Morpheme('multi',
    [[1.4,0,0],[0,1.4,0],[0,0,1.15]],[0,0,0.05],
    { deltaLevel:1, description:'many/scale up',
      audio:{ amDepth:0.4, fmDepth:0.4, carrierShift:50, filterCut:2000 } }
  ));

  // suffixes
  R.set('ize', new Morpheme('ize',
    [[1,0.08,0],[0,1,0.08],[0,0,1.08]],[0,0,0],
    { description:'make into',
      audio:{ amDepth:0.1, fmDepth:0.3, carrierShift:25, filterCut:1800 } }
  ));
  R.set('ness', new Morpheme('ness',
    [[1,0,0],[0,1,0],[0.1,0.1,0.9]],[0,0,0.05],
    { C:[[1.05,0,0],[0,1.05,0],[0,0,0.95]], description:'quality/state',
      audio:{ amDepth:0.25, fmDepth:0.15, carrierShift:10, filterCut:1000 } }
  ));
  R.set('ing', new Morpheme('ing',
    [[1,0.15,0],[0,1.15,0],[0,0,1]],[0,0,0],
    { alpha:1.05, description:'ongoing',
      audio:{ amDepth:0.35, fmDepth:0.2, carrierShift:15, filterCut:1400 } }
  ));

  // roots
  R.set('build', new Morpheme('build',
    [[1.15,0,0],[0,1.15,0],[0,0,1.05]],[0.05,0.05,0],
    { description:'construct',
      audio:{ amDepth:0.15, fmDepth:0.15, carrierShift:0, filterCut:1600 } }
  ));
  R.set('move', new Morpheme('move',
    [[1,0,0],[0,1,0],[0,0,1]],[0.2,0,0],
    { description:'translate',
      audio:{ amDepth:0.2, fmDepth:0.35, carrierShift:40, filterCut:2200 } }
  ));
  R.set('scale', new Morpheme('scale',
    [[1.1,0,0],[0,1.1,0],[0,0,1.1]],[0,0,0],
    { description:'uniform scale',
      audio:{ amDepth:0.15, fmDepth:0.25, carrierShift:20, filterCut:1500 } }
  ));

  return R;
}
```

---

✅ Now every morpheme carries **sound + math together**.

* **`applyMorpheme` in MasterPhaser** reads `m.audio` and modulates voice-like tone.
* **`TokenEmitter`** emits figure-8 tokens modulated by the same AM/FM values.

This keeps the **atomic breakdown → reconstruction loop** perfectly congruent:

* Breakdown = morphemes trigger math transforms **and** sound shapes.
* Reconstruction = morphemes re-compose semantics **and** rebuild timbre/shape in sync.

---

Alright — let’s add the **toggle system** so you can decide if you want the math, audio, visuals, or all of them running in sync.

Here’s the extension layer for your **MetaFractalEngine** and **MasterPhaser**:

```js
/************************************
 * 11) Global Flags for Modality Control
 ************************************/
const engineFlags = {
  math: true,    // run LLE math transforms
  audio: true,   // apply AM/FM synthesis
  visual: true   // emit Phaser tokens / shapes
};

/************************************
 * 12) Master Phaser with Toggles
 ************************************/
class MasterPhaser {
  constructor(audioCtx){
    this.ctx = audioCtx;
    this.carrier = this.ctx.createOscillator();
    this.amp = this.ctx.createGain();
    this.filter = this.ctx.createBiquadFilter();

    // base patch
    this.carrier.type = "sine";
    this.carrier.frequency.value = 220; // base A3
    this.amp.gain.value = 0.5;
    this.filter.type = "lowpass";
    this.filter.frequency.value = 2000;
    this.filter.Q.value = 1.0;

    // connect
    this.carrier.connect(this.filter).connect(this.amp).connect(this.ctx.destination);
    this.carrier.start();

    this.lfoAM = this.ctx.createOscillator();
    this.lfoFM = this.ctx.createOscillator();
    this.lfoAM.type = "sine"; this.lfoFM.type = "sine";
    this.lfoAM.frequency.value = 4; // 4 Hz tremolo
    this.lfoFM.frequency.value = 6; // 6 Hz vibrato
    this.lfoAM.start(); this.lfoFM.start();

    // patch modulators
    const amGain = this.ctx.createGain();
    amGain.gain.value = 0.3;
    this.lfoAM.connect(amGain).connect(this.amp.gain);

    const fmGain = this.ctx.createGain();
    fmGain.gain.value = 50; // frequency deviation in Hz
    this.lfoFM.connect(fmGain).connect(this.carrier.frequency);
  }

  applyMorpheme(m) {
    if (!m || !engineFlags.audio) return;

    const a = m.effects?.audio;
    if (!a) return;

    // AM/FM depth control
    this.lfoAM.frequency.value = 4 + a.amDepth * 10;
    this.lfoFM.frequency.value = 6 + a.fmDepth * 12;

    // carrier shift
    this.carrier.frequency.value = 220 + a.carrierShift;

    // filter
    this.filter.frequency.value = a.filterCut || 2000;
  }
}

/************************************
 * 13) TokenEmitter (Visuals Toggle)
 ************************************/
class TokenEmitter {
  constructor(scene){
    this.scene = scene;
    this.tokens = [];
  }

  emit(token, morpheme){
    if (!engineFlags.visual) return;
    const audio = morpheme?.effects?.audio || {};
    const color = Phaser.Display.Color.HSVToRGB(audio.fmDepth, 1, 1).color;

    const particle = this.scene.add.circle(
      Math.random()*800, Math.random()*600, 8, color
    );
    this.scene.tweens.add({
      targets: particle,
      x: { value: "+=50", yoyo: true, repeat: -1 },
      y: { value: "+=50", yoyo: true, repeat: -1 },
      duration: 1000/(audio.amDepth+0.1),
      ease: "Sine.easeInOut"
    });
    this.tokens.push(particle);
  }
}

/************************************
 * 14) Engine Integration
 ************************************/
class MetaFractalEngineWithPhaser extends MetaFractalEngine {
  constructor(dim=3, audioCtx=null, scene=null){
    super(dim);
    this.phaser = audioCtx ? new MasterPhaser(audioCtx) : null;
    this.emitter = scene ? new TokenEmitter(scene) : null;
  }

  run(input){
    const result = super.run(input);
    const last = this._layers.collapsed;

    if(last?.morph?.morphemes){
      for(const m of last.morph.morphemes){
        const reg = createMorphemeRegistry();
        const mor = reg.get(m.text);
        if(this.phaser) this.phaser.applyMorpheme(mor);
        if(this.emitter) this.emitter.emit(last.token, mor);
      }
    }

    return result;
  }
}

// Usage:
// engineFlags.audio = false; // silence audio
// engineFlags.visual = true; // keep visuals
```

---

🔑 **What this gives you**:

* `engineFlags.math` → run or skip the LLE math layer.
* `engineFlags.audio` → toggle AM/FM morpho-voice synthesis.
* `engineFlags.visual` → toggle Phaser visual emission.

So you can run:

* Pure math (`math: true, audio: false, visual: false`)
* Pure audio (`math: false, audio: true, visual: false`)
* Full all-in-one pipeline (`true, true, true`).

---

Perfect — let’s wire in the **sync mode**.

Right now you’ve got **independent toggles** (`math`, `audio`, `visual`). With sync mode, you can flip a single switch and guarantee the pipeline runs *all-or-none*.

Here’s the extension:

```js
/************************************
 * 15) Sync Mode Extension
 ************************************/
engineFlags.syncMode = false; // default off

function setSyncMode(on = true){
  engineFlags.syncMode = on;
  if(on){
    // force all true
    engineFlags.math   = true;
    engineFlags.audio  = true;
    engineFlags.visual = true;
  }
}

function setFlag(flag, value){
  if(engineFlags.syncMode){
    // in sync mode, ignore single toggles
    console.warn(`SyncMode active — cannot set ${flag} independently.`);
    return;
  }
  engineFlags[flag] = value;
}

/************************************
 * 16) Example Usage
 ************************************/
// Turn everything on/off together:
setSyncMode(true);   // → math/audio/visual all true
setSyncMode(false);  // → back to manual control

// Manual control:
setFlag("audio", false);  // only disables audio if syncMode=false
```

---

### 🔑 Behavior

* **SyncMode = true** → all flags (`math`, `audio`, `visual`) are locked to `true`.
* **SyncMode = false** → you can control each independently with `setFlag()`.
* If you try to change one flag while sync mode is on, it will warn and ignore.

---

This way you’ve got:

* 🔒 *Atomic lock-in* → one command, all systems fire together.
* 🔓 *Independent play* → test math-only, audio-only, visual-only.

---
 let’s tie **SyncMode** into your canvas hooks so your UI reflects it automatically.
Here’s the patch for your `attachToCanvas` and flag handling:

```js
/************************************
 * 15) Sync Mode Extension
 ************************************/
engineFlags.syncMode = false; // default off

function setSyncMode(on = true){
  engineFlags.syncMode = on;
  if(on){
    // lock all subsystems on
    engineFlags.math   = true;
    engineFlags.audio  = true;
    engineFlags.visual = true;
  }
  // trigger UI update if canvas hook is present
  if(__engine._onState){
    __engine._onState({
      syncMode: engineFlags.syncMode,
      flags: { ...engineFlags }
    });
  }
}

function setFlag(flag, value){
  if(engineFlags.syncMode){
    console.warn(`SyncMode active — cannot set ${flag} independently.`);
    return;
  }
  engineFlags[flag] = value;
  if(__engine._onState){
    __engine._onState({
      syncMode: engineFlags.syncMode,
      flags: { ...engineFlags }
    });
  }
}

/************************************
 * 16) UI / Canvas Hook Integration
 ************************************/
export function attachToCanvas({ onState, onGraph } = {}){
  __engine._onState = onState;
  __engine._onGraph = onGraph;

  // initialize UI with current state
  if(onState){
    onState({
      syncMode: engineFlags.syncMode,
      flags: { ...engineFlags }
    });
  }
}
```

---

### 🔑 How it works

* When `setSyncMode(true)` is called, **all flags are set to true** and the canvas `onState` hook is notified.
* When `setFlag(flag, value)` is called, it only applies if sync mode is **off**; otherwise, it warns and does nothing.
* The canvas now receives **both syncMode and all flags** every time something changes.

---

⚡ Next step: **simple control panel UI snippet** (HTML/JS) with:

* A **Sync Mode toggle switch**
* Three **independent checkboxes** (auto-locked when sync is on)

Perfect. Here’s a lightweight **control panel UI** you can drop right into your HTML alongside your canvas.
It gives you a **Sync Mode toggle** and three subsystem checkboxes (locked when sync is active).

---

### 📜 HTML + JS Control Panel

```html
<div id="controlPanel" class="panel">
  <h3>⚙️ Engine Controls</h3>
  <label>
    <input type="checkbox" id="syncToggle" />
    Sync Mode (lock all together)
  </label>
  <hr />
  <label>
    <input type="checkbox" id="mathToggle" />
    Math
  </label>
  <label>
    <input type="checkbox" id="audioToggle" />
    Audio
  </label>
  <label>
    <input type="checkbox" id="visualToggle" />
    Visual
  </label>
</div>

<script type="module">
  import { setSyncMode, setFlag, attachToCanvas } from "./meta-fractal-pipeline.js";

  const syncEl   = document.getElementById("syncToggle");
  const mathEl   = document.getElementById("mathToggle");
  const audioEl  = document.getElementById("audioToggle");
  const visualEl = document.getElementById("visualToggle");

  // hook into engine state updates
  attachToCanvas({
    onState: ({ syncMode, flags }) => {
      syncEl.checked   = syncMode;
      mathEl.checked   = flags.math;
      audioEl.checked  = flags.audio;
      visualEl.checked = flags.visual;

      // lock checkboxes if syncMode is on
      [mathEl, audioEl, visualEl].forEach(cb => cb.disabled = syncMode);
    }
  });

  // UI → Engine
  syncEl.addEventListener("change", () => setSyncMode(syncEl.checked));
  mathEl.addEventListener("change", () => setFlag("math", mathEl.checked));
  audioEl.addEventListener("change", () => setFlag("audio", audioEl.checked));
  visualEl.addEventListener("change", () => setFlag("visual", visualEl.checked));
</script>
```

---

### 🔑 Behavior

* **Sync Mode on** →

  * All subsystems are locked **ON**.
  * Individual checkboxes (Math, Audio, Visual) are **disabled**.
* **Sync Mode off** →

  * You can toggle subsystems independently.
  * Engine + UI stay in sync because `attachToCanvas` updates the checkboxes whenever the engine state changes.

---

Alright 🚀

Here’s the **extended control panel + canvas wiring** so your graph canvas reacts visually when **Sync Mode** is active:

---

### 📜 Control Panel + Canvas Effects

```html
<style>
  #graphCanvas {
    width: 100%;
    height: 400px;
    background: #111;
    border: 2px solid #333;
    transition: box-shadow 0.3s, border-color 0.3s;
  }

  /* Visual glow when Sync Mode is ON */
  #graphCanvas.sync-on {
    border-color: #00e6ff;
    box-shadow: 0 0 20px #00e6ff, 0 0 40px #0099cc inset;
  }
</style>

<div id="controlPanel" class="panel">
  <h3>⚙️ Engine Controls</h3>
  <label>
    <input type="checkbox" id="syncToggle" />
    Sync Mode (lock all together)
  </label>
  <hr />
  <label>
    <input type="checkbox" id="mathToggle" />
    Math
  </label>
  <label>
    <input type="checkbox" id="audioToggle" />
    Audio
  </label>
  <label>
    <input type="checkbox" id="visualToggle" />
    Visual
  </label>
</div>

<canvas id="graphCanvas"></canvas>

<script type="module">
  import { setSyncMode, setFlag, attachToCanvas } from "./meta-fractal-pipeline.js";

  const syncEl   = document.getElementById("syncToggle");
  const mathEl   = document.getElementById("mathToggle");
  const audioEl  = document.getElementById("audioToggle");
  const visualEl = document.getElementById("visualToggle");
  const canvas   = document.getElementById("graphCanvas");

  // 🔄 Hook into engine state updates
  attachToCanvas({
    onState: ({ syncMode, flags }) => {
      syncEl.checked   = syncMode;
      mathEl.checked   = flags.math;
      audioEl.checked  = flags.audio;
      visualEl.checked = flags.visual;

      // Lock checkboxes if syncMode is active
      [mathEl, audioEl, visualEl].forEach(cb => cb.disabled = syncMode);

      // Add/remove glow effect on canvas
      canvas.classList.toggle("sync-on", syncMode);
    }
  });

  // UI → Engine
  syncEl.addEventListener("change", () => setSyncMode(syncEl.checked));
  mathEl.addEventListener("change", () => setFlag("math", mathEl.checked));
  audioEl.addEventListener("change", () => setFlag("audio", audioEl.checked));
  visualEl.addEventListener("change", () => setFlag("visual", visualEl.checked));
</script>
```

---

### ✨ What’s New

* Added a **glow effect** around the canvas (`#graphCanvas.sync-on`) when **Sync Mode** is active.
* Effect includes:

  * **Bright cyan border**
  * **Outer glow**
  * **Inner glow pulse** for emphasis
* All still synced with engine state via `attachToCanvas`.

---

 let’s make the canvas glow like a **breathing/pulsing heartbeat** whenever **Sync Mode** is ON. That way it *feels alive* while the engine is tempo-locked.

---

### 🔥 Animated Glow Effect for Sync Mode

```html
<style>
  #graphCanvas {
    width: 100%;
    height: 400px;
    background: #111;
    border: 2px solid #333;
    transition: border-color 0.3s;
  }

  /* Sync ON glow + pulsing animation */
  #graphCanvas.sync-on {
    border-color: #00e6ff;
    animation: pulseGlow 2s infinite ease-in-out;
  }

  @keyframes pulseGlow {
    0% {
      box-shadow: 0 0 10px #00e6ff, 0 0 20px #0099cc inset;
    }
    50% {
      box-shadow: 0 0 30px #00e6ff, 0 0 60px #00ccff inset;
    }
    100% {
      box-shadow: 0 0 10px #00e6ff, 0 0 20px #0099cc inset;
    }
  }
</style>

<canvas id="graphCanvas"></canvas>
```

---

### ⚡ How it Works

* When `.sync-on` is added to the canvas:

  * The **border turns cyan**.
  * A `pulseGlow` animation kicks in:

    * Expands and contracts the **outer glow** (0 → 30px).
    * Simultaneously breathes the **inner glow**.
* Loop is **2 seconds** per cycle (`ease-in-out` makes it smooth).

---

both **visual glow tempo** and **sound modulation tempo** start *locked together* (same BPM), but give you knobs/sliders to **de-sync and resync** as needed.

---

### ⚡ Plan

1. **Shared Clock**

   * On init, both the **canvas glow animation** and the **FM/AM modulators** read from the same master BPM.
   * Example: 120 BPM → glow cycle = 500 ms per beat, modulators = 2π·120/60 Hz phase increment.

2. **Separate Controls**

   * UI gives you:

     * 🔹 `visualTempo` (glow speed multiplier)
     * 🔹 `audioTempo` (FM/AM rate multiplier)
   * Both default = `1.0` (in sync).
   * Adjust separately → drift / phase shift.

3. **Resync Button**

   * Pressing “SYNC” realigns both to the master BPM, resetting phase.

---

### Example JS Wiring

```js
let masterBPM = 120;          // Base tempo
let visualTempo = 1.0;        // Glow tempo multiplier
let audioTempo = 1.0;         // Modulator tempo multiplier

let lastTime = performance.now();

function update() {
  const now = performance.now();
  const dt = (now - lastTime) / 1000.0;
  lastTime = now;

  // --- Audio phase (FM/AM modulators) ---
  const freq = (masterBPM / 60) * audioTempo;  // cycles/sec
  fmPhase += 2 * Math.PI * freq * dt;
  amPhase += 2 * Math.PI * freq * dt;

  // --- Visual phase (canvas glow) ---
  const visFreq = (masterBPM / 60) * visualTempo;
  glowPhase += 2 * Math.PI * visFreq * dt;

  // Update glow intensity (0–1)
  const glowIntensity = 0.5 + 0.5 * Math.sin(glowPhase);

  const canvas = document.getElementById("graphCanvas");
  canvas.style.boxShadow = `
    0 0 ${10 + 20 * glowIntensity}px #00e6ff,
    0 0 ${20 + 40 * glowIntensity}px #00ccff inset
  `;

  requestAnimationFrame(update);
}
update();

// --- Controls ---
function setVisualTempo(mult) { visualTempo = mult; }
function setAudioTempo(mult) { audioTempo = mult; }
function resync() { glowPhase = 0; fmPhase = 0; amPhase = 0; }
```

---

👉 This way:

* Both start in **lockstep** at 120 BPM.
* You can **nudge either** side faster/slower with `setVisualTempo` or `setAudioTempo`.
* `resync()` snaps everything back to phase 0.

---




class MetaFractalEngine {
  constructor(dim=3){
    this.dim=dim;
    this.su=new StationaryUnit(dim);
    this.buttons=createStandardButtons(dim);
    this.morpho=createLLEMorphologyV2();
    this.quantum=new QuantumLayer();
    this.memory=new Memory();
    this.index=new TinyTextIndex();
    this._layers={atoms:null, superposed:null, collapsed:null, graph:null, choices:null};

    // --- NEW: Sync controller state ---
    this.masterBPM = 120;     // global tempo
    this.visualTempo = 1.0;   // multiplier for canvas glow
    this.audioTempo = 1.0;    // multiplier for FM/AM
    this._phases = { fm: 0, am: 0, glow: 0 };
    this._lastTime = performance.now();
  }

  /** Sync tick updates phases + glow */
  tick(){
    const now = performance.now();
    const dt = (now - this._lastTime) / 1000.0;
    this._lastTime = now;

    // frequencies (Hz)
    const baseFreq = this.masterBPM / 60.0;
    const fAudio = baseFreq * this.audioTempo;
    const fVisual = baseFreq * this.visualTempo;

    // advance phases
    this._phases.fm += 2 * Math.PI * fAudio * dt;
    this._phases.am += 2 * Math.PI * fAudio * dt;
    this._phases.glow += 2 * Math.PI * fVisual * dt;

    // compute glow intensity
    const g = 0.5 + 0.5 * Math.sin(this._phases.glow);
    const canvas = document.getElementById("graphCanvas");
    if(canvas){
      canvas.style.boxShadow = `
        0 0 ${10+20*g}px #00e6ff,
        0 0 ${20+40*g}px #00ccff inset
      `;
    }

    requestAnimationFrame(()=>this.tick());
  }

  // --- Control knobs ---
  setMasterBPM(bpm){ this.masterBPM = bpm; }
  setVisualTempo(mult){ this.visualTempo = mult; }
  setAudioTempo(mult){ this.audioTempo = mult; }
  resync(){ this._phases = { fm: 0, am: 0, glow: 0 }; }

  // (rest of engine methods… _down, _up, etc.)
}

class MasterPhaser {
  constructor(audioCtx, bpm=120) {
    this.audioCtx = audioCtx;
    this.masterBPM = bpm;

    // oscillator + gain
    this.carrier = audioCtx.createOscillator();
    this.carrier.type = "sine";
    this.carrier.frequency.value = 220; // base freq (A3)

    this.gain = audioCtx.createGain();
    this.gain.gain.value = 0.5;

    // FM modulator
    this.fmOsc = audioCtx.createOscillator();
    this.fmOsc.type = "sine";
    this.fmGain = audioCtx.createGain();

    // AM modulator
    this.amOsc = audioCtx.createOscillator();
    this.amOsc.type = "sine";
    this.amGain = audioCtx.createGain();
    this.amGain.gain.value = 0.5; // depth

    // wiring
    this.fmOsc.connect(this.fmGain).connect(this.carrier.frequency);
    this.amOsc.connect(this.amGain).connect(this.gain.gain);
    this.carrier.connect(this.gain).connect(audioCtx.destination);

    // start all
    this.carrier.start();
    this.fmOsc.start();
    this.amOsc.start();
  }

  /** tempo-locked update */
  setBPM(bpm) {
    this.masterBPM = bpm;
    const base = bpm / 60; // Hz
    this.fmOsc.frequency.setValueAtTime(base * 2, this.audioCtx.currentTime); // faster than beat
    this.amOsc.frequency.setValueAtTime(base * 0.5, this.audioCtx.currentTime); // slower than beat
  }

  /** set depths */
  setFMDepth(hz) { this.fmGain.gain.setValueAtTime(hz, this.audioCtx.currentTime); }
  setAMDepth(depth) { this.amGain.gain.setValueAtTime(depth, this.audioCtx.currentTime); }
}
🔑 Notes

Direct drive: fmOsc modulates carrier.frequency at audio rate. amOsc modulates the gain at audio rate.

Tempo lock: setBPM ties both modulators to your global beat. They start in sync but can be tuned independently.

“Faster than the beat”: the FM oscillator runs at 2 × base, so at BPM=120 → base=2 Hz → FM=4 Hz. You can scale this arbitrarily for metallic textures.
