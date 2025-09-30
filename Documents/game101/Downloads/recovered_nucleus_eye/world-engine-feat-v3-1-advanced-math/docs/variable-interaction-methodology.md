# **Variables in Experiments: Guidelines & Protocol Worksheet**

## Variable Interaction Map Overview

```
                    Variable Interaction Map
              Relationships, Tensions and Boundaries

┌─────────────────┐    Affects    ┌─────────────────┐
│   Independent   │ ──────────────► │   Dependent     │
│    Variable     │                │    Variable     │
└─────────┬───────┘                └─────────┬───────┘
          │                                  │
          │       ┌─────────────────┐        │
          │       │Misidentification│ ───────┤
          │       │     (Risk)      │        │
          │       └─────────────────┘        │
          │                                  │
          ▼                                  ▼
┌─────────────────┐   Neutralizes   ┌─────────────────┐
│    Control      │◄───────────────►│  Confounding    │
│    Variable     │                 │    Variable     │
└─────────┬───────┘                 └─────────┬───────┘
          │                                   │
          │ Neutroduce                        │ Biases results/
          │                                   │ Can morph into
          ▼                                   ▼
┌─────────────────┐   Indronce effect ┌─────────────────┐
│    Boundary     │◄─────────────────│   Moderator     │
│                 │                   │    Variable     │
└─────────┬───────┘                   └─────────┬───────┘
          │                                     │ Changes
          │ Gray boundary                       │ effect
          │                                     ▼
          │     ┌─────────────────┐      ┌─────────────────┐
          │     │Misidentification│      │    Mediator     │
          │     │ (uncomfortably) │      │    Variable     │
          │     └─────────────────┘      └─────────┬───────┘
          │             │                          │
          │             │ Explains effect          │ Explains
          │             ▼                          │ effect
          └────► ┌─────────────────┐               │
                 │     Latent      │◄──────────────┘
                 │    Variable     │
                 └─────────────────┘

Sensitive Areas (Red):      Sensitive Areas (Lines):
🔴 Sensitive areas         ──── Sensitive Areas
--- Gray boundary          ---- Gray boundary
```

## 1. Key Characteristics

* **The Effect**:

  * **Independent Variable (IV):** The "cause" that you manipulate.
  * **Dependent Variable (DV):** The "effect" that you measure.
* **Measurement**:

  * You record the DV after changing the IV; you don't control it directly.
* **Graphing Convention**:

  * X-axis = Independent Variable
  * Y-axis = Dependent Variable

---

## 2. Core Examples in Physics

* **Gravity Experiment**:

  * IV: Drop height
  * DV: Time to hit ground

* **Inclined Motion**:

  * IV: Ramp angle
  * DV: Speed of toy car

* **Arrow Flight**:

  * IV: Launch angle
  * DV: Range (distance traveled)

---

## 3. Types of Variables

### Controlled Variables

* Held constant to prevent influencing the DV.
* Example: Fertilizer experiment → Control sunlight & water.

### Extraneous Variables

* Unplanned influences that add "noise."
* Example: Teaching method → Prior knowledge, motivation, classroom temperature.

### Confounding Variables

* Affect both IV and DV, making causal inference tricky.
* Example: Ice cream sales vs. sunburn → Temperature is the confounder.

---

## 4. Other Variable Categories

* **Quantitative**: Numerical values.

  * Continuous (temperature, height)
  * Discrete (number of students)
* **Categorical**: Qualitative groupings.

  * Nominal (eye color)
  * Ordinal (rankings)
  * Binary (yes/no)
* **Role-based**:

  * Moderator: Changes strength/direction of relationship.
  * Mediator: Explains the process between IV and DV.
  * Latent: Hidden or unmeasurable (e.g., intelligence).

---

## 5. Controlling Confounders

### Experimental Design Strategies

* **Randomization**: Random assignment to spread out confounders.
* **Restriction**: Narrow sample to reduce variability.
* **Matching**: Pair participants with similar traits.
* **Blinding**: Prevent bias (single-blind, double-blind).
* **Placebo Use**: Isolate treatment effect from expectation.

### Statistical Strategies

* **Stratification**: Analyze by layers (e.g., age groups).
* **Multivariate Analysis**: Regression models adjusting for confounders.

---

## 6. Causal Diagrams (Directed Acyclic Graphs – DAGs)

* **Strengths**: Visualize assumptions, show pathways of influence.
* **Limitations**:

  * Depend on prior knowledge.
  * Cannot show magnitude, feedback loops, or chance confounding.
  * Risk of collider bias (adjusting for wrong variable).

---

## 7. Alternatives When Knowledge Is Limited

* **Quasi-Experimental Designs**:

  * Instrumental Variables, Difference-in-Differences, Regression Discontinuity, Synthetic Controls.
* **Statistical Approaches**:

  * Sensitivity Analysis, Inverse Probability Weighting, Causal Discovery algorithms.
* **Qualitative Approaches**:

  * Case studies, process tracing, domain knowledge.

---

# **Protocol Rules to Follow**

1. **Always identify the IV and DV clearly.**
2. **List control variables** before starting the experiment.
3. **Consider possible extraneous and confounding variables.**
4. **Use experimental design strategies first** (randomization, blinding, etc.) to minimize bias.
5. **Apply statistical adjustments after data collection** if confounders remain.
6. **Graph properly**: IV on x-axis, DV on y-axis.
7. **Document assumptions** if using DAGs or causal diagrams.
8. **When knowledge is incomplete**, lean on quasi-experiments, sensitivity analyses, or proxies.
9. **Be transparent** about uncertainty and limitations.

---

# **Variable Interaction Worksheet: Relationships, Tensions & Boundaries**

## 1. Independent vs. Dependent Variables

* **Work With Each Other**:

  * IV is the cause; DV is the measurable effect. The whole experiment lives in this dance.
* **Sensitive Area**:

  * Misidentification: calling the effect the cause (reverse causation).
* **Gray Boundary**:

  * In complex systems, feedback loops can blur which variable is "independent" and which is "dependent" (e.g., stress affecting sleep, and sleep affecting stress).

---

## 2. Control Variables in the Mix

* **Work With IV & DV**:

  * By staying constant, they keep the spotlight on the IV–DV relationship.
* **Work Against Confounders**:

  * They neutralize potential rivals that might sneak in.
* **Sensitive Area**:

  * Over-controlling: if you control a variable that actually *should* be part of the causal chain, you cut off the true effect.
* **Gray Boundary**:

  * Some variables can be both "control" and "confounder" depending on context (e.g., socioeconomic status in health studies).

---

## 3. Extraneous Variables

* **Work Against the Experiment**:

  * They introduce noise, making the IV–DV signal harder to detect.
* **Sometimes Work With Confounders**:

  * They can morph into confounders if they systematically bias results instead of just adding random scatter.
* **Sensitive Area**:

  * Unmeasured extraneous factors can make results unreliable.
* **Gray Boundary**:

  * Deciding what is "extraneous" vs. "meaningful context" (e.g., mood in a classroom—distraction or essential factor?).

---

## 4. Confounding Variables

* **Work Against Clarity**:

  * They mimic or mask causal relationships.
* **Work With Both IV & DV**:

  * By influencing both, they create false correlations.
* **Sensitive Area**:

  * If not controlled, they can entirely flip conclusions (e.g., "ice cream causes sunburn").
* **Gray Boundary**:

  * Confounders may also be mediators (part of the causal path). Mislabeling changes the analysis strategy.

---

## 5. Moderator Variables

* **Work With IV & DV**:

  * They change the *strength or direction* of the effect (e.g., exercise improves mood, but more strongly for younger people).
* **Sensitive Area**:

  * Ignoring moderators may hide important subgroup effects.
* **Gray Boundary**:

  * Can look like confounders unless carefully analyzed.

---

## 6. Mediator Variables

* **Work Between IV & DV**:

  * They explain *how* or *why* the effect happens (e.g., job stress → poor sleep → lower productivity).
* **Work With Confounders (uncomfortably)**:

  * Misidentifying a mediator as a confounder could erase the causal chain.
* **Sensitive Area**:

  * Over-adjustment: removing mediators from analysis when they're the very mechanism of interest.
* **Gray Boundary**:

  * Mediators vs. confounders overlap when timing/order is unclear.

---

## 7. Latent Variables

* **Work Beneath Everything**:

  * They lurk, unmeasured but influential (e.g., "intelligence," "motivation").
* **Sensitive Area**:

  * Their invisibility forces reliance on proxies, which may distort relationships.
* **Gray Boundary**:

  * The line between a latent construct and a poorly measured extraneous variable is fuzzy.

---

# **Sensitive Areas (Red Zones)**

* Misidentifying confounders vs. mediators.
* Over-controlling variables that actually belong in the causal path.
* Treating feedback loops as one-directional cause-and-effect.
* Ignoring moderators that flip relationships under certain conditions.

---

# **Gray Areas (Unclear Boundaries)**

* When an extraneous variable becomes a confounder.
* When a mediator is mistaken for a confounder.
* When latent variables are assumed but never tested.
* Deciding whether to simplify complexity (ignoring noise) or honor it (embracing multi-variable designs).

---

## Application to Nexus Physics System

### Physics Variables in Nexus Forge:

**Independent Variables (Manipulated):**
- Force application (nudge buttons)
- Mass settings
- Gravity strength
- Initial position/velocity

**Dependent Variables (Measured):**
- Position over time
- Velocity changes
- Collision outcomes
- Energy dissipation

**Control Variables (Held Constant):**
- Time step (dt = 1/60)
- Ground plane position
- Cube dimensions
- Damping coefficients

**Potential Confounders:**
- Browser performance variations
- Frame rate inconsistencies
- User interaction timing
- Environmental factors (system load)

**Moderator Variables:**
- Theme settings (affecting visibility/perception)
- Camera angle/position
- Lighting conditions

**Mediator Variables:**
- Air resistance calculations
- Collision detection accuracy
- Numerical integration precision

This methodology provides a framework for analyzing and improving the experimental validity of physics simulations in the Nexus system.
