# Lexiscale Micro-Rationale & Design Notes

## Dimensional rationale

- **mode**: A discrete, operational setting (safe/debug/maintenance/dark/airplane), sibling to "state" for software. Underlying layer: feature flags, config keys; reverse lens: continuous, adaptive context.
- **eligibility**: Criteria-based access dimension, found under "condition". Underlying: rules, scoring, audits; reverse lens: universal access.

## Pipeline

- Triangle: Detects rhythm/structure (loops, functions, variables).
- Square: Persists word and constraint records.
- Circle: Renders/animates type graph vectors.
- Loop: Repeats with new text or patterns.

## Usage

```sh
pip install numpy scikit-learn
python -m lexiscale.self_scaler
```

## Interop

- Output: `/out/type_vectors.json`
- JS hydration (`hydrateFromArtifact`): Loads type vectors and updates Upflow index.