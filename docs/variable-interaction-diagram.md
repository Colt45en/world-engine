# Variable Interaction Map - Visual Reference

## Diagram Description

This ASCII diagram represents the Variable Interaction Map showing relationships, tensions, and boundaries between different variable types in experimental design:

```
Variable Interaction Map - Relationships, Tensions and Boundaries

┌─────────────────┐         Affects         ┌─────────────────┐
│   Independent   │ ─────────────────────► │   Dependent     │
│    Variable     │                         │    Variable     │
└─────────┬───────┘                         └─────────┬───────┘
          │                                           │
          │            ┌─────────────────┐           │
          │            │Misidentification│ ──────────┤
          │            │     (Risk)      │           │
          │            └─────────────────┘           │
          │                                          │
          ▼                                          ▼
┌─────────────────┐      Neutralizes       ┌─────────────────┐
│    Control      │ ◄─────────────────────►│  Confounding    │
│    Variable     │                        │    Variable     │
└─────────┬───────┘                        └─────────┬───────┘
          │                                          │
          │ Neutroduce                               │ Biases results/
          │                                          │ Can morph into
          ▼                                          ▼
┌─────────────────┐      Indronce effect    ┌─────────────────┐
│    Boundary     │ ◄─────────────────────  │   Moderator     │
│                 │                         │    Variable     │
└─────────┬───────┘                         └─────────┬───────┘
          │                                           │ Changes
          │ Gray boundary                             │ effect
          │                                           ▼
          │        ┌─────────────────┐         ┌─────────────────┐
          │        │Misidentification│         │    Mediator     │
          │        │ (uncomfortably) │         │    Variable     │
          │        └─────────────────┘         └─────────┬───────┘
          │                │                            │
          │                │ Explains effect            │ Explains
          │                ▼                            │ effect
          └─────────► ┌─────────────────┐               │
                      │     Latent      │ ◄─────────────┘
                      │    Variable     │
                      └─────────────────┘

Legend:
🔴 Sensitive areas (marked in red on original diagram)
──── Sensitive Areas (solid lines)
---- Gray boundary (dashed lines)
```

## Key Relationships Illustrated:

### Primary Flow:
- **Independent Variable → Dependent Variable**: Core causal relationship

### Control Mechanisms:
- **Control Variable**: Neutralizes confounding effects
- **Control Variable ↔ Confounding Variable**: Bidirectional tension

### Risk Factors:
- **Misidentification**: Risk of confusing cause/effect relationships
- **Confounding Variable**: Can bias results and morph into other variable types

### Advanced Relationships:
- **Moderator Variable**: Changes the strength/direction of effects
- **Mediator Variable**: Explains the mechanism of effects
- **Latent Variable**: Underlying, unmeasured influences

### Boundary Conditions:
- **Gray Boundaries**: Areas where variable classifications become unclear
- **Sensitive Areas**: High-risk zones requiring careful analysis

This visual framework helps identify potential issues in experimental design and guides proper variable classification and control strategies.
