# Nexus System Architecture Flow

_Compiled October 7, 2025_

This chart summarizes how the Nexus meta-layer, AI engines, vector limbs, and
knowledge vaults interact, based on the Layer5 documentation
(`README-NEXUS-FORGE-UNIFIED.md`, `README-integrated-system.md`, and related
files).

## 🌐 High-Level Flow

```mermaid
flowchart TD
    subgraph MetaLayer[Meta Layer Core]
        MLOrchestrator[Studio Orchestrator / Meta Scheduler]
        MLQueue[Priority Queues & Blockers]
        MLRegistry[Engine Registry & Health]
    end

    subgraph KnowledgeVaults[Knowledge Vaults & Librarians]
        KVIngress[Ingress Vaults (Raw Inputs)]
        KVProcessing[Librarian Processing \n (Validation, Tagging, Embeddings)]
        KVEgress[Egress Vaults (Structured Outputs)]
    end

    subgraph Engines[Specialized AI Engines]
        NexusCore[Nexus Forge Primary AI]
        VectorNet[Vector Network Engine]
        PainOpp[Pain / Opportunity System]
        Conscious[Consciousness Analyzer]
        Fractal[Fractal / Auxiliary Units]
    end

    ExternalInputs[External Inputs \n (Users, Sensors, Files)] -->|1. Submit| KVIngress
    KVIngress -->|2. Librarian Enqueues \n + Metadata| KVProcessing
    KVProcessing -->|3. Publish to Meta Layer| MLQueue
    MLQueue -->|4. Schedule Tasks| MLOrchestrator
    MLOrchestrator -->|5. Dispatch via Vector Limbs| Engines
    Engines -->|6. Produce Results| KnowledgeVaults
    KnowledgeVaults -->|7. Aggregated Insights| MLRegistry
    MLRegistry -->|8. Combined Decision / Response| MetaLayer
    MetaLayer -->|9. Feedback & Commands| Engines
    MetaLayer -->|10. External Outputs| ExternalOutputs[Dashboards, Clients, Storage]
```

## 🔁 Detailed Interaction Steps

1. **Input arrival** (docs: `README-NEXUS-FORGE-UNIFIED.md` – demo flow):
   - User or automated events push content into the relevant vector limb.
   - Data lands in an ingress vault. Librarians label it and ensure schema compliance.

2. **Vault processing** (`README-integrated-system.md` – orchestrator section):
   - Librarians transform raw data into structured records.
   - Blockers/throttles prevent engines from flooding the meta layer.

3. **Meta layer scheduling**:
    - Orchestrator reads queued tasks, checks engine availability, and applies
       backpressure policies.
    - Registry tracks engine status (healthy, busy, degraded).

4. **Engine execution**:
    - Each engine (Vector Network, Pain/Opportunity, Consciousness Analyzer,
       fractal helpers) receives curated tasks.
    - Engines run autonomously; they do not call each other directly. They
       reply via their egress limbs.

5. **Knowledge consolidation**:
   - Results return to vaults. Librarians validate, summarize, and store provenance.
   - Aggregated reports/headlines are provided to the meta layer registry.

6. **Decision + feedback loop**:
   - Nexus Forge primary AI synthesizes multi-engine outputs.
   - Meta layer issues commands, updates dashboards, and triggers additional
      tasks as needed.

## 🧭 Debugging Aid (Crosswalk with Docs)

**Crosswalk with docs:**

- **Meta layer orchestrator** — see `README-integrated-system.md`
   (StudioOrchestrator); details queue management, retry logic, and engine
   pools.
- **Vector limbs & blockers** — also in `README-integrated-system.md` (event
   bus, backpressure); implemented via priority queues and policy guards.
- **Knowledge vaults & librarians** — `Layer5/documentation/README.md`
   (World Engine Studio); covers data structuring, schema enforcement, and
   indexing.
- **Nexus Forge engine suite** — `README-NEXUS-FORGE-UNIFIED.md`; describes
   math, AI pattern, terrain, and beat engines.
- **Subsystem scripts (`*.py`, `*.js`)** — individual subsystem READMEs (for
   example `services/pain/README`); provide runtime instructions.
- **Dashboards / visualization** — `Layer5/public/nexus-master-control-center.html`;
   displays status indicators and control knobs.

## ✅ Usage Tips

- Keep `ports.json` authoritative for all WebSocket services so dashboards,
  engines, and tests stay aligned.
- Run subsystems individually (using `.venv311`) when the launcher flags them,
  then restart `nexus_launcher.py` to re-evaluate status.
- Document fixes in `README_DEV.md` so future passes through the flow stay smooth.

> This flowchart can be expanded with additional limbs or vaults as more
> engines are wired in; follow the same pattern (ingress → librarian → meta
> scheduling → engine → egress).
