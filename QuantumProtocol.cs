using UnityEngine;
using System.Collections.Generic;

public class QuantumProtocol : MonoBehaviour {
    public static QuantumProtocol Instance;

    private void Awake() {
        if (Instance == null) Instance = this;
        else Destroy(gameObject);
    }

    // --- EVENT: A single agent has collapsed ---
    public void OnAgentCollapse(string agentID) {
        Debug.Log($"[QuantumProtocol] COLLAPSE EVENT for agent: {agentID}");

        MemoryGhost.Spawn(agentID);                           // Replay its path
        QuantumUI.ShowScoreGlyph(agentID);                    // Show floating score
        QuantumVisuals.BurstAt(agentID);                      // Optional: particle shock
        QuantumLore.Archive(agentID);                         // Log to memory scroll
    }

    // --- EVENT: All paths have collapsed ---
    public void OnCollapseAll() {
        Debug.Log("[QuantumProtocol] GLOBAL COLLAPSE triggered.");
        
        QuantumFeaturePanel.Instance.ReplayAllGhosts();       // Replays all agent paths
        QuantumVisuals.FadeWorld();                           // Optional: darken world
        QuantumAudio.PlayEchoField();                         // Echoing collapse sfx
    }

    // --- EVENT: Math Function Changed ---
    public void OnFunctionChanged(MathFunctionType newFunc) {
        Debug.Log($"[QuantumProtocol] Function shift to: {newFunc}");

        Shader.SetGlobalInt("_FunctionID", (int)newFunc);     // Global shader behavior
        QuantumAudio.PlayFunctionTone(newFunc);               // SFX variation
        QuantumUI.UpdateFunctionDisplay(newFunc);             // UI glow/panel color
        QuantumVisuals.SyncTrailPalette(newFunc);             // All trails adapt
    }

    // --- EVENT: Agent reached step limit ---
    public void OnAgentComplete(string agentID) {
        Debug.Log($"[QuantumProtocol] Agent {agentID} completed its journey.");
        OnAgentCollapse(agentID);                             // Trigger collapse flow
    }

    // --- EVENT: Resource state change ---
    public void OnResourceStateChange(string resourceID, ResourceState newState) {
        Debug.Log($"[QuantumProtocol] Resource {resourceID} transitioned to: {newState}");
        
        QuantumVisuals.UpdateResourceVisuals(resourceID, newState);
        QuantumAudio.PlayResourceTone(resourceID, newState);
        
        if (newState == ResourceState.Quantum) {
            QuantumLore.ArchiveResourceTransition(resourceID, "Quantum superposition achieved");
        }
    }
}

public enum ResourceState {
    Classical,
    Quantum,
    Entangled,
    Collapsed
}

public enum MathFunctionType {
    Mirror,
    Cosine,
    Chaos,
    Absorb,
    Wave,
    Fractal
}