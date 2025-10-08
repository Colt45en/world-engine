using UnityEngine;
using System.Collections.Generic;
using System;

public static class QuantumLore {
    private static List<QuantumEvent> eventHistory = new List<QuantumEvent>();
    private static Dictionary<string, AgentMemory> agentMemories = new Dictionary<string, AgentMemory>();
    private static int maxHistorySize = 1000;
    
    [System.Serializable]
    public class QuantumEvent {
        public string eventId;
        public string eventType;
        public string agentId;
        public string description;
        public DateTime timestamp;
        public Vector3 worldPosition;
        public Dictionary<string, object> metadata;
        
        public QuantumEvent(string type, string agent, string desc, Vector3 pos) {
            eventId = Guid.NewGuid().ToString();
            eventType = type;
            agentId = agent;
            description = desc;
            timestamp = DateTime.Now;
            worldPosition = pos;
            metadata = new Dictionary<string, object>();
        }
    }
    
    [System.Serializable]
    public class AgentMemory {
        public string agentId;
        public List<Vector3> pathHistory;
        public List<string> decisionHistory;
        public float totalConsciousnessGained;
        public int collisionCount;
        public DateTime firstSeen;
        public DateTime lastActive;
        public MathFunctionType lastFunction;
        
        public AgentMemory(string id) {
            agentId = id;
            pathHistory = new List<Vector3>();
            decisionHistory = new List<string>();
            totalConsciousnessGained = 0f;
            collisionCount = 0;
            firstSeen = DateTime.Now;
            lastActive = DateTime.Now;
        }
    }

    public static void Archive(string agentID) {
        GameObject agent = GameObject.Find(agentID);
        Vector3 position = agent != null ? agent.transform.position : Vector3.zero;
        
        QuantumEvent collapseEvent = new QuantumEvent(
            "AGENT_COLLAPSE", 
            agentID, 
            $"Agent {agentID} collapsed into classical state",
            position
        );
        
        // Add metadata about the collapse
        if (agent != null) {
            SigilPathAgent pathAgent = agent.GetComponent<SigilPathAgent>();
            if (pathAgent != null) {
                collapseEvent.metadata["steps_taken"] = pathAgent.stepCount;
                collapseEvent.metadata["consciousness_level"] = pathAgent.consciousness;
                collapseEvent.metadata["final_function"] = pathAgent.currentFunction.ToString();
            }
        }
        
        AddEvent(collapseEvent);
        UpdateAgentMemory(agentID, position, "COLLAPSED");
        
        Debug.Log($"[QuantumLore] Agent '{agentID}' archived in the memory scroll. Total events: {eventHistory.Count}");
    }
    
    public static void ArchiveResourceTransition(string resourceID, string description) {
        GameObject resource = GameObject.Find(resourceID);
        Vector3 position = resource != null ? resource.transform.position : Vector3.zero;
        
        QuantumEvent resourceEvent = new QuantumEvent(
            "RESOURCE_TRANSITION",
            resourceID,
            description,
            position
        );
        
        AddEvent(resourceEvent);
        Debug.Log($"[QuantumLore] Resource transition archived: {description}");
    }
    
    public static void ArchiveFunctionChange(MathFunctionType oldFunc, MathFunctionType newFunc) {
        QuantumEvent functionEvent = new QuantumEvent(
            "FUNCTION_CHANGE",
            "GLOBAL_SYSTEM",
            $"Mathematical function shifted from {oldFunc} to {newFunc}",
            Vector3.zero
        );
        
        functionEvent.metadata["old_function"] = oldFunc.ToString();
        functionEvent.metadata["new_function"] = newFunc.ToString();
        functionEvent.metadata["transition_time"] = Time.time;
        
        AddEvent(functionEvent);
        
        // Update all active agent memories with new function
        foreach (var memory in agentMemories.Values) {
            memory.lastFunction = newFunc;
            memory.lastActive = DateTime.Now;
        }
    }
    
    public static void ArchiveAgentStep(string agentID, Vector3 position, string decision) {
        if (!agentMemories.ContainsKey(agentID)) {
            agentMemories[agentID] = new AgentMemory(agentID);
        }
        
        AgentMemory memory = agentMemories[agentID];
        memory.pathHistory.Add(position);
        memory.decisionHistory.Add(decision);
        memory.lastActive = DateTime.Now;
        
        // Limit path history size for performance
        if (memory.pathHistory.Count > 500) {
            memory.pathHistory.RemoveAt(0);
            memory.decisionHistory.RemoveAt(0);
        }
    }
    
    public static void ArchiveConsciousnessGain(string agentID, float amount) {
        if (!agentMemories.ContainsKey(agentID)) {
            agentMemories[agentID] = new AgentMemory(agentID);
        }
        
        agentMemories[agentID].totalConsciousnessGained += amount;
        
        QuantumEvent consciousnessEvent = new QuantumEvent(
            "CONSCIOUSNESS_GAIN",
            agentID,
            $"Agent {agentID} gained {amount:F3} consciousness",
            Vector3.zero
        );
        
        consciousnessEvent.metadata["gain_amount"] = amount;
        consciousnessEvent.metadata["total_consciousness"] = agentMemories[agentID].totalConsciousnessGained;
        
        AddEvent(consciousnessEvent);
    }
    
    private static void AddEvent(QuantumEvent quantumEvent) {
        eventHistory.Add(quantumEvent);
        
        // Maintain history size limit
        if (eventHistory.Count > maxHistorySize) {
            eventHistory.RemoveAt(0);
        }
    }
    
    private static void UpdateAgentMemory(string agentID, Vector3 position, string finalAction) {
        if (!agentMemories.ContainsKey(agentID)) {
            agentMemories[agentID] = new AgentMemory(agentID);
        }
        
        AgentMemory memory = agentMemories[agentID];
        memory.pathHistory.Add(position);
        memory.decisionHistory.Add(finalAction);
        memory.lastActive = DateTime.Now;
    }

    public static List<QuantumEvent> GetEventHistory() {
        return new List<QuantumEvent>(eventHistory);
    }
    
    public static List<QuantumEvent> GetEventsOfType(string eventType) {
        List<QuantumEvent> filteredEvents = new List<QuantumEvent>();
        foreach (var evt in eventHistory) {
            if (evt.eventType == eventType) {
                filteredEvents.Add(evt);
            }
        }
        return filteredEvents;
    }
    
    public static List<QuantumEvent> GetAgentEvents(string agentID) {
        List<QuantumEvent> agentEvents = new List<QuantumEvent>();
        foreach (var evt in eventHistory) {
            if (evt.agentId == agentID) {
                agentEvents.Add(evt);
            }
        }
        return agentEvents;
    }
    
    public static AgentMemory GetAgentMemory(string agentID) {
        return agentMemories.ContainsKey(agentID) ? agentMemories[agentID] : null;
    }
    
    public static List<string> GetCollapseHistory() {
        List<string> collapseHistory = new List<string>();
        foreach (var evt in eventHistory) {
            if (evt.eventType == "AGENT_COLLAPSE") {
                collapseHistory.Add(evt.agentId);
            }
        }
        return collapseHistory;
    }
    
    public static void ExportLoreData() {
        string timestamp = DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
        string filename = $"QuantumLore_Export_{timestamp}.json";
        
        var exportData = new {
            export_timestamp = DateTime.Now,
            total_events = eventHistory.Count,
            total_agents = agentMemories.Count,
            events = eventHistory,
            agent_memories = agentMemories
        };
        
        string jsonData = JsonUtility.ToJson(exportData, true);
        
        // In a real implementation, save to persistent data path
        string path = Application.persistentDataPath + "/" + filename;
        System.IO.File.WriteAllText(path, jsonData);
        
        Debug.Log($"[QuantumLore] Exported lore data to: {path}");
    }
    
    public static QuantumLoreStats GetStatistics() {
        var stats = new QuantumLoreStats();
        
        stats.totalEvents = eventHistory.Count;
        stats.totalAgents = agentMemories.Count;
        stats.totalCollapses = GetEventsOfType("AGENT_COLLAPSE").Count;
        stats.totalFunctionChanges = GetEventsOfType("FUNCTION_CHANGE").Count;
        
        // Calculate total consciousness gained across all agents
        foreach (var memory in agentMemories.Values) {
            stats.totalConsciousnessGained += memory.totalConsciousnessGained;
        }
        
        // Find most active agent
        string mostActiveAgent = "";
        int maxSteps = 0;
        foreach (var memory in agentMemories.Values) {
            if (memory.pathHistory.Count > maxSteps) {
                maxSteps = memory.pathHistory.Count;
                mostActiveAgent = memory.agentId;
            }
        }
        stats.mostActiveAgent = mostActiveAgent;
        stats.maxStepsTaken = maxSteps;
        
        // Calculate average session time
        if (agentMemories.Count > 0) {
            TimeSpan totalTime = TimeSpan.Zero;
            foreach (var memory in agentMemories.Values) {
                totalTime += memory.lastActive - memory.firstSeen;
            }
            stats.averageSessionTime = totalTime.TotalSeconds / agentMemories.Count;
        }
        
        return stats;
    }
    
    public static void ClearHistory() {
        eventHistory.Clear();
        agentMemories.Clear();
        Debug.Log("[QuantumLore] History cleared.");
    }
    
    // Real-time lore insights
    public static string GetRealtimeInsight() {
        if (eventHistory.Count == 0) return "The quantum field awaits its first disturbance...";
        
        var recent = eventHistory[eventHistory.Count - 1];
        
        switch (recent.eventType) {
            case "AGENT_COLLAPSE":
                return $"Agent {recent.agentId} has returned to the classical realm, leaving echoes in the quantum field.";
            case "FUNCTION_CHANGE":
                return $"The mathematical foundation shifts, reality bends to new rules: {recent.metadata["new_function"]}";
            case "CONSCIOUSNESS_GAIN":
                return $"Consciousness expands as {recent.agentId} transcends another threshold.";
            case "RESOURCE_TRANSITION":
                return $"Resource {recent.agentId} undergoes quantum transformation: {recent.description}";
            default:
                return "The quantum tapestry weaves new patterns...";
        }
    }
}

[System.Serializable]
public class QuantumLoreStats {
    public int totalEvents;
    public int totalAgents;
    public int totalCollapses;
    public int totalFunctionChanges;
    public float totalConsciousnessGained;
    public string mostActiveAgent;
    public int maxStepsTaken;
    public double averageSessionTime;
}