using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class QuantumFeaturePanel : MonoBehaviour {
    [Header("UI Controls")]
    public Button cosineScatterButton;
    public Button toggleChaosButton;
    public Button showScoresButton;
    public Button loadMathFunctionButton;
    public Button projectTrailsButton;
    public Button collapseNowButton;
    public Button exportLoreButton;
    public Button clearHistoryButton;
    
    [Header("Quantum Systems")]
    public GlyphAmplitudeResolver amplitudeResolver;
    
    [Header("Agent Spawning")]
    [SerializeField] GameObject trailAgentPrefab;
    [SerializeField] Transform spawnOrigin;
    [SerializeField] int agentCount = 12;
    [SerializeField] float spawnRadius = 2f;
    
    [Header("Function Selection")]
    [SerializeField] Dropdown functionDropdown;
    
    // Static flags for quantum behaviors
    public static bool useCosineScatter = false;
    public static bool chaosEnabled = false;
    
    // Active spawned agents
    private List<GameObject> spawnedAgents = new List<GameObject>();
    
    public static QuantumFeaturePanel Instance { get; private set; }

    void Awake() {
        if (Instance == null) {
            Instance = this;
        } else {
            Destroy(gameObject);
        }
    }

    void Start() {
        SetupButtonListeners();
        SetupFunctionDropdown();
        InitializeQuantumSystems();
    }
    
    void SetupButtonListeners() {
        if (cosineScatterButton) cosineScatterButton.onClick.AddListener(InjectCosineScatter);
        if (toggleChaosButton) toggleChaosButton.onClick.AddListener(ToggleChaosMode);
        if (showScoresButton) showScoresButton.onClick.AddListener(ShowPathScores);
        if (loadMathFunctionButton) loadMathFunctionButton.onClick.AddListener(LoadMathFunction);
        if (projectTrailsButton) projectTrailsButton.onClick.AddListener(SpawnTrailAgents);
        if (collapseNowButton) collapseNowButton.onClick.AddListener(TriggerGlobalCollapse);
        if (exportLoreButton) exportLoreButton.onClick.AddListener(ExportQuantumLore);
        if (clearHistoryButton) clearHistoryButton.onClick.AddListener(ClearQuantumHistory);
    }
    
    void SetupFunctionDropdown() {
        if (functionDropdown != null) {
            functionDropdown.options.Clear();
            functionDropdown.options.Add(new Dropdown.OptionData("Mirror Field"));
            functionDropdown.options.Add(new Dropdown.OptionData("Cosine Wave"));
            functionDropdown.options.Add(new Dropdown.OptionData("Chaos Matrix"));
            functionDropdown.options.Add(new Dropdown.OptionData("Absorb Void"));
            functionDropdown.options.Add(new Dropdown.OptionData("Wave Pulse"));
            functionDropdown.options.Add(new Dropdown.OptionData("Fractal Nexus"));
            
            functionDropdown.onValueChanged.AddListener(OnFunctionSelected);
        }
    }
    
    void InitializeQuantumSystems() {
        // Start quantum ambient audio
        QuantumAudio.StartQuantumAmbient();
        
        // Initialize default function
        QuantumProtocol.Instance.OnFunctionChanged(MathFunctionType.Mirror);
        
        Debug.Log("[QuantumFeaturePanel] Quantum systems initialized");
    }

    void InjectCosineScatter() {
        useCosineScatter = !useCosineScatter;
        
        // Update button visual feedback
        if (cosineScatterButton != null) {
            ColorBlock colors = cosineScatterButton.colors;
            colors.normalColor = useCosineScatter ? Color.cyan : Color.white;
            cosineScatterButton.colors = colors;
        }
        
        Debug.Log("[QuantumFeaturePanel] Cosine-weighted scattering " + (useCosineScatter ? "ENABLED" : "DISABLED"));
        
        // Archive the change
        QuantumLore.ArchiveFunctionChange(MathFunctionType.Mirror, MathFunctionType.Cosine);
    }

    void ToggleChaosMode() {
        chaosEnabled = !chaosEnabled;
        
        // Update global shader parameters
        Shader.SetGlobalFloat("_QuantumChaosFactor", chaosEnabled ? 1f : 0f);
        
        // Update button visual feedback
        if (toggleChaosButton != null) {
            ColorBlock colors = toggleChaosButton.colors;
            colors.normalColor = chaosEnabled ? Color.red : Color.white;
            toggleChaosButton.colors = colors;
        }
        
        // Apply chaos visual effects to all active agents
        ApplyChaosToActiveAgents();
        
        Debug.Log("[QuantumFeaturePanel] Chaos Mode: " + (chaosEnabled ? "ENABLED" : "DISABLED"));
    }
    
    void ApplyChaosToActiveAgents() {
        SigilPathAgent[] allAgents = FindObjectsOfType<SigilPathAgent>();
        foreach (var agent in allAgents) {
            if (chaosEnabled) {
                // Apply chaotic behavior
                agent.GetComponent<Renderer>().material.color = Random.ColorHSV();
                
                // Randomize agent parameters
                agent.stepSize *= Random.Range(0.5f, 2f);
                agent.rotationSpeed *= Random.Range(0.2f, 3f);
            } else {
                // Restore normal behavior
                agent.GetComponent<Renderer>().material.color = Color.white;
            }
        }
    }

    void ShowPathScores() {
        Debug.Log("[QuantumFeaturePanel] Displaying quantum path scores");
        
        // Find all active agents and show their scores
        SigilPathAgent[] agents = FindObjectsOfType<SigilPathAgent>();
        
        foreach (var agent in agents) {
            if (agent != null) {
                QuantumUI.ShowScoreGlyph(agent.name);
                
                // Show additional quantum metrics
                string tooltipText = $"Ψ: {agent.consciousness:F2}\\nSteps: {agent.stepCount}\\nFunction: {agent.currentFunction}";
                QuantumUI.ShowQuantumTooltip(tooltipText, agent.transform.position + Vector3.up * 1.5f);
            }
        }
        
        // Display lore statistics
        var stats = QuantumLore.GetStatistics();
        Debug.Log($"Quantum Statistics - Agents: {stats.totalAgents}, Events: {stats.totalEvents}, Consciousness: {stats.totalConsciousnessGained:F2}");
    }

    void LoadMathFunction() {
        Debug.Log("[QuantumFeaturePanel] Opening mathematical function interface");
        
        // Show function dropdown if available
        if (functionDropdown != null) {
            functionDropdown.Show();
        }
        
        // Alternative: Cycle through functions
        MathFunctionType[] functions = (MathFunctionType[])System.Enum.GetValues(typeof(MathFunctionType));
        MathFunctionType randomFunc = functions[Random.Range(0, functions.Length)];
        
        QuantumProtocol.Instance.OnFunctionChanged(randomFunc);
    }
    
    void OnFunctionSelected(int index) {
        MathFunctionType selectedFunction = (MathFunctionType)index;
        QuantumProtocol.Instance.OnFunctionChanged(selectedFunction);
        
        Debug.Log($"[QuantumFeaturePanel] Function selected: {selectedFunction}");
    }

    void SpawnTrailAgents() {
        Debug.Log("[QuantumFeaturePanel] Spawning quantum trail agents");
        
        // Clear existing agents first
        ClearSpawnedAgents();
        
        if (trailAgentPrefab == null || spawnOrigin == null) {
            Debug.LogWarning("[QuantumFeaturePanel] Missing prefab or spawn origin!");
            return;
        }
        
        // Spawn agents in radial pattern
        float angleStep = 360f / agentCount;
        
        for (int i = 0; i < agentCount; i++) {
            float angle = i * angleStep * Mathf.Deg2Rad;
            Vector3 direction = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            Vector3 spawnPosition = spawnOrigin.position + direction * spawnRadius;
            
            GameObject newAgent = Instantiate(trailAgentPrefab, spawnPosition, Quaternion.LookRotation(direction));
            newAgent.name = $"QuantumAgent_Radial_{i:D2}";
            
            // Configure agent properties
            SigilPathAgent pathAgent = newAgent.GetComponent<SigilPathAgent>();
            if (pathAgent != null) {
                pathAgent.agentId = newAgent.name;
                pathAgent.consciousness = Random.Range(0.1f, 0.5f);
                
                // Apply current quantum states
                if (chaosEnabled) {
                    pathAgent.GetComponent<Renderer>().material.color = Random.ColorHSV();
                }
            }
            
            spawnedAgents.Add(newAgent);
            
            // Archive the spawn event
            QuantumLore.ArchiveAgentStep(newAgent.name, spawnPosition, "SPAWNED");
        }
        
        Debug.Log($"[QuantumFeaturePanel] Spawned {agentCount} quantum agents");
    }
    
    void ClearSpawnedAgents() {
        foreach (var agent in spawnedAgents) {
            if (agent != null) {
                QuantumProtocol.Instance.OnAgentCollapse(agent.name);
                Destroy(agent);
            }
        }
        spawnedAgents.Clear();
    }
    
    void TriggerGlobalCollapse() {
        Debug.Log("[QuantumFeaturePanel] Triggering quantum collapse event");
        
        if (amplitudeResolver != null) {
            amplitudeResolver.ResolveAndCollapse();
        }
        
        // Trigger protocol collapse
        QuantumProtocol.Instance.OnCollapseAll();
        
        // Clear all spawned agents
        ClearSpawnedAgents();
    }
    
    void ExportQuantumLore() {
        Debug.Log("[QuantumFeaturePanel] Exporting quantum lore data");
        QuantumLore.ExportLoreData();
        
        // Show confirmation
        var stats = QuantumLore.GetStatistics();
        QuantumUI.ShowQuantumTooltip($"Exported {stats.totalEvents} events, {stats.totalAgents} agents", transform.position);
    }
    
    void ClearQuantumHistory() {
        Debug.Log("[QuantumFeaturePanel] Clearing quantum history");
        QuantumLore.ClearHistory();
        
        // Show confirmation
        QuantumUI.ShowQuantumTooltip("Quantum history cleared", transform.position);
    }
    
    // Method called by QuantumProtocol for ghost replay
    public void ReplayAllGhosts() {
        Debug.Log("[QuantumFeaturePanel] Replaying all quantum ghost paths");
        
        var agentMemories = QuantumLore.GetStatistics();
        
        // For each collapsed agent, replay its path
        var collapseHistory = QuantumLore.GetCollapseHistory();
        foreach (string agentId in collapseHistory) {
            var memory = QuantumLore.GetAgentMemory(agentId);
            if (memory != null && memory.pathHistory.Count > 0) {
                StartCoroutine(ReplayAgentPath(agentId, memory.pathHistory));
            }
        }
    }
    
    System.Collections.IEnumerator ReplayAgentPath(string agentId, List<Vector3> pathHistory) {
        // Create ghost trail visualization
        GameObject ghost = new GameObject($"Ghost_{agentId}");
        LineRenderer line = ghost.AddComponent<LineRenderer>();
        
        line.material = Resources.Load<Material>("Materials/GhostTrail");
        line.color = new Color(1, 1, 1, 0.3f);
        line.width = 0.05f;
        line.positionCount = 0;
        
        // Animate path replay
        for (int i = 0; i < pathHistory.Count; i++) {
            line.positionCount = i + 1;
            line.SetPosition(i, pathHistory[i]);
            yield return new WaitForSeconds(0.05f);
        }
        
        // Fade out ghost after replay
        yield return new WaitForSeconds(2f);
        
        float fadeTime = 1f;
        float elapsed = 0f;
        Color startColor = line.color;
        
        while (elapsed < fadeTime) {
            elapsed += Time.deltaTime;
            float alpha = Mathf.Lerp(startColor.a, 0f, elapsed / fadeTime);
            line.color = new Color(startColor.r, startColor.g, startColor.b, alpha);
            yield return null;
        }
        
        Destroy(ghost);
    }

    // Cosine-weighted scattering function for quantum mechanics
    public static Vector3 CosineScatter(Vector3 normal) {
        if (!useCosineScatter) {
            // Default random scatter
            return Random.onUnitSphere;
        }
        
        // Cosine-weighted hemisphere sampling
        float u = Random.value;
        float v = Random.value;

        float r = Mathf.Sqrt(u);
        float theta = 2f * Mathf.PI * v;

        float x = r * Mathf.Cos(theta);
        float y = r * Mathf.Sin(theta);
        float z = Mathf.Sqrt(1f - u);

        // Create orthonormal basis around normal
        Vector3 tangent = Vector3.Cross(normal, Vector3.up);
        if (tangent.magnitude < 0.001f) {
            tangent = Vector3.Cross(normal, Vector3.right);
        }
        tangent = tangent.normalized;
        
        Vector3 bitangent = Vector3.Cross(normal, tangent).normalized;

        return (x * tangent + y * bitangent + z * normal).normalized;
    }
    
    void Update() {
        // Real-time quantum insights
        if (Input.GetKeyDown(KeyCode.I)) {
            string insight = QuantumLore.GetRealtimeInsight();
            QuantumUI.ShowQuantumTooltip(insight, Camera.main.transform.position + Camera.main.transform.forward * 2f);
        }
        
        // Quick function switching
        if (Input.GetKeyDown(KeyCode.F)) {
            LoadMathFunction();
        }
        
        // Emergency collapse
        if (Input.GetKeyDown(KeyCode.C)) {
            TriggerGlobalCollapse();
        }
    }
    
    void OnDestroy() {
        // Clean up spawned agents
        ClearSpawnedAgents();
        
        // Stop ambient audio
        QuantumAudio.StopQuantumAmbient();
    }
}