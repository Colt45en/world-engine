using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public static class QuantumUI {
    private static Canvas worldCanvas;
    private static Dictionary<string, GameObject> activeGlyphs = new Dictionary<string, GameObject>();
    private static GameObject functionDisplayPanel;
    
    static QuantumUI() {
        InitializeUI();
    }
    
    private static void InitializeUI() {
        // Create world space canvas for 3D UI elements
        GameObject canvasGO = new GameObject("QuantumWorldCanvas");
        worldCanvas = canvasGO.AddComponent<Canvas>();
        worldCanvas.renderMode = RenderMode.WorldSpace;
        worldCanvas.worldCamera = Camera.main;
        
        CanvasScaler scaler = canvasGO.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);
        
        GraphicRaycaster raycaster = canvasGO.AddComponent<GraphicRaycaster>();
    }

    public static void ShowScoreGlyph(string agentID) {
        GameObject agent = GameObject.Find(agentID);
        if (agent == null) return;

        // Remove existing glyph if present
        if (activeGlyphs.ContainsKey(agentID)) {
            Object.Destroy(activeGlyphs[agentID]);
            activeGlyphs.Remove(agentID);
        }

        // Create new score glyph
        GameObject glyphContainer = new GameObject("ScoreGlyph_" + agentID);
        glyphContainer.transform.SetParent(worldCanvas.transform, false);
        
        // Position above agent
        Vector3 worldPos = agent.transform.position + Vector3.up * 2.5f;
        glyphContainer.transform.position = worldPos;
        glyphContainer.transform.LookAt(Camera.main.transform);
        glyphContainer.transform.Rotate(0, 180, 0); // Face camera correctly
        
        // Create background panel
        GameObject panel = new GameObject("GlyphPanel");
        panel.transform.SetParent(glyphContainer.transform, false);
        
        Image panelImage = panel.AddComponent<Image>();
        panelImage.color = new Color(0, 0, 0, 0.7f);
        
        RectTransform panelRect = panel.GetComponent<RectTransform>();
        panelRect.sizeDelta = new Vector2(200, 80);
        
        // Create score text
        GameObject textGO = new GameObject("ScoreText");
        textGO.transform.SetParent(panel.transform, false);
        
        Text scoreText = textGO.AddComponent<Text>();
        scoreText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        scoreText.text = $"Echo: {agentID}\nΨ: {Random.Range(0.1f, 1.0f):F2}";
        scoreText.color = Color.cyan;
        scoreText.fontSize = 16;
        scoreText.alignment = TextAnchor.MiddleCenter;
        
        RectTransform textRect = scoreText.GetComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = Vector2.zero;
        textRect.offsetMax = Vector2.zero;
        
        // Add quantum glow effect
        Outline outline = scoreText.gameObject.AddComponent<Outline>();
        outline.effectColor = Color.cyan;
        outline.effectDistance = new Vector2(1, 1);
        
        // Animate appearance
        CanvasGroup canvasGroup = glyphContainer.AddComponent<CanvasGroup>();
        canvasGroup.alpha = 0f;
        
        LeanTween.alphaCanvas(canvasGroup, 1f, 0.5f).setEaseOutCubic();
        LeanTween.scale(glyphContainer, Vector3.one * 1.2f, 0.3f)
                .setEaseOutElastic()
                .setOnComplete(() => {
                    LeanTween.scale(glyphContainer, Vector3.one, 0.2f);
                });
        
        // Auto-destroy after delay
        Object.Destroy(glyphContainer, 5f);
        activeGlyphs[agentID] = glyphContainer;
        
        // Add pulsing animation
        LeanTween.scale(glyphContainer, Vector3.one * 1.05f, 1f)
                .setLoopPingPong()
                .setEaseInOutSine();
    }

    public static void UpdateFunctionDisplay(MathFunctionType func) {
        Debug.Log($"[QuantumUI] UI display now themed to: {func}");
        
        // Update function display panel
        UpdateFunctionPanel(func);
        
        // Update global UI theme
        ApplyFunctionTheme(func);
        
        // Update shader globals for UI materials
        UpdateUIShaderGlobals(func);
    }
    
    private static void UpdateFunctionPanel(MathFunctionType func) {
        if (functionDisplayPanel == null) {
            CreateFunctionDisplayPanel();
        }
        
        // Get panel components
        Image panelBG = functionDisplayPanel.GetComponent<Image>();
        Text functionText = functionDisplayPanel.GetComponentInChildren<Text>();
        
        if (panelBG == null || functionText == null) return;
        
        // Update colors and text based on function
        switch (func) {
            case MathFunctionType.Mirror:
                panelBG.color = new Color(0, 1, 1, 0.3f); // Cyan
                functionText.text = "◊ MIRROR FIELD ◊";
                functionText.color = Color.cyan;
                break;
                
            case MathFunctionType.Cosine:
                panelBG.color = new Color(1, 0, 1, 0.3f); // Magenta
                functionText.text = "∿ COSINE WAVE ∿";
                functionText.color = Color.magenta;
                break;
                
            case MathFunctionType.Chaos:
                panelBG.color = new Color(Random.value, Random.value, Random.value, 0.3f);
                functionText.text = "※ CHAOS FIELD ※";
                functionText.color = Color.red;
                break;
                
            case MathFunctionType.Absorb:
                panelBG.color = new Color(0.1f, 0.1f, 0.1f, 0.8f); // Dark
                functionText.text = "● ABSORB VOID ●";
                functionText.color = Color.gray;
                break;
                
            case MathFunctionType.Wave:
                panelBG.color = new Color(0, 0, 1, 0.3f); // Blue
                functionText.text = "～ WAVE MATRIX ～";
                functionText.color = Color.blue;
                break;
                
            case MathFunctionType.Fractal:
                panelBG.color = new Color(1, 0.5f, 0, 0.3f); // Orange
                functionText.text = "◈ FRACTAL NEXUS ◈";
                functionText.color = Color.yellow;
                break;
        }
        
        // Animate panel transition
        AnimateFunctionTransition(functionDisplayPanel);
    }
    
    private static void CreateFunctionDisplayPanel() {
        GameObject panelGO = new GameObject("FunctionDisplayPanel");
        panelGO.transform.SetParent(worldCanvas.transform, false);
        
        // Position in top-right of screen
        RectTransform rect = panelGO.AddComponent<RectTransform>();
        rect.anchorMin = new Vector2(1, 1);
        rect.anchorMax = new Vector2(1, 1);
        rect.pivot = new Vector2(1, 1);
        rect.anchoredPosition = new Vector2(-20, -20);
        rect.sizeDelta = new Vector2(300, 60);
        
        // Background
        Image bg = panelGO.AddComponent<Image>();
        bg.color = new Color(0, 0, 0, 0.7f);
        
        // Text
        GameObject textGO = new GameObject("FunctionText");
        textGO.transform.SetParent(panelGO.transform, false);
        
        Text text = textGO.AddComponent<Text>();
        text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        text.text = "◊ QUANTUM FIELD ◊";
        text.color = Color.white;
        text.fontSize = 18;
        text.alignment = TextAnchor.MiddleCenter;
        text.fontStyle = FontStyle.Bold;
        
        RectTransform textRect = text.GetComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = Vector2.zero;
        textRect.offsetMax = Vector2.zero;
        
        functionDisplayPanel = panelGO;
    }
    
    private static void AnimateFunctionTransition(GameObject panel) {
        // Scale pulse animation
        LeanTween.scale(panel, Vector3.one * 1.1f, 0.2f)
                .setEaseOutCubic()
                .setOnComplete(() => {
                    LeanTween.scale(panel, Vector3.one, 0.3f).setEaseOutElastic();
                });
        
        // Glow effect
        Image bg = panel.GetComponent<Image>();
        if (bg != null) {
            Color originalColor = bg.color;
            Color glowColor = new Color(originalColor.r, originalColor.g, originalColor.b, 0.8f);
            
            LeanTween.value(panel, originalColor.a, glowColor.a, 0.2f)
                    .setOnUpdate((float alpha) => {
                        bg.color = new Color(originalColor.r, originalColor.g, originalColor.b, alpha);
                    })
                    .setOnComplete(() => {
                        LeanTween.value(panel, glowColor.a, originalColor.a, 0.5f)
                                .setOnUpdate((float alpha) => {
                                    bg.color = new Color(originalColor.r, originalColor.g, originalColor.b, alpha);
                                });
                    });
        }
    }
    
    private static void ApplyFunctionTheme(MathFunctionType func) {
        // Update all UI elements with function-specific theming
        Button[] allButtons = Object.FindObjectsOfType<Button>();
        
        foreach (Button button in allButtons) {
            if (button.gameObject.name.Contains("Quantum")) {
                ColorBlock colors = button.colors;
                
                switch (func) {
                    case MathFunctionType.Mirror:
                        colors.normalColor = new Color(0, 0.8f, 0.8f, 1f);
                        colors.highlightedColor = new Color(0, 1f, 1f, 1f);
                        break;
                    case MathFunctionType.Cosine:
                        colors.normalColor = new Color(0.8f, 0, 0.8f, 1f);
                        colors.highlightedColor = new Color(1f, 0, 1f, 1f);
                        break;
                    case MathFunctionType.Chaos:
                        colors.normalColor = new Color(0.8f, 0.2f, 0.2f, 1f);
                        colors.highlightedColor = new Color(1f, 0.3f, 0.3f, 1f);
                        break;
                    case MathFunctionType.Absorb:
                        colors.normalColor = new Color(0.3f, 0.3f, 0.3f, 1f);
                        colors.highlightedColor = new Color(0.5f, 0.5f, 0.5f, 1f);
                        break;
                    case MathFunctionType.Wave:
                        colors.normalColor = new Color(0, 0.3f, 0.8f, 1f);
                        colors.highlightedColor = new Color(0, 0.5f, 1f, 1f);
                        break;
                    case MathFunctionType.Fractal:
                        colors.normalColor = new Color(0.8f, 0.4f, 0, 1f);
                        colors.highlightedColor = new Color(1f, 0.6f, 0, 1f);
                        break;
                }
                
                button.colors = colors;
            }
        }
    }
    
    private static void UpdateUIShaderGlobals(MathFunctionType func) {
        // Set global shader properties for UI materials
        Shader.SetGlobalFloat("_UIFunctionType", (float)func);
        
        Vector4 functionColor = Vector4.one;
        switch (func) {
            case MathFunctionType.Mirror:
                functionColor = new Vector4(0, 1, 1, 1); // Cyan
                break;
            case MathFunctionType.Cosine:
                functionColor = new Vector4(1, 0, 1, 1); // Magenta
                break;
            case MathFunctionType.Chaos:
                functionColor = new Vector4(1, 0.3f, 0.3f, 1); // Red
                break;
            case MathFunctionType.Absorb:
                functionColor = new Vector4(0.3f, 0.3f, 0.3f, 1); // Gray
                break;
            case MathFunctionType.Wave:
                functionColor = new Vector4(0, 0.5f, 1, 1); // Blue
                break;
            case MathFunctionType.Fractal:
                functionColor = new Vector4(1, 0.6f, 0, 1); // Orange
                break;
        }
        
        Shader.SetGlobalVector("_UIFunctionColor", functionColor);
    }
    
    public static void ShowQuantumTooltip(string text, Vector3 worldPosition) {
        GameObject tooltip = new GameObject("QuantumTooltip");
        tooltip.transform.SetParent(worldCanvas.transform, false);
        tooltip.transform.position = worldPosition;
        
        // Create tooltip background
        Image bg = tooltip.AddComponent<Image>();
        bg.color = new Color(0, 0, 0, 0.9f);
        
        RectTransform rect = tooltip.GetComponent<RectTransform>();
        rect.sizeDelta = new Vector2(text.Length * 8 + 20, 30);
        
        // Create tooltip text
        GameObject textGO = new GameObject("TooltipText");
        textGO.transform.SetParent(tooltip.transform, false);
        
        Text tooltipText = textGO.AddComponent<Text>();
        tooltipText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        tooltipText.text = text;
        tooltipText.color = Color.white;
        tooltipText.fontSize = 12;
        tooltipText.alignment = TextAnchor.MiddleCenter;
        
        RectTransform textRect = tooltipText.GetComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = Vector2.zero;
        textRect.offsetMax = Vector2.zero;
        
        // Auto-destroy
        Object.Destroy(tooltip, 3f);
    }
}