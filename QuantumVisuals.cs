using UnityEngine;
using UnityEngine.Rendering;

public static class QuantumVisuals {
    private static ParticleSystem collapseBurstPrefab;
    private static Material quantumTrailMaterial;
    
    static QuantumVisuals() {
        LoadResources();
    }
    
    private static void LoadResources() {
        collapseBurstPrefab = Resources.Load<ParticleSystem>("Effects/CollapseBurst");
        quantumTrailMaterial = Resources.Load<Material>("Materials/QuantumTrail");
    }

    public static void BurstAt(string agentID) {
        GameObject agent = GameObject.Find(agentID);
        if (agent == null) return;

        ParticleSystem prefab = Resources.Load<ParticleSystem>("CollapseBurst");
        if (prefab) {
            ParticleSystem ps = Object.Instantiate(prefab, agent.transform.position, Quaternion.identity);
            ps.Play();
            Object.Destroy(ps.gameObject, 2f); // Clean up
        }
    }

    public static void FadeWorld() {
        Debug.Log("[QuantumVisuals] Fading world visuals for collapse...");
        
        // Apply post-processing fade effect
        Volume postProcessVolume = Object.FindObjectOfType<Volume>();
        if (postProcessVolume != null) {
            // Trigger fade animation
            AnimateWorldFade(postProcessVolume);
        }
        
        // Alternative: Screen overlay fade
        CreateScreenFadeOverlay();
    }
    
    private static void CreateScreenFadeOverlay() {
        GameObject fadeOverlay = new GameObject("QuantumFadeOverlay");
        Canvas canvas = fadeOverlay.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 9999;
        
        GameObject fadePanel = new GameObject("FadePanel");
        fadePanel.transform.SetParent(canvas.transform, false);
        
        UnityEngine.UI.Image image = fadePanel.AddComponent<UnityEngine.UI.Image>();
        image.color = new Color(0, 0, 0, 0);
        
        RectTransform rect = fadePanel.GetComponent<RectTransform>();
        rect.anchorMin = Vector2.zero;
        rect.anchorMax = Vector2.one;
        rect.offsetMin = Vector2.zero;
        rect.offsetMax = Vector2.zero;
        
        // Animate fade in/out
        LeanTween.alpha(rect, 0.7f, 1f).setEaseInOutCubic()
                .setOnComplete(() => {
                    LeanTween.alpha(rect, 0f, 2f).setEaseInOutCubic()
                            .setOnComplete(() => Object.Destroy(fadeOverlay));
                });
    }
    
    private static void AnimateWorldFade(Volume volume) {
        // Animate post-processing effects for world fade
        // This would require specific post-processing profile setup
    }

    public static void SyncTrailPalette(MathFunctionType func) {
        Color startColor = Color.white;
        Color endColor = Color.gray;
        float width = 0.1f;
        bool useGlow = false;

        switch (func) {
            case MathFunctionType.Mirror:
                startColor = Color.cyan;
                endColor = Color.white;
                width = 0.1f;
                useGlow = true;
                break;
            case MathFunctionType.Cosine:
                startColor = new Color(1f, 0f, 1f); // Magenta
                endColor = Color.yellow;
                width = 0.15f;
                break;
            case MathFunctionType.Chaos:
                startColor = Random.ColorHSV();
                endColor = Random.ColorHSV();
                width = 0.25f;
                break;
            case MathFunctionType.Absorb:
                startColor = Color.black;
                endColor = new Color(0.2f, 0.2f, 0.2f);
                width = 0.05f;
                break;
            case MathFunctionType.Wave:
                startColor = Color.blue;
                endColor = Color.cyan;
                width = 0.12f;
                useGlow = true;
                break;
            case MathFunctionType.Fractal:
                startColor = new Color(1f, 0.5f, 0f); // Orange
                endColor = Color.red;
                width = 0.2f;
                break;
        }

        // Update all quantum trail renderers
        QuantumTrailRenderer[] trails = Object.FindObjectsOfType<QuantumTrailRenderer>();
        foreach (QuantumTrailRenderer trail in trails) {
            LineRenderer lr = trail.GetComponent<LineRenderer>();
            if (lr != null) {
                Gradient gradient = new Gradient();
                gradient.SetKeys(
                    new GradientColorKey[] {
                        new GradientColorKey(startColor, 0f),
                        new GradientColorKey(endColor, 1f)
                    },
                    new GradientAlphaKey[] {
                        new GradientAlphaKey(1f, 0f),
                        new GradientAlphaKey(0.3f, 1f)
                    }
                );
                lr.colorGradient = gradient;
                lr.widthMultiplier = width;
                
                // Apply glow effect if needed
                if (useGlow && quantumTrailMaterial != null) {
                    lr.material = quantumTrailMaterial;
                    lr.material.SetFloat("_GlowIntensity", 2f);
                }
            }
        }
        
        // Global shader update for function type
        Shader.SetGlobalFloat("_QuantumFunctionType", (float)func);
        Shader.SetGlobalColor("_QuantumPrimaryColor", startColor);
        Shader.SetGlobalColor("_QuantumSecondaryColor", endColor);
    }
    
    public static void UpdateResourceVisuals(string resourceID, ResourceState state) {
        GameObject resource = GameObject.Find(resourceID);
        if (resource == null) return;
        
        Renderer renderer = resource.GetComponent<Renderer>();
        if (renderer == null) return;
        
        switch (state) {
            case ResourceState.Classical:
                renderer.material.color = Color.white;
                renderer.material.SetFloat("_EmissionIntensity", 0f);
                break;
                
            case ResourceState.Quantum:
                renderer.material.color = Color.cyan;
                renderer.material.SetFloat("_EmissionIntensity", 1.5f);
                CreateQuantumAura(resource.transform);
                break;
                
            case ResourceState.Entangled:
                renderer.material.color = Color.magenta;
                renderer.material.SetFloat("_EmissionIntensity", 2f);
                CreateEntanglementEffect(resource.transform);
                break;
                
            case ResourceState.Collapsed:
                renderer.material.color = Color.gray;
                renderer.material.SetFloat("_EmissionIntensity", 0.2f);
                CreateCollapseRipple(resource.transform);
                break;
        }
    }
    
    private static void CreateQuantumAura(Transform target) {
        GameObject aura = new GameObject("QuantumAura");
        aura.transform.SetParent(target);
        aura.transform.localPosition = Vector3.zero;
        
        ParticleSystem ps = aura.AddComponent<ParticleSystem>();
        var main = ps.main;
        main.startLifetime = 3f;
        main.startSpeed = 0.5f;
        main.startSize = 0.2f;
        main.startColor = Color.cyan;
        main.maxParticles = 50;
        
        var emission = ps.emission;
        emission.rateOverTime = 15f;
        
        var shape = ps.shape;
        shape.shapeType = ParticleSystemShapeType.Sphere;
        shape.radius = 1f;
    }
    
    private static void CreateEntanglementEffect(Transform target) {
        // Create quantum entanglement visual links
        Debug.Log($"[QuantumVisuals] Creating entanglement effect for {target.name}");
    }
    
    private static void CreateCollapseRipple(Transform target) {
        // Create wave collapse ripple effect
        Debug.Log($"[QuantumVisuals] Creating collapse ripple for {target.name}");
    }
}