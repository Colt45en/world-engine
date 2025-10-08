using UnityEngine;
using System.Collections.Generic;

public static class QuantumAudio {
    private static Dictionary<MathFunctionType, AudioClip> functionTones;
    private static Dictionary<ResourceState, AudioClip> resourceTones;
    private static AudioSource globalAudioSource;
    
    static QuantumAudio() {
        InitializeAudio();
    }
    
    private static void InitializeAudio() {
        // Create global audio source
        GameObject audioGO = new GameObject("QuantumAudioManager");
        globalAudioSource = audioGO.AddComponent<AudioSource>();
        globalAudioSource.volume = 0.7f;
        globalAudioSource.spatialBlend = 0f; // 2D audio
        Object.DontDestroyOnLoad(audioGO);
        
        LoadAudioClips();
    }
    
    private static void LoadAudioClips() {
        functionTones = new Dictionary<MathFunctionType, AudioClip>();
        resourceTones = new Dictionary<ResourceState, AudioClip>();
        
        // Load function-specific tones
        functionTones[MathFunctionType.Mirror] = Resources.Load<AudioClip>("Audio/Tones/Tone_Mirror");
        functionTones[MathFunctionType.Cosine] = Resources.Load<AudioClip>("Audio/Tones/Tone_Cosine");
        functionTones[MathFunctionType.Chaos] = Resources.Load<AudioClip>("Audio/Tones/Tone_Chaos");
        functionTones[MathFunctionType.Absorb] = Resources.Load<AudioClip>("Audio/Tones/Tone_Absorb");
        functionTones[MathFunctionType.Wave] = Resources.Load<AudioClip>("Audio/Tones/Tone_Wave");
        functionTones[MathFunctionType.Fractal] = Resources.Load<AudioClip>("Audio/Tones/Tone_Fractal");
        
        // Load resource state tones
        resourceTones[ResourceState.Classical] = Resources.Load<AudioClip>("Audio/Resources/Classical_Hum");
        resourceTones[ResourceState.Quantum] = Resources.Load<AudioClip>("Audio/Resources/Quantum_Resonance");
        resourceTones[ResourceState.Entangled] = Resources.Load<AudioClip>("Audio/Resources/Entanglement_Harmonic");
        resourceTones[ResourceState.Collapsed] = Resources.Load<AudioClip>("Audio/Resources/Collapse_Echo");
    }

    public static void PlayFunctionTone(MathFunctionType func) {
        if (functionTones.ContainsKey(func) && functionTones[func] != null) {
            AudioClip clip = functionTones[func];
            AudioSource.PlayClipAtPoint(clip, Vector3.zero, 0.8f);
        } else {
            // Generate procedural tone based on function type
            GenerateProceduralTone(func);
        }
    }
    
    private static void GenerateProceduralTone(MathFunctionType func) {
        float frequency = 440f; // Base A note
        float duration = 1f;
        
        switch (func) {
            case MathFunctionType.Mirror:
                frequency = 523.25f; // C5
                break;
            case MathFunctionType.Cosine:
                frequency = 659.25f; // E5
                break;
            case MathFunctionType.Chaos:
                frequency = Random.Range(200f, 800f); // Random frequency
                break;
            case MathFunctionType.Absorb:
                frequency = 196f; // G3 (lower, darker)
                break;
            case MathFunctionType.Wave:
                frequency = 440f; // A4
                break;
            case MathFunctionType.Fractal:
                frequency = 783.99f; // G5
                break;
        }
        
        PlayProceduralTone(frequency, duration);
    }
    
    private static void PlayProceduralTone(float frequency, float duration) {
        if (globalAudioSource == null) return;
        
        int sampleRate = AudioSettings.outputSampleRate;
        int samples = Mathf.RoundToInt(sampleRate * duration);
        
        AudioClip clip = AudioClip.Create("ProceduralTone", samples, 1, sampleRate, false);
        float[] data = new float[samples];
        
        for (int i = 0; i < samples; i++) {
            float t = (float)i / sampleRate;
            data[i] = Mathf.Sin(2f * Mathf.PI * frequency * t) * 0.3f;
            
            // Apply envelope
            float envelope = 1f;
            if (t < 0.1f) envelope = t / 0.1f; // Fade in
            else if (t > duration - 0.2f) envelope = (duration - t) / 0.2f; // Fade out
            
            data[i] *= envelope;
        }
        
        clip.SetData(data, 0);
        globalAudioSource.PlayOneShot(clip);
    }

    public static void PlayEchoField() {
        AudioClip clip = Resources.Load<AudioClip>("Audio/Effects/EchoCollapse");
        if (clip != null) {
            AudioSource.PlayClipAtPoint(clip, Vector3.zero, 1f);
        } else {
            // Generate procedural echo field
            GenerateEchoField();
        }
    }
    
    private static void GenerateEchoField() {
        // Create layered echo effect with multiple frequencies
        float[] frequencies = { 220f, 330f, 440f, 660f };
        
        foreach (float freq in frequencies) {
            float delay = Random.Range(0f, 0.5f);
            MonoBehaviourHelper.Instance.StartCoroutine(
                PlayDelayedTone(freq, 2f, delay)
            );
        }
    }
    
    private static System.Collections.IEnumerator PlayDelayedTone(float frequency, float duration, float delay) {
        yield return new WaitForSeconds(delay);
        PlayProceduralTone(frequency, duration);
    }
    
    public static void PlayResourceTone(string resourceID, ResourceState state) {
        if (resourceTones.ContainsKey(state) && resourceTones[state] != null) {
            AudioClip clip = resourceTones[state];
            
            // Get resource position for 3D audio
            GameObject resource = GameObject.Find(resourceID);
            Vector3 position = resource != null ? resource.transform.position : Vector3.zero;
            
            AudioSource.PlayClipAtPoint(clip, position, 0.6f);
        } else {
            // Generate procedural resource tone
            GenerateResourceTone(state);
        }
    }
    
    private static void GenerateResourceTone(ResourceState state) {
        float frequency = 440f;
        float duration = 0.8f;
        
        switch (state) {
            case ResourceState.Classical:
                frequency = 261.63f; // C4 - stable, grounded
                break;
            case ResourceState.Quantum:
                frequency = 523.25f; // C5 - higher, energetic
                // Add harmonic overtones for quantum superposition
                PlayProceduralTone(frequency * 1.5f, duration * 0.5f); // Fifth harmonic
                break;
            case ResourceState.Entangled:
                frequency = 698.46f; // F5 - complex harmony
                // Play two intertwined frequencies
                PlayProceduralTone(frequency, duration);
                PlayProceduralTone(frequency * 1.25f, duration); // Major third
                return;
            case ResourceState.Collapsed:
                frequency = 146.83f; // D3 - low, final
                break;
        }
        
        PlayProceduralTone(frequency, duration);
    }
    
    // Ambient quantum field audio
    public static void StartQuantumAmbient() {
        if (globalAudioSource == null) return;
        
        AudioClip ambientClip = Resources.Load<AudioClip>("Audio/Ambient/QuantumField");
        if (ambientClip != null) {
            globalAudioSource.clip = ambientClip;
            globalAudioSource.loop = true;
            globalAudioSource.volume = 0.3f;
            globalAudioSource.Play();
        }
    }
    
    public static void StopQuantumAmbient() {
        if (globalAudioSource != null && globalAudioSource.isPlaying) {
            globalAudioSource.Stop();
        }
    }
}

// Helper MonoBehaviour for coroutines in static context
public class MonoBehaviourHelper : MonoBehaviour {
    public static MonoBehaviourHelper Instance { get; private set; }
    
    private void Awake() {
        if (Instance == null) {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        } else {
            Destroy(gameObject);
        }
    }
}