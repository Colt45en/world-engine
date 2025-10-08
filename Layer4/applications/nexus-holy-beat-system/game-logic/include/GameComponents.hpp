#pragma once

#include "GameEntity.hpp"
#include <array>
#include <cmath>

namespace NexusGame {

/**
 * Transform Component - Position, rotation, scale in 3D space
 * Syncs with NEXUS worldEngine and artEngine for visual effects
 */
class Transform : public Component {
public:
    struct Vector3 {
        double x = 0.0, y = 0.0, z = 0.0;

        Vector3() = default;
        Vector3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

        Vector3 operator+(const Vector3& other) const {
            return Vector3(x + other.x, y + other.y, z + other.z);
        }

        Vector3 operator*(double scalar) const {
            return Vector3(x * scalar, y * scalar, z * scalar);
        }

        double magnitude() const {
            return std::sqrt(x*x + y*y + z*z);
        }
    };

private:
    Vector3 m_position;
    Vector3 m_rotation; // Euler angles in radians
    Vector3 m_scale{1.0, 1.0, 1.0};

    mutable bool m_matrixDirty = true;
    mutable std::array<std::array<double, 4>, 4> m_transformMatrix;

public:
    Transform(GameEntity* entity) : Component(entity) {}

    // Position
    const Vector3& GetPosition() const { return m_position; }
    void SetPosition(const Vector3& pos) { m_position = pos; m_matrixDirty = true; }
    void SetPosition(double x, double y, double z) { SetPosition(Vector3(x, y, z)); }

    // Rotation
    const Vector3& GetRotation() const { return m_rotation; }
    void SetRotation(const Vector3& rot) { m_rotation = rot; m_matrixDirty = true; }
    void SetRotation(double x, double y, double z) { SetRotation(Vector3(x, y, z)); }

    // Scale
    const Vector3& GetScale() const { return m_scale; }
    void SetScale(const Vector3& scale) { m_scale = scale; m_matrixDirty = true; }
    void SetScale(double x, double y, double z) { SetScale(Vector3(x, y, z)); }
    void SetScale(double uniform) { SetScale(Vector3(uniform, uniform, uniform)); }

    // Transform operations
    void Translate(const Vector3& delta) { SetPosition(m_position + delta); }
    void Rotate(const Vector3& deltaRot) { SetRotation(m_rotation + deltaRot); }

    // Matrix calculation
    const std::array<std::array<double, 4>, 4>& GetTransformMatrix() const;

private:
    void CalculateTransformMatrix() const;
};

/**
 * AudioSync Component - Synchronizes entity behavior with NEXUS audioEngine
 * Responds to BPM, harmonics, and audio analysis data
 */
class AudioSync : public Component {
public:
    enum class SyncMode {
        BPM_PULSE,      // Pulse/scale with BPM
        HARMONIC_WAVE,  // Wave motion based on harmonics
        AMPLITUDE_SCALE, // Scale with audio amplitude
        FREQUENCY_COLOR  // Change properties based on frequency
    };

private:
    SyncMode m_syncMode = SyncMode::BPM_PULSE;
    double m_intensity = 1.0;
    double m_phase = 0.0;
    double m_lastBeatTime = 0.0;

    // Audio data from NEXUS system
    double m_currentBPM = 120.0;
    int m_harmonics = 6;
    double m_amplitude = 0.0;
    double m_frequency = 440.0;

public:
    AudioSync(GameEntity* entity) : Component(entity) {}

    void Update(double deltaTime) override;

    // Configuration
    void SetSyncMode(SyncMode mode) { m_syncMode = mode; }
    SyncMode GetSyncMode() const { return m_syncMode; }

    void SetIntensity(double intensity) { m_intensity = intensity; }
    double GetIntensity() const { return m_intensity; }

    void SetPhase(double phase) { m_phase = phase; }
    double GetPhase() const { return m_phase; }

    // Audio data sync (called by engine)
    void UpdateAudioData(double bpm, int harmonics, double amplitude, double frequency);

    // Get calculated values
    double GetBeatPhase() const;
    double GetHarmonicValue(int harmonic) const;
    bool IsOnBeat() const;

private:
    void ApplyBPMPulse(double deltaTime);
    void ApplyHarmonicWave(double deltaTime);
    void ApplyAmplitudeScale(double deltaTime);
    void ApplyFrequencyColor(double deltaTime);
};

/**
 * ArtSync Component - Synchronizes with NEXUS artEngine for visual effects
 * Controls petal patterns, colors, and artistic transformations
 */
class ArtSync : public Component {
public:
    struct Color {
        double r = 1.0, g = 1.0, b = 1.0, a = 1.0;

        Color() = default;
        Color(double r_, double g_, double b_, double a_ = 1.0)
            : r(r_), g(g_), b(b_), a(a_) {}
    };

    enum class PatternMode {
        PETAL_FORMATION, // Follow petal count patterns
        SPIRAL_MOTION,   // Spiral based on golden ratio
        MANDALA_SYNC,    // Mandala-like patterns
        FRACTAL_DANCE    // Fractal transformations
    };

private:
    PatternMode m_patternMode = PatternMode::PETAL_FORMATION;
    Color m_baseColor{0.33, 0.94, 0.72, 1.0}; // NEXUS green
    Color m_accentColor{1.0, 0.41, 0.71, 1.0}; // NEXUS pink

    int m_petalCount = 8;
    double m_spiralRate = 1.618; // Golden ratio
    double m_animationTime = 0.0;

public:
    ArtSync(GameEntity* entity) : Component(entity) {}

    void Update(double deltaTime) override;
    void Render() override;

    // Configuration
    void SetPatternMode(PatternMode mode) { m_patternMode = mode; }
    PatternMode GetPatternMode() const { return m_patternMode; }

    void SetBaseColor(const Color& color) { m_baseColor = color; }
    const Color& GetBaseColor() const { return m_baseColor; }

    void SetAccentColor(const Color& color) { m_accentColor = color; }
    const Color& GetAccentColor() const { return m_accentColor; }

    void SetPetalCount(int count) { m_petalCount = count; }
    int GetPetalCount() const { return m_petalCount; }

    // Art data sync (called by engine)
    void UpdateArtData(int petalCount, double terrainRoughness);

    // Get calculated values
    Color GetCurrentColor() const;
    Transform::Vector3 GetPetalPosition(int petalIndex) const;
    double GetSpiralAngle() const;

private:
    void UpdatePetalFormation(double deltaTime);
    void UpdateSpiralMotion(double deltaTime);
    void UpdateMandalaSync(double deltaTime);
    void UpdateFractalDance(double deltaTime);
};

/**
 * Physics Component - Basic physics simulation
 * Integrates with NEXUS worldEngine for terrain and collision
 */
class Physics : public Component {
private:
    Transform::Vector3 m_velocity;
    Transform::Vector3 m_acceleration;
    double m_mass = 1.0;
    double m_drag = 0.98;
    bool m_useGravity = true;
    bool m_isGrounded = false;

    static constexpr double GRAVITY = -9.81;

public:
    Physics(GameEntity* entity) : Component(entity) {}

    void Update(double deltaTime) override;

    // Velocity
    const Transform::Vector3& GetVelocity() const { return m_velocity; }
    void SetVelocity(const Transform::Vector3& vel) { m_velocity = vel; }
    void AddVelocity(const Transform::Vector3& vel) { m_velocity = m_velocity + vel; }

    // Forces
    void AddForce(const Transform::Vector3& force);
    void AddImpulse(const Transform::Vector3& impulse);

    // Properties
    double GetMass() const { return m_mass; }
    void SetMass(double mass) { m_mass = mass; }

    double GetDrag() const { return m_drag; }
    void SetDrag(double drag) { m_drag = drag; }

    bool GetUseGravity() const { return m_useGravity; }
    void SetUseGravity(bool use) { m_useGravity = use; }

    bool IsGrounded() const { return m_isGrounded; }

private:
    void ApplyGravity(double deltaTime);
    void ApplyDrag(double deltaTime);
    void CheckGroundCollision();
};

} // namespace NexusGame
