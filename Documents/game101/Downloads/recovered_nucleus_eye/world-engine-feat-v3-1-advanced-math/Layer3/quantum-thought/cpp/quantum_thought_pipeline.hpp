#ifndef QUANTUM_THOUGHT_PIPELINE_HPP
#define QUANTUM_THOUGHT_PIPELINE_HPP

#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <iostream>

struct ThoughtZone {
    std::string name;
    std::string type;
    std::array<float, 3> position{};
    std::array<float, 3> scale{};
    std::string color;
};

struct Agent {
    std::string id;
    std::array<float, 3> position{};
    int   maxSteps = 0;
    float stepSize = 1.0f;
    std::string trailStartColor;
    std::string trailEndColor;
};

class QuantumThoughtPipeline {
public:
    QuantumThoughtPipeline() { buildField(); }

    void buildField() {
        zones.push_back(createZone("QuantumGround", "Ground",
                                   {0.f, 0.f, 0.f}, {2.f, 1.f, 2.f}, "gray"));
        zones.push_back(createZone("MirrorZone", "Mirror",
                                   {6.f, 0.1f, 0.f}, {3.f, 0.2f, 3.f}, "cyan"));
        zones.push_back(createZone("ChaosZone", "Chaos",
                                   {0.f, 0.1f, 6.f}, {3.f, 0.2f, 3.f}, "magenta"));
        zones.push_back(createZone("AbsorbPit", "Absorb",
                                   {0.f, -2.0f, 0.f}, {1.f, 1.f, 1.f}, "black"));
        agent = createAgent("agent_1", {-4.f, 1.f, -4.f}, 10, 1.5f, "cyan", "blue");

        std::cout << "🔮 Quantum Thought Pipeline Initialized\n";
    }

    const std::vector<ThoughtZone>& getZones() const { return zones; }
    const Agent& getAgent() const { return agent; }

private:
    std::vector<ThoughtZone> zones;
    Agent agent;

    static ThoughtZone createZone(const std::string& name, const std::string& type,
                                  std::array<float,3> pos, std::array<float,3> scale,
                                  const std::string& color) {
        ThoughtZone z{name, type, pos, scale, color};
        return z;
    }

    static Agent createAgent(const std::string& id, std::array<float,3> pos,
                             int maxSteps, float stepSize,
                             const std::string& colorStart, const std::string& colorEnd) {
        Agent a{id, pos, maxSteps, stepSize, colorStart, colorEnd};
        return a;
    }
};

#endif // QUANTUM_THOUGHT_PIPELINE_HPP
