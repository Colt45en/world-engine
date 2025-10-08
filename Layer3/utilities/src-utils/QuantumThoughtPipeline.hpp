
#ifndef QUANTUM_THOUGHT_PIPELINE_HPP
#define QUANTUM_THOUGHT_PIPELINE_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <iostream>

struct ThoughtZone {
    std::string name;
    std::string type;
    float position[3];
    float scale[3];
    std::string color;
};

struct Agent {
    std::string id;
    float position[3];
    int maxSteps;
    float stepSize;
    std::string trailStartColor;
    std::string trailEndColor;
};

class QuantumThoughtPipeline {
public:
    QuantumThoughtPipeline() {
        buildField();
    }

    void buildField() {
        zones.push_back(createZone("QuantumGround", "Ground", {0, 0, 0}, {2, 1, 2}, "gray"));
        zones.push_back(createZone("MirrorZone", "Mirror", {6, 0.1f, 0}, {3, 0.2f, 3}, "cyan"));
        zones.push_back(createZone("ChaosZone", "Chaos", {0, 0.1f, 6}, {3, 0.2f, 3}, "magenta"));
        zones.push_back(createZone("AbsorbPit", "Absorb", {0, -2.0f, 0}, {1, 1, 1}, "black"));
        agent = createAgent("agent_1", {-4, 1, -4}, 10, 1.5f, "cyan", "blue");

        std::cout << "🔮 Quantum Thought Pipeline Initialized
";
    }

    const std::vector<ThoughtZone>& getZones() const { return zones; }
    const Agent& getAgent() const { return agent; }

private:
    std::vector<ThoughtZone> zones;
    Agent agent;

    ThoughtZone createZone(const std::string& name, const std::string& type,
                           std::initializer_list<float> pos, std::initializer_list<float> scale,
                           const std::string& color) {
        ThoughtZone zone{name, type, {}, {}, color};
        std::copy(pos.begin(), pos.end(), zone.position);
        std::copy(scale.begin(), scale.end(), zone.scale);
        return zone;
    }

    Agent createAgent(const std::string& id, std::initializer_list<float> pos,
                      int maxSteps, float stepSize, const std::string& colorStart,
                      const std::string& colorEnd) {
        Agent a{id, {}, maxSteps, stepSize, colorStart, colorEnd};
        std::copy(pos.begin(), pos.end(), a.position);
        return a;
    }
};

#endif // QUANTUM_THOUGHT_PIPELINE_HPP
