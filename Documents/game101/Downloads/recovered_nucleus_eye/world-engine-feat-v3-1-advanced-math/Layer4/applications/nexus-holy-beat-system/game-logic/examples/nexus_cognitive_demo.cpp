#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <vector>
#include <string>

// Include NEXUS headers
#include "NexusProtocol.hpp"
#include "NexusVisuals.hpp"
#include "NexusTrailRenderer.hpp"
#include "NexusRecursiveKeeperEngine.hpp"

using namespace NEXUS;

// Cognitive topic generator for demonstration
class CognitiveTopicGenerator {
public:
    CognitiveTopicGenerator() : gen_(rd_()), topic_dist_(0, topics_.size() - 1) {}

    std::string GetRandomTopic() {
        return topics_[topic_dist_(gen_)];
    }

    std::string GetPhilosophicalTopic() {
        std::vector<std::string> philosophical = {
            "The Nature of Consciousness",
            "Digital Transcendence",
            "Quantum Entanglement of Ideas",
            "Temporal Perception Loops",
            "Emergent Intelligence",
            "Collective Memory Networks",
            "Recursive Reality Structures",
            "Harmonic Cognitive Resonance"
        };
        std::uniform_int_distribution<size_t> dist(0, philosophical.size() - 1);
        return philosophical[dist(gen_)];
    }

    std::string GetTechnicalTopic() {
        std::vector<std::string> technical = {
            "Neural Network Optimization",
            "Distributed Computing Paradigms",
            "Quantum Information Processing",
            "Adaptive System Architecture",
            "Recursive Algorithm Design",
            "Real-Time Data Synchronization",
            "Multi-Dimensional State Management",
            "Dynamic Resource Allocation"
        };
        std::uniform_int_distribution<size_t> dist(0, technical.size() - 1);
        return technical[dist(gen_)];
    }

private:
    std::vector<std::string> topics_ = {
        "Artificial Intelligence",
        "Human Consciousness",
        "Digital Art Creation",
        "Music and Cognition",
        "Recursive Thinking",
        "Quantum Computing",
        "Evolutionary Algorithms",
        "Collective Intelligence",
        "Temporal Dynamics",
        "Emergent Complexity"
    };

    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_int_distribution<size_t> topic_dist_;
};

// Enhanced audio/art simulator for cognitive demo
class CognitiveDataSimulator {
public:
    CognitiveDataSimulator() : gen_(rd_()), param_dist_(0.0f, 1.0f) {}

    AudioData GenerateCognitiveAudio(const std::string& topic) {
        AudioData data;

        // Topic influences audio characteristics
        float topic_hash = std::hash<std::string>{}(topic) / static_cast<float>(SIZE_MAX);

        data.volume = 0.3f + 0.7f * (std::sin(time_ * 0.8f + topic_hash * 10.0f) + 1.0f) * 0.5f;
        data.bass = 0.2f + 0.8f * (std::sin(time_ * 1.2f + topic_hash * 7.0f) + 1.0f) * 0.5f;
        data.midrange = 0.3f + 0.7f * (std::sin(time_ * 1.0f + topic_hash * 5.0f) + 1.0f) * 0.5f;
        data.treble = 0.1f + 0.9f * (std::sin(time_ * 1.5f + topic_hash * 3.0f) + 1.0f) * 0.5f;
        data.bpm = 100.0f + 60.0f * (std::sin(time_ * 0.3f + topic_hash) + 1.0f) * 0.5f;
        data.beat_detected = param_dist_(gen_) > (0.8f - data.volume * 0.3f);
        data.spectral_centroid = 0.3f + 0.4f * topic_hash + 0.3f * data.treble;
        data.spectral_rolloff = 0.4f + 0.6f * data.midrange;
        data.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        time_ += 0.1f;
        return data;
    }

    ArtData GenerateCognitiveArt(const std::string& topic) {
        ArtData data;

        float topic_hash = std::hash<std::string>{}(topic) / static_cast<float>(SIZE_MAX);

        data.complexity = 0.4f + 0.6f * topic_hash;
        data.brightness = 0.3f + 0.7f * (std::sin(art_time_ * 0.7f + topic_hash * 8.0f) + 1.0f) * 0.5f;
        data.contrast = 0.2f + 0.8f * (std::sin(art_time_ * 0.9f + topic_hash * 6.0f) + 1.0f) * 0.5f;
        data.saturation = 0.5f + 0.5f * (std::sin(art_time_ * 1.1f + topic_hash * 4.0f) + 1.0f) * 0.5f;

        // Topic-based art style
        if (topic.find("Quantum") != std::string::npos || topic.find("Digital") != std::string::npos) {
            data.style = "geometric";
        } else if (topic.find("Consciousness") != std::string::npos || topic.find("Emergent") != std::string::npos) {
            data.style = "abstract";
        } else {
            data.style = "organic";
        }

        data.dominant_color[0] = 0.5f + 0.5f * std::sin(topic_hash * 12.0f);
        data.dominant_color[1] = 0.5f + 0.5f * std::sin(topic_hash * 15.0f);
        data.dominant_color[2] = 0.5f + 0.5f * std::sin(topic_hash * 18.0f);

        data.texture_intensity = 0.3f + 0.7f * topic_hash;
        data.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        art_time_ += 0.05f;
        return data;
    }

private:
    std::random_device rd_;
    std::mt19937 gen_;
    std::uniform_real_distribution<float> param_dist_;
    float time_ = 0.0f;
    float art_time_ = 0.0f;
};

// Cognitive trail entity that visualizes thought evolution
class CognitiveTrailEntity {
public:
    CognitiveTrailEntity(const std::string& name, const std::string& initial_topic)
        : name_(name), current_topic_(initial_topic) {
        trail_ = NexusTrailManager::Instance().CreateTrail(name);
        trail_->SetMaxTrailPoints(30);
        trail_->SetTrailLifetime(5.0f);
        trail_->SetProcessingModeSync(true);
        trail_->SetAudioReactive(true);
        trail_->SetArtReactive(true);

        // Initialize position based on topic hash
        float topic_hash = std::hash<std::string>{}(initial_topic) / static_cast<float>(SIZE_MAX);
        position_[0] = 10.0f * (topic_hash - 0.5f);
        position_[1] = 8.0f * (std::sin(topic_hash * 6.28f));
        position_[2] = 0.0f;
    }

    void UpdateWithCognitiveNode(const CognitiveNode& node, float deltaTime) {
        current_topic_ = node.topic;

        // Movement based on cognitive properties
        float complexity_factor = node.conceptual_complexity * 2.0f;
        float intensity_factor = node.cognitive_intensity * 0.5f;

        static float time = 0.0f;
        time += deltaTime;

        // Topic-influenced movement
        float topic_hash = std::hash<std::string>{}(node.topic) / static_cast<float>(SIZE_MAX);

        position_[0] += complexity_factor * std::cos(time * intensity_factor + topic_hash * 10.0f) * deltaTime;
        position_[1] += complexity_factor * std::sin(time * intensity_factor + topic_hash * 8.0f) * deltaTime;
        position_[2] = 2.0f * std::sin(time * 0.5f + topic_hash * 5.0f);

        // Add trail point with cognitive-influenced width
        float trail_width = 0.5f + node.cognitive_intensity;
        trail_->AddPoint(position_, trail_width);
        trail_->Update(deltaTime);
    }

    void Render() const {
        std::cout << "[" << name_ << "] Topic: \"" << current_topic_ << "\" ";
        trail_->Render();
    }

    const std::string& GetCurrentTopic() const { return current_topic_; }

private:
    std::string name_;
    std::string current_topic_;
    std::shared_ptr<NexusTrailRenderer> trail_;
    float position_[3];
};

// Main cognitive demonstration
void RunCognitiveEvolutionDemo() {
    std::cout << "\n🧠 NEXUS Recursive Keeper Engine - Cognitive Evolution Demo\n";
    std::cout << "==========================================================\n";

    // Initialize systems
    NexusRecursiveKeeperEngine keeper;
    CognitiveTopicGenerator topic_gen;
    CognitiveDataSimulator data_sim;
    auto& protocol = NexusProtocol::Instance();

    protocol.SetDebugMode(true);
    protocol.SetAutoModeEnabled(false); // Manual cognitive control

    // Create cognitive trail entities
    std::vector<std::unique_ptr<CognitiveTrailEntity>> cognitive_entities;
    cognitive_entities.push_back(std::make_unique<CognitiveTrailEntity>("Philosopher", topic_gen.GetPhilosophicalTopic()));
    cognitive_entities.push_back(std::make_unique<CognitiveTrailEntity>("Technician", topic_gen.GetTechnicalTopic()));
    cognitive_entities.push_back(std::make_unique<CognitiveTrailEntity>("Artist", "Digital Art Creation"));

    std::cout << "🚀 Starting cognitive evolution simulation...\n";
    std::cout << "Entities: " << cognitive_entities.size() << "\n";
    std::cout << "Processing modes will be driven by cognitive analysis\n\n";

    auto last_time = std::chrono::steady_clock::now();
    int iteration = 0;

    // Main cognitive evolution loop
    while (iteration < 50) { // ~25 seconds at 2 Hz
        auto current_time = std::chrono::steady_clock::now();
        float delta_time = std::chrono::duration<float>(current_time - last_time).count();
        last_time = current_time;

        // Select entity for cognitive processing
        int entity_index = iteration % cognitive_entities.size();
        auto& entity = cognitive_entities[entity_index];
        std::string current_topic = entity->GetCurrentTopic();

        // Generate contextual audio/art data
        AudioData audio = data_sim.GenerateCognitiveAudio(current_topic);
        ArtData art = data_sim.GenerateCognitiveArt(current_topic);

        // Update NEXUS protocol with cognitive-influenced data
        protocol.BroadcastAudioData(audio);
        protocol.BroadcastArtData(art);

        // Trigger cognitive analysis
        keeper.TriggerCognitiveAnalysis(current_topic);

        // Get the latest cognitive node from memory
        const auto& memory = keeper.Memory().Forward();
        if (!memory.empty()) {
            const CognitiveNode& latest_node = memory.back();
            entity->UpdateWithCognitiveNode(latest_node, delta_time);

            // Clear screen periodically for animation
            if (iteration % 3 == 0) {
                system("cls"); // Windows - use "clear" on Linux/macOS

                std::cout << "🧠 NEXUS Cognitive Evolution [Iteration " << iteration << "]\n";
                std::cout << "Current Mode: " << ProcessingModeToString(protocol.GetProcessingMode()) << "\n";
                std::cout << "Cognitive Consensus: " << ProcessingModeToString(keeper.GetCognitiveConsensusMode()) << "\n";
                std::cout << "System Complexity: " << std::fixed << std::setprecision(2)
                          << keeper.GetSystemComplexity() << "\n";
                std::cout << "System Intensity: " << keeper.GetSystemIntensity() << "\n\n";

                // Show current cognitive analysis
                std::cout << "🔬 Latest Cognitive Analysis:\n";
                std::cout << "Topic: \"" << latest_node.topic << "\"\n";
                std::cout << "Suggested Mode: " << ProcessingModeToString(latest_node.suggested_mode) << "\n";
                std::cout << "Cognitive Intensity: " << latest_node.cognitive_intensity << "\n";
                std::cout << "Conceptual Complexity: " << latest_node.conceptual_complexity << "\n\n";

                std::cout << "🌀 State Analysis:\n";
                std::cout << "Solid: " << latest_node.solid_state << "\n";
                std::cout << "Liquid: " << latest_node.liquid_state << "\n";
                std::cout << "Gas: " << latest_node.gas_state << "\n\n";

                std::cout << "🔮 Derived Evolution:\n";
                std::cout << "Next Topic: \"" << latest_node.derived_topic << "\"\n\n";

                // Show cognitive trails
                std::cout << "🌈 Cognitive Trails:\n";
                for (const auto& ent : cognitive_entities) {
                    ent->Render();
                }

                // Show memory statistics
                std::cout << "\n📊 Cognitive Memory:\n";
                std::cout << "Total Nodes: " << memory.size() << "\n";
                std::cout << "Imprints: " << keeper.Imprints().All().size() << "\n";
                std::cout << "Dominant Mode: " << ProcessingModeToString(keeper.Memory().GetDominantMode()) << "\n";

                auto palette = NexusVisuals::GetCurrentPalette();
                std::cout << "Current Palette Intensity: " << palette.intensity << "\n";
            }
        }

        iteration++;
        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 2 Hz
    }

    std::cout << "\n🎯 Cognitive Evolution Demo Complete!\n";
    std::cout << "=====================================\n";
}

// Interactive cognitive exploration
void RunInteractiveCognitiveExploration() {
    std::cout << "\n🌟 Interactive Cognitive Exploration\n";
    std::cout << "====================================\n";

    NexusRecursiveKeeperEngine keeper;
    CognitiveDataSimulator data_sim;
    auto& protocol = NexusProtocol::Instance();

    protocol.SetDebugMode(false);

    std::cout << "Enter topics to explore (type 'quit' to exit):\n\n";

    std::string input;
    while (true) {
        std::cout << "🔍 Enter topic: ";
        std::getline(std::cin, input);

        if (input == "quit" || input == "exit") break;
        if (input.empty()) continue;

        std::cout << "\n🧠 Analyzing: \"" << input << "\"\n";
        std::cout << std::string(50, '-') << "\n";

        // Generate contextual data
        AudioData audio = data_sim.GenerateCognitiveAudio(input);
        ArtData art = data_sim.GenerateCognitiveArt(input);

        protocol.BroadcastAudioData(audio);
        protocol.BroadcastArtData(art);

        // Run cognitive processing
        keeper.ProcessTopic(input, 3);

        // Display results
        const auto& memory = keeper.Memory().Forward();
        if (!memory.empty()) {
            for (size_t i = std::max(0, static_cast<int>(memory.size()) - 3); i < memory.size(); ++i) {
                const auto& node = memory[i];
                std::cout << "\n🔬 Analysis " << (i + 1) << ":\n";
                std::cout << "Topic: " << node.topic << "\n";
                std::cout << "Mode: " << ProcessingModeToString(node.suggested_mode) << "\n";
                std::cout << "Intensity: " << node.cognitive_intensity << "\n";
                std::cout << "Complexity: " << node.conceptual_complexity << "\n";
                std::cout << "Next Evolution: " << node.derived_topic << "\n";
            }
        }

        std::cout << "\n📊 Current System State:\n";
        std::cout << "Memory Nodes: " << keeper.Memory().Forward().size() << "\n";
        std::cout << "Cognitive Consensus: " << ProcessingModeToString(keeper.GetCognitiveConsensusMode()) << "\n";
        std::cout << "Average Complexity: " << keeper.GetSystemComplexity() << "\n\n";
    }

    std::cout << "\n✅ Interactive exploration completed!\n";
}

// Cognitive memory analysis
void DemonstrateCognitiveMemoryPatterns() {
    std::cout << "\n🧬 Cognitive Memory Pattern Analysis\n";
    std::cout << "===================================\n";

    NexusRecursiveKeeperEngine keeper;
    CognitiveTopicGenerator topic_gen;
    CognitiveDataSimulator data_sim;

    // Process multiple related topics
    std::vector<std::string> topic_chain = {
        "Machine Learning",
        "Neural Networks",
        "Artificial Consciousness",
        "Digital Sentience",
        "Quantum Cognition",
        "Recursive Self-Awareness"
    };

    std::cout << "Processing cognitive topic chain...\n\n";

    for (const std::string& topic : topic_chain) {
        std::cout << "🔄 Processing: \"" << topic << "\"\n";

        // Generate contextual data
        AudioData audio = data_sim.GenerateCognitiveAudio(topic);
        ArtData art = data_sim.GenerateCognitiveArt(topic);

        auto& protocol = NexusProtocol::Instance();
        protocol.BroadcastAudioData(audio);
        protocol.BroadcastArtData(art);

        keeper.ProcessTopic(topic, 2);

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // Analyze patterns
    std::cout << "\n📈 Memory Pattern Analysis:\n";
    std::cout << "Total Memory Nodes: " << keeper.Memory().Forward().size() << "\n";
    std::cout << "Dominant Processing Mode: " << ProcessingModeToString(keeper.Memory().GetDominantMode()) << "\n";
    std::cout << "Average Cognitive Intensity: " << keeper.Memory().GetAverageIntensity() << "\n";
    std::cout << "Average Conceptual Complexity: " << keeper.Memory().GetAverageComplexity() << "\n";

    // Show imprint analysis
    std::cout << "\n🔮 Eternal Imprint Analysis:\n";
    auto recent_imprints = keeper.Imprints().GetRecentImprints(10000); // Last 10 seconds
    std::cout << "Recent Imprints: " << recent_imprints.size() << "\n";
    std::cout << "Most Influential Mode: " << ProcessingModeToString(keeper.Imprints().GetMostInfluentialMode()) << "\n";

    std::cout << "\n✨ Memory patterns demonstrate cognitive evolution!\n";
}

int main() {
    std::cout << "🚀 NEXUS Recursive Keeper Engine Integration Demo\n";
    std::cout << "=================================================\n";
    std::cout << "Cognitive Processing meets Audio-Reactive Quantum Protocols\n\n";

    try {
        // Initialize NEXUS systems
        auto& protocol = NexusProtocol::Instance();
        std::cout << "✅ NEXUS Protocol initialized\n";
        std::cout << "✅ Recursive Keeper Engine ready\n\n";

        // Demo 1: Cognitive evolution with visual trails
        RunCognitiveEvolutionDemo();

        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Demo 2: Memory pattern analysis
        DemonstrateCognitiveMemoryPatterns();

        std::this_thread::sleep_for(std::chrono::seconds(2));

        // Demo 3: Interactive exploration
        std::cout << "\nPress Enter to start interactive exploration (or Ctrl+C to exit)...";
        std::cin.get();
        RunInteractiveCognitiveExploration();

        // Final system statistics
        std::cout << "\n📊 Final System Statistics:\n";
        auto stats = protocol.GetStatistics();
        std::cout << "NEXUS Mode Changes: " << stats.mode_changes << "\n";
        std::cout << "Audio Updates: " << stats.audio_updates << "\n";
        std::cout << "Art Updates: " << stats.art_updates << "\n";
        std::cout << "Beats Detected: " << stats.beats_detected << "\n";

        std::cout << "\n🎉 NEXUS Recursive Keeper Engine integration completed!\n";
        std::cout << "Cognitive processing now seamlessly integrated with quantum protocols.\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
