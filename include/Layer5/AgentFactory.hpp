#pragma once

#include <vector>
#include <memory>
#include <string>
#include <random>
#include <unordered_map>
#include <chrono>
#include "../Layer0/OverseerBrain.hpp"

namespace WorldEngine::Layer5 {

/**
 * Agent Types and Behavioral Patterns
 */
enum class AgentType {
    COOPERATIVE,      // Forest-like environments, resource sharing
    SOCIAL,          // City environments, communication focused
    ADAPTIVE,        // Ocean environments, exploration based
    ANALYTICAL,      // Space environments, problem solving
    CREATIVE,        // Playground environments, play-based learning
    HYBRID          // Mixed behavioral patterns
};

enum class LearningMode {
    RESOURCE_SHARING,
    COMMUNICATION,
    EXPLORATION,
    PROBLEM_SOLVING,
    PLAY_BASED,
    REINFORCEMENT,
    IMITATION,
    EVOLUTIONARY
};

/**
 * Core Agent Structure
 */
struct Agent {
    std::string id;
    AgentType type;
    LearningMode learning;

    // Spatial properties
    float x, y, z;
    float orientation;
    float speed;

    // Learning metrics
    float experience;
    float adaptability;
    float cooperation_score;
    float innovation_score;
    float survival_rating;

    // Neural network weights (simplified)
    std::vector<float> neural_weights;
    std::vector<float> memory_state;

    // Behavioral traits
    float aggression;
    float curiosity;
    float social_tendency;
    float risk_tolerance;

    // Performance tracking
    int successful_interactions;
    int failed_interactions;
    std::chrono::steady_clock::time_point birth_time;
    std::chrono::steady_clock::time_point last_update;

    // Genetic markers for evolution
    std::string genetic_signature;
    int generation;
    std::vector<std::string> parent_ids;

    Agent() : experience(0.0f), adaptability(0.5f), cooperation_score(0.5f),
              innovation_score(0.5f), survival_rating(0.5f),
              aggression(0.5f), curiosity(0.5f), social_tendency(0.5f),
              risk_tolerance(0.5f), successful_interactions(0),
              failed_interactions(0), generation(0) {
        birth_time = std::chrono::steady_clock::now();
        last_update = birth_time;
        neural_weights.resize(64);  // Basic neural network
        memory_state.resize(32);
    }
};

/**
 * Environment Context for Agent Spawning
 */
struct Environment {
    std::string id;
    std::string prompt;
    AgentType preferred_agent_type;
    LearningMode primary_learning_mode;

    // Environmental parameters
    float resource_abundance;
    float danger_level;
    float social_complexity;
    float task_difficulty;

    // Active agents in this environment
    std::vector<std::shared_ptr<Agent>> agents;

    // Performance metrics
    int total_interactions;
    float average_learning_rate;
    float environment_effectiveness;

    std::chrono::steady_clock::time_point created_time;
};

/**
 * Agent Factory - Core Layer 5 Component
 * Responsible for spawning, evolving, and managing autonomous agents
 */
class AgentFactory {
private:
    std::mt19937 rng;
    std::shared_ptr<Layer0::OverseerBrain> overseer;

    // Agent management
    std::vector<std::shared_ptr<Agent>> active_agents;
    std::vector<std::shared_ptr<Agent>> retired_agents;
    std::vector<std::shared_ptr<Environment>> environments;

    // Factory parameters
    int max_agents_per_environment;
    int max_total_agents;
    float mutation_rate;
    float crossover_probability;
    float retirement_threshold;

    // Agent ID generation
    std::unordered_map<AgentType, int> type_counters;
    int global_agent_counter;

    // Performance tracking
    struct FactoryMetrics {
        int agents_created;
        int agents_retired;
        int successful_evolutions;
        int failed_spawns;
        float average_agent_lifespan;
        std::chrono::steady_clock::time_point last_cleanup;
    } metrics;

public:
    AgentFactory(std::shared_ptr<Layer0::OverseerBrain> overseer_brain);
    ~AgentFactory() = default;

    // Core factory operations
    std::shared_ptr<Agent> spawnAgent(const Environment& env);
    std::shared_ptr<Agent> spawnAgent(AgentType type, LearningMode learning);
    std::vector<std::shared_ptr<Agent>> spawnAgentBatch(int count, const Environment& env);

    // Evolution and breeding
    std::shared_ptr<Agent> evolveAgent(const Agent& parent);
    std::shared_ptr<Agent> crossbreedAgents(const Agent& parent1, const Agent& parent2);
    void mutateAgent(Agent& agent);

    // Environment management
    std::shared_ptr<Environment> createEnvironment(const std::string& prompt,
                                                  AgentType preferred_type);
    void updateEnvironment(const std::string& env_id);
    void retireEnvironment(const std::string& env_id);

    // Agent lifecycle management
    void updateAgent(Agent& agent, float delta_time);
    void evaluateAgentPerformance(Agent& agent);
    void retireAgent(const std::string& agent_id);
    void cleanupRetiredAgents();

    // Learning and training
    void facilitateAgentInteraction(Agent& agent1, Agent& agent2);
    void facilitateHumanAgentInteraction(Agent& agent, const std::string& human_action);
    void rewardAgent(Agent& agent, float reward);
    void punishAgent(Agent& agent, float penalty);

    // Selection and survival
    std::vector<std::shared_ptr<Agent>> selectEliteAgents(int count);
    std::vector<std::shared_ptr<Agent>> selectAgentsForEvolution();
    void applyNaturalSelection();

    // Query and analysis
    std::vector<std::shared_ptr<Agent>> getActiveAgents() const;
    std::vector<std::shared_ptr<Agent>> getAgentsByType(AgentType type) const;
    std::vector<std::shared_ptr<Agent>> getAgentsByEnvironment(const std::string& env_id) const;

    // Performance metrics
    FactoryMetrics getMetrics() const { return metrics; }
    float getAverageAgentFitness() const;
    float getPopulationDiversity() const;

    // Configuration
    void setMaxAgents(int max_total, int max_per_env);
    void setEvolutionParameters(float mutation_rate, float crossover_prob);
    void setRetirementThreshold(float threshold);

    // Integration with other layers
    void connectMorphologyEngine(void* morph_engine);
    void connectAudioVisualEngine(void* av_engine);
    void connectSynthesisEngine(void* synth_engine);

    // Export/Import for persistence
    std::string exportAgentGenome(const Agent& agent) const;
    std::shared_ptr<Agent> importAgentGenome(const std::string& genome);
    void saveFactoryState(const std::string& filepath);
    void loadFactoryState(const std::string& filepath);

private:
    // Internal helper methods
    std::string generateAgentId(AgentType type);
    std::string generateGeneticSignature();
    void initializeAgentTraits(Agent& agent, const Environment& env);
    void initializeNeuralWeights(Agent& agent);
    float calculateAgentFitness(const Agent& agent) const;
    float calculateGeneticDistance(const Agent& agent1, const Agent& agent2) const;

    // Neural network operations
    std::vector<float> processNeuralNetwork(const Agent& agent,
                                          const std::vector<float>& inputs);
    void backpropagateError(Agent& agent, const std::vector<float>& error);

    // Behavioral modeling
    void updateAgentBehavior(Agent& agent, const Environment& env);
    float calculateSocialCompatibility(const Agent& agent1, const Agent& agent2) const;

    // Layer 0 compliance
    bool validateAgentSpawn(const Agent& proposed_agent);
    void enforceCanvasLaws(Agent& agent);
    void reportToOverseer(const std::string& event, const Agent& agent);
};

} // namespace WorldEngine::Layer5
