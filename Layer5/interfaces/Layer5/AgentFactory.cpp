#include "AgentFactory.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>

namespace WorldEngine::Layer5 {

AgentFactory::AgentFactory(std::shared_ptr<Layer0::OverseerBrain> overseer_brain)
    : overseer(overseer_brain), rng(std::chrono::steady_clock::now().time_since_epoch().count()),
      max_agents_per_environment(10), max_total_agents(50), mutation_rate(0.05f),
      crossover_probability(0.7f), retirement_threshold(0.1f), global_agent_counter(0) {

    // Initialize type counters
    type_counters[AgentType::COOPERATIVE] = 0;
    type_counters[AgentType::SOCIAL] = 0;
    type_counters[AgentType::ADAPTIVE] = 0;
    type_counters[AgentType::ANALYTICAL] = 0;
    type_counters[AgentType::CREATIVE] = 0;
    type_counters[AgentType::HYBRID] = 0;

    // Initialize factory metrics
    metrics = {};
    metrics.last_cleanup = std::chrono::steady_clock::now();

    // Report initialization to Layer 0 Overseer
    if (overseer) {
        overseer->reportLayerStatus("Layer5", "AgentFactory", "INITIALIZED");
    }
}

std::shared_ptr<Agent> AgentFactory::spawnAgent(const Environment& env) {
    // Check Layer 0 canvas laws
    if (active_agents.size() >= max_total_agents) {
        reportToOverseer("SPAWN_FAILED", Agent());
        metrics.failed_spawns++;
        return nullptr;
    }

    auto agent = std::make_shared<Agent>();
    agent->id = generateAgentId(env.preferred_agent_type);
    agent->type = env.preferred_agent_type;
    agent->learning = env.primary_learning_mode;

    // Initialize spatial properties within environment bounds
    std::uniform_real_distribution<float> pos_dist(5.0f, 95.0f);
    agent->x = pos_dist(rng);
    agent->y = pos_dist(rng);
    agent->z = pos_dist(rng);

    std::uniform_real_distribution<float> orient_dist(0.0f, 2.0f * M_PI);
    agent->orientation = orient_dist(rng);
    agent->speed = 1.0f + pos_dist(rng) / 100.0f;

    // Initialize traits based on environment
    initializeAgentTraits(*agent, env);
    initializeNeuralWeights(*agent);

    agent->genetic_signature = generateGeneticSignature();
    agent->generation = 1;

    // Validate with Layer 0 Overseer
    if (!validateAgentSpawn(*agent)) {
        reportToOverseer("SPAWN_REJECTED", *agent);
        metrics.failed_spawns++;
        return nullptr;
    }

    // Add to active agents
    active_agents.push_back(agent);
    metrics.agents_created++;

    reportToOverseer("AGENT_SPAWNED", *agent);
    return agent;
}

std::shared_ptr<Agent> AgentFactory::spawnAgent(AgentType type, LearningMode learning) {
    if (active_agents.size() >= max_total_agents) {
        metrics.failed_spawns++;
        return nullptr;
    }

    auto agent = std::make_shared<Agent>();
    agent->id = generateAgentId(type);
    agent->type = type;
    agent->learning = learning;

    // Default spatial initialization
    std::uniform_real_distribution<float> pos_dist(10.0f, 90.0f);
    agent->x = pos_dist(rng);
    agent->y = pos_dist(rng);
    agent->z = pos_dist(rng);

    // Initialize traits based on type
    Environment default_env;
    default_env.preferred_agent_type = type;
    default_env.primary_learning_mode = learning;
    default_env.resource_abundance = 0.5f;
    default_env.danger_level = 0.3f;
    default_env.social_complexity = 0.6f;
    default_env.task_difficulty = 0.4f;

    initializeAgentTraits(*agent, default_env);
    initializeNeuralWeights(*agent);

    agent->genetic_signature = generateGeneticSignature();
    agent->generation = 1;

    if (validateAgentSpawn(*agent)) {
        active_agents.push_back(agent);
        metrics.agents_created++;
        reportToOverseer("AGENT_SPAWNED", *agent);
        return agent;
    }

    metrics.failed_spawns++;
    return nullptr;
}

std::vector<std::shared_ptr<Agent>> AgentFactory::spawnAgentBatch(int count, const Environment& env) {
    std::vector<std::shared_ptr<Agent>> batch;
    batch.reserve(count);

    for (int i = 0; i < count && active_agents.size() < max_total_agents; ++i) {
        auto agent = spawnAgent(env);
        if (agent) {
            batch.push_back(agent);
        }
    }

    return batch;
}

std::shared_ptr<Agent> AgentFactory::evolveAgent(const Agent& parent) {
    auto offspring = std::make_shared<Agent>(parent);
    offspring->id = generateAgentId(parent.type);
    offspring->generation = parent.generation + 1;
    offspring->parent_ids = {parent.id};

    // Reset experience and metrics for new generation
    offspring->experience = 0.0f;
    offspring->successful_interactions = 0;
    offspring->failed_interactions = 0;
    offspring->birth_time = std::chrono::steady_clock::now();

    // Apply mutations
    mutateAgent(*offspring);

    offspring->genetic_signature = generateGeneticSignature();

    if (validateAgentSpawn(*offspring)) {
        metrics.successful_evolutions++;
        reportToOverseer("AGENT_EVOLVED", *offspring);
        return offspring;
    }

    return nullptr;
}

std::shared_ptr<Agent> AgentFactory::crossbreedAgents(const Agent& parent1, const Agent& parent2) {
    // Only crossbreed compatible types
    float compatibility = calculateSocialCompatibility(parent1, parent2);
    if (compatibility < 0.3f) {
        return nullptr;
    }

    auto offspring = std::make_shared<Agent>();
    offspring->id = generateAgentId(parent1.type);
    offspring->generation = std::max(parent1.generation, parent2.generation) + 1;
    offspring->parent_ids = {parent1.id, parent2.id};

    // Inherit mixed traits
    offspring->type = (calculateAgentFitness(parent1) > calculateAgentFitness(parent2)) ?
                     parent1.type : parent2.type;
    offspring->learning = parent1.learning; // Primary parent's learning mode

    // Blend behavioral traits
    offspring->aggression = (parent1.aggression + parent2.aggression) / 2.0f;
    offspring->curiosity = (parent1.curiosity + parent2.curiosity) / 2.0f;
    offspring->social_tendency = (parent1.social_tendency + parent2.social_tendency) / 2.0f;
    offspring->risk_tolerance = (parent1.risk_tolerance + parent2.risk_tolerance) / 2.0f;

    // Crossover neural weights
    offspring->neural_weights.resize(parent1.neural_weights.size());
    std::uniform_real_distribution<float> crossover_dist(0.0f, 1.0f);

    for (size_t i = 0; i < parent1.neural_weights.size(); ++i) {
        offspring->neural_weights[i] = (crossover_dist(rng) < crossover_probability) ?
                                     parent1.neural_weights[i] : parent2.neural_weights[i];
    }

    // Apply light mutations
    mutateAgent(*offspring);

    offspring->genetic_signature = generateGeneticSignature();
    offspring->birth_time = std::chrono::steady_clock::now();

    if (validateAgentSpawn(*offspring)) {
        reportToOverseer("AGENT_CROSSBRED", *offspring);
        return offspring;
    }

    return nullptr;
}

void AgentFactory::mutateAgent(Agent& agent) {
    std::uniform_real_distribution<float> mutation_dist(-mutation_rate, mutation_rate);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    // Mutate behavioral traits
    if (prob_dist(rng) < 0.3f) {
        agent.aggression = std::clamp(agent.aggression + mutation_dist(rng), 0.0f, 1.0f);
    }
    if (prob_dist(rng) < 0.3f) {
        agent.curiosity = std::clamp(agent.curiosity + mutation_dist(rng), 0.0f, 1.0f);
    }
    if (prob_dist(rng) < 0.3f) {
        agent.social_tendency = std::clamp(agent.social_tendency + mutation_dist(rng), 0.0f, 1.0f);
    }
    if (prob_dist(rng) < 0.3f) {
        agent.risk_tolerance = std::clamp(agent.risk_tolerance + mutation_dist(rng), 0.0f, 1.0f);
    }

    // Mutate neural weights
    for (auto& weight : agent.neural_weights) {
        if (prob_dist(rng) < mutation_rate * 2) {
            weight += mutation_dist(rng);
            weight = std::clamp(weight, -2.0f, 2.0f);
        }
    }

    // Mutate core metrics slightly
    agent.adaptability = std::clamp(agent.adaptability + mutation_dist(rng) * 0.1f, 0.0f, 1.0f);
    agent.speed = std::clamp(agent.speed + mutation_dist(rng) * 0.1f, 0.1f, 3.0f);
}

std::shared_ptr<Environment> AgentFactory::createEnvironment(const std::string& prompt,
                                                           AgentType preferred_type) {
    auto env = std::make_shared<Environment>();
    env->id = "env_" + std::to_string(environments.size() + 1);
    env->prompt = prompt;
    env->preferred_agent_type = preferred_type;

    // Determine learning mode from agent type
    switch (preferred_type) {
        case AgentType::COOPERATIVE:
            env->primary_learning_mode = LearningMode::RESOURCE_SHARING;
            env->resource_abundance = 0.7f;
            env->social_complexity = 0.8f;
            break;
        case AgentType::SOCIAL:
            env->primary_learning_mode = LearningMode::COMMUNICATION;
            env->social_complexity = 0.9f;
            env->task_difficulty = 0.6f;
            break;
        case AgentType::ADAPTIVE:
            env->primary_learning_mode = LearningMode::EXPLORATION;
            env->danger_level = 0.6f;
            env->task_difficulty = 0.7f;
            break;
        case AgentType::ANALYTICAL:
            env->primary_learning_mode = LearningMode::PROBLEM_SOLVING;
            env->task_difficulty = 0.8f;
            env->social_complexity = 0.4f;
            break;
        case AgentType::CREATIVE:
            env->primary_learning_mode = LearningMode::PLAY_BASED;
            env->resource_abundance = 0.8f;
            env->danger_level = 0.2f;
            break;
        default:
            env->primary_learning_mode = LearningMode::REINFORCEMENT;
            break;
    }

    env->created_time = std::chrono::steady_clock::now();
    environments.push_back(env);

    if (overseer) {
        overseer->reportLayerStatus("Layer5", "Environment", "CREATED");
    }

    return env;
}

void AgentFactory::updateAgent(Agent& agent, float delta_time) {
    agent.last_update = std::chrono::steady_clock::now();

    // Update experience based on interactions
    if (agent.successful_interactions > 0) {
        float learning_bonus = 0.01f * delta_time * agent.curiosity;
        agent.experience += learning_bonus;
    }

    // Update survival rating based on age and performance
    auto age = std::chrono::duration_cast<std::chrono::seconds>(
        agent.last_update - agent.birth_time).count();

    float performance_ratio = (agent.successful_interactions > 0) ?
        static_cast<float>(agent.successful_interactions) /
        (agent.successful_interactions + agent.failed_interactions) : 0.5f;

    agent.survival_rating = 0.3f * performance_ratio + 0.2f * agent.experience +
                           0.1f * std::min(1.0f, age / 3600.0f); // Factor in age up to 1 hour

    // Enforce canvas laws
    enforceCanvasLaws(agent);
}

void AgentFactory::facilitateAgentInteraction(Agent& agent1, Agent& agent2) {
    // Calculate interaction success based on compatibility and traits
    float compatibility = calculateSocialCompatibility(agent1, agent2);
    float success_probability = 0.5f + 0.3f * compatibility +
                               0.1f * (agent1.social_tendency + agent2.social_tendency);

    std::uniform_real_distribution<float> success_dist(0.0f, 1.0f);
    bool interaction_successful = success_dist(rng) < success_probability;

    if (interaction_successful) {
        agent1.successful_interactions++;
        agent2.successful_interactions++;

        // Both agents learn from successful interaction
        float learning_amount = 0.05f * (1.0f + compatibility);
        agent1.experience += learning_amount;
        agent2.experience += learning_amount;

        // Update cooperation scores
        agent1.cooperation_score += 0.02f;
        agent2.cooperation_score += 0.02f;

        reportToOverseer("AGENT_INTERACTION_SUCCESS", agent1);
    } else {
        agent1.failed_interactions++;
        agent2.failed_interactions++;

        reportToOverseer("AGENT_INTERACTION_FAILED", agent1);
    }

    // Update agent positions to simulate interaction
    float avg_x = (agent1.x + agent2.x) / 2.0f;
    float avg_y = (agent1.y + agent2.y) / 2.0f;

    agent1.x = avg_x + (agent1.x - avg_x) * 0.9f;
    agent1.y = avg_y + (agent1.y - avg_y) * 0.9f;
    agent2.x = avg_x + (agent2.x - avg_x) * 0.9f;
    agent2.y = avg_y + (agent2.y - avg_y) * 0.9f;
}

void AgentFactory::facilitateHumanAgentInteraction(Agent& agent, const std::string& human_action) {
    // Simulate human-agent interaction based on action type
    bool positive_interaction = human_action.find("help") != std::string::npos ||
                               human_action.find("teach") != std::string::npos ||
                               human_action.find("cooperate") != std::string::npos;

    if (positive_interaction) {
        agent.successful_interactions++;
        agent.experience += 0.1f; // Higher learning from humans
        agent.social_tendency += 0.01f;
        agent.cooperation_score += 0.05f;

        reportToOverseer("HUMAN_AGENT_SUCCESS", agent);
    } else {
        agent.failed_interactions++;
        agent.adaptability += 0.02f; // Learn to adapt even from negative interactions

        reportToOverseer("HUMAN_AGENT_INTERACTION", agent);
    }
}

std::vector<std::shared_ptr<Agent>> AgentFactory::selectEliteAgents(int count) {
    std::vector<std::shared_ptr<Agent>> sorted_agents = active_agents;

    std::sort(sorted_agents.begin(), sorted_agents.end(),
              [this](const std::shared_ptr<Agent>& a, const std::shared_ptr<Agent>& b) {
                  return calculateAgentFitness(*a) > calculateAgentFitness(*b);
              });

    int elite_count = std::min(count, static_cast<int>(sorted_agents.size()));
    return std::vector<std::shared_ptr<Agent>>(sorted_agents.begin(),
                                             sorted_agents.begin() + elite_count);
}

void AgentFactory::applyNaturalSelection() {
    // Remove agents below retirement threshold
    active_agents.erase(
        std::remove_if(active_agents.begin(), active_agents.end(),
                      [this](const std::shared_ptr<Agent>& agent) {
                          float fitness = calculateAgentFitness(*agent);
                          if (fitness < retirement_threshold) {
                              retired_agents.push_back(agent);
                              reportToOverseer("AGENT_RETIRED", *agent);
                              metrics.agents_retired++;
                              return true;
                          }
                          return false;
                      }),
        active_agents.end());

    // Maintain population by evolving elite agents if below threshold
    if (active_agents.size() < max_total_agents * 0.7f) {
        auto elite = selectEliteAgents(5);
        for (const auto& parent : elite) {
            if (active_agents.size() < max_total_agents) {
                auto offspring = evolveAgent(*parent);
                if (offspring) {
                    active_agents.push_back(offspring);
                }
            }
        }
    }
}

// Helper method implementations

std::string AgentFactory::generateAgentId(AgentType type) {
    std::string type_prefix;
    switch (type) {
        case AgentType::COOPERATIVE: type_prefix = "COOP"; break;
        case AgentType::SOCIAL: type_prefix = "SOC"; break;
        case AgentType::ADAPTIVE: type_prefix = "ADP"; break;
        case AgentType::ANALYTICAL: type_prefix = "ANA"; break;
        case AgentType::CREATIVE: type_prefix = "CRE"; break;
        case AgentType::HYBRID: type_prefix = "HYB"; break;
    }

    return type_prefix + "_" + std::to_string(++type_counters[type]) +
           "_" + std::to_string(++global_agent_counter);
}

std::string AgentFactory::generateGeneticSignature() {
    std::stringstream signature;
    std::uniform_int_distribution<int> hex_dist(0, 15);

    for (int i = 0; i < 16; ++i) {
        signature << std::hex << hex_dist(rng);
    }

    return signature.str();
}

void AgentFactory::initializeAgentTraits(Agent& agent, const Environment& env) {
    std::uniform_real_distribution<float> trait_dist(0.2f, 0.8f);

    // Base traits influenced by environment
    agent.aggression = trait_dist(rng) * (1.0f - env.social_complexity);
    agent.curiosity = trait_dist(rng) * (1.0f + env.task_difficulty);
    agent.social_tendency = trait_dist(rng) * env.social_complexity;
    agent.risk_tolerance = trait_dist(rng) * (1.0f - env.danger_level);

    // Core metrics
    agent.adaptability = trait_dist(rng);
    agent.cooperation_score = agent.social_tendency * 0.8f + trait_dist(rng) * 0.2f;
    agent.innovation_score = agent.curiosity * 0.7f + agent.risk_tolerance * 0.3f;
    agent.survival_rating = 0.5f;

    // Clamp all values to valid ranges
    agent.aggression = std::clamp(agent.aggression, 0.0f, 1.0f);
    agent.curiosity = std::clamp(agent.curiosity, 0.0f, 1.0f);
    agent.social_tendency = std::clamp(agent.social_tendency, 0.0f, 1.0f);
    agent.risk_tolerance = std::clamp(agent.risk_tolerance, 0.0f, 1.0f);
    agent.cooperation_score = std::clamp(agent.cooperation_score, 0.0f, 1.0f);
    agent.innovation_score = std::clamp(agent.innovation_score, 0.0f, 1.0f);
}

void AgentFactory::initializeNeuralWeights(Agent& agent) {
    std::normal_distribution<float> weight_dist(0.0f, 0.5f);

    for (auto& weight : agent.neural_weights) {
        weight = weight_dist(rng);
    }

    // Initialize memory state
    std::fill(agent.memory_state.begin(), agent.memory_state.end(), 0.0f);
}

float AgentFactory::calculateAgentFitness(const Agent& agent) const {
    float age_factor = std::chrono::duration_cast<std::chrono::minutes>(
        std::chrono::steady_clock::now() - agent.birth_time).count() / 60.0f;

    float interaction_ratio = (agent.successful_interactions > 0) ?
        static_cast<float>(agent.successful_interactions) /
        (agent.successful_interactions + agent.failed_interactions) : 0.0f;

    return 0.3f * agent.experience +
           0.25f * interaction_ratio +
           0.2f * agent.survival_rating +
           0.15f * agent.cooperation_score +
           0.1f * std::min(1.0f, age_factor);
}

float AgentFactory::calculateSocialCompatibility(const Agent& agent1, const Agent& agent2) const {
    float trait_similarity = 1.0f - (
        std::abs(agent1.aggression - agent2.aggression) +
        std::abs(agent1.curiosity - agent2.curiosity) +
        std::abs(agent1.social_tendency - agent2.social_tendency) +
        std::abs(agent1.risk_tolerance - agent2.risk_tolerance)
    ) / 4.0f;

    float type_compatibility = (agent1.type == agent2.type) ? 1.0f : 0.5f;
    float learning_compatibility = (agent1.learning == agent2.learning) ? 1.0f : 0.7f;

    return 0.5f * trait_similarity + 0.3f * type_compatibility + 0.2f * learning_compatibility;
}

bool AgentFactory::validateAgentSpawn(const Agent& proposed_agent) {
    // Check with Layer 0 Overseer
    if (!overseer) return false;

    // Validate agent parameters are within acceptable ranges
    if (proposed_agent.aggression < 0.0f || proposed_agent.aggression > 1.0f) return false;
    if (proposed_agent.curiosity < 0.0f || proposed_agent.curiosity > 1.0f) return false;
    if (proposed_agent.social_tendency < 0.0f || proposed_agent.social_tendency > 1.0f) return false;
    if (proposed_agent.risk_tolerance < 0.0f || proposed_agent.risk_tolerance > 1.0f) return false;

    // Check spatial bounds
    if (proposed_agent.x < 0.0f || proposed_agent.x > 100.0f) return false;
    if (proposed_agent.y < 0.0f || proposed_agent.y > 100.0f) return false;
    if (proposed_agent.z < 0.0f || proposed_agent.z > 100.0f) return false;

    // Validate neural network integrity
    if (proposed_agent.neural_weights.size() != 64) return false;

    return true;
}

void AgentFactory::enforceCanvasLaws(Agent& agent) {
    // Ensure agent stays within canvas bounds
    agent.x = std::clamp(agent.x, 0.0f, 100.0f);
    agent.y = std::clamp(agent.y, 0.0f, 100.0f);
    agent.z = std::clamp(agent.z, 0.0f, 100.0f);

    // Enforce trait bounds
    agent.aggression = std::clamp(agent.aggression, 0.0f, 1.0f);
    agent.curiosity = std::clamp(agent.curiosity, 0.0f, 1.0f);
    agent.social_tendency = std::clamp(agent.social_tendency, 0.0f, 1.0f);
    agent.risk_tolerance = std::clamp(agent.risk_tolerance, 0.0f, 1.0f);

    // Prevent excessive trait drift
    if (agent.experience > 10.0f) agent.experience = 10.0f;
    if (agent.cooperation_score > 1.0f) agent.cooperation_score = 1.0f;
    if (agent.innovation_score > 1.0f) agent.innovation_score = 1.0f;
    if (agent.survival_rating > 1.0f) agent.survival_rating = 1.0f;
}

void AgentFactory::reportToOverseer(const std::string& event, const Agent& agent) {
    if (overseer) {
        std::string message = event + " - Agent: " + agent.id +
                             " Type: " + std::to_string(static_cast<int>(agent.type)) +
                             " Fitness: " + std::to_string(calculateAgentFitness(agent));
        overseer->reportLayerStatus("Layer5", "AgentFactory", message);
    }
}

std::vector<std::shared_ptr<Agent>> AgentFactory::getActiveAgents() const {
    return active_agents;
}

float AgentFactory::getAverageAgentFitness() const {
    if (active_agents.empty()) return 0.0f;

    float total_fitness = 0.0f;
    for (const auto& agent : active_agents) {
        total_fitness += calculateAgentFitness(*agent);
    }

    return total_fitness / active_agents.size();
}

float AgentFactory::getPopulationDiversity() const {
    if (active_agents.size() < 2) return 0.0f;

    float total_distance = 0.0f;
    int pairs = 0;

    for (size_t i = 0; i < active_agents.size(); ++i) {
        for (size_t j = i + 1; j < active_agents.size(); ++j) {
            total_distance += calculateGeneticDistance(*active_agents[i], *active_agents[j]);
            pairs++;
        }
    }

    return (pairs > 0) ? total_distance / pairs : 0.0f;
}

float AgentFactory::calculateGeneticDistance(const Agent& agent1, const Agent& agent2) const {
    float trait_distance = std::abs(agent1.aggression - agent2.aggression) +
                          std::abs(agent1.curiosity - agent2.curiosity) +
                          std::abs(agent1.social_tendency - agent2.social_tendency) +
                          std::abs(agent1.risk_tolerance - agent2.risk_tolerance);

    // Neural weight distance (simplified)
    float weight_distance = 0.0f;
    for (size_t i = 0; i < std::min(agent1.neural_weights.size(), agent2.neural_weights.size()); ++i) {
        weight_distance += std::abs(agent1.neural_weights[i] - agent2.neural_weights[i]);
    }
    weight_distance /= agent1.neural_weights.size();

    return 0.7f * trait_distance + 0.3f * weight_distance;
}

} // namespace WorldEngine::Layer5
