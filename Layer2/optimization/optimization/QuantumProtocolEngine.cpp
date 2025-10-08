/**
 * Quantum Protocol Engine Implementation
 * =====================================
 */

#include "QuantumProtocolEngine.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <random>

namespace QuantumProtocol {

// ============================================================================
// QuantumRayFieldManager Implementation
// ============================================================================

QuantumRayFieldManager& QuantumRayFieldManager::instance() {
    static QuantumRayFieldManager instance;
    return instance;
}

QuantumRayFieldManager::QuantumRayFieldManager() {
    // Initialize with default settings
}

void QuantumRayFieldManager::register_agent(const std::string& id, const Vector3& initial_position) {
    std::lock_guard<std::mutex> lock(agents_mutex_);

    QuantumAgent agent;
    agent.id = id;
    agent.position = initial_position;
    agent.velocity = Vector3(0, 0, 0);
    agent.energy_level = 1.0f;
    agent.coherence = 1.0f;
    agent.step_count = 0;
    agent.max_steps = 1000;
    agent.is_active = true;
    agent.is_collapsed = false;
    agent.last_update = std::chrono::system_clock::now();

    agents_[id] = std::move(agent);
    std::cout << "[QuantumRayFieldManager] Registered agent: " << id << std::endl;
}

void QuantumRayFieldManager::remove_agent(const std::string& id) {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    agents_.erase(id);
}

QuantumAgent* QuantumRayFieldManager::get_agent(const std::string& id) {
    std::lock_guard<std::mutex> lock(agents_mutex_);
    auto it = agents_.find(id);
    return it != agents_.end() ? &it->second : nullptr;
}

void QuantumRayFieldManager::register_step(const Vector3& position, const std::string& agent_id,
                                          int step_number, float energy_level) {
    std::lock_guard<std::mutex> steps_lock(steps_mutex_);
    std::lock_guard<std::mutex> agents_lock(agents_mutex_);

    QuantumStep step;
    step.position = position;
    step.agent_id = agent_id;
    step.step_number = step_number;
    step.timestamp = std::chrono::system_clock::now();
    step.energy_level = energy_level;
    step.coherence = 1.0f;
    step.entanglement_strength = 0.5f;
    step.is_collapsed = false;

    quantum_steps_.push_back(step);

    // Update agent if it exists
    auto it = agents_.find(agent_id);
    if (it != agents_.end()) {
        it->second.position = position;
        it->second.step_count = step_number;
        it->second.energy_level = energy_level;
        it->second.path_history.push_back(step);
        it->second.last_update = step.timestamp;
    }
}

std::vector<QuantumStep> QuantumRayFieldManager::get_agent_path(const std::string& agent_id) const {
    std::lock_guard<std::mutex> lock(steps_mutex_);
    std::vector<QuantumStep> path;

    for (const auto& step : quantum_steps_) {
        if (step.agent_id == agent_id) {
            path.push_back(step);
        }
    }

    return path;
}

std::vector<QuantumStep> QuantumRayFieldManager::get_all_steps() const {
    std::lock_guard<std::mutex> lock(steps_mutex_);
    return quantum_steps_;
}

void QuantumRayFieldManager::update(float delta_time) {
    check_collapse_conditions();
    apply_quantum_evolution(delta_time);
}

void QuantumRayFieldManager::set_function_type(MathFunctionType function) {
    if (current_function_ != function) {
        current_function_ = function;

        // Notify callbacks
        for (auto& callback : function_callbacks_) {
            callback(function);
        }

        std::cout << "[QuantumRayFieldManager] Function changed to: " << static_cast<int>(function) << std::endl;
    }
}

void QuantumRayFieldManager::on_agent_collapse(CollapseCallback callback) {
    collapse_callbacks_.push_back(callback);
}

void QuantumRayFieldManager::on_function_change(FunctionChangeCallback callback) {
    function_callbacks_.push_back(callback);
}

void QuantumRayFieldManager::check_collapse_conditions() {
    std::lock_guard<std::mutex> lock(agents_mutex_);

    for (auto& [id, agent] : agents_) {
        if (!agent.is_active || agent.is_collapsed) continue;

        // Check step limit
        if (agent.step_count >= agent.max_steps) {
            agent.is_collapsed = true;
            agent.is_active = false;

            for (auto& callback : collapse_callbacks_) {
                callback(id);
            }
        }

        // Check energy depletion
        if (agent.energy_level <= 0.0f) {
            agent.is_collapsed = true;
            agent.is_active = false;

            for (auto& callback : collapse_callbacks_) {
                callback(id);
            }
        }
    }
}

void QuantumRayFieldManager::apply_quantum_evolution(float delta_time) {
    std::lock_guard<std::mutex> lock(agents_mutex_);

    for (auto& [id, agent] : agents_) {
        if (!agent.is_active) continue;

        // Apply quantum decay
        agent.coherence *= std::exp(-delta_time * 0.1f);
        agent.energy_level -= delta_time * 0.01f;

        // Clamp values
        agent.coherence = std::max(0.0f, std::min(1.0f, agent.coherence));
        agent.energy_level = std::max(0.0f, agent.energy_level);
    }
}

// ============================================================================
// EnvironmentalEventSystem Implementation
// ============================================================================

EnvironmentalEventSystem::EnvironmentalEventSystem() {
    setup_default_effects();
}

void EnvironmentalEventSystem::spawn_event(EnvironmentalEventType type, const Vector3& origin,
                                         float radius, float duration) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    EnvironmentalEvent event;
    event.type = type;
    event.origin = origin;
    event.radius = radius;
    event.duration = duration;
    event.time_elapsed = 0.0f;
    event.intensity = base_intensity_;

    // Set up effect function based on type
    switch (type) {
        case EnvironmentalEventType::STORM:
            event.apply_effect = [this](QuantumAgent& agent) { apply_storm_effect(agent); };
            break;
        case EnvironmentalEventType::FLUX_SURGE:
            event.apply_effect = [this](QuantumAgent& agent) { apply_flux_surge_effect(agent); };
            break;
        case EnvironmentalEventType::MEMORY_ECHO:
            event.apply_effect = [this](QuantumAgent& agent) { apply_memory_echo_effect(agent); };
            break;
        default:
            event.apply_effect = [](QuantumAgent&) { /* No effect */ };
    }

    active_events_.push_back(std::move(event));
    std::cout << "[EnvironmentalEventSystem] Spawned event type " << static_cast<int>(type)
              << " at (" << origin.x << "," << origin.y << "," << origin.z << ")" << std::endl;
}

void EnvironmentalEventSystem::update_events(float delta_time) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    // Update and filter expired events
    active_events_.erase(
        std::remove_if(active_events_.begin(), active_events_.end(),
            [delta_time](EnvironmentalEvent& event) {
                event.time_elapsed += delta_time;
                return event.time_elapsed >= event.duration;
            }),
        active_events_.end()
    );
}

void EnvironmentalEventSystem::apply_events_to_agents(std::vector<QuantumAgent*>& agents) {
    std::lock_guard<std::mutex> lock(events_mutex_);

    for (auto& event : active_events_) {
        for (auto* agent : agents) {
            if (agent && event.affects(agent->position)) {
                event.apply_effect(*agent);
            }
        }
    }
}

void EnvironmentalEventSystem::setup_default_effects() {
    // Default effects are set up in spawn_event method
}

void EnvironmentalEventSystem::apply_storm_effect(QuantumAgent& agent) {
    agent.energy_level *= 0.95f;
    agent.coherence *= 0.9f;
}

void EnvironmentalEventSystem::apply_flux_surge_effect(QuantumAgent& agent) {
    agent.energy_level += 1.0f;
    agent.quantum_state["mutated"] = 1.0f;
}

void EnvironmentalEventSystem::apply_memory_echo_effect(QuantumAgent& agent) {
    agent.quantum_state["memory_awakened"] = 1.0f;
    agent.coherence += 0.1f;
}

// ============================================================================
// GlyphAmplitudeResolver Implementation
// ============================================================================

GlyphAmplitudeResolver::GlyphAmplitudeResolver() {
    scoring_function_ = [this](const QuantumAgent& agent) {
        return default_scoring_function(agent);
    };
}

GlyphAmplitudeResolver::AmplitudeResult GlyphAmplitudeResolver::resolve_and_collapse() {
    auto& manager = QuantumRayFieldManager::instance();
    AmplitudeResult result;
    result.resolution_time = std::chrono::system_clock::now();
    result.max_amplitude = std::numeric_limits<float>::lowest();

    // Get all agents and calculate scores
    auto all_steps = manager.get_all_steps();
    std::unordered_map<std::string, std::vector<QuantumStep>> agent_paths;

    for (const auto& step : all_steps) {
        agent_paths[step.agent_id].push_back(step);
    }

    for (const auto& [agent_id, path] : agent_paths) {
        auto* agent = manager.get_agent(agent_id);
        if (agent) {
            float score = scoring_function_(*agent);
            result.all_scores[agent_id] = score;

            if (score > result.max_amplitude) {
                result.max_amplitude = score;
                result.winner_id = agent_id;
            }
        }
    }

    if (!result.winner_id.empty()) {
        std::cout << "[GlyphAmplitudeResolver] Winner: " << result.winner_id
                  << " with amplitude: " << result.max_amplitude << std::endl;
        emit_collapse_signal(result);
    }

    return result;
}

void GlyphAmplitudeResolver::emit_collapse_signal(const AmplitudeResult& result) {
    for (auto& callback : signal_callbacks_) {
        callback(result);
    }
}

void GlyphAmplitudeResolver::on_collapse_signal(CollapseSignalCallback callback) {
    signal_callbacks_.push_back(callback);
}

float GlyphAmplitudeResolver::default_scoring_function(const QuantumAgent& agent) {
    // Heuristic: longer path with more steps and higher energy = higher amplitude
    float base_score = static_cast<float>(agent.step_count);
    float energy_bonus = agent.energy_level * 10.0f;
    float coherence_bonus = agent.coherence * 5.0f;

    return base_score + energy_bonus + coherence_bonus;
}

// ============================================================================
// RecursiveInfrastructureFlow Implementation
// ============================================================================

RecursiveInfrastructureFlow::RecursiveInfrastructureFlow() {
    // Initialize with default settings
}

RecursiveInfrastructureNode RecursiveInfrastructureFlow::analyze_topic(const std::string& topic, int depth) {
    RecursiveInfrastructureNode node;
    node.topic = topic;
    node.timestamp = std::chrono::system_clock::now();
    node.iteration_depth = depth;

    // Generate infrastructure aspects
    node.visible_infrastructure = "Tangible structures supporting " + topic;
    node.unseen_infrastructure = "Intangible frameworks supporting " + topic;
    node.solid_state = "Fixed and rigid forms of " + topic;
    node.liquid_state = "Adaptive and evolving forms of " + topic;
    node.gas_state = "Dispersed and pervasive forms of " + topic;
    node.derived_topic = "The Evolution of " + topic + " in the Next Cycle";

    // Record in memory
    std::lock_guard<std::mutex> lock(memory_mutex_);
    memory_.forward_memory.push_back(node);
    memory_.reverse_memory.insert(memory_.reverse_memory.begin(), node);

    return node;
}

std::vector<RecursiveInfrastructureNode> RecursiveInfrastructureFlow::recursive_analysis(
    const std::string& starting_topic, int iterations) {

    std::vector<RecursiveInfrastructureNode> results;
    std::string current_topic = starting_topic;

    for (int i = 0; i < iterations; ++i) {
        auto node = analyze_topic(current_topic, i);
        results.push_back(node);
        record_imprint(node);
        current_topic = node.derived_topic;
    }

    return results;
}

void RecursiveInfrastructureFlow::record_imprint(const RecursiveInfrastructureNode& node) {
    imprint_registry_[node.topic] = node;
}

RecursiveInfrastructureNode* RecursiveInfrastructureFlow::get_imprint(const std::string& topic) {
    auto it = imprint_registry_.find(topic);
    return it != imprint_registry_.end() ? &it->second : nullptr;
}

void RecursiveInfrastructureFlow::assign_symbol(const std::string& topic, const std::string& symbol) {
    symbol_map_[topic] = symbol;
}

std::string RecursiveInfrastructureFlow::get_symbol(const std::string& topic) {
    auto it = symbol_map_.find(topic);
    return it != symbol_map_.end() ? it->second : "";
}

// ============================================================================
// SwarmMind Implementation
// ============================================================================

SwarmMind::SwarmMind() {
    // Initialize with default settings
}

SwarmMind::SwarmAnalysis SwarmMind::analyze_nexus_memory(
    const std::vector<RecursiveInfrastructureNode>& memory) {

    SwarmAnalysis analysis;
    analysis.total_nodes = memory.size();
    analysis.latest_timestamp = std::chrono::system_clock::time_point::min();

    std::set<std::string> unique_topics_set;

    for (const auto& node : memory) {
        unique_topics_set.insert(node.topic);
        analysis.topic_frequency[node.topic]++;

        if (node.timestamp > analysis.latest_timestamp) {
            analysis.latest_timestamp = node.timestamp;
        }
    }

    analysis.unique_topics.assign(unique_topics_set.begin(), unique_topics_set.end());
    analysis.convergence_metric = calculate_convergence_metric(analysis);
    analysis.dominant_theme = determine_dominant_theme(analysis.topic_frequency);

    analysis_history_.push_back(analysis);
    last_analysis_ = analysis;

    return analysis;
}

SwarmMind::SwarmAnalysis SwarmMind::analyze_combined_system() {
    // Combine analysis from all integrated systems
    auto& quantum_manager = QuantumRayFieldManager::instance();
    auto all_steps = quantum_manager.get_all_steps();

    SwarmAnalysis analysis;
    analysis.total_nodes = all_steps.size();

    std::set<std::string> agents_set;
    for (const auto& step : all_steps) {
        agents_set.insert(step.agent_id);
    }

    analysis.unique_topics.assign(agents_set.begin(), agents_set.end());
    analysis.convergence_metric = calculate_convergence_metric(analysis);

    return analysis;
}

bool SwarmMind::detect_convergence(float threshold) {
    return last_analysis_.convergence_metric >= threshold;
}

float SwarmMind::calculate_convergence_metric(const SwarmAnalysis& analysis) {
    if (analysis.topic_frequency.empty()) return 0.0f;

    // Calculate distribution entropy as convergence metric
    float total_count = static_cast<float>(analysis.total_nodes);
    float entropy = 0.0f;

    for (const auto& [topic, count] : analysis.topic_frequency) {
        float probability = count / total_count;
        if (probability > 0) {
            entropy -= probability * std::log2(probability);
        }
    }

    // Normalize entropy to 0-1 range (inverted so higher means more convergent)
    float max_entropy = std::log2(static_cast<float>(analysis.topic_frequency.size()));
    return max_entropy > 0 ? (max_entropy - entropy) / max_entropy : 1.0f;
}

std::string SwarmMind::determine_dominant_theme(const std::unordered_map<std::string, int>& frequencies) {
    if (frequencies.empty()) return "None";

    auto max_element = std::max_element(frequencies.begin(), frequencies.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    return max_element->first;
}

// ============================================================================
// QuantumProtocolCore Implementation
// ============================================================================

QuantumProtocolCore& QuantumProtocolCore::instance() {
    static QuantumProtocolCore instance;
    return instance;
}

QuantumProtocolCore::QuantumProtocolCore() {
    // Set up inter-system callbacks
    ray_field_manager_.on_agent_collapse([this](const std::string& agent_id) {
        on_agent_collapse(agent_id);
    });

    ray_field_manager_.on_function_change([this](MathFunctionType function) {
        on_function_changed(function);
    });

    amplitude_resolver_.on_collapse_signal([this](const GlyphAmplitudeResolver::AmplitudeResult& result) {
        std::unordered_map<std::string, std::string> data;
        data["winner_id"] = result.winner_id;
        data["amplitude"] = std::to_string(result.max_amplitude);
        dispatch_event(QuantumEventType::AGENT_COLLAPSE, data);
    });
}

QuantumProtocolCore::~QuantumProtocolCore() {
    stop();
}

void QuantumProtocolCore::start() {
    if (!running_) {
        running_ = true;
        update_thread_ = std::thread(&QuantumProtocolCore::update_loop, this);
        std::cout << "[QuantumProtocolCore] Started" << std::endl;
    }
}

void QuantumProtocolCore::stop() {
    if (running_) {
        running_ = false;
        if (update_thread_.joinable()) {
            update_thread_.join();
        }
        std::cout << "[QuantumProtocolCore] Stopped" << std::endl;
    }
}

void QuantumProtocolCore::update(float delta_time) {
    ray_field_manager_.update(delta_time);
    environmental_system_.update_events(delta_time);

    // Apply environmental effects to quantum agents
    std::vector<QuantumAgent*> agents;
    // Note: This would need access to all agents - implementation detail
    environmental_system_.apply_events_to_agents(agents);
}

void QuantumProtocolCore::update_loop() {
    auto last_time = std::chrono::high_resolution_clock::now();
    const auto frame_duration = std::chrono::nanoseconds(static_cast<int64_t>(1e9 / update_frequency_));

    while (running_) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto delta_time = std::chrono::duration<float>(current_time - last_time).count();
        last_time = current_time;

        update(delta_time);

        // Sleep to maintain target frequency
        auto sleep_time = frame_duration - (std::chrono::high_resolution_clock::now() - current_time);
        if (sleep_time > std::chrono::nanoseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        }
    }
}

void QuantumProtocolCore::on_agent_collapse(const std::string& agent_id) {
    std::cout << "[QuantumProtocol] COLLAPSE EVENT for agent: " << agent_id << std::endl;

    std::unordered_map<std::string, std::string> data;
    data["agent_id"] = agent_id;
    dispatch_event(QuantumEventType::AGENT_COLLAPSE, data);

    // Trigger memory ghost spawn (placeholder)
    // Trigger quantum UI score glyph (placeholder)
    // Trigger quantum visuals burst (placeholder)
    // Archive in quantum lore (placeholder)
}

void QuantumProtocolCore::on_collapse_all() {
    std::cout << "[QuantumProtocol] GLOBAL COLLAPSE triggered." << std::endl;

    dispatch_event(QuantumEventType::SWARM_CONVERGENCE, {});

    // Trigger global replay system
    // Trigger world fade visuals
    // Trigger echo field audio
}

void QuantumProtocolCore::on_function_changed(MathFunctionType new_function) {
    std::cout << "[QuantumProtocol] Function shift to: " << static_cast<int>(new_function) << std::endl;

    std::unordered_map<std::string, std::string> data;
    data["function_type"] = std::to_string(static_cast<int>(new_function));
    dispatch_event(QuantumEventType::FUNCTION_CHANGE, data);

    // Set global shader parameters (placeholder)
    // Trigger function tone audio (placeholder)
    // Update UI function display (placeholder)
    // Sync trail palette visuals (placeholder)
}

void QuantumProtocolCore::on_agent_complete(const std::string& agent_id) {
    std::cout << "[QuantumProtocol] Agent " << agent_id << " completed its journey." << std::endl;
    on_agent_collapse(agent_id);
}

void QuantumProtocolCore::activate_nexus(const std::string& seed_topic) {
    std::cout << "[QuantumProtocol] Activating Nexus with seed: " << seed_topic << std::endl;

    // Process recursive analysis
    recursive_flow_.recursive_analysis(seed_topic, 5);

    // Analyze with swarm mind
    auto memory = recursive_flow_.get_memory();
    swarm_mind_.analyze_nexus_memory(memory.forward_memory);

    dispatch_event(QuantumEventType::SWARM_CONVERGENCE, {{"seed_topic", seed_topic}});
}

SwarmMind::SwarmAnalysis QuantumProtocolCore::get_nexus_analysis() {
    return swarm_mind_.analyze_combined_system();
}

void QuantumProtocolCore::dispatch_event(QuantumEventType type,
                                       const std::unordered_map<std::string, std::string>& data) {
    notify_event_callbacks(type, data);
}

void QuantumProtocolCore::on_protocol_event(ProtocolEventCallback callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    event_callbacks_.push_back(callback);
}

void QuantumProtocolCore::notify_event_callbacks(QuantumEventType type,
                                                const std::unordered_map<std::string, std::string>& data) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    for (auto& callback : event_callbacks_) {
        callback(type, data);
    }
}

} // namespace QuantumProtocol
