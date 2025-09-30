/**
 * State Management Engine - Complex application state coordination and synchronization
 * ===================================================================================
 *
 * Features:
 * - Centralized state management with reactive updates
 * - State versioning and time-travel debugging
 * - Conflict resolution and state merging
 * - Multi-threaded state synchronization
 * - State persistence and serialization
 * - Event-sourced state reconstruction
 * - Distributed state synchronization
 */

#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <condition_variable>
#include <thread>
#include <any>

namespace StateManagement {

using StateId = std::string;
using Version = uint64_t;
using TimePoint = std::chrono::system_clock::time_point;

enum class StateChangeType {
    CREATE,
    UPDATE,
    DELETE,
    MERGE,
    ROLLBACK
};

enum class ConflictResolution {
    LAST_WRITE_WINS,
    FIRST_WRITE_WINS,
    MERGE_STRATEGIES,
    CUSTOM_RESOLVER,
    REJECT_CHANGE
};

struct StateValue {
    std::any value;
    std::string type_name;
    Version version;
    TimePoint timestamp;
    std::string source;
    std::unordered_map<std::string, std::string> metadata;

    template<typename T>
    T get() const {
        return std::any_cast<T>(value);
    }

    template<typename T>
    bool is_type() const {
        return value.type() == typeid(T);
    }
};

struct StateChange {
    StateId state_id;
    StateChangeType change_type;
    StateValue previous_value;
    StateValue new_value;
    Version change_version;
    TimePoint change_time;
    std::string actor; // Who made the change
    std::string reason; // Why the change was made

    // For conflict resolution
    std::vector<StateChange> conflicting_changes;
    ConflictResolution resolution_strategy;
};

struct StateSnapshot {
    Version snapshot_version;
    TimePoint snapshot_time;
    std::unordered_map<StateId, StateValue> states;
    std::string snapshot_name;
    std::string description;
};

class StateValidator {
public:
    virtual ~StateValidator() = default;
    virtual bool validate(const StateValue& value) const = 0;
    virtual std::string get_validation_error(const StateValue& value) const = 0;
};

template<typename T>
class TypedStateValidator : public StateValidator {
public:
    using ValidationFunction = std::function<bool(const T&)>;
    using ErrorFunction = std::function<std::string(const T&)>;

    TypedStateValidator(ValidationFunction validator, ErrorFunction error_func)
        : validator_(validator), error_func_(error_func) {}

    bool validate(const StateValue& value) const override {
        if (!value.is_type<T>()) {
            return false;
        }
        return validator_(value.get<T>());
    }

    std::string get_validation_error(const StateValue& value) const override {
        if (!value.is_type<T>()) {
            return "Type mismatch";
        }
        return error_func_(value.get<T>());
    }

private:
    ValidationFunction validator_;
    ErrorFunction error_func_;
};

class StateStore {
public:
    StateStore();
    ~StateStore();

    // State management
    template<typename T>
    bool set_state(const StateId& id, const T& value, const std::string& source = "");

    template<typename T>
    std::optional<T> get_state(const StateId& id) const;

    bool has_state(const StateId& id) const;
    bool remove_state(const StateId& id, const std::string& source = "");

    // Versioned access
    template<typename T>
    std::optional<T> get_state_at_version(const StateId& id, Version version) const;

    std::vector<Version> get_state_versions(const StateId& id) const;
    StateValue get_state_value(const StateId& id) const;

    // Batch operations
    template<typename T>
    void set_states(const std::unordered_map<StateId, T>& states, const std::string& source = "");

    std::unordered_map<StateId, StateValue> get_all_states() const;
    std::vector<StateId> get_state_ids() const;

    // State validation
    void add_validator(const StateId& id, std::unique_ptr<StateValidator> validator);
    void remove_validator(const StateId& id);
    bool is_valid_state(const StateId& id, const StateValue& value) const;

    // Change tracking
    std::vector<StateChange> get_changes_since(Version version) const;
    std::vector<StateChange> get_changes_between(Version from, Version to) const;
    std::vector<StateChange> get_state_history(const StateId& id) const;

    // Conflict resolution
    void set_conflict_resolution(ConflictResolution strategy);
    void set_custom_resolver(std::function<StateValue(const std::vector<StateChange>&)> resolver);

    // Snapshots
    Version create_snapshot(const std::string& name = "", const std::string& description = "");
    bool restore_snapshot(Version snapshot_version);
    std::vector<StateSnapshot> get_snapshots() const;
    bool remove_snapshot(Version snapshot_version);

    // Callbacks
    using StateChangeCallback = std::function<void(const StateChange&)>;
    using StateValidationCallback = std::function<void(const StateId&, const std::string&)>;

    void on_state_changed(StateChangeCallback callback);
    void on_validation_failed(StateValidationCallback callback);
    void remove_change_callback(StateChangeCallback callback);

    // Serialization
    std::string serialize_state(const StateId& id) const;
    std::string serialize_all_states() const;
    bool deserialize_state(const StateId& id, const std::string& serialized_data);
    bool deserialize_all_states(const std::string& serialized_data);

    // Statistics
    struct StateStatistics {
        size_t total_states;
        size_t total_versions;
        Version latest_version;
        TimePoint oldest_change;
        TimePoint latest_change;
        std::unordered_map<StateId, size_t> state_version_counts;
        std::unordered_map<std::string, size_t> changes_by_source;
    };

    StateStatistics get_statistics() const;

private:
    struct StateEntry {
        std::vector<StateValue> versions;
        std::unique_ptr<StateValidator> validator;
        mutable std::shared_mutex access_mutex;
    };

    mutable std::shared_mutex store_mutex_;
    std::unordered_map<StateId, std::unique_ptr<StateEntry>> states_;
    std::vector<StateChange> change_history_;
    std::vector<StateSnapshot> snapshots_;

    std::atomic<Version> current_version_{0};
    ConflictResolution conflict_resolution_{ConflictResolution::LAST_WRITE_WINS};
    std::function<StateValue(const std::vector<StateChange>&)> custom_resolver_;

    std::vector<StateChangeCallback> change_callbacks_;
    std::vector<StateValidationCallback> validation_callbacks_;

    // Helper methods
    Version next_version();
    StateChange create_change_record(const StateId& id, StateChangeType type,
                                   const StateValue& old_val, const StateValue& new_val,
                                   const std::string& source);
    void notify_state_changed(const StateChange& change);
    void notify_validation_failed(const StateId& id, const std::string& error);

    StateValue resolve_conflict(const std::vector<StateChange>& conflicts);
    bool apply_change(const StateChange& change);
};

class ReactiveStateManager {
public:
    ReactiveStateManager();
    ~ReactiveStateManager();

    // Reactive subscriptions
    template<typename T>
    using StateSubscription = std::function<void(const StateId&, const T&, const T&)>;

    template<typename T>
    void subscribe(const StateId& id, StateSubscription<T> callback);

    void unsubscribe(const StateId& id);
    void unsubscribe_all();

    // Computed states
    template<typename T, typename... Dependencies>
    void create_computed_state(const StateId& id,
                              std::function<T(const Dependencies&...)> computer,
                              const std::vector<StateId>& dependency_ids);

    // State derivation
    template<typename T, typename U>
    void derive_state(const StateId& derived_id, const StateId& source_id,
                     std::function<U(const T&)> deriver);

    // Batch updates (avoid cascading updates)
    void begin_batch_update();
    void end_batch_update();
    bool is_in_batch_update() const;

    // Access to underlying store
    StateStore& get_store() { return store_; }
    const StateStore& get_store() const { return store_; }

private:
    StateStore store_;

    struct Subscription {
        std::string type_name;
        std::function<void(const StateChange&)> callback;
    };

    std::unordered_map<StateId, std::vector<Subscription>> subscriptions_;
    std::unordered_map<StateId, std::vector<StateId>> dependencies_;
    std::unordered_map<StateId, std::vector<StateId>> dependents_;

    std::atomic<bool> in_batch_update_{false};
    std::vector<StateChange> batch_changes_;

    mutable std::shared_mutex subscriptions_mutex_;

    void handle_state_change(const StateChange& change);
    void update_computed_states(const StateId& changed_id);
    void propagate_change(const StateChange& change);
};

class DistributedStateSync {
public:
    struct Node {
        std::string node_id;
        std::string address;
        TimePoint last_seen;
        Version last_synced_version;
        bool is_active;
    };

    DistributedStateSync(const std::string& node_id);
    ~DistributedStateSync();

    // Node management
    void add_node(const Node& node);
    void remove_node(const std::string& node_id);
    std::vector<Node> get_active_nodes() const;

    // Synchronization
    void start_sync(ReactiveStateManager& state_manager);
    void stop_sync();
    void force_sync();

    // Conflict resolution for distributed scenarios
    enum class DistributedConflictResolution {
        VECTOR_CLOCK,
        LAMPORT_TIMESTAMP,
        CUSTOM_PRIORITY
    };

    void set_distributed_conflict_resolution(DistributedConflictResolution strategy);

    // Network callbacks
    using MessageSender = std::function<void(const std::string& node_id, const std::string& message)>;
    using MessageReceiver = std::function<void(const std::string& from_node, const std::string& message)>;

    void set_message_sender(MessageSender sender);
    void set_message_receiver(MessageReceiver receiver);

private:
    std::string node_id_;
    std::vector<Node> nodes_;
    ReactiveStateManager* state_manager_;

    std::atomic<bool> sync_active_{false};
    std::thread sync_thread_;

    MessageSender message_sender_;
    MessageReceiver message_receiver_;

    mutable std::mutex nodes_mutex_;

    void sync_loop();
    void handle_sync_message(const std::string& from_node, const std::string& message);
    void broadcast_state_changes(const std::vector<StateChange>& changes);
};

// Global state manager instance
class GlobalStateManager {
public:
    static GlobalStateManager& instance();

    ReactiveStateManager& get_manager() { return manager_; }
    DistributedStateSync& get_sync() { return sync_; }

    // Convenience methods
    template<typename T>
    bool set_global_state(const StateId& id, const T& value) {
        return manager_.get_store().set_state(id, value, "global");
    }

    template<typename T>
    std::optional<T> get_global_state(const StateId& id) {
        return manager_.get_store().get_state<T>(id);
    }

    template<typename T>
    void subscribe_global_state(const StateId& id,
                               typename ReactiveStateManager::StateSubscription<T> callback) {
        manager_.subscribe(id, callback);
    }

private:
    GlobalStateManager();
    ~GlobalStateManager();

    ReactiveStateManager manager_;
    DistributedStateSync sync_;
};

// Template implementations
template<typename T>
bool StateStore::set_state(const StateId& id, const T& value, const std::string& source) {
    std::unique_lock<std::shared_mutex> lock(store_mutex_);

    StateValue new_state_value;
    new_state_value.value = std::any(value);
    new_state_value.type_name = typeid(T).name();
    new_state_value.version = next_version();
    new_state_value.timestamp = std::chrono::system_clock::now();
    new_state_value.source = source;

    // Validate if validator exists
    if (states_.find(id) != states_.end() && states_[id]->validator) {
        if (!states_[id]->validator->validate(new_state_value)) {
            notify_validation_failed(id, states_[id]->validator->get_validation_error(new_state_value));
            return false;
        }
    }

    StateValue old_value;
    StateChangeType change_type = StateChangeType::CREATE;

    if (states_.find(id) != states_.end()) {
        change_type = StateChangeType::UPDATE;
        if (!states_[id]->versions.empty()) {
            old_value = states_[id]->versions.back();
        }
    } else {
        states_[id] = std::make_unique<StateEntry>();
    }

    states_[id]->versions.push_back(new_state_value);

    StateChange change = create_change_record(id, change_type, old_value, new_state_value, source);
    change_history_.push_back(change);

    notify_state_changed(change);

    return true;
}

template<typename T>
std::optional<T> StateStore::get_state(const StateId& id) const {
    std::shared_lock<std::shared_mutex> lock(store_mutex_);

    auto it = states_.find(id);
    if (it == states_.end() || it->second->versions.empty()) {
        return std::nullopt;
    }

    const StateValue& latest = it->second->versions.back();
    if (!latest.is_type<T>()) {
        return std::nullopt;
    }

    return latest.get<T>();
}

// Convenience macros
#define SET_GLOBAL_STATE(id, value) \
    StateManagement::GlobalStateManager::instance().set_global_state(id, value)

#define GET_GLOBAL_STATE(id, type) \
    StateManagement::GlobalStateManager::instance().get_global_state<type>(id)

#define SUBSCRIBE_GLOBAL_STATE(id, type, callback) \
    StateManagement::GlobalStateManager::instance().subscribe_global_state<type>(id, callback)

} // namespace StateManagement
