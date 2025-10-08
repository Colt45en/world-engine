#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <sstream>
#include <memory>
#include <mutex>
#include <functional>
#include <typeinfo>
#include <any>
#include <chrono>
#include "NexusLogger.hpp"

namespace NEXUS {

// Configuration value types
enum class ConfigType {
    BOOL,
    INTEGER,
    FLOAT,
    STRING,
    VECTOR3,
    COLOR,
    ARRAY
};

struct ConfigValue {
    std::any value;
    ConfigType type;
    std::string description;
    std::string category;
    bool is_readonly;
    std::function<bool(const std::any&)> validator;

    ConfigValue() : type(ConfigType::STRING), is_readonly(false) {}

    template<typename T>
    ConfigValue(const T& val, ConfigType t, const std::string& desc = "",
                const std::string& cat = "General", bool readonly = false)
        : value(val), type(t), description(desc), category(cat), is_readonly(readonly) {}

    template<typename T>
    T get() const {
        try {
            return std::any_cast<T>(value);
        } catch (const std::bad_any_cast& e) {
            NEXUS_LOG_ERROR("CONFIG", "Bad cast for config value: " + std::string(e.what()));
            return T{};
        }
    }

    template<typename T>
    bool set(const T& new_value) {
        if (is_readonly) {
            NEXUS_LOG_WARN("CONFIG", "Attempted to modify readonly config value");
            return false;
        }

        if (validator && !validator(std::any(new_value))) {
            NEXUS_LOG_WARN("CONFIG", "Config value failed validation");
            return false;
        }

        value = new_value;
        return true;
    }
};

struct Vec3Config {
    float x, y, z;
    Vec3Config(float x = 0.0f, float y = 0.0f, float z = 0.0f) : x(x), y(y), z(z) {}

    std::string toString() const {
        std::ostringstream oss;
        oss << x << "," << y << "," << z;
        return oss.str();
    }

    static Vec3Config fromString(const std::string& str) {
        std::istringstream iss(str);
        std::string token;
        std::vector<float> values;

        while (std::getline(iss, token, ',') && values.size() < 3) {
            try {
                values.push_back(std::stof(token));
            } catch (...) {
                values.push_back(0.0f);
            }
        }

        while (values.size() < 3) values.push_back(0.0f);
        return Vec3Config(values[0], values[1], values[2]);
    }
};

struct ColorConfig {
    float r, g, b, a;
    ColorConfig(float r = 1.0f, float g = 1.0f, float b = 1.0f, float a = 1.0f)
        : r(r), g(g), b(b), a(a) {}

    std::string toString() const {
        std::ostringstream oss;
        oss << r << "," << g << "," << b << "," << a;
        return oss.str();
    }

    static ColorConfig fromString(const std::string& str) {
        std::istringstream iss(str);
        std::string token;
        std::vector<float> values;

        while (std::getline(iss, token, ',') && values.size() < 4) {
            try {
                values.push_back(std::stof(token));
            } catch (...) {
                values.push_back(1.0f);
            }
        }

        while (values.size() < 4) values.push_back(1.0f);
        return ColorConfig(values[0], values[1], values[2], values[3]);
    }
};

class ConfigPreset {
private:
    std::string name;
    std::string description;
    std::unordered_map<std::string, ConfigValue> values;

public:
    ConfigPreset(const std::string& preset_name, const std::string& desc = "")
        : name(preset_name), description(desc) {}

    void addValue(const std::string& key, const ConfigValue& value) {
        values[key] = value;
    }

    bool hasValue(const std::string& key) const {
        return values.find(key) != values.end();
    }

    const ConfigValue& getValue(const std::string& key) const {
        static ConfigValue empty;
        auto it = values.find(key);
        return (it != values.end()) ? it->second : empty;
    }

    const std::string& getName() const { return name; }
    const std::string& getDescription() const { return description; }

    std::vector<std::string> getKeys() const {
        std::vector<std::string> keys;
        for (const auto& pair : values) {
            keys.push_back(pair.first);
        }
        return keys;
    }
};

class NexusConfig {
private:
    std::unordered_map<std::string, ConfigValue> config_values;
    std::unordered_map<std::string, ConfigPreset> presets;
    std::unordered_map<std::string, std::function<void(const std::string&, const ConfigValue&)>> change_callbacks;
    std::string config_file_path;
    std::string active_preset;
    mutable std::mutex config_mutex;
    static std::unique_ptr<NexusConfig> instance;

public:
    static NexusConfig& getInstance() {
        static std::once_flag once_flag;
        std::call_once(once_flag, []() {
            instance = std::unique_ptr<NexusConfig>(new NexusConfig());
        });
        return *instance;
    }

    // Configuration value management
    template<typename T>
    void set(const std::string& key, const T& value, ConfigType type,
             const std::string& description = "", const std::string& category = "General",
             bool readonly = false) {
        std::lock_guard<std::mutex> lock(config_mutex);

        ConfigValue config_val(value, type, description, category, readonly);
        config_values[key] = config_val;

        // Notify callbacks
        auto callback_it = change_callbacks.find(key);
        if (callback_it != change_callbacks.end()) {
            callback_it->second(key, config_val);
        }

        NEXUS_LOG_DEBUG("CONFIG", "Set config value: " + key);
    }

    template<typename T>
    T get(const std::string& key, const T& default_value = T{}) const {
        std::lock_guard<std::mutex> lock(config_mutex);

        auto it = config_values.find(key);
        if (it != config_values.end()) {
            return it->second.get<T>();
        }

        NEXUS_LOG_WARN("CONFIG", "Config key not found, using default: " + key);
        return default_value;
    }

    template<typename T>
    bool update(const std::string& key, const T& new_value) {
        std::lock_guard<std::mutex> lock(config_mutex);

        auto it = config_values.find(key);
        if (it == config_values.end()) {
            NEXUS_LOG_ERROR("CONFIG", "Cannot update non-existent config key: " + key);
            return false;
        }

        if (!it->second.set(new_value)) {
            return false;
        }

        // Notify callbacks
        auto callback_it = change_callbacks.find(key);
        if (callback_it != change_callbacks.end()) {
            callback_it->second(key, it->second);
        }

        NEXUS_LOG_DEBUG("CONFIG", "Updated config value: " + key);
        return true;
    }

    bool exists(const std::string& key) const {
        std::lock_guard<std::mutex> lock(config_mutex);
        return config_values.find(key) != config_values.end();
    }

    void addValidator(const std::string& key, std::function<bool(const std::any&)> validator) {
        std::lock_guard<std::mutex> lock(config_mutex);

        auto it = config_values.find(key);
        if (it != config_values.end()) {
            it->second.validator = validator;
            NEXUS_LOG_DEBUG("CONFIG", "Added validator for: " + key);
        }
    }

    void addChangeCallback(const std::string& key,
                          std::function<void(const std::string&, const ConfigValue&)> callback) {
        std::lock_guard<std::mutex> lock(config_mutex);
        change_callbacks[key] = callback;
        NEXUS_LOG_DEBUG("CONFIG", "Added change callback for: " + key);
    }

    // Preset management
    void createPreset(const std::string& name, const std::string& description = "") {
        std::lock_guard<std::mutex> lock(config_mutex);

        ConfigPreset preset(name, description);

        // Copy current config values to preset
        for (const auto& pair : config_values) {
            if (!pair.second.is_readonly) {
                preset.addValue(pair.first, pair.second);
            }
        }

        presets[name] = preset;
        NEXUS_LOG_INFO("CONFIG", "Created preset: " + name);
    }

    bool loadPreset(const std::string& name) {
        std::lock_guard<std::mutex> lock(config_mutex);

        auto it = presets.find(name);
        if (it == presets.end()) {
            NEXUS_LOG_ERROR("CONFIG", "Preset not found: " + name);
            return false;
        }

        const ConfigPreset& preset = it->second;

        // Apply preset values
        for (const auto& key : preset.getKeys()) {
            const ConfigValue& preset_value = preset.getValue(key);
            auto config_it = config_values.find(key);
            if (config_it != config_values.end() && !config_it->second.is_readonly) {
                config_it->second.value = preset_value.value;

                // Notify callbacks
                auto callback_it = change_callbacks.find(key);
                if (callback_it != change_callbacks.end()) {
                    callback_it->second(key, config_it->second);
                }
            }
        }

        active_preset = name;
        NEXUS_LOG_INFO("CONFIG", "Loaded preset: " + name);
        return true;
    }

    std::vector<std::string> getPresetNames() const {
        std::lock_guard<std::mutex> lock(config_mutex);
        std::vector<std::string> names;
        for (const auto& pair : presets) {
            names.push_back(pair.first);
        }
        return names;
    }

    const std::string& getActivePreset() const {
        std::lock_guard<std::mutex> lock(config_mutex);
        return active_preset;
    }

    // File I/O
    void setConfigFile(const std::string& file_path) {
        std::lock_guard<std::mutex> lock(config_mutex);
        config_file_path = file_path;
    }

    bool saveToFile(const std::string& file_path = "") {
        std::lock_guard<std::mutex> lock(config_mutex);

        std::string path = file_path.empty() ? config_file_path : file_path;
        if (path.empty()) {
            NEXUS_LOG_ERROR("CONFIG", "No config file path specified");
            return false;
        }

        std::ofstream file(path);
        if (!file.is_open()) {
            NEXUS_LOG_ERROR("CONFIG", "Failed to open config file for writing: " + path);
            return false;
        }

        file << "# NEXUS Configuration File\n";
        file << "# Generated: " << std::chrono::system_clock::now().time_since_epoch().count() << "\n\n";

        // Group by categories
        std::unordered_map<std::string, std::vector<std::pair<std::string, ConfigValue>>> categories;
        for (const auto& pair : config_values) {
            categories[pair.second.category].push_back({pair.first, pair.second});
        }

        for (const auto& category : categories) {
            file << "[" << category.first << "]\n";

            for (const auto& config_pair : category.second) {
                const std::string& key = config_pair.first;
                const ConfigValue& value = config_pair.second;

                if (!value.description.empty()) {
                    file << "# " << value.description << "\n";
                }

                file << key << " = " << configValueToString(value) << "\n";

                if (value.is_readonly) {
                    file << "# ^ READ ONLY\n";
                }

                file << "\n";
            }
        }

        // Save presets
        if (!presets.empty()) {
            file << "\n# PRESETS\n";
            for (const auto& preset_pair : presets) {
                const ConfigPreset& preset = preset_pair.second;
                file << "[PRESET:" << preset.getName() << "]\n";
                file << "# " << preset.getDescription() << "\n";

                for (const auto& key : preset.getKeys()) {
                    const ConfigValue& value = preset.getValue(key);
                    file << key << " = " << configValueToString(value) << "\n";
                }
                file << "\n";
            }
        }

        file.close();
        NEXUS_LOG_INFO("CONFIG", "Configuration saved to: " + path);
        return true;
    }

    bool loadFromFile(const std::string& file_path = "") {
        std::lock_guard<std::mutex> lock(config_mutex);

        std::string path = file_path.empty() ? config_file_path : file_path;
        if (path.empty()) {
            NEXUS_LOG_ERROR("CONFIG", "No config file path specified");
            return false;
        }

        std::ifstream file(path);
        if (!file.is_open()) {
            NEXUS_LOG_WARN("CONFIG", "Config file not found, using defaults: " + path);
            return false;
        }

        std::string line;
        std::string current_category = "General";
        std::string current_preset = "";

        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t"));
            line.erase(line.find_last_not_of(" \t") + 1);

            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') {
                continue;
            }

            // Category header
            if (line[0] == '[' && line.back() == ']') {
                std::string header = line.substr(1, line.length() - 2);

                if (header.substr(0, 7) == "PRESET:") {
                    current_preset = header.substr(7);
                    current_category = "";
                } else {
                    current_category = header;
                    current_preset = "";
                }
                continue;
            }

            // Key-value pair
            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos) continue;

            std::string key = line.substr(0, eq_pos);
            std::string value_str = line.substr(eq_pos + 1);

            // Trim key and value
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value_str.erase(0, value_str.find_first_not_of(" \t"));
            value_str.erase(value_str.find_last_not_of(" \t") + 1);

            if (!current_preset.empty()) {
                // Loading preset value - will implement preset loading
                continue;
            }

            // Update existing config value
            auto it = config_values.find(key);
            if (it != config_values.end()) {
                setConfigValueFromString(it->second, value_str);
                NEXUS_LOG_DEBUG("CONFIG", "Loaded config: " + key + " = " + value_str);
            }
        }

        file.close();
        NEXUS_LOG_INFO("CONFIG", "Configuration loaded from: " + path);
        return true;
    }

    // Debug and introspection
    void printAll() const {
        std::lock_guard<std::mutex> lock(config_mutex);

        NEXUS_LOG_INFO("CONFIG", "=== Configuration Values ===");

        std::unordered_map<std::string, std::vector<std::pair<std::string, ConfigValue>>> categories;
        for (const auto& pair : config_values) {
            categories[pair.second.category].push_back({pair.first, pair.second});
        }

        for (const auto& category : categories) {
            NEXUS_LOG_INFO("CONFIG", "[" + category.first + "]");

            for (const auto& config_pair : category.second) {
                const std::string& key = config_pair.first;
                const ConfigValue& value = config_pair.second;

                std::string readonly_marker = value.is_readonly ? " [READONLY]" : "";
                std::string desc = value.description.empty() ? "" : " - " + value.description;

                NEXUS_LOG_INFO("CONFIG", "  " + key + " = " + configValueToString(value) +
                              readonly_marker + desc);
            }
        }

        if (!presets.empty()) {
            NEXUS_LOG_INFO("CONFIG", "=== Presets ===");
            for (const auto& preset_pair : presets) {
                std::string active_marker = (preset_pair.first == active_preset) ? " [ACTIVE]" : "";
                NEXUS_LOG_INFO("CONFIG", preset_pair.first + active_marker + " - " +
                              preset_pair.second.getDescription());
            }
        }
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(config_mutex);
        return config_values.size();
    }

private:
    NexusConfig() : config_file_path("nexus_config.ini") {
        setupDefaultValues();
    }

    void setupDefaultValues() {
        // Engine defaults
        set("engine.max_entities", 10000, ConfigType::INTEGER, "Maximum number of entities", "Engine");
        set("engine.target_fps", 60.0f, ConfigType::FLOAT, "Target frame rate", "Engine");
        set("engine.vsync", true, ConfigType::BOOL, "Enable vertical sync", "Engine");

        // Audio defaults
        set("audio.master_volume", 1.0f, ConfigType::FLOAT, "Master audio volume", "Audio");
        set("audio.sample_rate", 44100, ConfigType::INTEGER, "Audio sample rate", "Audio");
        set("audio.buffer_size", 1024, ConfigType::INTEGER, "Audio buffer size", "Audio");

        // Quantum Protocol defaults
        set("quantum.default_mode", 0, ConfigType::INTEGER, "Default processing mode", "Quantum");
        set("quantum.transition_speed", 2.0f, ConfigType::FLOAT, "Mode transition speed", "Quantum");
        set("quantum.intensity_threshold", 0.5f, ConfigType::FLOAT, "Intensity threshold", "Quantum");

        // Cognitive defaults
        set("cognitive.max_thoughts", 100, ConfigType::INTEGER, "Maximum concurrent thoughts", "Cognitive");
        set("cognitive.decay_rate", 0.01f, ConfigType::FLOAT, "Memory decay rate", "Cognitive");
        set("cognitive.learning_rate", 0.1f, ConfigType::FLOAT, "Learning rate", "Cognitive");

        // Visual defaults
        ColorConfig bg_color(0.1f, 0.1f, 0.2f, 1.0f);
        set("visual.background_color", bg_color, ConfigType::COLOR, "Background color", "Visual");
        set("visual.trail_length", 50, ConfigType::INTEGER, "Trail render length", "Visual");
        set("visual.glow_intensity", 0.8f, ConfigType::FLOAT, "Glow effect intensity", "Visual");

        Vec3Config world_size(1000.0f, 1000.0f, 1000.0f);
        set("world.size", world_size, ConfigType::VECTOR3, "World boundary size", "World");

        // Debug defaults
        set("debug.enable_profiler", true, ConfigType::BOOL, "Enable performance profiler", "Debug");
        set("debug.log_level", 2, ConfigType::INTEGER, "Minimum log level (0=TRACE, 5=CRITICAL)", "Debug");
        set("debug.save_dumps", false, ConfigType::BOOL, "Save debug dumps", "Debug");

        // System info (readonly)
        set("system.version", std::string("NEXUS v1.0"), ConfigType::STRING, "System version", "System", true);
        set("system.build_date", std::string(__DATE__), ConfigType::STRING, "Build date", "System", true);

        NEXUS_LOG_INFO("CONFIG", "Default configuration initialized");
    }

    std::string configValueToString(const ConfigValue& value) const {
        switch (value.type) {
            case ConfigType::BOOL:
                return value.get<bool>() ? "true" : "false";
            case ConfigType::INTEGER:
                return std::to_string(value.get<int>());
            case ConfigType::FLOAT:
                return std::to_string(value.get<float>());
            case ConfigType::STRING:
                return value.get<std::string>();
            case ConfigType::VECTOR3: {
                Vec3Config vec = value.get<Vec3Config>();
                return vec.toString();
            }
            case ConfigType::COLOR: {
                ColorConfig color = value.get<ColorConfig>();
                return color.toString();
            }
            default:
                return "unknown";
        }
    }

    void setConfigValueFromString(ConfigValue& config_val, const std::string& str) {
        try {
            switch (config_val.type) {
                case ConfigType::BOOL:
                    config_val.set(str == "true" || str == "1");
                    break;
                case ConfigType::INTEGER:
                    config_val.set(std::stoi(str));
                    break;
                case ConfigType::FLOAT:
                    config_val.set(std::stof(str));
                    break;
                case ConfigType::STRING:
                    config_val.set(str);
                    break;
                case ConfigType::VECTOR3:
                    config_val.set(Vec3Config::fromString(str));
                    break;
                case ConfigType::COLOR:
                    config_val.set(ColorConfig::fromString(str));
                    break;
            }
        } catch (const std::exception& e) {
            NEXUS_LOG_ERROR("CONFIG", "Failed to parse config value: " + std::string(e.what()));
        }
    }
};

std::unique_ptr<NexusConfig> NexusConfig::instance = nullptr;

// Convenience macros
#define NEXUS_CONFIG_GET(key, default_val) \
    NEXUS::NexusConfig::getInstance().get(key, default_val)

#define NEXUS_CONFIG_SET(key, value, type, desc, cat) \
    NEXUS::NexusConfig::getInstance().set(key, value, type, desc, cat)

#define NEXUS_CONFIG_UPDATE(key, value) \
    NEXUS::NexusConfig::getInstance().update(key, value)

} // namespace NEXUS
