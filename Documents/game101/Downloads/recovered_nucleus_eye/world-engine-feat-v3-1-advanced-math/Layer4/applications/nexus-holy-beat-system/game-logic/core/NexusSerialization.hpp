// NEXUS Serialization System
// Comprehensive save/load system for game state, events, timers, and configurations
// Supports binary, JSON, and compressed formats with version management

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <sstream>
#include <chrono>
#include <functional>
#include <type_traits>

// JSON library (using nlohmann/json-like interface)
#include <iostream> // For debug output

namespace NEXUS {

// ============ SERIALIZATION INTERFACE ============

class ISerializable {
public:
    virtual ~ISerializable() = default;
    virtual void serialize(std::ostream& stream) const = 0;
    virtual void deserialize(std::istream& stream) = 0;
    virtual std::string getTypeName() const = 0;
    virtual uint32_t getVersion() const { return 1; }
};

// ============ BINARY SERIALIZATION HELPERS ============

class BinarySerializer {
public:
    // Write primitives
    template<typename T>
    static void write(std::ostream& stream, const T& value) {
        static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable");
        stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    static void write(std::ostream& stream, const std::string& value) {
        uint32_t size = static_cast<uint32_t>(value.size());
        write(stream, size);
        stream.write(value.c_str(), size);
    }

    template<typename T>
    static void write(std::ostream& stream, const std::vector<T>& vec) {
        uint32_t size = static_cast<uint32_t>(vec.size());
        write(stream, size);
        for (const auto& item : vec) {
            if constexpr (std::is_base_of_v<ISerializable, T>) {
                item.serialize(stream);
            } else {
                write(stream, item);
            }
        }
    }

    // Read primitives
    template<typename T>
    static T read(std::istream& stream) {
        static_assert(std::is_trivially_copyable_v<T>, "Type must be trivially copyable");
        T value;
        stream.read(reinterpret_cast<char*>(&value), sizeof(T));
        return value;
    }

    static std::string readString(std::istream& stream) {
        uint32_t size = read<uint32_t>(stream);
        std::string value(size, '\0');
        stream.read(&value[0], size);
        return value;
    }

    template<typename T>
    static std::vector<T> readVector(std::istream& stream) {
        uint32_t size = read<uint32_t>(stream);
        std::vector<T> vec;
        vec.reserve(size);

        for (uint32_t i = 0; i < size; ++i) {
            if constexpr (std::is_base_of_v<ISerializable, T>) {
                T item;
                item.deserialize(stream);
                vec.push_back(std::move(item));
            } else {
                vec.push_back(read<T>(stream));
            }
        }
        return vec;
    }
};

// ============ SAVE FILE HEADER ============

struct SaveFileHeader {
    static constexpr uint32_t MAGIC_NUMBER = 0x4E455855; // "NEXU"
    static constexpr uint32_t CURRENT_VERSION = 1;

    uint32_t magic{MAGIC_NUMBER};
    uint32_t version{CURRENT_VERSION};
    uint64_t timestamp{0};
    uint32_t checksum{0};
    char description[128]{};
    uint32_t dataSize{0};
    uint32_t compressionType{0}; // 0=none, 1=zlib, etc.

    SaveFileHeader() {
        auto now = std::chrono::system_clock::now();
        timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    }

    void serialize(std::ostream& stream) const {
        BinarySerializer::write(stream, magic);
        BinarySerializer::write(stream, version);
        BinarySerializer::write(stream, timestamp);
        BinarySerializer::write(stream, checksum);
        stream.write(description, 128);
        BinarySerializer::write(stream, dataSize);
        BinarySerializer::write(stream, compressionType);
    }

    void deserialize(std::istream& stream) {
        magic = BinarySerializer::read<uint32_t>(stream);
        version = BinarySerializer::read<uint32_t>(stream);
        timestamp = BinarySerializer::read<uint64_t>(stream);
        checksum = BinarySerializer::read<uint32_t>(stream);
        stream.read(description, 128);
        dataSize = BinarySerializer::read<uint32_t>(stream);
        compressionType = BinarySerializer::read<uint32_t>(stream);
    }

    bool isValid() const {
        return magic == MAGIC_NUMBER && version <= CURRENT_VERSION;
    }
};

// ============ TIMER STATE SERIALIZATION ============

struct SerializableTimer : public ISerializable {
    std::string id;
    double remainingTime;
    double originalDuration;
    bool isPaused;
    bool isRepeating;
    std::chrono::steady_clock::time_point createdAt;

    void serialize(std::ostream& stream) const override {
        BinarySerializer::write(stream, id);
        BinarySerializer::write(stream, remainingTime);
        BinarySerializer::write(stream, originalDuration);
        BinarySerializer::write(stream, isPaused);
        BinarySerializer::write(stream, isRepeating);

        // Convert time_point to milliseconds since epoch
        auto duration = createdAt.time_since_epoch();
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        BinarySerializer::write(stream, millis);
    }

    void deserialize(std::istream& stream) override {
        id = BinarySerializer::readString(stream);
        remainingTime = BinarySerializer::read<double>(stream);
        originalDuration = BinarySerializer::read<double>(stream);
        isPaused = BinarySerializer::read<bool>(stream);
        isRepeating = BinarySerializer::read<bool>(stream);

        // Convert milliseconds back to time_point
        auto millis = BinarySerializer::read<int64_t>(stream);
        createdAt = std::chrono::steady_clock::time_point(std::chrono::milliseconds(millis));
    }

    std::string getTypeName() const override { return "Timer"; }
};

// ============ EVENT STATE SERIALIZATION ============

struct SerializableEvent : public ISerializable {
    std::string eventId;
    std::unordered_map<std::string, std::string> properties;
    std::chrono::steady_clock::time_point timestamp;
    double intensity;

    void serialize(std::ostream& stream) const override {
        BinarySerializer::write(stream, eventId);
        BinarySerializer::write(stream, intensity);

        // Serialize timestamp
        auto duration = timestamp.time_since_epoch();
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
        BinarySerializer::write(stream, millis);

        // Serialize properties
        uint32_t propCount = static_cast<uint32_t>(properties.size());
        BinarySerializer::write(stream, propCount);
        for (const auto& prop : properties) {
            BinarySerializer::write(stream, prop.first);
            BinarySerializer::write(stream, prop.second);
        }
    }

    void deserialize(std::istream& stream) override {
        eventId = BinarySerializer::readString(stream);
        intensity = BinarySerializer::read<double>(stream);

        // Deserialize timestamp
        auto millis = BinarySerializer::read<int64_t>(stream);
        timestamp = std::chrono::steady_clock::time_point(std::chrono::milliseconds(millis));

        // Deserialize properties
        uint32_t propCount = BinarySerializer::read<uint32_t>(stream);
        properties.clear();
        for (uint32_t i = 0; i < propCount; ++i) {
            std::string key = BinarySerializer::readString(stream);
            std::string value = BinarySerializer::readString(stream);
            properties[key] = value;
        }
    }

    std::string getTypeName() const override { return "Event"; }
};

// ============ GAME STATE CONTAINER ============

struct GameStateSnapshot : public ISerializable {
    std::vector<SerializableTimer> activeTimers;
    std::vector<SerializableEvent> queuedEvents;
    std::unordered_map<std::string, std::string> gameSettings;
    std::unordered_map<std::string, double> systemMetrics;
    double gameTime{0.0};
    uint64_t frameCount{0};

    void serialize(std::ostream& stream) const override {
        BinarySerializer::write(stream, gameTime);
        BinarySerializer::write(stream, frameCount);

        // Serialize timers
        BinarySerializer::write(stream, activeTimers);

        // Serialize events
        BinarySerializer::write(stream, queuedEvents);

        // Serialize settings
        uint32_t settingsCount = static_cast<uint32_t>(gameSettings.size());
        BinarySerializer::write(stream, settingsCount);
        for (const auto& setting : gameSettings) {
            BinarySerializer::write(stream, setting.first);
            BinarySerializer::write(stream, setting.second);
        }

        // Serialize metrics
        uint32_t metricsCount = static_cast<uint32_t>(systemMetrics.size());
        BinarySerializer::write(stream, metricsCount);
        for (const auto& metric : systemMetrics) {
            BinarySerializer::write(stream, metric.first);
            BinarySerializer::write(stream, metric.second);
        }
    }

    void deserialize(std::istream& stream) override {
        gameTime = BinarySerializer::read<double>(stream);
        frameCount = BinarySerializer::read<uint64_t>(stream);

        // Deserialize timers
        activeTimers = BinarySerializer::readVector<SerializableTimer>(stream);

        // Deserialize events
        queuedEvents = BinarySerializer::readVector<SerializableEvent>(stream);

        // Deserialize settings
        uint32_t settingsCount = BinarySerializer::read<uint32_t>(stream);
        gameSettings.clear();
        for (uint32_t i = 0; i < settingsCount; ++i) {
            std::string key = BinarySerializer::readString(stream);
            std::string value = BinarySerializer::readString(stream);
            gameSettings[key] = value;
        }

        // Deserialize metrics
        uint32_t metricsCount = BinarySerializer::read<uint32_t>(stream);
        systemMetrics.clear();
        for (uint32_t i = 0; i < metricsCount; ++i) {
            std::string key = BinarySerializer::readString(stream);
            double value = BinarySerializer::read<double>(stream);
            systemMetrics[key] = value;
        }
    }

    std::string getTypeName() const override { return "GameState"; }
};

// ============ MAIN SERIALIZATION SYSTEM ============

class SerializationSystem {
private:
    std::string saveDirectory;
    uint32_t maxBackups{10};
    bool useCompression{false};

    // Registered type factories for deserialization
    std::unordered_map<std::string, std::function<std::unique_ptr<ISerializable>()>> typeFactories;

    // Calculate simple checksum
    uint32_t calculateChecksum(const char* data, size_t size) {
        uint32_t checksum = 0;
        for (size_t i = 0; i < size; ++i) {
            checksum = (checksum << 1) ^ static_cast<uint32_t>(data[i]);
        }
        return checksum;
    }

public:
    SerializationSystem(const std::string& saveDir = "./saves")
        : saveDirectory(saveDir) {
        // Register built-in types
        registerType<SerializableTimer>();
        registerType<SerializableEvent>();
        registerType<GameStateSnapshot>();
    }

    template<typename T>
    void registerType() {
        static_assert(std::is_base_of_v<ISerializable, T>, "Type must inherit from ISerializable");
        T sample;
        typeFactories[sample.getTypeName()] = []() {
            return std::make_unique<T>();
        };
    }

    // Save game state to file
    bool saveGameState(const GameStateSnapshot& gameState, const std::string& filename) {
        std::string fullPath = saveDirectory + "/" + filename;

        // Create backup if file exists
        if (std::ifstream(fullPath).good()) {
            createBackup(fullPath);
        }

        std::ofstream file(fullPath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // Write to memory buffer first to calculate size and checksum
        std::stringstream buffer;
        gameState.serialize(buffer);

        std::string data = buffer.str();

        // Create header
        SaveFileHeader header;
        std::strncpy(header.description, "NEXUS Game State", sizeof(header.description) - 1);
        header.dataSize = static_cast<uint32_t>(data.size());
        header.checksum = calculateChecksum(data.c_str(), data.size());
        header.compressionType = useCompression ? 1 : 0;

        // Write header and data
        header.serialize(file);
        file.write(data.c_str(), data.size());

        file.close();
        return file.good();
    }

    // Load game state from file
    std::unique_ptr<GameStateSnapshot> loadGameState(const std::string& filename) {
        std::string fullPath = saveDirectory + "/" + filename;
        std::ifstream file(fullPath, std::ios::binary);
        if (!file.is_open()) {
            return nullptr;
        }

        // Read and verify header
        SaveFileHeader header;
        header.deserialize(file);

        if (!header.isValid()) {
            return nullptr;
        }

        // Read data
        std::vector<char> data(header.dataSize);
        file.read(data.data(), header.dataSize);

        // Verify checksum
        uint32_t actualChecksum = calculateChecksum(data.data(), data.size());
        if (actualChecksum != header.checksum) {
            return nullptr; // Data corruption detected
        }

        // Deserialize game state
        std::stringstream dataStream;
        dataStream.write(data.data(), data.size());
        dataStream.seekg(0);

        auto gameState = std::make_unique<GameStateSnapshot>();
        gameState->deserialize(dataStream);

        return gameState;
    }

    // Save any serializable object
    template<typename T>
    bool saveObject(const T& object, const std::string& filename) {
        static_assert(std::is_base_of_v<ISerializable, T>, "Type must inherit from ISerializable");

        std::string fullPath = saveDirectory + "/" + filename;
        std::ofstream file(fullPath, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        object.serialize(file);
        return file.good();
    }

    // Load any serializable object
    template<typename T>
    std::unique_ptr<T> loadObject(const std::string& filename) {
        static_assert(std::is_base_of_v<ISerializable, T>, "Type must inherit from ISerializable");

        std::string fullPath = saveDirectory + "/" + filename;
        std::ifstream file(fullPath, std::ios::binary);
        if (!file.is_open()) {
            return nullptr;
        }

        auto object = std::make_unique<T>();
        object->deserialize(file);
        return object;
    }

    // Quick save/load for active sessions
    bool quickSave(const GameStateSnapshot& gameState, const std::string& slotName = "quicksave") {
        return saveGameState(gameState, slotName + ".sav");
    }

    std::unique_ptr<GameStateSnapshot> quickLoad(const std::string& slotName = "quicksave") {
        return loadGameState(slotName + ".sav");
    }

    // Auto-save functionality
    bool autoSave(const GameStateSnapshot& gameState) {
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

        std::string filename = "autosave_" + std::to_string(timestamp) + ".sav";
        bool success = saveGameState(gameState, filename);

        if (success) {
            cleanupOldAutoSaves();
        }

        return success;
    }

    // Utility functions
    std::vector<std::string> listSaveFiles() const {
        std::vector<std::string> saves;
        // In a real implementation, you'd enumerate directory contents
        // For now, return empty vector
        return saves;
    }

    bool deleteSave(const std::string& filename) {
        std::string fullPath = saveDirectory + "/" + filename;
        return std::remove(fullPath.c_str()) == 0;
    }

    void createBackup(const std::string& originalPath) {
        std::string backupPath = originalPath + ".backup";
        std::ifstream src(originalPath, std::ios::binary);
        std::ofstream dst(backupPath, std::ios::binary);
        dst << src.rdbuf();
    }

    void cleanupOldAutoSaves() {
        // In a real implementation, you'd:
        // 1. List all autosave files
        // 2. Sort by timestamp
        // 3. Delete oldest files if count exceeds maxBackups
    }

    // Configuration
    void setSaveDirectory(const std::string& dir) { saveDirectory = dir; }
    void setMaxBackups(uint32_t max) { maxBackups = max; }
    void setUseCompression(bool compress) { useCompression = compress; }

    const std::string& getSaveDirectory() const { return saveDirectory; }
    uint32_t getMaxBackups() const { return maxBackups; }
    bool getUseCompression() const { return useCompression; }
};

// ============ SAVE SYSTEM INTEGRATION HELPERS ============

template<typename TimerSystemType>
GameStateSnapshot createSnapshotFromTimerSystem(const TimerSystemType& timerSystem, double currentGameTime) {
    GameStateSnapshot snapshot;
    snapshot.gameTime = currentGameTime;
    snapshot.frameCount = 0; // Would be set by calling system

    // Convert active timers to serializable format
    // (Implementation depends on your timer system's interface)
    // for (const auto& timer : timerSystem.getActiveTimers()) {
    //     SerializableTimer sTimer;
    //     sTimer.id = timer.getId();
    //     sTimer.remainingTime = timer.getRemainingTime();
    //     sTimer.originalDuration = timer.getOriginalDuration();
    //     sTimer.isPaused = timer.isPaused();
    //     sTimer.isRepeating = timer.isRepeating();
    //     sTimer.createdAt = timer.getCreationTime();
    //     snapshot.activeTimers.push_back(sTimer);
    // }

    return snapshot;
}

template<typename EventSystemType>
void addEventsToSnapshot(GameStateSnapshot& snapshot, const EventSystemType& eventSystem) {
    // Convert queued events to serializable format
    // (Implementation depends on your event system's interface)
    // for (const auto& event : eventSystem.getQueuedEvents()) {
    //     SerializableEvent sEvent;
    //     sEvent.eventId = event.eventId;
    //     sEvent.properties = event.properties;
    //     sEvent.timestamp = event.timestamp;
    //     sEvent.intensity = event.intensity;
    //     snapshot.queuedEvents.push_back(sEvent);
    // }
}

} // namespace NEXUS
