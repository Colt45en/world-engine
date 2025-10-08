// NEXUS WebSocket Bridge - Connects C++ Game Engine to Web Dashboard
// This enables real-time communication between the game logic and web interfaces

#pragma once

#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <chrono>
#include <sstream>
#include <iomanip>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

namespace NEXUS {

struct AudioData {
    double bpm = 128.0;
    float amplitude = 0.0f;
    float frequency = 440.0f;
    float spectral_centroid = 0.0f;
    std::vector<float> spectrum;
    int current_mode = 0;
    std::string mode_name = "MIRROR";

    std::string toJson() const {
        std::ostringstream json;
        json << "{"
             << "\"bpm\":" << bpm << ","
             << "\"amplitude\":" << amplitude << ","
             << "\"frequency\":" << frequency << ","
             << "\"spectral_centroid\":" << spectral_centroid << ","
             << "\"current_mode\":" << current_mode << ","
             << "\"mode_name\":\"" << mode_name << "\","
             << "\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count()
             << "}";
        return json.str();
    }
};

struct VisualData {
    struct Color {
        float r, g, b, a;
        Color(float r = 1.0f, float g = 1.0f, float b = 1.0f, float a = 1.0f) : r(r), g(g), b(b), a(a) {}
    };

    std::vector<Color> palette;
    float glow_intensity = 0.8f;
    int petal_count = 12;
    float trail_length = 50.0f;
    std::string active_effect = "quantum_flow";

    std::string toJson() const {
        std::ostringstream json;
        json << "{"
             << "\"glow_intensity\":" << glow_intensity << ","
             << "\"petal_count\":" << petal_count << ","
             << "\"trail_length\":" << trail_length << ","
             << "\"active_effect\":\"" << active_effect << "\","
             << "\"palette\":[";

        for (size_t i = 0; i < palette.size(); ++i) {
            if (i > 0) json << ",";
            json << "{\"r\":" << palette[i].r << ",\"g\":" << palette[i].g
                 << ",\"b\":" << palette[i].b << ",\"a\":" << palette[i].a << "}";
        }

        json << "]"
             << "}";
        return json.str();
    }
};

struct SystemMetrics {
    float cpu_usage = 0.0f;
    size_t memory_usage = 0;
    float fps = 60.0f;
    size_t active_entities = 0;
    size_t active_thoughts = 0;
    size_t active_resources = 0;
    std::string system_status = "ONLINE";

    std::string toJson() const {
        std::ostringstream json;
        json << "{"
             << "\"cpu_usage\":" << cpu_usage << ","
             << "\"memory_usage\":" << memory_usage << ","
             << "\"fps\":" << fps << ","
             << "\"active_entities\":" << active_entities << ","
             << "\"active_thoughts\":" << active_thoughts << ","
             << "\"active_resources\":" << active_resources << ","
             << "\"system_status\":\"" << system_status << "\""
             << "}";
        return json.str();
    }
};

struct NexusMessage {
    enum Type {
        AUDIO_DATA,
        VISUAL_DATA,
        SYSTEM_METRICS,
        COMMAND,
        HEARTBEAT
    };

    Type type;
    std::string payload;
    std::chrono::system_clock::time_point timestamp;

    NexusMessage(Type t, const std::string& p)
        : type(t), payload(p), timestamp(std::chrono::system_clock::now()) {}

    std::string toWebSocketMessage() const {
        std::ostringstream message;
        message << "{"
                << "\"type\":\"" << typeToString() << "\","
                << "\"payload\":" << payload << ","
                << "\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(
                       timestamp.time_since_epoch()).count()
                << "}";
        return message.str();
    }

private:
    std::string typeToString() const {
        switch (type) {
            case AUDIO_DATA: return "audio_data";
            case VISUAL_DATA: return "visual_data";
            case SYSTEM_METRICS: return "system_metrics";
            case COMMAND: return "command";
            case HEARTBEAT: return "heartbeat";
            default: return "unknown";
        }
    }
};

class NexusWebSocketBridge {
private:
    std::atomic<bool> running_;
    std::atomic<bool> connected_;
    std::thread server_thread_;
    std::thread message_processor_;

    std::queue<NexusMessage> outbound_queue_;
    std::queue<std::string> inbound_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    int server_socket_;
    int client_socket_;
    int port_;

    // Callbacks
    std::function<void(const std::string&)> command_callback_;
    std::function<AudioData()> audio_data_provider_;
    std::function<VisualData()> visual_data_provider_;
    std::function<SystemMetrics()> metrics_provider_;

public:
    NexusWebSocketBridge(int port = 8080)
        : running_(false), connected_(false), server_socket_(-1), client_socket_(-1), port_(port) {

#ifdef _WIN32
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
    }

    ~NexusWebSocketBridge() {
        stop();

#ifdef _WIN32
        WSACleanup();
#endif
    }

    bool start() {
        if (running_) return false;

        running_ = true;
        server_thread_ = std::thread(&NexusWebSocketBridge::serverLoop, this);
        message_processor_ = std::thread(&NexusWebSocketBridge::messageProcessorLoop, this);

        return true;
    }

    void stop() {
        running_ = false;
        connected_ = false;

        if (client_socket_ != -1) {
#ifdef _WIN32
            closesocket(client_socket_);
#else
            close(client_socket_);
#endif
            client_socket_ = -1;
        }

        if (server_socket_ != -1) {
#ifdef _WIN32
            closesocket(server_socket_);
#else
            close(server_socket_);
#endif
            server_socket_ = -1;
        }

        queue_cv_.notify_all();

        if (server_thread_.joinable()) server_thread_.join();
        if (message_processor_.joinable()) message_processor_.join();
    }

    // Set data providers
    void setAudioDataProvider(std::function<AudioData()> provider) {
        audio_data_provider_ = provider;
    }

    void setVisualDataProvider(std::function<VisualData()> provider) {
        visual_data_provider_ = provider;
    }

    void setMetricsProvider(std::function<SystemMetrics()> provider) {
        metrics_provider_ = provider;
    }

    void setCommandCallback(std::function<void(const std::string&)> callback) {
        command_callback_ = callback;
    }

    // Send data to web client
    void sendAudioData(const AudioData& data) {
        queueMessage(NexusMessage(NexusMessage::AUDIO_DATA, data.toJson()));
    }

    void sendVisualData(const VisualData& data) {
        queueMessage(NexusMessage(NexusMessage::VISUAL_DATA, data.toJson()));
    }

    void sendMetrics(const SystemMetrics& metrics) {
        queueMessage(NexusMessage(NexusMessage::SYSTEM_METRICS, metrics.toJson()));
    }

    void sendHeartbeat() {
        queueMessage(NexusMessage(NexusMessage::HEARTBEAT, "{\"status\":\"alive\"}"));
    }

    bool isConnected() const {
        return connected_.load();
    }

    void update() {
        // Automatic data sending if providers are set
        if (!connected_) return;

        if (audio_data_provider_) {
            sendAudioData(audio_data_provider_());
        }

        if (visual_data_provider_) {
            sendVisualData(visual_data_provider_());
        }

        if (metrics_provider_) {
            sendMetrics(metrics_provider_());
        }
    }

private:
    void serverLoop() {
        // Create server socket
        server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
        if (server_socket_ < 0) {
            return;
        }

        // Enable reuse of address
        int opt = 1;
        setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));

        // Bind to port
        sockaddr_in server_addr{};
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(port_);

        if (bind(server_socket_, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            return;
        }

        // Listen for connections
        if (listen(server_socket_, 1) < 0) {
            return;
        }

        while (running_) {
            // Accept client connection
            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);

            client_socket_ = accept(server_socket_, (sockaddr*)&client_addr, &client_len);
            if (client_socket_ < 0) {
                if (running_) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                continue;
            }

            connected_ = true;

            // Handle client communication
            handleClient();

            connected_ = false;

#ifdef _WIN32
            closesocket(client_socket_);
#else
            close(client_socket_);
#endif
            client_socket_ = -1;
        }
    }

    void handleClient() {
        // Send initial handshake response (simplified WebSocket handshake)
        std::string handshake_response =
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Accept: dummy\r\n\r\n";

        send(client_socket_, handshake_response.c_str(), handshake_response.length(), 0);

        // Communication loop
        char buffer[4096];
        while (running_ && connected_) {
            // Check for incoming messages
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(client_socket_, &readfds);

            timeval timeout{0, 100000}; // 100ms timeout
            int activity = select(client_socket_ + 1, &readfds, nullptr, nullptr, &timeout);

            if (activity > 0 && FD_ISSET(client_socket_, &readfds)) {
                int bytes_received = recv(client_socket_, buffer, sizeof(buffer) - 1, 0);
                if (bytes_received <= 0) {
                    break;
                }

                buffer[bytes_received] = '\0';
                processIncomingMessage(std::string(buffer, bytes_received));
            }
        }
    }

    void processIncomingMessage(const std::string& message) {
        // Simple command processing (in a real implementation, you'd parse WebSocket frames)
        if (command_callback_ && message.find("command") != std::string::npos) {
            command_callback_(message);
        }

        std::lock_guard<std::mutex> lock(queue_mutex_);
        inbound_queue_.push(message);
        queue_cv_.notify_one();
    }

    void messageProcessorLoop() {
        while (running_) {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait_for(lock, std::chrono::milliseconds(16)); // ~60fps

            // Process outbound messages
            while (!outbound_queue_.empty() && connected_) {
                NexusMessage message = outbound_queue_.front();
                outbound_queue_.pop();

                std::string ws_message = message.toWebSocketMessage();

                // Send WebSocket frame (simplified)
                if (client_socket_ != -1) {
                    std::string frame = createWebSocketFrame(ws_message);
                    send(client_socket_, frame.c_str(), frame.length(), 0);
                }
            }

            lock.unlock();
        }
    }

    void queueMessage(const NexusMessage& message) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        outbound_queue_.push(message);
        queue_cv_.notify_one();
    }

    std::string createWebSocketFrame(const std::string& payload) {
        // Simplified WebSocket frame creation (text frame)
        std::string frame;
        frame.push_back(0x81); // Text frame, final fragment

        if (payload.length() < 126) {
            frame.push_back(static_cast<char>(payload.length()));
        } else if (payload.length() < 65536) {
            frame.push_back(126);
            frame.push_back(static_cast<char>((payload.length() >> 8) & 0xFF));
            frame.push_back(static_cast<char>(payload.length() & 0xFF));
        } else {
            frame.push_back(127);
            for (int i = 7; i >= 0; --i) {
                frame.push_back(static_cast<char>((payload.length() >> (8 * i)) & 0xFF));
            }
        }

        frame += payload;
        return frame;
    }
};

// Convenience class for easy integration
class NexusWebIntegration {
private:
    std::unique_ptr<NexusWebSocketBridge> bridge_;
    std::chrono::steady_clock::time_point last_update_;

public:
    NexusWebIntegration(int port = 8080) : bridge_(std::make_unique<NexusWebSocketBridge>(port)) {
        last_update_ = std::chrono::steady_clock::now();
    }

    bool initialize() {
        return bridge_->start();
    }

    void shutdown() {
        if (bridge_) {
            bridge_->stop();
        }
    }

    void update() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update_);

        // Update at ~60fps
        if (elapsed.count() >= 16) {
            bridge_->update();
            last_update_ = now;
        }
    }

    // Easy setup methods
    template<typename GameEngine, typename Protocol, typename Visuals, typename Profiler>
    void connectSystems(GameEngine& engine, Protocol& protocol, Visuals& visuals, Profiler& profiler) {
        // Set up automatic data providers
        bridge_->setAudioDataProvider([&protocol]() -> AudioData {
            AudioData data;
            data.current_mode = protocol.getCurrentMode();
            auto features = protocol.getLastAudioFeatures();
            data.amplitude = features.amplitude;
            data.frequency = features.frequency;
            data.spectral_centroid = features.spectral_centroid;
            return data;
        });

        bridge_->setVisualDataProvider([&visuals]() -> VisualData {
            VisualData data;
            auto palette = visuals.getCurrentPalette();
            for (const auto& color : palette) {
                data.palette.emplace_back(color.r, color.g, color.b, color.a);
            }
            return data;
        });

        bridge_->setMetricsProvider([&engine, &profiler]() -> SystemMetrics {
            SystemMetrics metrics;
            metrics.active_entities = engine.getEntityCount();
            metrics.memory_usage = profiler.getCurrentMemoryUsage();
            metrics.fps = 60.0f; // Would get from actual FPS counter
            return metrics;
        });

        bridge_->setCommandCallback([&engine, &protocol](const std::string& command) {
            // Parse and handle web commands
            if (command.find("setMode") != std::string::npos) {
                // Extract mode number and set it
                // This would be more sophisticated in practice
            }
        });
    }

    bool isConnected() const {
        return bridge_->isConnected();
    }
};

} // namespace NEXUS
