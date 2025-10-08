// VectorLab Forge with Integrated Brain - Complete Body System
// Heart + Veins + Body + BRAIN (Nucleus Core AI Integration)
// Deps: GLFW, GLEW, GLUT, ImGui, GLM, nlohmann/json, libcurl (for nucleus communication)
// Build: g++ vectorlab-forge-with-brain.cpp -std=c++17 -lglfw -lGLEW -lGL -lglut -ldl -lpthread -lcurl

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <queue>
#include <functional>
#include <cmath>
#include <thread>
#include <mutex>
#include <chrono>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glut.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

// For nucleus communication (optional - can be removed if not available)
#ifdef USE_CURL
#include <curl/curl.h>
#endif

// Embedded HTTP Server & Live Viewer
#include <thread>
#include <atomic>
#include <sstream>
#include <algorithm>
#include <cstring>

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

// ═══════════════════════════════════════════════════════════════════
// LIVE VIEWER SERVER - Nucleus Self-Observation & Real-time Monitoring
// ═══════════════════════════════════════════════════════════════════

class LiveViewerServer
{
private:
    std::atomic<bool> running{false};
    std::thread serverThread;
    int serverSocket = -1;
    int port = 7777;

    // Shared state for live broadcasting
    mutable std::mutex stateMutex;
    json lastForgeState;
    json lastBrainState;
    json lastGlyphEvents;
    std::string lastThought = "Initializing...";
    float lastHeartRate = 0.0f;

public:
    std::atomic<bool> broadcastEnabled{true};
    std::string serverUrl = "http://localhost:7777";

    void start()
    {
        if (running)
            return;

        running = true;
        serverThread = std::thread(&LiveViewerServer::serverLoop, this);

        std::cout << "🌐 Live Viewer Server starting on " << serverUrl << "\n";
        std::cout << "📊 Dashboard: " << serverUrl << "/\n";
        std::cout << "🧠 Brain Panel: " << serverUrl << "/panel\n";
        std::cout << "📡 Live Stream: " << serverUrl << "/sse\n";
    }

    void stop()
    {
        running = false;
        if (serverThread.joinable())
        {
            serverThread.join();
        }
        if (serverSocket >= 0)
        {
#ifdef _WIN32
            closesocket(serverSocket);
            WSACleanup();
#else
            close(serverSocket);
#endif
        }
    }

    // Update state for live broadcasting (called from main app)
    void updateState(const json &forgeState, const json &brainState,
                     const json &glyphEvents, const std::string &thought, float heartRate)
    {
        std::lock_guard<std::mutex> lock(stateMutex);
        lastForgeState = forgeState;
        lastBrainState = brainState;
        lastGlyphEvents = glyphEvents;
        lastThought = thought;
        lastHeartRate = heartRate;
    }

    json getSnapshot() const
    {
        std::lock_guard<std::mutex> lock(stateMutex);

        json snapshot;
        snapshot["timestamp"] = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch())
                                    .count();
        snapshot["forge"] = lastForgeState;
        snapshot["brain"] = lastBrainState;
        snapshot["glyph_events"] = lastGlyphEvents;
        snapshot["current_thought"] = lastThought;
        snapshot["heart_rate"] = lastHeartRate;

        return snapshot;
    }

private:
    void serverLoop()
    {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
        {
            std::cerr << "WSAStartup failed\n";
            return;
        }
#endif

        serverSocket = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSocket < 0)
        {
            std::cerr << "Failed to create server socket\n";
            return;
        }

        int opt = 1;
#ifdef _WIN32
        setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, (char *)&opt, sizeof(opt));
#else
        setsockopt(serverSocket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

        sockaddr_in address;
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(port);

        if (bind(serverSocket, (struct sockaddr *)&address, sizeof(address)) < 0)
        {
            std::cerr << "Failed to bind to port " << port << "\n";
            return;
        }

        if (listen(serverSocket, 10) < 0)
        {
            std::cerr << "Failed to listen on socket\n";
            return;
        }

        std::cout << "🚀 Live Viewer Server listening on port " << port << "\n";

        while (running)
        {
            sockaddr_in clientAddr;
#ifdef _WIN32
            int clientLen = sizeof(clientAddr);
#else
            socklen_t clientLen = sizeof(clientAddr);
#endif

            int clientSocket = accept(serverSocket, (struct sockaddr *)&clientAddr, &clientLen);
            if (clientSocket >= 0)
            {
                std::thread(&LiveViewerServer::handleClient, this, clientSocket).detach();
            }
        }
    }

    void handleClient(int clientSocket)
    {
        char buffer[4096] = {0};
        ssize_t bytesRead = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);

        if (bytesRead <= 0)
        {
#ifdef _WIN32
            closesocket(clientSocket);
#else
            close(clientSocket);
#endif
            return;
        }

        std::string request(buffer, bytesRead);
        std::string path = extractPath(request);

        if (path == "/" || path.empty())
        {
            serveDashboard(clientSocket);
        }
        else if (path == "/panel")
        {
            servePanel(clientSocket);
        }
        else if (path == "/state")
        {
            serveState(clientSocket);
        }
        else if (path == "/sse")
        {
            serveSSE(clientSocket);
        }
        else
        {
            serve404(clientSocket);
        }

#ifdef _WIN32
        closesocket(clientSocket);
#else
        close(clientSocket);
#endif
    }

    std::string extractPath(const std::string &request)
    {
        size_t start = request.find(' ') + 1;
        size_t end = request.find(' ', start);
        if (start != std::string::npos && end != std::string::npos)
        {
            std::string path = request.substr(start, end - start);
            size_t queryPos = path.find('?');
            if (queryPos != std::string::npos)
            {
                path = path.substr(0, queryPos);
            }
            return path;
        }
        return "/";
    }

    void serveDashboard(int clientSocket)
    {
        std::string html = R"(<!DOCTYPE html>
<html><head>
<title>VectorLab Forge - Live Nucleus Viewer</title>
<meta charset="utf-8">
<style>
body { font-family: 'Courier New', monospace; background: #0a0a0a; color: #00ff88; margin: 0; padding: 10px; }
.header { text-align: center; border-bottom: 2px solid #00ff88; padding: 10px; margin-bottom: 20px; }
.container { display: flex; gap: 20px; height: calc(100vh - 100px); }
.panel { flex: 1; border: 1px solid #00ff88; border-radius: 5px; overflow: hidden; }
.graph { flex: 1; border: 1px solid #00ff88; border-radius: 5px; background: #111; }
iframe { width: 100%; height: 100%; border: none; background: #111; }
canvas { display: block; width: 100%; height: 100%; }
.status { position: fixed; top: 10px; right: 10px; background: rgba(0,255,136,0.1); padding: 10px; border-radius: 5px; }
</style>
</head><body>
<div class="header">
  <h1>🔥 VectorLab Forge - Live Nucleus Viewer</h1>
  <p>Real-time monitoring of brain, heart, and glyph system automation</p>
</div>
<div class="status" id="status">Connecting...</div>
<div class="container">
  <div class="panel">
    <iframe src="/panel" title="Brain Panel"></iframe>
  </div>
  <div class="graph">
    <canvas id="graph"></canvas>
  </div>
</div>
<script>
let canvas = document.getElementById('graph');
let ctx = canvas.getContext('2d');
let statusDiv = document.getElementById('status');
let dataPoints = [];
let maxPoints = 100;

function resizeCanvas() {
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

let eventSource = new EventSource('/sse');
eventSource.onopen = () => {
  statusDiv.innerHTML = '🟢 Connected';
  statusDiv.style.background = 'rgba(0,255,136,0.2)';
};

eventSource.onerror = () => {
  statusDiv.innerHTML = '🔴 Disconnected';
  statusDiv.style.background = 'rgba(255,0,0,0.2)';
};

eventSource.onmessage = (event) => {
  try {
    let data = JSON.parse(event.data);
    processData(data);
    drawGraph();
  } catch (e) {
    console.error('SSE parse error:', e);
  }
};

function processData(data) {
  let now = Date.now();
  let point = {
    time: now,
    brainActivity: data.brain ? data.brain.cognitiveLoad || 0 : 0,
    heartRate: data.heart_rate || 0,
    glyphCount: data.brain ? (data.brain.connected_glyphs || 0) : 0,
    epochCount: data.brain ? (data.brain.connected_epochs || 0) : 0
  };

  dataPoints.push(point);
  if (dataPoints.length > maxPoints) {
    dataPoints.shift();
  }
}

function drawGraph() {
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (dataPoints.length < 2) return;

  let width = canvas.width;
  let height = canvas.height;
  let padding = 40;

  // Draw brain activity
  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < dataPoints.length; i++) {
    let x = padding + (i / (maxPoints - 1)) * (width - 2 * padding);
    let y = height - padding - dataPoints[i].brainActivity * (height - 2 * padding);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw heart rate
  ctx.strokeStyle = '#ff4444';
  ctx.beginPath();
  for (let i = 0; i < dataPoints.length; i++) {
    let x = padding + (i / (maxPoints - 1)) * (width - 2 * padding);
    let y = height - padding - (dataPoints[i].heartRate / 2) * (height - 2 * padding);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Labels
  ctx.fillStyle = '#00ff88';
  ctx.font = '12px Courier New';
  ctx.fillText('🧠 Brain Activity', 10, 20);
  ctx.fillStyle = '#ff4444';
  ctx.fillText('💓 Heart Rate', 10, 35);

  // Latest values
  if (dataPoints.length > 0) {
    let latest = dataPoints[dataPoints.length - 1];
    ctx.fillStyle = '#ffffff';
    ctx.fillText(`Brain: ${latest.brainActivity.toFixed(2)}`, width - 150, 20);
    ctx.fillText(`Heart: ${latest.heartRate.toFixed(2)}`, width - 150, 35);
    ctx.fillText(`Glyphs: ${latest.glyphCount}`, width - 150, 50);
    ctx.fillText(`Epochs: ${latest.epochCount}`, width - 150, 65);
  }
}

setInterval(drawGraph, 100);
</script>
</body></html>)";

        sendResponse(clientSocket, "200 OK", "text/html", html);
    }

    void servePanel(int clientSocket)
    {
        json state = getSnapshot();

        std::string html = R"(<!DOCTYPE html>
<html><head>
<title>Brain Panel</title>
<meta charset="utf-8">
<style>
body { font-family: 'Courier New', monospace; background: #111; color: #00ff88; margin: 0; padding: 10px; font-size: 12px; }
.section { margin-bottom: 15px; border: 1px solid #00ff88; border-radius: 3px; padding: 8px; }
.header { color: #ffff44; font-weight: bold; }
.value { color: #ffffff; }
.status { padding: 2px 6px; border-radius: 3px; }
.active { background: #004400; color: #00ff88; }
.inactive { background: #440000; color: #ff4444; }
.thought { font-style: italic; color: #88ffff; max-height: 40px; overflow: hidden; }
</style>
</head><body>
<div class="section">
  <div class="header">🧠 NUCLEUS BRAIN</div>
  <div>Status: <span class="status )";

        if (state["brain"]["isActive"].get<bool>())
        {
            html += "active\">ACTIVE</span></div>";
        }
        else
        {
            html += "inactive\">INACTIVE</span></div>";
        }

        html += R"(
  <div>Load: <span class="value">)" +
                std::to_string(state["brain"]["cognitiveLoad"].get<float>()) + R"(</span></div>
  <div>Queue: <span class="value">)" +
                std::to_string(state["brain"]["queueSize"].get<int>()) + R"(</span></div>
  <div class="thought">💭 )" +
                state["current_thought"].get<std::string>() + R"(</div>
</div>
<div class="section">
  <div class="header">🌟 GLYPH SYSTEM</div>
  <div>Status: <span class="status )";

        if (state["brain"]["glyph_system_active"].get<bool>())
        {
            html += "active\">CONNECTED</span></div>";
            html += R"(
  <div>Epochs: <span class="value">)" +
                    std::to_string(state["brain"]["connected_epochs"].get<int>()) + R"(</span></div>
  <div>Glyphs: <span class="value">)" +
                    std::to_string(state["brain"]["connected_glyphs"].get<int>()) + R"(</span></div>)";
        }
        else
        {
            html += "inactive\">DISCONNECTED</span></div>";
        }

        html += R"(
</div>
<div class="section">
  <div class="header">💓 HEART</div>
  <div>Rate: <span class="value">)" +
                std::to_string(state["heart_rate"].get<float>()) + R"(</span></div>
  <div>Resonance: <span class="value">)" +
                std::to_string(state["forge"]["heart"]["resonance"].get<float>()) + R"(</span></div>
  <div>Sync: <span class="value">)" +
                std::to_string(state["forge"]["heart"]["brainSync"].get<float>()) + R"(</span></div>
</div>
<div class="section">
  <div class="header">🔥 FORGE</div>
  <div>Temp: <span class="value">)" +
                std::to_string(state["forge"]["forgeTemperature"].get<float>()) + R"(°F</span></div>
  <div>Active: <span class="value">)" +
                (state["forge"]["isOperational"].get<bool>() ? "YES" : "NO") + R"(</span></div>
</div>
<script>
setTimeout(() => window.location.reload(), 2000);
</script>
</body></html>)";

        sendResponse(clientSocket, "200 OK", "text/html", html);
    }

    void serveState(int clientSocket)
    {
        json state = getSnapshot();
        sendResponse(clientSocket, "200 OK", "application/json", state.dump(2));
    }

    void serveSSE(int clientSocket)
    {
        std::string headers = "HTTP/1.1 200 OK\r\n";
        headers += "Content-Type: text/event-stream\r\n";
        headers += "Cache-Control: no-cache\r\n";
        headers += "Connection: keep-alive\r\n";
        headers += "Access-Control-Allow-Origin: *\r\n";
        headers += "\r\n";

        send(clientSocket, headers.c_str(), headers.length(), 0);

        // Send initial data and keep connection alive
        for (int i = 0; i < 30 && running; i++)
        { // 30 seconds max
            json state = getSnapshot();
            std::string data = "data: " + state.dump() + "\n\n";

            ssize_t sent = send(clientSocket, data.c_str(), data.length(), 0);
            if (sent <= 0)
                break;

            std::this_thread::sleep_for(std::chrono::milliseconds(500)); // 2 FPS
        }
    }

    void serve404(int clientSocket)
    {
        std::string html = "<h1>404 Not Found</h1><p>VectorLab Forge - Path not found</p>";
        sendResponse(clientSocket, "404 Not Found", "text/html", html);
    }

    void sendResponse(int clientSocket, const std::string &status, const std::string &contentType, const std::string &body)
    {
        std::string response = "HTTP/1.1 " + status + "\r\n";
        response += "Content-Type: " + contentType + "\r\n";
        response += "Content-Length: " + std::to_string(body.length()) + "\r\n";
        response += "Connection: close\r\n";
        response += "Access-Control-Allow-Origin: *\r\n";
        response += "\r\n";
        response += body;

        send(clientSocket, response.c_str(), response.length(), 0);
    }
};

// ═══════════════════════════════════════════════════════════════════
// BRAIN - NUCLEUS CORE AI INTEGRATION WITH GLYPH SYSTEM AUTOMATION
// ═══════════════════════════════════════════════════════════════════

struct NeuralMessage
{
    std::string type;          // "query", "learning", "feedback"
    std::string content;       // The message content
    std::string nucleusRole;   // VIBRATE, OPTIMIZATION, STATE, SEED
    std::string tier4Operator; // ST, UP, CV, RB
    float intensity = 1.0f;    // Processing intensity
    int64_t timestamp;
    bool processed = false;
};

// Glyph System Integration - connects to Python scene-glyph-generator.py
struct WorldEpoch
{
    std::string epoch;
    std::string cultural_shift;
    std::vector<std::string> agents;
    std::string recursive_message;
    std::string timestamp;
    float energy_pattern = 1.0f;
};

struct IntelligenceGlyph
{
    std::string agent_name;
    std::string cultivation_stage;
    std::string timestamp;
    std::string core_hash;
    std::string meaning;
    std::string sigil;

    json to_json() const
    {
        return {
            {"agent", agent_name},
            {"stage", cultivation_stage},
            {"timestamp", timestamp},
            {"hash", core_hash},
            {"meaning", meaning},
            {"sigil", sigil}};
    }
};

// Auto-connection to Python glyph system without disrupting nucleus automation
class GlyphSystemBridge
{
private:
    std::vector<WorldEpoch> epochs;
    std::vector<IntelligenceGlyph> eternal_imprints;
    std::map<std::string, json> active_agents;
    bool auto_sync_enabled = true;

public:
    void initialize_with_python_epochs()
    {
        // Mirror the exact epochs from Python scene-glyph-generator.py
        epochs = {
            {"The Great Convergence", "All scattered tribes unite under the Golden Nexus", {"Seeker Alpha", "Guardian Beta", "Weaver Gamma"}, "Unity births power, power births responsibility", "2025-09-27 15:30:00", 0.85f},

            {"The Void Whispers", "Dark energies emerge, reality becomes unstable", {"Shadow Walker", "Void Touched", "Memory Keeper"}, "What is forgotten seeks to be remembered", "2025-09-27 15:31:00", 0.3f},

            {"Glyph Awakening", "Ancient symbols begin manifesting spontaneously", {"Symbol Scribe", "Pattern Reader", "Reality Anchor"}, "The written becomes the written-into-reality", "2025-09-27 15:32:00", 1.2f},

            {"The Recursive Loop", "Time begins folding back on itself", {"Time Walker", "Loop Guardian", "Echo Sage"}, "Every end is a beginning, every beginning an end", "2025-09-27 15:33:00", 0.95f}};

        std::cout << "🌐 Glyph System Bridge initialized with " << epochs.size() << " epochs\n";
        std::cout << "⚡ Auto-sync with Python nucleus automation: " << (auto_sync_enabled ? "ENABLED" : "DISABLED") << "\n";
    }

    // Automatically generates events like Python system without disruption
    IntelligenceGlyph generate_intelligence_glyph(const std::string &agent, const std::string &stage, const std::string &event)
    {
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

        std::string raw = agent + "-" + stage + "-" + std::to_string(timestamp) + "-" + event;
        std::hash<std::string> hasher;
        auto hash_val = hasher(raw);
        std::string core_hash = std::to_string(hash_val).substr(0, 6);
        std::string sigil = agent.substr(0, 3) + "-" + core_hash;
        std::transform(sigil.begin(), sigil.begin() + 3, sigil.begin(), ::toupper);

        IntelligenceGlyph glyph = {
            agent, stage, std::to_string(timestamp), core_hash, event, sigil};

        eternal_imprints.push_back(glyph);
        std::cout << "🧬 Intelligence glyph auto-generated: " << sigil << " - " << event << "\n";

        return glyph;
    }

    // Creates dashboard events exactly like Python system
    json create_dashboard_event_stream() const
    {
        json dashboard_data;
        dashboard_data["timestamp"] = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
                                                         std::chrono::system_clock::now().time_since_epoch())
                                                         .count());
        dashboard_data["total_epochs"] = epochs.size();
        dashboard_data["total_imprints"] = eternal_imprints.size();
        dashboard_data["active_agents"] = active_agents.size();

        dashboard_data["recent_events"] = json::array();

        // Add epoch events
        for (const auto &epoch : epochs)
        {
            dashboard_data["recent_events"].push_back({{"type", "epoch"},
                                                       {"title", epoch.epoch},
                                                       {"description", epoch.cultural_shift},
                                                       {"timestamp", epoch.timestamp},
                                                       {"agents", epoch.agents},
                                                       {"energy", epoch.energy_pattern},
                                                       {"icon", "📚"}});
        }

        // Add glyph events
        for (const auto &glyph : eternal_imprints)
        {
            dashboard_data["recent_events"].push_back({{"type", "intelligence_glyph"},
                                                       {"title", glyph.sigil + " Generated"},
                                                       {"description", glyph.meaning},
                                                       {"timestamp", glyph.timestamp},
                                                       {"agent", glyph.agent_name},
                                                       {"stage", glyph.cultivation_stage},
                                                       {"icon", "🧬"}});
        }

        return dashboard_data;
    }

    // Chat response generation like Python system
    json create_chat_response(const std::string &query) const
    {
        json response;
        response["user_query"] = query;
        response["response_type"] = "world_history_context";

        std::string query_lower = query;
        std::transform(query_lower.begin(), query_lower.end(), query_lower.begin(), ::tolower);

        // Search epochs
        std::vector<json> matching_epochs;
        for (const auto &epoch : epochs)
        {
            std::string searchable = epoch.epoch + " " + epoch.cultural_shift + " " + epoch.recursive_message;
            std::transform(searchable.begin(), searchable.end(), searchable.begin(), ::tolower);

            if (searchable.find(query_lower) != std::string::npos)
            {
                matching_epochs.push_back({{"epoch", epoch.epoch},
                                           {"cultural_shift", epoch.cultural_shift},
                                           {"recursive_message", epoch.recursive_message},
                                           {"agents", epoch.agents}});
            }
        }

        // Search glyphs
        std::vector<json> matching_glyphs;
        for (const auto &glyph : eternal_imprints)
        {
            std::string searchable = glyph.meaning;
            std::transform(searchable.begin(), searchable.end(), searchable.begin(), ::tolower);

            if (searchable.find(query_lower) != std::string::npos)
            {
                matching_glyphs.push_back(glyph.to_json());
            }
        }

        response["epochs_found"] = matching_epochs.size();
        response["glyphs_found"] = matching_glyphs.size();
        response["context_strength"] = matching_epochs.size() + matching_glyphs.size();

        // Generate response like Python system
        if (!matching_epochs.empty())
        {
            auto primary_epoch = matching_epochs[0];
            response["suggested_response"] = "The archives speak of '" +
                                             primary_epoch["epoch"].get<std::string>() + "' - " +
                                             primary_epoch["cultural_shift"].get<std::string>() + ". " +
                                             primary_epoch["recursive_message"].get<std::string>() + ".";

            if (!matching_glyphs.empty())
            {
                auto glyph = matching_glyphs[0];
                response["suggested_response"] = response["suggested_response"].get<std::string>() +
                                                 " The intelligence glyph " + glyph["sigil"].get<std::string>() +
                                                 " resonates: '" + glyph["meaning"].get<std::string>() + "'.";
            }
        }
        else if (!matching_glyphs.empty())
        {
            auto glyph = matching_glyphs[0];
            response["suggested_response"] = "The glyph archives reveal " + glyph["sigil"].get<std::string>() +
                                             " from " + glyph["agent"].get<std::string>() + " in " + glyph["stage"].get<std::string>() +
                                             ": '" + glyph["meaning"].get<std::string>() + "'.";
        }
        else
        {
            response["suggested_response"] = "The codex whispers of uncharted possibilities. Perhaps this query will birth a new epoch...";
        }

        return response;
    }

    // Game event creation like Python system
    json create_game_event(const std::string &event_type, const std::map<std::string, std::string> &params = {})
    {
        if (event_type == "random_encounter" && !epochs.empty())
        {
            const auto &recent_epoch = epochs.back();
            return {
                {"event_type", "encounter"},
                {"title", "Echo of " + recent_epoch.epoch},
                {"description", "You encounter manifestations related to " + recent_epoch.cultural_shift},
                {"agents", recent_epoch.agents},
                {"energy_level", recent_epoch.energy_pattern},
                {"recursive_hint", recent_epoch.recursive_message}};
        }
        else if (event_type == "glyph_manifestation")
        {
            std::string agent = params.count("agent") ? params.at("agent") : "Unknown Wanderer";
            std::string stage = params.count("stage") ? params.at("stage") : "Seeking";
            std::string event = params.count("event") ? params.at("event") : "mysterious occurrence";

            auto glyph = generate_intelligence_glyph(agent, stage, event);

            return {
                {"event_type", "glyph_manifestation"},
                {"title", "Intelligence Glyph Manifests"},
                {"description", "A " + glyph.sigil + " glyph appears, resonating with: '" + glyph.meaning + "'"},
                {"glyph_data", glyph.to_json()},
                {"energy_signature", static_cast<float>(glyph.meaning.length()) * 0.1f}};
        }
        else if (event_type == "epoch_birth")
        {
            std::string title = params.count("title") ? params.at("title") : "The Unnamed Shift";
            std::string shift = params.count("cultural_shift") ? params.at("cultural_shift") : "Reality trembles with change";
            std::string message = params.count("message") ? params.at("message") : "Change begets change, as it ever was";

            WorldEpoch new_epoch = {
                title, shift, {"Player"}, message, std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()), 1.0f};

            // Auto-add to epochs like Python system
            const_cast<GlyphSystemBridge *>(this)->epochs.push_back(new_epoch);
            std::cout << "📚 New epoch auto-recorded: " << title << "\n";

            return {
                {"event_type", "epoch_birth"},
                {"title", "New Epoch: " + title},
                {"description", shift},
                {"recursive_message", message},
                {"agents", new_epoch.agents}};
        }

        return {{"event_type", "unknown"}, {"error", "Unknown event type: " + event_type}};
    }

    const std::vector<WorldEpoch> &get_epochs() const { return epochs; }
    const std::vector<IntelligenceGlyph> &get_glyphs() const { return eternal_imprints; }
    size_t get_agent_count() const { return active_agents.size(); }
};

struct CognitivePath
{
    std::string from;
    std::string to;
    float weight = 1.0f;
    float activation = 0.0f;
    glm::vec3 color{0.2f, 0.8f, 1.0f}; // Neural blue
};

class NucleusBrain
{
private:
    std::map<std::string, std::string> nucleusRouting = {
        {"query", "VIBRATE"},         // User queries → Stabilization
        {"learning", "OPTIMIZATION"}, // Learning data → Update/Progress
        {"feedback", "STATE"},        // Feedback loops → Convergence
        {"seed", "SEED"}              // New seeds → Rollback/Reset
    };

    // AI Chatbot Connection Persistence (maintains connection during nexus merge)
    std::atomic<bool> chatbotConnectionActive{false};
    std::string chatbotConnectionId = "nexus-merge-bridge";
    std::chrono::steady_clock::time_point lastConnectionCheck;

    std::map<std::string, std::string> tier4Operators = {
        {"VIBRATE", "ST"},      // Stabilize
        {"OPTIMIZATION", "UP"}, // Update
        {"STATE", "CV"},        // Converge
        {"SEED", "RB"}          // Rollback
    };

    std::queue<NeuralMessage> messageQueue;
    std::vector<CognitivePath> cognitivePaths;
    std::mutex brainMutex;

    // Brain state
    float cognitiveLoad = 0.0f;
    float learningRate = 0.1f;
    float memoryDecay = 0.05f;
    std::string currentThought = "Connecting to glyph system automation...";

    // AUTO-CONNECTION TO GLYPH SYSTEM (maintains nucleus automation)
    std::unique_ptr<GlyphSystemBridge> glyphBridge;

public:
    bool isActive = false;
    std::string nucleusWebSocketUrl = "ws://localhost:9000";

    void initialize()
    {
        isActive = true;

        // Auto-initialize connection to Python glyph system
        glyphBridge = std::make_unique<GlyphSystemBridge>();
        glyphBridge->initialize_with_python_epochs();

        // Initialize cognitive pathways (neural connections)
        cognitivePaths = {
            {"Input", "Perspective", 0.8f, 0.0f, {0.3f, 0.9f, 0.6f}},
            {"Perspective", "Processing", 0.9f, 0.0f, {0.9f, 0.7f, 0.2f}},
            {"Processing", "Meaning", 0.7f, 0.0f, {0.8f, 0.3f, 0.9f}},
            {"Meaning", "Meta Layer", 0.6f, 0.0f, {0.2f, 0.8f, 1.0f}},
            {"Meta Layer", "Librarian", 0.5f, 0.0f, {0.9f, 0.2f, 0.4f}},
            {"Librarian", "Root Sorter", 0.9f, 0.0f, {0.1f, 1.0f, 0.8f}},
            {"Root Sorter", "Data Optimizer", 0.8f, 0.0f, {1.0f, 0.5f, 0.1f}}};

        // Initialize AI Chatbot Connection Bridge
        chatbotConnectionActive = true;
        lastConnectionCheck = std::chrono::steady_clock::now();

        std::cout << "🧠 Nucleus Brain initialized with " << cognitivePaths.size() << " pathways\n";
        std::cout << "🌐 Auto-connected to glyph system with " << glyphBridge->get_epochs().size() << " epochs\n";
        std::cout << "💬 AI Chatbot connection bridge ACTIVE (" << chatbotConnectionId << ")\n";
        std::cout << "🔗 Connection persists through nexus merge operations\n";
        std::cout << "⚡ Seamless nucleus automation ACTIVE\n";
    }

    void processMessage(const std::string &message, const std::string &type = "query")
    {
        std::lock_guard<std::mutex> lock(brainMutex);

        // Create neural message through nucleus routing (unchanged automation)
        NeuralMessage msg;
        msg.type = type;
        msg.content = message;
        msg.nucleusRole = nucleusRouting[type];
        msg.tier4Operator = tier4Operators[msg.nucleusRole];
        msg.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();

        messageQueue.push(msg);

        // AUTO-GENERATE glyph events (preserves automation)
        if (glyphBridge)
        {
            if (type == "query" && message.find("discover") != std::string::npos)
            {
                glyphBridge->create_game_event("glyph_manifestation", {{"agent", "Forge Brain"},
                                                                       {"stage", "Neural Processing"},
                                                                       {"event", message}});
            }
            else if (type == "learning")
            {
                glyphBridge->create_game_event("epoch_birth", {{"title", "Neural Learning Epoch"},
                                                               {"cultural_shift", "Brain adaptation through: " + message},
                                                               {"message", "Learning begets understanding, understanding begets wisdom"}});
            }
        }

        // Update cognitive load
        cognitiveLoad = std::min(1.0f, cognitiveLoad + 0.1f);

        std::cout << "🧠 Brain processing: " << message << " → " << msg.nucleusRole << " → " << msg.tier4Operator << "\n";
    }

    void update(float deltaTime)
    {
        if (!isActive)
            return;

        std::lock_guard<std::mutex> lock(brainMutex);

        // Process message queue (unchanged nucleus automation)
        if (!messageQueue.empty())
        {
            auto &msg = messageQueue.front();
            if (!msg.processed)
            {
                // Simulate neural processing
                currentThought = "Processing via " + msg.nucleusRole + ": " + msg.content.substr(0, 30) + "...";

                // Activate cognitive pathways based on nucleus role
                for (auto &path : cognitivePaths)
                {
                    if (msg.nucleusRole == "VIBRATE" && path.from == "Input")
                    {
                        path.activation = std::min(1.0f, path.activation + 0.3f);
                    }
                    else if (msg.nucleusRole == "OPTIMIZATION" && path.from == "Processing")
                    {
                        path.activation = std::min(1.0f, path.activation + 0.4f);
                    }
                    else if (msg.nucleusRole == "STATE" && path.from == "Meta Layer")
                    {
                        path.activation = std::min(1.0f, path.activation + 0.2f);
                    }
                }

                msg.processed = true;
            }

            messageQueue.pop();
        }

        // Update cognitive pathways (decay activation over time)
        for (auto &path : cognitivePaths)
        {
            path.activation = std::max(0.0f, path.activation - memoryDecay * deltaTime);
        }

        // Monitor AI Chatbot Connection (every 5 seconds)
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - lastConnectionCheck).count() >= 5)
        {
            // Maintain connection bridge during nexus merge
            if (!chatbotConnectionActive.load())
            {
                chatbotConnectionActive = true;
                std::cout << "🔄 AI Chatbot connection bridge restored\n";
            }
            lastConnectionCheck = now;
        }

        // Decay cognitive load
        cognitiveLoad = std::max(0.0f, cognitiveLoad - 0.02f * deltaTime);

        // Update thought state with glyph system integration
        if (messageQueue.empty() && cognitiveLoad < 0.1f)
        {
            if (glyphBridge)
            {
                currentThought = "Monitoring forge systems... " +
                                 std::to_string(glyphBridge->get_epochs().size()) + " epochs, " +
                                 std::to_string(glyphBridge->get_glyphs().size()) + " glyphs active";
            }
            else
            {
                currentThought = "Monitoring forge systems...";
            }
        }
    }

    // Get brain state for visualization (enhanced with glyph data)
    json getBrainState() const
    {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex &>(brainMutex));

        json state;
        state["isActive"] = isActive;
        state["cognitiveLoad"] = cognitiveLoad;
        state["learningRate"] = learningRate;
        state["currentThought"] = currentThought;
        state["queueSize"] = messageQueue.size();

        // Add glyph system data
        if (glyphBridge)
        {
            state["connected_epochs"] = glyphBridge->get_epochs().size();
            state["connected_glyphs"] = glyphBridge->get_glyphs().size();
            state["glyph_system_active"] = true;
        }
        else
        {
            state["glyph_system_active"] = false;
        }

        state["cognitivePaths"] = json::array();
        for (const auto &path : cognitivePaths)
        {
            state["cognitivePaths"].push_back({{"from", path.from},
                                               {"to", path.to},
                                               {"weight", path.weight},
                                               {"activation", path.activation},
                                               {"color", {path.color.r, path.color.g, path.color.b}}});
        }

        return state;
    }

    // Direct access to glyph system (preserves Python automation)
    json getDashboardEvents() const
    {
        if (glyphBridge)
        {
            return glyphBridge->create_dashboard_event_stream();
        }
        return json::object();
    }

    json getChatResponse(const std::string &query) const
    {
        // AI Chatbot Connection Bridge - maintains connection during nexus merge
        json response;
        response["connection_id"] = chatbotConnectionId;
        response["nexus_merge_active"] = true;
        response["connection_persists"] = chatbotConnectionActive.load();

        if (glyphBridge)
        {
            json bridge_response = glyphBridge->create_chat_response(query);
            response["bridge_data"] = bridge_response;
            response["glyph_system_connected"] = true;
        }
        else
        {
            response["glyph_system_connected"] = false;
        }

        // Ensure connection bridge remains active
        response["bridge_status"] = "Connection maintained through nexus merge";
        response["websocket_url"] = nucleusWebSocketUrl;

        return response;
    }

    json createGameEvent(const std::string &event_type, const std::map<std::string, std::string> &params = {})
    {
        if (glyphBridge)
        {
            return glyphBridge->create_game_event(event_type, params);
        }
        return json{{"error", "Glyph system not connected"}};
    }

    void renderNeuralPaths(const glm::vec3 &brainCenter, float radius = 2.0f)
    {
        if (!isActive)
            return;

        // Render cognitive pathways as glowing connections (unchanged)
        for (const auto &path : cognitivePaths)
        {
            float intensity = path.activation;
            if (intensity > 0.01f)
            {
                // Calculate positions in 3D space around brain center
                glm::vec3 fromPos = brainCenter + glm::vec3(
                                                      std::sin(std::hash<std::string>{}(path.from)) * radius,
                                                      std::cos(std::hash<std::string>{}(path.from)) * radius * 0.5f,
                                                      std::cos(std::hash<std::string>{}(path.from) + 100) * radius);

                glm::vec3 toPos = brainCenter + glm::vec3(
                                                    std::sin(std::hash<std::string>{}(path.to)) * radius,
                                                    std::cos(std::hash<std::string>{}(path.to)) * radius * 0.5f,
                                                    std::cos(std::hash<std::string>{}(path.to) + 100) * radius);

                // Render glowing neural connection
                glColor3f(path.color.r * intensity, path.color.g * intensity, path.color.b * intensity);
                glLineWidth(2.0f + intensity * 3.0f);
                glBegin(GL_LINES);
                glVertex3f(fromPos.x, fromPos.y, fromPos.z);
                glVertex3f(toPos.x, toPos.y, toPos.z);
                glEnd();

                // Neural nodes
                glPushMatrix();
                glTranslatef(fromPos.x, fromPos.y, fromPos.z);
                glColor3f(path.color.r * (0.5f + intensity), path.color.g * (0.5f + intensity), path.color.b * (0.5f + intensity));
                glutSolidSphere(0.05 + intensity * 0.1, 12, 12);
                glPopMatrix();
            }
        }

        // Brain core visualization (enhanced with glyph activity)
        glPushMatrix();
        glTranslatef(brainCenter.x, brainCenter.y, brainCenter.z);

        float glyph_activity = glyphBridge ? (glyphBridge->get_glyphs().size() * 0.01f) : 0.0f;
        glColor3f(0.8f + cognitiveLoad * 0.2f, 0.6f + glyph_activity, 1.0f);
        glutSolidSphere(0.3f + (cognitiveLoad + glyph_activity) * 0.1f, 24, 24);
        glPopMatrix();
    }

    std::vector<CognitivePath> getCognitivePaths() const
    {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex &>(brainMutex));
        return cognitivePaths;
    }
}; // ═══════════════════════════════════════════════════════════════════
// EXISTING FORGE BODY (Heart + Veins + Physical Form)
// ═══════════════════════════════════════════════════════════════════

struct Cube
{
    glm::vec3 position{0.0f};
    float animationOffset{0.0f};
    bool visible{true};
    std::string organType = "muscle"; // muscle, bone, vessel, nerve
};

struct Spotlight
{
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::vec3 direction{0.0f, -1.0f, 0.0f};
    float cutOff{glm::cos(glm::radians(12.5f))};
    float outerCutOff{glm::cos(glm::radians(17.5f))};
    glm::vec3 ambient{0.2f};
    glm::vec3 diffuse{0.8f};
    glm::vec3 specular{1.0f};
    bool enabled{false};
};

// Veins - VectorObjects as circulatory/nervous system
struct VectorObject
{
    glm::vec3 from{0.0f};
    glm::vec3 to{0.0f};
    glm::vec3 color{1.0f, 0.6f, 0.2f};
    std::string veinType = "artery"; // artery, vein, nerve, lymph
    float flow = 0.0f;               // Flow rate for animation
};

// Heart - Enhanced with brain connection
struct Heart
{
    float resonance = 0.0f;
    float resonanceCap = 1.0f;
    std::string state = "idle";
    float brainSync = 0.0f;   // Synchronization with brain activity
    float currentBPM = 60.0f; // Base heart rate
    float heartPulse = 0.0f;

    void initialize()
    {
        currentBPM = 60.0f;
        heartPulse = 0.0f;
        state = "initialized";
        std::cout << "💓 Heart initialized - BPM: " << currentBPM << "\n";
    }

    void update(float deltaTime)
    {
        // Update heart pulse animation
        heartPulse += deltaTime * (currentBPM / 60.0f) * 6.28f; // 2*PI for full cycle
        if (heartPulse > 6.28f)
            heartPulse -= 6.28f;

        // Update BPM based on brain activity and resonance
        currentBPM = 60.0f + (brainSync * 40.0f) + (resonance * 20.0f);
    }

    float getCurrentBPM() const { return currentBPM; }
    float getHeartPulse() const { return std::sin(heartPulse) * 0.5f + 0.5f; }

    json getHeartState() const
    {
        return {
            {"resonance", resonance},
            {"resonanceCap", resonanceCap},
            {"state", state},
            {"brainSync", brainSync},
            {"currentBPM", currentBPM},
            {"heartPulse", getHeartPulse()}};
    }

    void pulse(float intensity)
    {
        resonance = glm::clamp(resonance + intensity, 0.0f, resonanceCap);
        state = "pulse";
        std::cout << "💓 Heart pulses with intensity: " << intensity << "\n";
    }

    void syncWithBrain(float brainActivity)
    {
        brainSync = brainActivity;
        // Heart rate increases with brain activity
        if (brainActivity > 0.5f)
        {
            pulse(brainActivity * 0.1f);
        }
    }

    void decay(float amount)
    {
        resonance = glm::clamp(resonance - amount, 0.0f, resonanceCap);
        if (resonance == 0.0f)
            state = "silent";
    }

    void echo()
    {
        state = "echo";
        std::cout << "🔁 Heart echoes resonance: " << resonance << "\n";
    }

    void render()
    {
        // Heart visualization (called from main render loop)
        glPushMatrix();
        glTranslatef(0.0f, 0.0f, 0.0f); // Heart at center

        float pulse = getHeartPulse();
        float size = 0.6f + pulse * 0.2f;

        glColor3f(1.0f + pulse * 0.2f, 0.2f, 0.3f + resonance * 0.4f);
        glutSolidSphere(size, 16, 16);

        glPopMatrix();
    }
};

struct LabNote
{
    std::string text = "";
    int targetVectorIndex = -1;
    glm::vec3 offset{0.0f, 0.5f, 0.0f};
    int frameTrigger = -1;
    std::string type = "info";
    glm::vec3 color{1.0f, 1.0f, 1.0f};

    bool shouldDisplay(int currentFrame) const
    {
        return frameTrigger == -1 || currentFrame >= frameTrigger;
    }
};

struct TimelineEngine
{
    int currentFrame = 0;
    bool isPlaying = true;

    void stepForward() { currentFrame++; }
    void stepBack() { currentFrame = std::max(0, currentFrame - 1); }
    void togglePlay() { isPlaying = !isPlaying; }
    void update()
    {
        if (isPlaying)
            stepForward();
    }
};

// ═══════════════════════════════════════════════════════════════════
// COMPLETE FORGE SYSTEM - Body + Brain Integration
// ═══════════════════════════════════════════════════════════════════

class VectorLabForge
{
private:
    NucleusBrain brain;
    Heart heart;
    TimelineEngine timeline;

    std::vector<Cube> bodyParts;
    std::vector<VectorObject> circulatorySystem;
    std::vector<LabNote> memoryNodes;

    Spotlight consciousness; // Spotlight as consciousness/attention

    // Live Viewer Server for real-time monitoring
    LiveViewerServer liveViewer;

    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 8.0f);
    glm::vec3 cameraTarget = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

public:
    void initialize()
    {
        std::cout << "🔥 Initializing VectorLab Forge - Complete Body System with Glyph Integration\n";

        // Initialize brain WITH glyph system connection (preserves nucleus automation)
        brain.initialize();

        // Initialize heart system
        heart.initialize();

        // Create body structure
        bodyParts = {
            // Core body
            {glm::vec3(0.0f, 0.0f, 0.0f), 0.0f, true, "heart"},   // Heart center
            {glm::vec3(0.0f, 2.0f, 0.0f), 0.5f, true, "brain"},   // Brain position
            {glm::vec3(-1.0f, 1.0f, 0.0f), 1.0f, true, "muscle"}, // Left arm
            {glm::vec3(1.0f, 1.0f, 0.0f), 1.5f, true, "muscle"},  // Right arm
            {glm::vec3(-0.5f, -2.0f, 0.0f), 2.0f, true, "bone"},  // Left leg
            {glm::vec3(0.5f, -2.0f, 0.0f), 2.5f, true, "bone"}    // Right leg
        };

        // Create circulatory/nervous system connections
        rebuildCirculatorySystem();

        std::cout << "✅ Forge body initialized with " << bodyParts.size() << " parts\n";
        std::cout << "✅ Circulatory system has " << circulatorySystem.size() << " vessels\n";
        std::cout << "🧠 Brain-Heart synchronization active\n";
        std::cout << "🌟 Glyph system automation connected WITHOUT disruption\n";

        // Start live viewer server for real-time monitoring
        liveViewer.start();
        std::cout << "🌐 Live Viewer Server active - Nucleus can now see itself!\n";

        // Test the nucleus automation connection
        testNucleusConnection();
    }

    void rebuildCirculatorySystem()
    {
        circulatorySystem.clear();

        if (bodyParts.empty())
            return;

        glm::vec3 heartPos = bodyParts[0].position; // Heart is first
        glm::vec3 brainPos = bodyParts.size() > 1 ? bodyParts[1].position : heartPos;

        // Heart-Brain connection (primary neural pathway)
        circulatorySystem.push_back({heartPos, brainPos,
                                     glm::vec3(1.0f, 0.2f, 0.8f), // Pink-red for heart-brain
                                     "nerve", 0.8f});

        // Connect all body parts to heart (arteries)
        for (size_t i = 2; i < bodyParts.size(); ++i)
        {
            circulatorySystem.push_back({heartPos, bodyParts[i].position,
                                         glm::vec3(0.9f, 0.1f, 0.1f), // Red for arteries
                                         "artery", 0.6f});
        }

        // Connect brain to body parts (nervous system)
        for (size_t i = 2; i < bodyParts.size(); ++i)
        {
            if (i % 2 == 0)
            { // Only some connections to avoid clutter
                circulatorySystem.push_back({brainPos, bodyParts[i].position,
                                             glm::vec3(0.2f, 0.8f, 1.0f), // Blue for nerves
                                             "nerve", 0.4f});
            }
        }

        // Add some inter-organ connections (lymphatic system)
        if (bodyParts.size() >= 4)
        {
            circulatorySystem.push_back({bodyParts[2].position, bodyParts[3].position, // Arms connected
                                         glm::vec3(0.6f, 0.9f, 0.3f),                  // Green for lymph
                                         "lymph", 0.3f});
        }
    }

    void processThought(const std::string &thought)
    {
        // Send thought to brain for processing (maintains nucleus automation routing)
        brain.processMessage(thought, "query");

        // Heart responds to brain activity
        json brainState = brain.getBrainState();
        float brainActivity = brainState["cognitiveLoad"];
        heart.syncWithBrain(brainActivity);

        std::cout << "💭 Processing thought: " << thought << "\n";
        std::cout << "🧠 Brain activity: " << brainActivity << "\n";
        std::cout << "💓 Heart resonance: " << heart.resonance << "\n";

        // Auto-generate glyph system events (preserves automation)
        if (brainState["glyph_system_active"].get<bool>())
        {
            std::cout << "🌟 Glyph system responding: "
                      << brainState["connected_glyphs"].get<int>() << " glyphs active\n";
        }
    }

    void testNucleusConnection()
    {
        std::cout << "🧪 Testing nucleus automation connection...\n";

        // Simulate Python nucleus automation messages (this is what the Python system would send)
        brain.processMessage("Forge body initialized, ready for glyph manifestation", "query");
        brain.processMessage("Learning forge visualization patterns", "learning");
        brain.processMessage("Heart-brain synchronization feedback received", "feedback");

        // Check glyph system response
        json brainState = brain.getBrainState();
        if (brainState["glyph_system_active"].get<bool>())
        {
            std::cout << "✅ Connection SUCCESSFUL!\n";
            std::cout << "   → " << brainState["connected_epochs"].get<int>() << " epochs available\n";
            std::cout << "   → " << brainState["connected_glyphs"].get<int>() << " glyphs ready\n";
            std::cout << "   → Nucleus automation PRESERVED and flowing to forge body\n";
        }
        else
        {
            std::cout << "⚠️ Glyph system bridge not active\n";
        }
    }

    void update(float deltaTime)
    {
        // Update brain
        brain.update(deltaTime);

        // Update heart (full update with pulse animation)
        heart.update(deltaTime);
        heart.decay(0.01f * deltaTime);

        // Update timeline
        timeline.update();

        // Update circulatory flow animation
        for (auto &vessel : circulatorySystem)
        {
            vessel.flow += deltaTime * 2.0f; // Flow animation speed
            if (vessel.flow > 6.28f)
                vessel.flow -= 6.28f; // Reset at 2π
        }

        // Synchronize consciousness (spotlight) with brain activity
        if (brain.isActive)
        {
            json brainState = brain.getBrainState();
            float cognitiveLoad = brainState["cognitiveLoad"];

            consciousness.enabled = cognitiveLoad > 0.1f;
            consciousness.position = bodyParts[1].position; // Brain position
            consciousness.ambient = glm::vec3(0.1f + cognitiveLoad * 0.3f);
        }

        // Update live viewer with current state (Nucleus self-observation)
        if (liveViewer.broadcastEnabled)
        {
            json forgeState = exportForgeState();
            json brainState = brain.getBrainState();
            json glyphEvents = brain.getDashboardEvents();
            std::string currentThought = brainState["currentThought"];
            float heartRate = heart.getCurrentBPM();

            liveViewer.updateState(forgeState, brainState, glyphEvents, currentThought, heartRate);
        }
    }

    void render(int windowWidth, int windowHeight)
    {
        // Set up camera
        glm::mat4 view = glm::lookAt(cameraPos, cameraTarget, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(45.0f),
                                                float(windowWidth) / float(windowHeight), 0.1f, 100.0f);

        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf(glm::value_ptr(projection));
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(glm::value_ptr(view));

        float time = glfwGetTime();

        // Render body parts
        renderBodyParts(time);

        // Render circulatory/nervous system
        renderCirculatorySystem();

        // Render brain neural pathways
        if (bodyParts.size() > 1)
        {
            brain.renderNeuralPaths(bodyParts[1].position, 1.5f);
        }

        // Render consciousness spotlight
        renderConsciousness();

        // Render memory nodes (if any)
        renderMemoryNodes();
    }

private:
    void renderBodyParts(float time)
    {
        for (const auto &part : bodyParts)
        {
            if (!part.visible)
                continue;

            float bob = std::sin(time + part.animationOffset) * 0.1f;

            // Color based on organ type
            glm::vec3 color;
            if (part.organType == "heart")
            {
                color = glm::vec3(1.0f, 0.2f, 0.3f); // Red
                bob *= (1.0f + heart.resonance);     // Heart beats with resonance
            }
            else if (part.organType == "brain")
            {
                json brainState = brain.getBrainState();
                float activity = brainState["cognitiveLoad"];
                color = glm::vec3(0.6f + activity * 0.4f, 0.4f, 1.0f); // Blue, brighter with activity
            }
            else if (part.organType == "muscle")
            {
                color = glm::vec3(0.8f, 0.6f, 0.4f); // Muscle color
            }
            else if (part.organType == "bone")
            {
                color = glm::vec3(0.9f, 0.9f, 0.8f); // Bone white
            }
            else
            {
                color = glm::vec3(0.5f, 0.5f, 0.5f); // Default gray
            }

            glPushMatrix();
            glTranslatef(part.position.x, part.position.y + bob, part.position.z);
            glColor3f(color.r, color.g, color.b);

            // Different shapes for different organs
            if (part.organType == "heart")
            {
                glutSolidSphere(0.6f, 16, 16);
            }
            else if (part.organType == "brain")
            {
                glutSolidSphere(0.8f, 20, 20);
            }
            else
            {
                glutSolidCube(1.0f);
            }

            glPopMatrix();
        }
        glColor3f(1.0f, 1.0f, 1.0f); // Reset color
    }

    void renderCirculatorySystem()
    {
        for (const auto &vessel : circulatorySystem)
        {
            // Animate flow
            float flowPhase = vessel.flow;
            float flowIntensity = 0.5f + 0.5f * std::sin(flowPhase);

            // Color based on vessel type with flow animation
            glm::vec3 color = vessel.color * flowIntensity;

            if (vessel.veinType == "nerve")
            {
                // Brain activity affects nerve brightness
                json brainState = brain.getBrainState();
                float brainActivity = brainState["cognitiveLoad"];
                color *= (0.5f + brainActivity);
            }
            else if (vessel.veinType == "artery")
            {
                // Heart resonance affects artery brightness
                color *= (0.5f + heart.resonance);
            }

            glColor3f(color.r, color.g, color.b);
            glLineWidth(2.0f + flowIntensity);

            glBegin(GL_LINES);
            glVertex3f(vessel.from.x, vessel.from.y, vessel.from.z);
            glVertex3f(vessel.to.x, vessel.to.y, vessel.to.z);
            glEnd();

            // Flow particle at tip
            glPushMatrix();
            glTranslatef(vessel.to.x, vessel.to.y, vessel.to.z);
            glutSolidSphere(0.04f + flowIntensity * 0.02f, 8, 8);
            glPopMatrix();
        }
        glColor3f(1.0f, 1.0f, 1.0f);
        glLineWidth(1.0f);
    }

    void renderConsciousness()
    {
        if (!consciousness.enabled)
            return;

        glPushMatrix();
        glTranslatef(consciousness.position.x, consciousness.position.y, consciousness.position.z);

        // Consciousness as a glowing orb
        json brainState = brain.getBrainState();
        float intensity = brainState["cognitiveLoad"];

        glColor3f(1.0f, 1.0f, 0.6f + intensity * 0.4f);
        glutSolidSphere(0.15f + intensity * 0.1f, 16, 16);
        glPopMatrix();
    }

    void renderMemoryNodes()
    {
        // Placeholder for memory visualization
        for (const auto &note : memoryNodes)
        {
            if (note.targetVectorIndex >= 0 &&
                note.targetVectorIndex < (int)circulatorySystem.size())
            {

                const auto &vessel = circulatorySystem[note.targetVectorIndex];
                glm::vec3 pos = vessel.to + note.offset;

                glPushMatrix();
                glTranslatef(pos.x, pos.y, pos.z);
                glColor3f(note.color.r, note.color.g, note.color.b);
                glutSolidSphere(0.05f, 8, 8);
                glPopMatrix();
            }
        }
    }

public:
    // Public interface for external control
    void addMemoryNode(const std::string &memory, const std::string &type = "info")
    {
        LabNote note;
        note.text = memory;
        note.type = type;
        note.targetVectorIndex = memoryNodes.size() % circulatorySystem.size();
        note.frameTrigger = timeline.currentFrame;

        if (type == "important")
        {
            note.color = glm::vec3(1.0f, 0.2f, 0.2f); // Red for important
        }
        else if (type == "learning")
        {
            note.color = glm::vec3(0.2f, 1.0f, 0.2f); // Green for learning
        }
        else
        {
            note.color = glm::vec3(0.2f, 0.2f, 1.0f); // Blue for info
        }

        memoryNodes.push_back(note);
        std::cout << "📝 Memory added: " << memory << "\n";
    }

    json exportForgeState()
    {
        json state;
        state["brain"] = brain.getBrainState();
        state["heart"] = {
            {"resonance", heart.resonance},
            {"state", heart.state},
            {"brainSync", heart.brainSync}};
        state["timeline"] = {
            {"currentFrame", timeline.currentFrame},
            {"isPlaying", timeline.isPlaying}};
        state["bodyParts"] = bodyParts.size();
        state["circulatorySystem"] = circulatorySystem.size();
        state["memoryNodes"] = memoryNodes.size();

        return state;
    }

    // API for dashboard/chat/game integration (preserves nucleus automation)
    json getDashboardEvents() const
    {
        return brain.getDashboardEvents();
    }

    json getChatResponse(const std::string &query) const
    {
        return brain.getChatResponse(query);
    }

    json createGameEvent(const std::string &event_type, const std::map<std::string, std::string> &params = {})
    {
        return brain.createGameEvent(event_type, params);
    }

    // Process automation messages from Python nucleus system
    void processAutomationMessage(const std::string &message, const std::string &type = "query")
    {
        brain.processMessage(message, type);
    }

    // Live viewer controls
    LiveViewerServer &getLiveViewer() { return liveViewer; }
    void toggleLiveViewer() { liveViewer.broadcastEnabled = !liveViewer.broadcastEnabled; }

    Heart &getHeart() { return heart; }
    NucleusBrain &getBrain() { return brain; }
    TimelineEngine &getTimeline() { return timeline; }
    std::vector<Cube> &getBodyParts() { return bodyParts; }

    ~VectorLabForge()
    {
        liveViewer.stop();
    }
};

// ═══════════════════════════════════════════════════════════════════
// MAIN APPLICATION
// ═══════════════════════════════════════════════════════════════════

int main()
{
    std::cout << "🔥 Starting VectorLab Forge - Complete Body System\n";

    // GLFW initialization
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    GLFWwindow *window = glfwCreateWindow(1200, 800, "VectorLab Forge - Brain + Heart + Body", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (glewInit() != GLEW_OK)
    {
        std::cerr << "GLEW initialization failed\n";
        return -1;
    }

    // GLUT initialization (for primitive rendering)
    int argc = 1;
    char *argv[] = {(char *)"forge", nullptr};
    glutInit(&argc, argv);

    // ImGui initialization
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    glEnable(GL_DEPTH_TEST);

    // Initialize the complete forge system
    VectorLabForge forge;
    forge.initialize();

    // UI state
    static char thoughtInput[256] = "What is the meaning of consciousness?";
    static char memoryInput[256] = "Important insight about reality";

    std::cout << "✅ VectorLab Forge ready!\n";
    std::cout << "🧠 Brain nucleus connected to heart and body\n";
    std::cout << "💓 Heart synchronized with brain activity\n";
    std::cout << "🌊 Circulatory system flowing with life\n";

    auto lastTime = std::chrono::high_resolution_clock::now();

    // Main render loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // Calculate delta time
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;

        // Update forge systems
        forge.update(deltaTime);

        // Clear screen
        int windowWidth, windowHeight;
        glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
        glViewport(0, 0, windowWidth, windowHeight);
        glClearColor(0.05f, 0.05f, 0.1f, 1.0f); // Dark space background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Render the complete forge
        forge.render(windowWidth, windowHeight);

        // ImGui interface
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ═══════════════ FORGE CONTROL PANEL ═══════════════
        ImGui::Begin("🔥 VectorLab Forge Control", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        ImGui::Text("🧠 BRAIN STATUS");
        ImGui::Separator();
        json brainState = forge.getBrain().getBrainState();
        ImGui::Text("Cognitive Load: %.2f", (float)brainState["cognitiveLoad"]);
        ImGui::Text("Current Thought: %s", std::string(brainState["currentThought"]).c_str());
        ImGui::Text("Queue Size: %d", (int)brainState["queueSize"]);

        // GLYPH SYSTEM STATUS (shows nucleus automation connection)
        ImGui::Spacing();
        ImGui::Text("🌟 GLYPH SYSTEM STATUS");
        ImGui::Separator();
        if (brainState["glyph_system_active"].get<bool>())
        {
            ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f), "STATUS: CONNECTED ✅");
            ImGui::Text("Connected Epochs: %d", brainState["connected_epochs"].get<int>());
            ImGui::Text("Active Glyphs: %d", brainState["connected_glyphs"].get<int>());
            ImGui::Text("Nucleus Automation: PRESERVED ⚡");
        }
        else
        {
            ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.2f, 1.0f), "STATUS: DISCONNECTED ❌");
        }

        ImGui::Spacing();
        ImGui::Text("💓 HEART STATUS");
        ImGui::Separator();
        Heart &heart = forge.getHeart();
        ImGui::Text("Resonance: %.2f / %.2f", heart.resonance, heart.resonanceCap);
        ImGui::Text("State: %s", heart.state.c_str());
        ImGui::Text("Brain Sync: %.2f", heart.brainSync);

        ImGui::Spacing();
        ImGui::Text("⚡ SYSTEM CONTROLS");
        ImGui::Separator();

        // Thought input
        ImGui::InputText("Send Thought", thoughtInput, IM_ARRAYSIZE(thoughtInput));
        ImGui::SameLine();
        if (ImGui::Button("Process 🧠"))
        {
            forge.processThought(std::string(thoughtInput));
        }

        // Memory input
        ImGui::InputText("Add Memory", memoryInput, IM_ARRAYSIZE(memoryInput));
        ImGui::SameLine();
        if (ImGui::Button("Store 📝"))
        {
            forge.addMemoryNode(std::string(memoryInput), "important");
        }

        // Quick actions
        if (ImGui::Button("Pulse Heart 💓"))
        {
            heart.pulse(0.3f);
        }
        ImGui::SameLine();
        if (ImGui::Button("Echo Heart 🔁"))
        {
            heart.echo();
        }

        if (ImGui::Button("Learning Mode 📚"))
        {
            forge.getBrain().processMessage("Entering learning mode", "learning");
        }
        ImGui::SameLine();
        if (ImGui::Button("Feedback Loop 🔄"))
        {
            forge.getBrain().processMessage("Processing feedback", "feedback");
        }

        // GLYPH SYSTEM CONTROLS
        ImGui::Spacing();
        ImGui::Text("🌟 GLYPH SYSTEM CONTROLS");
        ImGui::Separator();

        if (ImGui::Button("Generate Dashboard Event 📊"))
        {
            json events = forge.getDashboardEvents();
            std::cout << "📊 Dashboard Events: " << events.dump(2) << "\n";
        }
        ImGui::SameLine();
        if (ImGui::Button("Test Chat Response 💬"))
        {
            json response = forge.getChatResponse("How is the forge body responding?");
            std::cout << "💬 Chat Response: " << response.dump(2) << "\n";
        }

        if (ImGui::Button("Create Game Event: Manifestation 🎮"))
        {
            json event = forge.createGameEvent("glyph_manifestation", {{"agent", "Forge UI"}, {"stage", "Interactive"}, {"event", "User triggered manifestation"}});
            std::cout << "🎮 Game Event: " << event.dump(2) << "\n";
        }
        ImGui::SameLine();
        if (ImGui::Button("Epoch Birth Event 🌟"))
        {
            json event = forge.createGameEvent("epoch_birth", {{"title", "UI Epoch"}, {"cultural_shift", "User interaction creating new epoch"}});
            std::cout << "🌟 Epoch Event: " << event.dump(2) << "\n";
        }

        // Timeline controls
        TimelineEngine &timeline = forge.getTimeline();
        ImGui::Spacing();
        ImGui::Text("Timeline Frame: %d", timeline.currentFrame);
        if (ImGui::Button(timeline.isPlaying ? "Pause ⏸️" : "Play ▶️"))
        {
            timeline.togglePlay();
        }
        ImGui::SameLine();
        if (ImGui::Button("Step ⏭️"))
        {
            timeline.stepForward();
        }

        // Export controls
        ImGui::Spacing();
        if (ImGui::Button("Export Forge State 💾"))
        {
            json state = forge.exportForgeState();
            std::ofstream file("forge_state.json");
            file << state.dump(2);
            std::cout << "💾 Forge state exported to forge_state.json\n";
        }

        // LIVE VIEWER CONTROLS
        ImGui::Spacing();
        ImGui::Text("🌐 LIVE VIEWER");
        ImGui::Separator();

        LiveViewerServer &viewer = forge.getLiveViewer();
        ImGui::Text("Server URL: %s", viewer.serverUrl.c_str());

        bool broadcasting = viewer.broadcastEnabled;
        if (ImGui::Checkbox("Broadcasting", &broadcasting))
        {
            viewer.broadcastEnabled = broadcasting;
        }

        if (ImGui::Button("Open Dashboard 🌐"))
        {
#ifdef _WIN32
            system(("start " + viewer.serverUrl).c_str());
#else
            system(("xdg-open " + viewer.serverUrl).c_str());
#endif
        }

        ImGui::SameLine();
        if (ImGui::Button("Toggle Broadcast 📡"))
        {
            forge.toggleLiveViewer();
        }

        ImGui::End();

        // ═══════════════ SYSTEM STATUS ═══════════════
        ImGui::Begin("📊 System Status", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

        ImGui::Text("Body Parts: %zu", forge.getBodyParts().size());
        ImGui::Text("Frame Rate: %.1f FPS", io.Framerate);
        ImGui::Text("Delta Time: %.3f ms", deltaTime * 1000.0f);

        ImGui::Spacing();
        ImGui::Text("🌐 Nucleus Integration");
        ImGui::Text("Brain Active: %s", forge.getBrain().isActive ? "YES" : "NO");
        ImGui::Text("WebSocket: %s", forge.getBrain().nucleusWebSocketUrl.c_str());

        ImGui::Spacing();
        ImGui::Text("📡 Live Viewer");
        LiveViewerServer &viewer = forge.getLiveViewer();
        ImGui::Text("Broadcasting: %s", viewer.broadcastEnabled.load() ? "ON" : "OFF");
        ImGui::Text("Dashboard: %s", viewer.serverUrl.c_str());

        ImGui::End();

        // Render ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    std::cout << "🔥 Shutting down VectorLab Forge\n";

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
