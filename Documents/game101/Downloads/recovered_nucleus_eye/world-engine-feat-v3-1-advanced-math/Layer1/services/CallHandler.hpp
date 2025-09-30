#ifndef CALL_HANDLER_HPP
#define CALL_HANDLER_HPP

#include <string>
#include <functional>
#include <map>
#include <vector>
#include <memory>
#include <mutex>
#include <future>

namespace AutomationNucleus {

enum class CallType {
    SYNC,
    ASYNC,
    DEFERRED,
    PRIORITY,
    BROADCAST
};

enum class CallStatus {
    PENDING,
    PROCESSING,
    COMPLETED,
    FAILED,
    TIMEOUT
};

struct CallContext {
    std::string callId;
    CallType type;
    std::string source;
    std::string target;
    std::string payload;
    int priority;
    long long timestamp;
    CallStatus status;
    std::string result;
    std::string error;
};

class CallHandler {
private:
    std::map<std::string, std::function<std::string(const std::string&)>> handlers_;
    std::vector<std::shared_ptr<CallContext>> callQueue_;
    std::mutex queueMutex_;
    bool isRunning_;
    std::thread processingThread_;

    void processCallQueue();
    std::string generateCallId();

public:
    CallHandler();
    ~CallHandler();

    // Core call handling methods
    void registerHandler(const std::string& callType,
                        std::function<std::string(const std::string&)> handler);

    std::string makeCall(const std::string& callType,
                        const std::string& payload,
                        CallType type = CallType::SYNC);

    std::future<std::string> makeAsyncCall(const std::string& callType,
                                          const std::string& payload);

    void broadcastCall(const std::string& callType, const std::string& payload);

    // Nova Omega integration methods
    std::string handleNovaOmegaCall(const std::string& operation,
                                   const std::string& data);

    // UDR (Universal Data Relay) methods
    void initializeUDR();
    std::string processUDRCall(const std::string& udrCommand,
                              const std::string& parameters);

    // Queue management
    std::vector<std::shared_ptr<CallContext>> getPendingCalls();
    void clearCompletedCalls();
    CallStatus getCallStatus(const std::string& callId);

    // Control methods
    void start();
    void stop();
    bool isRunning() const { return isRunning_; }
};

} // namespace AutomationNucleus

#endif // CALL_HANDLER_HPP
