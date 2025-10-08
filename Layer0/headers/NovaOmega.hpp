#ifndef NOVA_OMEGA_HPP
#define NOVA_OMEGA_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <functional>

namespace AutomationNucleus {

enum class OmegaOperation {
    INITIALIZE,
    PROCESS,
    TRANSFORM,
    OPTIMIZE,
    SYNCHRONIZE,
    TERMINATE
};

struct NovaContext {
    std::string contextId;
    std::string sourceDomain;
    std::string targetDomain;
    std::map<std::string, std::string> parameters;
    long long createdAt;
    long long lastModified;
};

class NovaOmega {
private:
    std::map<std::string, std::shared_ptr<NovaContext>> contexts_;
    std::mutex contextMutex_;
    std::vector<std::function<void(const std::string&, const std::string&)>> observers_;

    bool isInitialized_;
    std::string currentMode_;

    void notifyObservers(const std::string& event, const std::string& data);
    std::string processOperation(OmegaOperation op, const std::string& data);

public:
    NovaOmega();
    ~NovaOmega();

    // Core Nova Omega functionality
    bool initialize(const std::string& config = "");
    void shutdown();

    // Context management
    std::string createContext(const std::string& sourceDomain,
                             const std::string& targetDomain,
                             const std::map<std::string, std::string>& params = {});

    bool updateContext(const std::string& contextId,
                      const std::map<std::string, std::string>& updates);

    std::shared_ptr<NovaContext> getContext(const std::string& contextId);
    void removeContext(const std::string& contextId);

    // Operation handlers
    std::string handleCall(const std::string& operation, const std::string& data);
    std::string processTransformation(const std::string& input,
                                     const std::string& transformType);
    std::string optimizeData(const std::string& data, const std::string& criteria);

    // UDR Nova integration
    std::string handleUDRNova(const std::string& udrData, const std::string& novaParams);
    void synchronizeWithUDR(const std::string& udrEndpoint);

    // Observer pattern for call monitoring
    void addObserver(std::function<void(const std::string&, const std::string&)> observer);
    void removeAllObservers();

    // Status and diagnostics
    bool isOperational() const { return isInitialized_; }
    std::string getCurrentMode() const { return currentMode_; }
    std::vector<std::string> getActiveContexts();
    std::map<std::string, std::string> getDiagnostics();
};

} // namespace AutomationNucleus

#endif // NOVA_OMEGA_HPP
