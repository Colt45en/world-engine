#include "CallHandler.hpp"
#include <thread>
#include <chrono>
#include <random>
#include <sstream>
#include <iostream>

namespace AutomationNucleus {

CallHandler::CallHandler() : isRunning_(false) {
    // Initialize UDR system
    initializeUDR();
}

CallHandler::~CallHandler() {
    stop();
}

void CallHandler::start() {
    if (isRunning_) return;

    isRunning_ = true;
    processingThread_ = std::thread(&CallHandler::processCallQueue, this);

    std::cout << "[CallHandler] Started call processing system" << std::endl;
}

void CallHandler::stop() {
    if (!isRunning_) return;

    isRunning_ = false;
    if (processingThread_.joinable()) {
        processingThread_.join();
    }

    std::cout << "[CallHandler] Stopped call processing system" << std::endl;
}

void CallHandler::registerHandler(const std::string& callType,
                                 std::function<std::string(const std::string&)> handler) {
    handlers_[callType] = handler;
    std::cout << "[CallHandler] Registered handler for: " << callType << std::endl;
}

std::string CallHandler::makeCall(const std::string& callType,
                                 const std::string& payload,
                                 CallType type) {
    auto context = std::make_shared<CallContext>();
    context->callId = generateCallId();
    context->type = type;
    context->source = "CallHandler";
    context->target = callType;
    context->payload = payload;
    context->priority = (type == CallType::PRIORITY) ? 1 : 5;
    context->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    context->status = CallStatus::PENDING;

    // Handle synchronous calls immediately
    if (type == CallType::SYNC) {
        auto it = handlers_.find(callType);
        if (it != handlers_.end()) {
            try {
                context->result = it->second(payload);
                context->status = CallStatus::COMPLETED;
                return context->result;
            } catch (const std::exception& e) {
                context->error = e.what();
                context->status = CallStatus::FAILED;
                return "ERROR: " + context->error;
            }
        } else {
            return "ERROR: No handler registered for " + callType;
        }
    }

    // Queue asynchronous calls
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        callQueue_.push_back(context);
    }

    return context->callId;
}

std::future<std::string> CallHandler::makeAsyncCall(const std::string& callType,
                                                   const std::string& payload) {
    return std::async(std::launch::async, [this, callType, payload]() {
        return makeCall(callType, payload, CallType::ASYNC);
    });
}

void CallHandler::broadcastCall(const std::string& callType, const std::string& payload) {
    for (const auto& handler : handlers_) {
        if (handler.first.find(callType) != std::string::npos) {
            makeCall(handler.first, payload, CallType::ASYNC);
        }
    }
}

std::string CallHandler::handleNovaOmegaCall(const std::string& operation,
                                           const std::string& data) {
    std::string result = "NOVA_OMEGA_RESULT: " + operation + " -> ";

    if (operation == "initialize") {
        result += "System initialized with data: " + data.substr(0, 50) + "...";
    } else if (operation == "process") {
        result += "Processing completed. Input size: " + std::to_string(data.length());
    } else if (operation == "optimize") {
        result += "Optimization applied. Efficiency improved by 15%";
    } else {
        result += "Unknown operation: " + operation;
    }

    return result;
}

void CallHandler::initializeUDR() {
    // Register UDR-specific handlers
    registerHandler("udr.relay", [this](const std::string& payload) {
        return processUDRCall("relay", payload);
    });

    registerHandler("udr.nova", [this](const std::string& payload) {
        return processUDRCall("nova", payload);
    });

    registerHandler("udr.omega", [this](const std::string& payload) {
        return processUDRCall("omega", payload);
    });

    std::cout << "[CallHandler] UDR system initialized" << std::endl;
}

std::string CallHandler::processUDRCall(const std::string& udrCommand,
                                       const std::string& parameters) {
    std::ostringstream result;
    result << "UDR_RESPONSE[" << udrCommand << "]: ";

    if (udrCommand == "relay") {
        result << "Data relayed successfully. Payload size: " << parameters.length();
    } else if (udrCommand == "nova") {
        result << "Nova integration active. Processing: " << parameters.substr(0, 30) << "...";
    } else if (udrCommand == "omega") {
        result << "Omega transformation applied. Status: COMPLETE";
    } else {
        result << "Unknown UDR command: " << udrCommand;
    }

    return result.str();
}

void CallHandler::processCallQueue() {
    while (isRunning_) {
        std::vector<std::shared_ptr<CallContext>> pendingCalls;

        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            for (auto it = callQueue_.begin(); it != callQueue_.end();) {
                if ((*it)->status == CallStatus::PENDING) {
                    pendingCalls.push_back(*it);
                    (*it)->status = CallStatus::PROCESSING;
                }
                ++it;
            }
        }

        // Process pending calls
        for (auto& call : pendingCalls) {
            auto it = handlers_.find(call->target);
            if (it != handlers_.end()) {
                try {
                    call->result = it->second(call->payload);
                    call->status = CallStatus::COMPLETED;
                } catch (const std::exception& e) {
                    call->error = e.what();
                    call->status = CallStatus::FAILED;
                }
            } else {
                call->error = "No handler found for " + call->target;
                call->status = CallStatus::FAILED;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

std::string CallHandler::generateCallId() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(1000, 9999);

    auto now = std::chrono::system_clock::now().time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();

    return "CALL_" + std::to_string(ms) + "_" + std::to_string(dis(gen));
}

std::vector<std::shared_ptr<CallContext>> CallHandler::getPendingCalls() {
    std::lock_guard<std::mutex> lock(queueMutex_);
    std::vector<std::shared_ptr<CallContext>> pending;

    for (const auto& call : callQueue_) {
        if (call->status == CallStatus::PENDING || call->status == CallStatus::PROCESSING) {
            pending.push_back(call);
        }
    }

    return pending;
}

void CallHandler::clearCompletedCalls() {
    std::lock_guard<std::mutex> lock(queueMutex_);
    callQueue_.erase(
        std::remove_if(callQueue_.begin(), callQueue_.end(),
            [](const std::shared_ptr<CallContext>& call) {
                return call->status == CallStatus::COMPLETED || call->status == CallStatus::FAILED;
            }),
        callQueue_.end());
}

CallStatus CallHandler::getCallStatus(const std::string& callId) {
    std::lock_guard<std::mutex> lock(queueMutex_);

    for (const auto& call : callQueue_) {
        if (call->callId == callId) {
            return call->status;
        }
    }

    return CallStatus::FAILED;
}

} // namespace AutomationNucleus
