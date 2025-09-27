#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "AssetResourceManager.hpp"

class AssetResourceBridge {
public:
    using ResolveFn = std::function<void(const std::string& type, const std::string& id)>;
    using RejectFn = std::function<void(const std::string& type, const std::string& id, const std::string& reason)>;

    explicit AssetResourceBridge(double memLimitMB = 2048.0)
        : running(false) {
        arm.setMemoryLimit(memLimitMB);
    }

    ~AssetResourceBridge() { stop(); }

    void registerBasePath(const std::string& type, const std::string& path) {
        arm.registerBasePath(type, path);
    }

    void preload(const std::vector<std::pair<std::string, std::string>>& items) {
        arm.preloadAssets(items);
    }

    void request(const std::string& type,
                 const std::string& id,
                 int priority = 0,
                 ResolveFn on_ok = nullptr,
                 RejectFn on_err = nullptr) {
        {
            std::lock_guard<std::mutex> lk(cb_mtx);
            if (on_ok) {
                ok_map[{type, id}] = std::move(on_ok);
            }
            if (on_err) {
                err_map[{type, id}] = std::move(on_err);
            }
        }
        arm.requestAsset(type, id, priority);
    }

    void start(int hz = 30) {
        if (running.exchange(true)) {
            return;
        }
        pump = std::thread([this, hz] {
            const auto dt = std::chrono::milliseconds(1000 / std::max(1, hz));
            while (running.load()) {
                arm.processQueue();
                flushCallbacks();
                std::this_thread::sleep_for(dt);
            }
        });
    }

    void stop() {
        if (!running.exchange(false)) {
            return;
        }
        if (pump.joinable()) {
            pump.join();
        }
    }

private:
    struct Key {
        std::string type;
        std::string id;
        bool operator==(const Key& o) const { return type == o.type && id == o.id; }
    };

    struct KeyHasher {
        std::size_t operator()(const Key& k) const {
            return std::hash<std::string>()(k.type) ^ (std::hash<std::string>()(k.id) << 1);
        }
    };

    AssetResourceManager arm;
    std::atomic<bool> running;
    std::thread pump;

    std::mutex cb_mtx;
    std::unordered_map<Key, ResolveFn, KeyHasher> ok_map;
    std::unordered_map<Key, RejectFn, KeyHasher> err_map;

    void flushCallbacks() {
        // Hook into your AssetResourceManager status to determine loaded/error state.
        // This placeholder does not fire callbacks automatically because the current
        // workspace does not expose a public status getter. Integrate with your
        // internal signals by calling fireResolve/fireReject accordingly.
    }

    void fireResolve(const Key& k) {
        std::lock_guard<std::mutex> lk(cb_mtx);
        if (auto it = ok_map.find(k); it != ok_map.end()) {
            it->second(k.type, k.id);
            ok_map.erase(it);
            err_map.erase(k);
        }
    }

    void fireReject(const Key& k, const std::string& reason) {
        std::lock_guard<std::mutex> lk(cb_mtx);
        if (auto it = err_map.find(k); it != err_map.end()) {
            it->second(k.type, k.id, reason);
            ok_map.erase(k);
            err_map.erase(it);
        }
    }
};
