#include <napi.h>
#include <memory>
#include "NexusGameEngine.hpp"
#include "GameEntity.hpp"
#include "GameComponents.hpp"

using namespace NexusGame;

/**
 * Node.js Addon for NEXUS Game Logic
 * Provides JavaScript interface to C++ game engine
 */

class NexusGameEngineWrapper : public Napi::ObjectWrap<NexusGameEngineWrapper> {
private:
    std::unique_ptr<NexusGameEngine> m_engine;

public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    NexusGameEngineWrapper(const Napi::CallbackInfo& info);
    ~NexusGameEngineWrapper();

    // Engine methods
    Napi::Value Initialize(const Napi::CallbackInfo& info);
    Napi::Value Update(const Napi::CallbackInfo& info);
    Napi::Value Render(const Napi::CallbackInfo& info);
    Napi::Value Shutdown(const Napi::CallbackInfo& info);

    // System integration
    Napi::Value SyncWithNexusSystems(const Napi::CallbackInfo& info);
    Napi::Value GetSystemStatusJson(const Napi::CallbackInfo& info);

    // Parameter setters
    Napi::Value SetBPM(const Napi::CallbackInfo& info);
    Napi::Value SetHarmonics(const Napi::CallbackInfo& info);
    Napi::Value SetPetalCount(const Napi::CallbackInfo& info);
    Napi::Value SetTerrainRoughness(const Napi::CallbackInfo& info);

    // Entity management
    Napi::Value CreateEntity(const Napi::CallbackInfo& info);
    Napi::Value DestroyEntity(const Napi::CallbackInfo& info);
    Napi::Value GetEntities(const Napi::CallbackInfo& info);

    // Properties
    Napi::Value IsRunning(const Napi::CallbackInfo& info);
    Napi::Value GetDeltaTime(const Napi::CallbackInfo& info);
};

class GameEntityWrapper : public Napi::ObjectWrap<GameEntityWrapper> {
private:
    std::shared_ptr<GameEntity> m_entity;

public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    GameEntityWrapper(const Napi::CallbackInfo& info);

    void SetEntity(std::shared_ptr<GameEntity> entity) { m_entity = entity; }
    std::shared_ptr<GameEntity> GetEntity() const { return m_entity; }

    // Entity methods
    Napi::Value GetID(const Napi::CallbackInfo& info);
    Napi::Value GetName(const Napi::CallbackInfo& info);
    Napi::Value SetName(const Napi::CallbackInfo& info);
    Napi::Value IsActive(const Napi::CallbackInfo& info);
    Napi::Value SetActive(const Napi::CallbackInfo& info);

    // Component management
    Napi::Value AddTransform(const Napi::CallbackInfo& info);
    Napi::Value AddAudioSync(const Napi::CallbackInfo& info);
    Napi::Value AddArtSync(const Napi::CallbackInfo& info);
    Napi::Value AddPhysics(const Napi::CallbackInfo& info);

    Napi::Value GetTransform(const Napi::CallbackInfo& info);
    Napi::Value GetAudioSync(const Napi::CallbackInfo& info);
    Napi::Value GetArtSync(const Napi::CallbackInfo& info);
    Napi::Value GetPhysics(const Napi::CallbackInfo& info);
};

// NexusGameEngineWrapper Implementation
Napi::Object NexusGameEngineWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "NexusGameEngine", {
        InstanceMethod("initialize", &NexusGameEngineWrapper::Initialize),
        InstanceMethod("update", &NexusGameEngineWrapper::Update),
        InstanceMethod("render", &NexusGameEngineWrapper::Render),
        InstanceMethod("shutdown", &NexusGameEngineWrapper::Shutdown),
        InstanceMethod("syncWithNexusSystems", &NexusGameEngineWrapper::SyncWithNexusSystems),
        InstanceMethod("getSystemStatusJson", &NexusGameEngineWrapper::GetSystemStatusJson),
        InstanceMethod("setBPM", &NexusGameEngineWrapper::SetBPM),
        InstanceMethod("setHarmonics", &NexusGameEngineWrapper::SetHarmonics),
        InstanceMethod("setPetalCount", &NexusGameEngineWrapper::SetPetalCount),
        InstanceMethod("setTerrainRoughness", &NexusGameEngineWrapper::SetTerrainRoughness),
        InstanceMethod("createEntity", &NexusGameEngineWrapper::CreateEntity),
        InstanceMethod("destroyEntity", &NexusGameEngineWrapper::DestroyEntity),
        InstanceMethod("getEntities", &NexusGameEngineWrapper::GetEntities),
        InstanceMethod("isRunning", &NexusGameEngineWrapper::IsRunning),
        InstanceMethod("getDeltaTime", &NexusGameEngineWrapper::GetDeltaTime)
    });

    exports.Set("NexusGameEngine", func);
    return exports;
}

NexusGameEngineWrapper::NexusGameEngineWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<NexusGameEngineWrapper>(info) {
    m_engine = std::make_unique<NexusGameEngine>();
}

NexusGameEngineWrapper::~NexusGameEngineWrapper() {
    if (m_engine && m_engine->IsRunning()) {
        m_engine->Shutdown();
    }
}

Napi::Value NexusGameEngineWrapper::Initialize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    NexusGameEngine::SystemParameters params;

    if (info.Length() > 0 && info[0].IsObject()) {
        Napi::Object paramObj = info[0].As<Napi::Object>();

        if (paramObj.Has("bpm")) {
            params.bpm = paramObj.Get("bpm").As<Napi::Number>().DoubleValue();
        }
        if (paramObj.Has("harmonics")) {
            params.harmonics = paramObj.Get("harmonics").As<Napi::Number>().Int32Value();
        }
        if (paramObj.Has("petalCount")) {
            params.petalCount = paramObj.Get("petalCount").As<Napi::Number>().Int32Value();
        }
        if (paramObj.Has("terrainRoughness")) {
            params.terrainRoughness = paramObj.Get("terrainRoughness").As<Napi::Number>().DoubleValue();
        }
    }

    bool success = m_engine->Initialize(params);
    return Napi::Boolean::New(env, success);
}

Napi::Value NexusGameEngineWrapper::Update(const Napi::CallbackInfo& info) {
    m_engine->Update();
    return info.Env().Undefined();
}

Napi::Value NexusGameEngineWrapper::Render(const Napi::CallbackInfo& info) {
    m_engine->Render();
    return info.Env().Undefined();
}

Napi::Value NexusGameEngineWrapper::Shutdown(const Napi::CallbackInfo& info) {
    m_engine->Shutdown();
    return info.Env().Undefined();
}

Napi::Value NexusGameEngineWrapper::SyncWithNexusSystems(const Napi::CallbackInfo& info) {
    if (info.Length() > 0 && info[0].IsString()) {
        std::string jsonData = info[0].As<Napi::String>().Utf8Value();
        m_engine->SyncWithNexusSystems(jsonData);
    }
    return info.Env().Undefined();
}

Napi::Value NexusGameEngineWrapper::GetSystemStatusJson(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    std::string status = m_engine->GetSystemStatusJson();
    return Napi::String::New(env, status);
}

Napi::Value NexusGameEngineWrapper::SetBPM(const Napi::CallbackInfo& info) {
    if (info.Length() > 0 && info[0].IsNumber()) {
        double bpm = info[0].As<Napi::Number>().DoubleValue();
        m_engine->SetBPM(bpm);
    }
    return info.Env().Undefined();
}

Napi::Value NexusGameEngineWrapper::SetHarmonics(const Napi::CallbackInfo& info) {
    if (info.Length() > 0 && info[0].IsNumber()) {
        int harmonics = info[0].As<Napi::Number>().Int32Value();
        m_engine->SetHarmonics(harmonics);
    }
    return info.Env().Undefined();
}

Napi::Value NexusGameEngineWrapper::SetPetalCount(const Napi::CallbackInfo& info) {
    if (info.Length() > 0 && info[0].IsNumber()) {
        int count = info[0].As<Napi::Number>().Int32Value();
        m_engine->SetPetalCount(count);
    }
    return info.Env().Undefined();
}

Napi::Value NexusGameEngineWrapper::SetTerrainRoughness(const Napi::CallbackInfo& info) {
    if (info.Length() > 0 && info[0].IsNumber()) {
        double roughness = info[0].As<Napi::Number>().DoubleValue();
        m_engine->SetTerrainRoughness(roughness);
    }
    return info.Env().Undefined();
}

Napi::Value NexusGameEngineWrapper::CreateEntity(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    std::string name = "Entity";
    if (info.Length() > 0 && info[0].IsString()) {
        name = info[0].As<Napi::String>().Utf8Value();
    }

    auto entity = m_engine->CreateEntity(name);

    // Create JavaScript wrapper
    auto entityWrapper = GameEntityWrapper::constructor.New({});
    GameEntityWrapper* wrapper = GameEntityWrapper::Unwrap(entityWrapper);
    wrapper->SetEntity(entity);

    return entityWrapper;
}

Napi::Value NexusGameEngineWrapper::IsRunning(const Napi::CallbackInfo& info) {
    return Napi::Boolean::New(info.Env(), m_engine->IsRunning());
}

Napi::Value NexusGameEngineWrapper::GetDeltaTime(const Napi::CallbackInfo& info) {
    return Napi::Number::New(info.Env(), m_engine->GetDeltaTime());
}

// GameEntityWrapper Implementation
Napi::FunctionReference GameEntityWrapper::constructor;

Napi::Object GameEntityWrapper::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "GameEntity", {
        InstanceMethod("getID", &GameEntityWrapper::GetID),
        InstanceMethod("getName", &GameEntityWrapper::GetName),
        InstanceMethod("setName", &GameEntityWrapper::SetName),
        InstanceMethod("isActive", &GameEntityWrapper::IsActive),
        InstanceMethod("setActive", &GameEntityWrapper::SetActive),
        InstanceMethod("addTransform", &GameEntityWrapper::AddTransform),
        InstanceMethod("addAudioSync", &GameEntityWrapper::AddAudioSync),
        InstanceMethod("addArtSync", &GameEntityWrapper::AddArtSync),
        InstanceMethod("addPhysics", &GameEntityWrapper::AddPhysics),
        InstanceMethod("getTransform", &GameEntityWrapper::GetTransform),
        InstanceMethod("getAudioSync", &GameEntityWrapper::GetAudioSync),
        InstanceMethod("getArtSync", &GameEntityWrapper::GetArtSync),
        InstanceMethod("getPhysics", &GameEntityWrapper::GetPhysics)
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("GameEntity", func);
    return exports;
}

GameEntityWrapper::GameEntityWrapper(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<GameEntityWrapper>(info) {
}

Napi::Value GameEntityWrapper::GetID(const Napi::CallbackInfo& info) {
    if (m_entity) {
        return Napi::Number::New(info.Env(), static_cast<double>(m_entity->GetID()));
    }
    return info.Env().Undefined();
}

Napi::Value GameEntityWrapper::GetName(const Napi::CallbackInfo& info) {
    if (m_entity) {
        return Napi::String::New(info.Env(), m_entity->GetName());
    }
    return Napi::String::New(info.Env(), "");
}

Napi::Value GameEntityWrapper::SetName(const Napi::CallbackInfo& info) {
    if (m_entity && info.Length() > 0 && info[0].IsString()) {
        std::string name = info[0].As<Napi::String>().Utf8Value();
        m_entity->SetName(name);
    }
    return info.Env().Undefined();
}

Napi::Value GameEntityWrapper::IsActive(const Napi::CallbackInfo& info) {
    if (m_entity) {
        return Napi::Boolean::New(info.Env(), m_entity->IsActive());
    }
    return Napi::Boolean::New(info.Env(), false);
}

Napi::Value GameEntityWrapper::SetActive(const Napi::CallbackInfo& info) {
    if (m_entity && info.Length() > 0 && info[0].IsBoolean()) {
        bool active = info[0].As<Napi::Boolean>().Value();
        m_entity->SetActive(active);
    }
    return info.Env().Undefined();
}

Napi::Value GameEntityWrapper::AddTransform(const Napi::CallbackInfo& info) {
    if (m_entity) {
        m_entity->AddComponent<Transform>();
    }
    return info.Env().Undefined();
}

Napi::Value GameEntityWrapper::AddAudioSync(const Napi::CallbackInfo& info) {
    if (m_entity) {
        m_entity->AddComponent<AudioSync>();
    }
    return info.Env().Undefined();
}

Napi::Value GameEntityWrapper::AddArtSync(const Napi::CallbackInfo& info) {
    if (m_entity) {
        m_entity->AddComponent<ArtSync>();
    }
    return info.Env().Undefined();
}

Napi::Value GameEntityWrapper::AddPhysics(const Napi::CallbackInfo& info) {
    if (m_entity) {
        m_entity->AddComponent<Physics>();
    }
    return info.Env().Undefined();
}

// Component getter methods would need additional wrapper classes for full functionality
Napi::Value GameEntityWrapper::GetTransform(const Napi::CallbackInfo& info) {
    // For now, return boolean indicating presence
    if (m_entity) {
        bool hasComponent = m_entity->HasComponent<Transform>();
        return Napi::Boolean::New(info.Env(), hasComponent);
    }
    return Napi::Boolean::New(info.Env(), false);
}

Napi::Value GameEntityWrapper::GetAudioSync(const Napi::CallbackInfo& info) {
    if (m_entity) {
        bool hasComponent = m_entity->HasComponent<AudioSync>();
        return Napi::Boolean::New(info.Env(), hasComponent);
    }
    return Napi::Boolean::New(info.Env(), false);
}

Napi::Value GameEntityWrapper::GetArtSync(const Napi::CallbackInfo& info) {
    if (m_entity) {
        bool hasComponent = m_entity->HasComponent<ArtSync>();
        return Napi::Boolean::New(info.Env(), hasComponent);
    }
    return Napi::Boolean::New(info.Env(), false);
}

Napi::Value GameEntityWrapper::GetPhysics(const Napi::CallbackInfo& info) {
    if (m_entity) {
        bool hasComponent = m_entity->HasComponent<Physics>();
        return Napi::Boolean::New(info.Env(), hasComponent);
    }
    return Napi::Boolean::New(info.Env(), false);
}

// Module initialization
Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
    NexusGameEngineWrapper::Init(env, exports);
    GameEntityWrapper::Init(env, exports);
    return exports;
}

NODE_API_MODULE(nexus_game, InitAll)
