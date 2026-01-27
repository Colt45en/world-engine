#ifndef WORLD_ENGINE_TYPES_H
#define WORLD_ENGINE_TYPES_H

#include <cstdint>
#include <string>
#include <memory>

namespace WorldEngine {

// Entity ID type
using EntityID = uint64_t;

// Component ID type
using ComponentID = uint32_t;

// System ID type
using SystemID = uint32_t;

// Invalid entity ID
constexpr EntityID INVALID_ENTITY = 0;

// Version information
constexpr const char* VERSION = "1.0.0";
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

} // namespace WorldEngine

#endif // WORLD_ENGINE_TYPES_H
