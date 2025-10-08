// NEXUS Area Effects System
// Advanced AoE with falloff curves, spatial indexing, and natural encounter feel
// Supports circle, ring, box areas with linear, smoothstep, exponential falloffs

#pragma once
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace NEXUS {

// ============ AREA EFFECT STRUCTURES ============

struct Point2D { double x, y; };
struct Point3D { double x, y, z; };

enum class FalloffType {
    NONE,           // Hard cutoff
    LINEAR,         // Linear fade
    SMOOTHSTEP,     // Smooth hermite interpolation
    EXPONENTIAL,    // Exponential decay
    RING,           // Ring-shaped (inner/outer radius)
    INVERSE_SQUARE  // 1/distance² falloff
};

enum class AreaShape {
    CIRCLE,
    BOX,
    RING,
    ELLIPSE
};

// ============ AREA EFFECT DEFINITION ============

class AreaEffect {
public:
    Point3D center{0, 0, 0};
    double radius{1.0};
    double innerRadius{0.0};        // For ring falloff
    AreaShape shape{AreaShape::CIRCLE};
    FalloffType falloff{FalloffType::SMOOTHSTEP};

    // Box-specific
    Point3D boxSize{1, 1, 1};

    // Ellipse-specific
    double radiusX{1.0};
    double radiusY{1.0};

    // Falloff parameters
    double falloffPower{2.0};       // For exponential falloff
    double falloffOffset{0.0};      // Minimum strength

    AreaEffect() = default;
    AreaEffect(Point3D c, double r, FalloffType f = FalloffType::SMOOTHSTEP)
        : center(c), radius(r), falloff(f) {}

    // Calculate strength at a given position
    double getStrengthAt(const Point3D& pos) const {
        double distance = calculateDistance(pos);
        return applyFalloff(distance);
    }

    double getStrengthAt(const Point2D& pos) const {
        return getStrengthAt({pos.x, pos.y, center.z});
    }

private:
    double calculateDistance(const Point3D& pos) const {
        switch (shape) {
            case AreaShape::CIRCLE: {
                double dx = pos.x - center.x;
                double dy = pos.y - center.y;
                double dz = pos.z - center.z;
                return std::sqrt(dx*dx + dy*dy + dz*dz);
            }

            case AreaShape::BOX: {
                double dx = std::max(0.0, std::abs(pos.x - center.x) - boxSize.x * 0.5);
                double dy = std::max(0.0, std::abs(pos.y - center.y) - boxSize.y * 0.5);
                double dz = std::max(0.0, std::abs(pos.z - center.z) - boxSize.z * 0.5);
                return std::sqrt(dx*dx + dy*dy + dz*dz);
            }

            case AreaShape::RING:
            case AreaShape::CIRCLE:
            default: {
                double dx = pos.x - center.x;
                double dy = pos.y - center.y;
                return std::sqrt(dx*dx + dy*dy);
            }

            case AreaShape::ELLIPSE: {
                double dx = (pos.x - center.x) / radiusX;
                double dy = (pos.y - center.y) / radiusY;
                return std::sqrt(dx*dx + dy*dy);
            }
        }
    }

    double applyFalloff(double distance) const {
        switch (falloff) {
            case FalloffType::NONE:
                return distance <= radius ? 1.0 : 0.0;

            case FalloffType::LINEAR: {
                if (distance >= radius) return falloffOffset;
                return falloffOffset + (1.0 - falloffOffset) * (1.0 - distance / radius);
            }

            case FalloffType::SMOOTHSTEP: {
                if (distance >= radius) return falloffOffset;
                double t = 1.0 - distance / radius;
                t = std::clamp(t, 0.0, 1.0);
                double smooth = t * t * (3.0 - 2.0 * t);
                return falloffOffset + (1.0 - falloffOffset) * smooth;
            }

            case FalloffType::EXPONENTIAL: {
                if (distance >= radius) return falloffOffset;
                double t = distance / radius;
                double exp_val = std::exp(-falloffPower * t);
                return falloffOffset + (1.0 - falloffOffset) * exp_val;
            }

            case FalloffType::RING: {
                if (distance < innerRadius || distance > radius) return falloffOffset;
                double ringWidth = radius - innerRadius;
                if (ringWidth <= 0.0) return 1.0;
                double ringPos = (distance - innerRadius) / ringWidth;
                double strength = 1.0 - ringPos; // Fade out across ring
                return falloffOffset + (1.0 - falloffOffset) * strength;
            }

            case FalloffType::INVERSE_SQUARE: {
                if (distance <= 0.01) return 1.0; // Avoid division by zero
                double invSq = 1.0 / (distance * distance);
                double maxInvSq = 1.0 / (0.01 * 0.01);
                double normalized = std::min(invSq / maxInvSq, 1.0);
                return falloffOffset + (1.0 - falloffOffset) * normalized;
            }
        }
        return 0.0;
    }
};

// ============ SPATIAL HASH GRID FOR PERFORMANCE ============

template<typename EntityID>
class SpatialHashGrid {
private:
    double cellSize{4.0};
    std::unordered_map<uint64_t, std::vector<EntityID>> grid;
    std::unordered_map<EntityID, Point2D> entityPositions;

    uint64_t hash(double x, double y) const {
        int32_t ix = static_cast<int32_t>(std::floor(x / cellSize));
        int32_t iy = static_cast<int32_t>(std::floor(y / cellSize));
        return (static_cast<uint64_t>(static_cast<uint32_t>(ix)) << 32) |
               static_cast<uint64_t>(static_cast<uint32_t>(iy));
    }

    std::vector<uint64_t> getCellsInRadius(Point2D center, double radius) const {
        std::vector<uint64_t> cells;
        int cellsRadius = static_cast<int>(std::ceil(radius / cellSize));

        int centerX = static_cast<int>(std::floor(center.x / cellSize));
        int centerY = static_cast<int>(std::floor(center.y / cellSize));

        for (int dx = -cellsRadius; dx <= cellsRadius; ++dx) {
            for (int dy = -cellsRadius; dy <= cellsRadius; ++dy) {
                cells.push_back(hash((centerX + dx) * cellSize, (centerY + dy) * cellSize));
            }
        }
        return cells;
    }

public:
    SpatialHashGrid(double cell = 4.0) : cellSize(cell) {}

    void clear() {
        grid.clear();
        entityPositions.clear();
    }

    void insert(EntityID id, Point2D pos) {
        // Remove from old position if it exists
        remove(id);

        uint64_t key = hash(pos.x, pos.y);
        grid[key].push_back(id);
        entityPositions[id] = pos;
    }

    void remove(EntityID id) {
        auto it = entityPositions.find(id);
        if (it != entityPositions.end()) {
            uint64_t key = hash(it->second.x, it->second.y);
            auto& bucket = grid[key];
            bucket.erase(std::remove(bucket.begin(), bucket.end(), id), bucket.end());
            entityPositions.erase(it);
        }
    }

    void update(EntityID id, Point2D newPos) {
        auto it = entityPositions.find(id);
        if (it != entityPositions.end()) {
            Point2D oldPos = it->second;
            uint64_t oldKey = hash(oldPos.x, oldPos.y);
            uint64_t newKey = hash(newPos.x, newPos.y);

            if (oldKey != newKey) {
                // Move to new cell
                auto& oldBucket = grid[oldKey];
                oldBucket.erase(std::remove(oldBucket.begin(), oldBucket.end(), id), oldBucket.end());
                grid[newKey].push_back(id);
            }
            it->second = newPos;
        } else {
            insert(id, newPos);
        }
    }

    std::vector<EntityID> queryRadius(Point2D center, double radius) const {
        std::vector<EntityID> results;
        auto cells = getCellsInRadius(center, radius);

        for (uint64_t cellKey : cells) {
            auto it = grid.find(cellKey);
            if (it != grid.end()) {
                for (EntityID id : it->second) {
                    auto posIt = entityPositions.find(id);
                    if (posIt != entityPositions.end()) {
                        double dx = posIt->second.x - center.x;
                        double dy = posIt->second.y - center.y;
                        double dist = std::sqrt(dx*dx + dy*dy);
                        if (dist <= radius) {
                            results.push_back(id);
                        }
                    }
                }
            }
        }
        return results;
    }

    std::vector<EntityID> queryArea(const AreaEffect& area) const {
        // Use bounding circle for coarse filtering
        double boundingRadius = area.radius;
        if (area.shape == AreaShape::BOX) {
            boundingRadius = std::sqrt(area.boxSize.x * area.boxSize.x +
                                     area.boxSize.y * area.boxSize.y) * 0.5;
        } else if (area.shape == AreaShape::ELLIPSE) {
            boundingRadius = std::max(area.radiusX, area.radiusY);
        }

        auto candidates = queryRadius({area.center.x, area.center.y}, boundingRadius);

        // Fine filtering with actual area shape
        std::vector<EntityID> results;
        for (EntityID id : candidates) {
            auto posIt = entityPositions.find(id);
            if (posIt != entityPositions.end()) {
                Point3D pos3d = {posIt->second.x, posIt->second.y, area.center.z};
                if (area.getStrengthAt(pos3d) > 0.0) {
                    results.push_back(id);
                }
            }
        }
        return results;
    }

    // Statistics
    struct Stats {
        int totalEntities;
        int totalCells;
        double averageEntitiesPerCell;
        int maxEntitiesInCell;
    };

    Stats getStats() const {
        Stats stats = {};
        stats.totalEntities = entityPositions.size();
        stats.totalCells = grid.size();

        int totalInCells = 0;
        int maxInCell = 0;
        for (const auto& pair : grid) {
            int count = pair.second.size();
            totalInCells += count;
            maxInCell = std::max(maxInCell, count);
        }

        stats.averageEntitiesPerCell = stats.totalCells > 0 ?
            static_cast<double>(totalInCells) / stats.totalCells : 0.0;
        stats.maxEntitiesInCell = maxInCell;

        return stats;
    }
};

// ============ AREA EFFECT MANAGER ============

template<typename EntityID>
class AreaEffectManager {
private:
    std::vector<AreaEffect> activeEffects;
    SpatialHashGrid<EntityID> spatialGrid;

public:
    AreaEffectManager(double cellSize = 4.0) : spatialGrid(cellSize) {}

    void updateSpatialGrid() {
        // Grid is updated externally via addEntity/updateEntity calls
    }

    void addEntity(EntityID id, Point2D position) {
        spatialGrid.insert(id, position);
    }

    void updateEntity(EntityID id, Point2D position) {
        spatialGrid.update(id, position);
    }

    void removeEntity(EntityID id) {
        spatialGrid.remove(id);
    }

    int addAreaEffect(const AreaEffect& effect) {
        activeEffects.push_back(effect);
        return activeEffects.size() - 1;
    }

    void removeAreaEffect(int effectIndex) {
        if (effectIndex >= 0 && effectIndex < activeEffects.size()) {
            activeEffects.erase(activeEffects.begin() + effectIndex);
        }
    }

    void clearEffects() {
        activeEffects.clear();
    }

    // Apply effects to entities in range
    template<typename Callback>
    void applyEffects(Callback callback) {
        for (size_t i = 0; i < activeEffects.size(); ++i) {
            const auto& effect = activeEffects[i];
            auto affectedEntities = spatialGrid.queryArea(effect);

            for (EntityID id : affectedEntities) {
                auto posIt = spatialGrid.entityPositions.find(id);
                if (posIt != spatialGrid.entityPositions.end()) {
                    Point3D pos3d = {posIt->second.x, posIt->second.y, effect.center.z};
                    double strength = effect.getStrengthAt(pos3d);
                    if (strength > 0.0) {
                        callback(id, static_cast<int>(i), strength, effect);
                    }
                }
            }
        }
    }

    // Get all entities in a specific effect's area
    std::vector<std::pair<EntityID, double>> getEntitiesInEffect(int effectIndex) {
        std::vector<std::pair<EntityID, double>> results;

        if (effectIndex < 0 || effectIndex >= activeEffects.size()) {
            return results;
        }

        const auto& effect = activeEffects[effectIndex];
        auto affectedEntities = spatialGrid.queryArea(effect);

        for (EntityID id : affectedEntities) {
            auto posIt = spatialGrid.entityPositions.find(id);
            if (posIt != spatialGrid.entityPositions.end()) {
                Point3D pos3d = {posIt->second.x, posIt->second.y, effect.center.z};
                double strength = effect.getStrengthAt(pos3d);
                if (strength > 0.0) {
                    results.emplace_back(id, strength);
                }
            }
        }

        return results;
    }

    AreaEffect* getEffect(int effectIndex) {
        if (effectIndex >= 0 && effectIndex < activeEffects.size()) {
            return &activeEffects[effectIndex];
        }
        return nullptr;
    }

    int getEffectCount() const {
        return activeEffects.size();
    }

    auto getSpatialStats() const {
        return spatialGrid.getStats();
    }
};

} // namespace NEXUS
