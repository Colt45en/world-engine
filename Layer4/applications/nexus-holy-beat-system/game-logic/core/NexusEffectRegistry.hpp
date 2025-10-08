// NEXUS Effect Registry System
// Data-driven, composable effects for dynamic content creation
// Supports parameterized effects, combinations, and runtime registration

#pragma once
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <variant>

namespace NEXUS {

// ============ EFFECT PARAMETER SYSTEM ============

using EffectParam = std::variant<double, int, std::string, bool>;

class EffectParams {
private:
    std::unordered_map<std::string, EffectParam> params;

public:
    template<typename T>
    void set(const std::string& key, const T& value) {
        params[key] = value;
    }

    template<typename T>
    T get(const std::string& key, const T& defaultValue = T{}) const {
        auto it = params.find(key);
        if (it != params.end()) {
            if (std::holds_alternative<T>(it->second)) {
                return std::get<T>(it->second);
            }
        }
        return defaultValue;
    }

    bool has(const std::string& key) const {
        return params.find(key) != params.end();
    }

    void clear() { params.clear(); }

    std::vector<std::string> getKeys() const {
        std::vector<std::string> keys;
        for (const auto& pair : params) {
            keys.push_back(pair.first);
        }
        return keys;
    }
};

// ============ EFFECT CONTEXT ============

template<typename EntityType>
struct EffectContext {
    EntityType* target{nullptr};
    double strength{1.0};
    double deltaTime{0.0};
    void* worldContext{nullptr};      // Pointer to game world/system
    EffectParams params;

    // Statistics
    mutable int applicationsThisFrame{0};
    mutable double totalStrengthApplied{0.0};
};

// ============ EFFECT DEFINITION ============

template<typename EntityType>
class Effect {
public:
    using EffectFunction = std::function<void(EffectContext<EntityType>&)>;

private:
    std::string name;
    std::string description;
    EffectFunction function;
    EffectParams defaultParams;
    bool stackable{true};
    double cooldown{0.0};
    int priority{0};

public:
    Effect(const std::string& n, EffectFunction fn, const std::string& desc = "")
        : name(n), function(std::move(fn)), description(desc) {}

    void apply(EffectContext<EntityType>& context) const {
        if (function) {
            function(context);
        }
    }

    // Builder pattern methods
    Effect& withParam(const std::string& key, const EffectParam& value) {
        defaultParams.set(key, value);
        return *this;
    }

    Effect& setStackable(bool s) { stackable = s; return *this; }
    Effect& setCooldown(double c) { cooldown = c; return *this; }
    Effect& setPriority(int p) { priority = p; return *this; }
    Effect& setDescription(const std::string& d) { description = d; return *this; }

    // Accessors
    const std::string& getName() const { return name; }
    const std::string& getDescription() const { return description; }
    const EffectParams& getDefaultParams() const { return defaultParams; }
    bool isStackable() const { return stackable; }
    double getCooldown() const { return cooldown; }
    int getPriority() const { return priority; }
};

// ============ EFFECT REGISTRY ============

template<typename EntityType>
class EffectRegistry {
private:
    std::unordered_map<std::string, std::unique_ptr<Effect<EntityType>>> effects;
    std::unordered_map<std::string, std::vector<std::string>> categories;

public:
    // Register a new effect
    void registerEffect(std::unique_ptr<Effect<EntityType>> effect) {
        if (effect) {
            effects[effect->getName()] = std::move(effect);
        }
    }

    // Helper for creating and registering effects
    Effect<EntityType>& createEffect(const std::string& name,
                                   typename Effect<EntityType>::EffectFunction function,
                                   const std::string& description = "") {
        auto effect = std::make_unique<Effect<EntityType>>(name, std::move(function), description);
        Effect<EntityType>* ptr = effect.get();
        registerEffect(std::move(effect));
        return *ptr;
    }

    // Apply single effect
    bool applyEffect(const std::string& effectName, EffectContext<EntityType>& context) {
        auto it = effects.find(effectName);
        if (it != effects.end()) {
            // Merge default parameters with context parameters
            auto defaultParams = it->second->getDefaultParams();
            for (const auto& key : defaultParams.getKeys()) {
                if (!context.params.has(key)) {
                    auto value = defaultParams.get<EffectParam>(key);
                    context.params.set(key, value);
                }
            }

            it->second->apply(context);
            context.applicationsThisFrame++;
            context.totalStrengthApplied += context.strength;
            return true;
        }
        return false;
    }

    // Apply multiple effects
    void applyEffects(const std::vector<std::string>& effectNames, EffectContext<EntityType>& context) {
        // Sort by priority
        std::vector<Effect<EntityType>*> sortedEffects;
        for (const auto& name : effectNames) {
            auto it = effects.find(name);
            if (it != effects.end()) {
                sortedEffects.push_back(it->second.get());
            }
        }

        std::sort(sortedEffects.begin(), sortedEffects.end(),
                 [](const Effect<EntityType>* a, const Effect<EntityType>* b) {
                     return a->getPriority() > b->getPriority();
                 });

        for (auto* effect : sortedEffects) {
            EffectContext<EntityType> effectContext = context;
            effect->apply(effectContext);

            // Accumulate statistics
            context.applicationsThisFrame += effectContext.applicationsThisFrame;
            context.totalStrengthApplied += effectContext.totalStrengthApplied;
        }
    }

    // Category management
    void addToCategory(const std::string& category, const std::string& effectName) {
        categories[category].push_back(effectName);
    }

    std::vector<std::string> getEffectsInCategory(const std::string& category) const {
        auto it = categories.find(category);
        return it != categories.end() ? it->second : std::vector<std::string>{};
    }

    // Query effects
    const Effect<EntityType>* getEffect(const std::string& name) const {
        auto it = effects.find(name);
        return it != effects.end() ? it->second.get() : nullptr;
    }

    std::vector<std::string> getAllEffectNames() const {
        std::vector<std::string> names;
        for (const auto& pair : effects) {
            names.push_back(pair.first);
        }
        return names;
    }

    std::vector<std::string> getAllCategories() const {
        std::vector<std::string> cats;
        for (const auto& pair : categories) {
            cats.push_back(pair.first);
        }
        return cats;
    }

    // Remove effects
    bool removeEffect(const std::string& name) {
        return effects.erase(name) > 0;
    }

    void clear() {
        effects.clear();
        categories.clear();
    }

    // Statistics
    size_t getEffectCount() const { return effects.size(); }
    size_t getCategoryCount() const { return categories.size(); }
};

// ============ PREDEFINED EFFECT BUILDERS ============

template<typename EntityType>
class StandardEffects {
public:
    // Damage effect
    static std::unique_ptr<Effect<EntityType>> createDamageEffect() {
        return std::make_unique<Effect<EntityType>>(
            "damage",
            [](EffectContext<EntityType>& ctx) {
                double amount = ctx.params.get<double>("amount", 10.0);
                double multiplier = ctx.params.get<double>("multiplier", 1.0);
                double finalDamage = amount * multiplier * ctx.strength;

                // Apply damage (assuming entity has health)
                if (ctx.target) {
                    // ctx.target->takeDamage(finalDamage);
                }
            },
            "Deals damage to the target entity"
        ).get()->withParam("amount", 10.0)
         .withParam("multiplier", 1.0);
    }

    // Healing effect
    static std::unique_ptr<Effect<EntityType>> createHealEffect() {
        return std::make_unique<Effect<EntityType>>(
            "heal",
            [](EffectContext<EntityType>& ctx) {
                double amount = ctx.params.get<double>("amount", 10.0);
                double finalHeal = amount * ctx.strength;

                if (ctx.target) {
                    // ctx.target->heal(finalHeal);
                }
            },
            "Restores health to the target entity"
        ).get()->withParam("amount", 10.0);
    }

    // Speed modification effect
    static std::unique_ptr<Effect<EntityType>> createSpeedEffect() {
        return std::make_unique<Effect<EntityType>>(
            "speed_modify",
            [](EffectContext<EntityType>& ctx) {
                double multiplier = ctx.params.get<double>("multiplier", 1.5);
                double duration = ctx.params.get<double>("duration", 5.0);

                if (ctx.target) {
                    // ctx.target->modifySpeed(multiplier, duration);
                }
            },
            "Modifies movement speed of the target"
        ).get()->withParam("multiplier", 1.5)
         .withParam("duration", 5.0);
    }

    // Energy manipulation
    static std::unique_ptr<Effect<EntityType>> createEnergyEffect() {
        return std::make_unique<Effect<EntityType>>(
            "energy_modify",
            [](EffectContext<EntityType>& ctx) {
                double amount = ctx.params.get<double>("amount", 20.0);
                std::string type = ctx.params.get<std::string>("type", "add");

                double finalAmount = amount * ctx.strength;

                if (ctx.target) {
                    if (type == "add") {
                        // ctx.target->addEnergy(finalAmount);
                    } else if (type == "drain") {
                        // ctx.target->drainEnergy(finalAmount);
                    }
                }
            },
            "Modifies energy/mana of the target"
        ).get()->withParam("amount", 20.0)
         .withParam("type", std::string("add"));
    }
};

// ============ EFFECT COMBINATION SYSTEM ============

template<typename EntityType>
class EffectCombination {
private:
    std::vector<std::string> effectNames;
    EffectParams combinationParams;
    std::string name;
    double strengthModifier{1.0};

public:
    EffectCombination(const std::string& n) : name(n) {}

    EffectCombination& addEffect(const std::string& effectName) {
        effectNames.push_back(effectName);
        return *this;
    }

    EffectCombination& setStrengthModifier(double modifier) {
        strengthModifier = modifier;
        return *this;
    }

    template<typename T>
    EffectCombination& setParam(const std::string& key, const T& value) {
        combinationParams.set(key, value);
        return *this;
    }

    void apply(EffectRegistry<EntityType>& registry, EffectContext<EntityType>& context) {
        // Apply combination parameters to context
        for (const auto& key : combinationParams.getKeys()) {
            if (!context.params.has(key)) {
                context.params.set(key, combinationParams.get<EffectParam>(key));
            }
        }

        // Modify strength
        EffectContext<EntityType> modifiedContext = context;
        modifiedContext.strength *= strengthModifier;

        // Apply all effects
        registry.applyEffects(effectNames, modifiedContext);

        // Copy statistics back
        context.applicationsThisFrame += modifiedContext.applicationsThisFrame;
        context.totalStrengthApplied += modifiedContext.totalStrengthApplied;
    }

    const std::vector<std::string>& getEffects() const { return effectNames; }
    const std::string& getName() const { return name; }
    double getStrengthModifier() const { return strengthModifier; }
};

} // namespace NEXUS
