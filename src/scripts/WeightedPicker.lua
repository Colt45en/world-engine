-- Script: Weighted random item picker
local Rarities = require(--[[Module script path as string]])

local function GetItem()
    local totalWeight = 0
    for _, v in ipairs(Rarities) do
        totalWeight = totalWeight + v[2]
    end

    local RNG = Random.new()
    local chosen = RNG:NextNumber(0, totalWeight)
    local cumulative = 0

    for _, v in ipairs(Rarities) do
        cumulative = cumulative + v[2]
        if chosen <= cumulative then
            return v[1]
        end
    end
    -- Fallback: Should not happen unless all weights are 0
    return Rarities[#Rarities][1]
end

-- Example usage:
local picked = GetItem()
print("Picked:", picked)