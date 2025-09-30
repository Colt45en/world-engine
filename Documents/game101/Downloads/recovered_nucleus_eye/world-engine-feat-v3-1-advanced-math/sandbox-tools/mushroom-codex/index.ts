import MushroomCodex from './MushroomCodex';

export default MushroomCodex;

export type {
    MushroomParams,
    MushroomProfile,
    BuiltMushroom
} from './MushroomCodex';

// Re-export utility functions for advanced usage
export {
    buildMushroom,
    createToonMaterial,
    generateSpots,
    createOutlineGroup,
    MUSHROOM_PROFILES
} from './MushroomCodex';
