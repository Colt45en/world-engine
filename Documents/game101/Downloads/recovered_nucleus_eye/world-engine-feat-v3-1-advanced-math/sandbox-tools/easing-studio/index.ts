import EasingStudio from './EasingStudio';

export default EasingStudio;

// Export easing functions for external use
export { EASING_FUNCTIONS } from './EasingStudio';

// Export presets
export { EASING_PRESETS } from './EasingStudio';

// Export utility functions
export {
    sampleEasingFunction,
    bezierFunction,
    generateAnimationPath,
    clamp
} from './EasingStudio';

// Export types for TypeScript integration
export type {
    EasingFunction,
    EasingPreset,
    CurvePoint,
    AnimationPath
} from './EasingStudio';
