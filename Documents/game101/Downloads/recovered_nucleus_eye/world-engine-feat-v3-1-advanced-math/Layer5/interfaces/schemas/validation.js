/**
 * Shared schemas and validators for World Engine Studio
 * Lightweight validation without external dependencies
 */

export const Schemas = {
    // Run record validation
    Run: (x) => x &&
        typeof x.runId === 'string' &&
        typeof x.ts === 'number' &&
        typeof x.input === 'string' &&
        (x.outcome === null || typeof x.outcome === 'object'),

    // Clip record validation
    Clip: (x) => x &&
        typeof x.clipId === 'string' &&
        typeof x.timestamp === 'number' &&
        typeof x.size === 'number' &&
        (typeof x.url === 'string' || x.url === null),

    // Marker validation
    Mark: (x) => x &&
        typeof x.id === 'string' &&
        typeof x.tag === 'string' &&
        typeof x.timestamp === 'number' &&
        (x.runId === null || typeof x.runId === 'string') &&
        (x.clipId === null || typeof x.clipId === 'string'),

    // Command validation
    Command: (x) => x &&
        typeof x.type === 'string' &&
        (x.args === undefined || typeof x.args === 'string'),

    // Engine result validation
    EngineResult: (x) => x &&
        (x.items === undefined || Array.isArray(x.items)) &&
        (x.count === undefined || typeof x.count === 'number'),

    // Session export validation
    SessionExport: (x) => x &&
        typeof x.schema === 'string' &&
        x.schema.startsWith('world-engine-session.') &&
        typeof x.sessionId === 'string' &&
        typeof x.ts === 'number'
};

// Type assertion helper
export function assertShape(guard, obj, msg = 'Invalid shape') {
    if (!guard(obj)) {
        throw new Error(`${msg}: ${JSON.stringify(obj).slice(0, 100)}`);
    }
    return obj;
}

// Batch validation
export function validateBatch(guard, items, msg = 'Invalid batch item') {
    const errors = [];
    items.forEach((item, i) => {
        try {
            assertShape(guard, item, `${msg} at index ${i}`);
        } catch (err) {
            errors.push(err.message);
        }
    });
    if (errors.length) {
        throw new Error(`Batch validation failed:\n${errors.join('\n')}`);
    }
    return items;
}

// Safe object construction with validation
export function createRun(runId, input, outcome = null, meta = {}) {
    const run = {
        runId,
        ts: Date.now(),
        input,
        outcome,
        clipId: null,
        meta
    };
    return assertShape(Schemas.Run, run, 'Invalid run object');
}

export function createClip(clipId, url, size, meta = {}) {
    const clip = {
        clipId,
        url,
        size,
        timestamp: Date.now(),
        duration: 0,
        meta,
        marks: []
    };
    return assertShape(Schemas.Clip, clip, 'Invalid clip object');
}

export function createMark(tag, runId = null, clipId = null) {
    const mark = {
        id: `mark_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        tag,
        runId,
        clipId,
        timestamp: Date.now()
    };
    return assertShape(Schemas.Mark, mark, 'Invalid mark object');
}
