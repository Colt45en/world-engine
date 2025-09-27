/**
 * Utility functions for World Engine Studio
 * Common helpers used across components
 */

export const Utils = {
    // ID generation
    generateId: (prefix = 'id') => {
        return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    },

    // Text processing
    sanitizeText: (text) => {
        if (typeof text !== 'string') return '';
        return text.trim().replace(/\s+/g, ' ');
    },

    tokenizeBasic: (text) => {
        return text.toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(t => t.length > 0);
    },

    // Time utilities
    formatDuration: (ms) => {
        if (ms < 1000) return `${ms}ms`;
        if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
        return `${(ms / 60000).toFixed(1)}m`;
    },

    getCurrentTimestamp: () => new Date().toISOString(),

    // Event utilities
    debounce: (func, wait) => {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    throttle: (func, limit) => {
        let inThrottle;
        return function (...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    // Data validation
    isValidJSON: (str) => {
        try {
            JSON.parse(str);
            return true;
        } catch {
            return false;
        }
    },

    safeJSONParse: (str, fallback = null) => {
        try {
            return JSON.parse(str);
        } catch {
            return fallback;
        }
    },

    // Array utilities
    chunkArray: (array, size) => {
        const chunks = [];
        for (let i = 0; i < array.length; i += size) {
            chunks.push(array.slice(i, i + size));
        }
        return chunks;
    },

    unique: (array) => [...new Set(array)],

    groupBy: (array, key) => {
        return array.reduce((groups, item) => {
            const group = (groups[item[key]] = groups[item[key]] || []);
            group.push(item);
            return groups;
        }, {});
    },

    // Object utilities
    deepClone: (obj) => {
        if (obj === null || typeof obj !== 'object') return obj;
        if (obj instanceof Date) return new Date(obj.getTime());
        if (obj instanceof Array) return obj.map(item => Utils.deepClone(item));
        if (typeof obj === 'object') {
            const clonedObj = {};
            for (const key in obj) {
                if (obj.hasOwnProperty(key)) {
                    clonedObj[key] = Utils.deepClone(obj[key]);
                }
            }
            return clonedObj;
        }
    },

    mergeDeep: (target, source) => {
        const output = Object.assign({}, target);
        if (Utils.isObject(target) && Utils.isObject(source)) {
            Object.keys(source).forEach(key => {
                if (Utils.isObject(source[key])) {
                    if (!(key in target))
                        Object.assign(output, { [key]: source[key] });
                    else
                        output[key] = Utils.mergeDeep(target[key], source[key]);
                } else {
                    Object.assign(output, { [key]: source[key] });
                }
            });
        }
        return output;
    },

    isObject: (item) => {
        return item && typeof item === 'object' && !Array.isArray(item);
    },

    // Storage utilities (localStorage wrapper)
    storage: {
        get: (key, fallback = null) => {
            try {
                const item = localStorage.getItem(key);
                return item ? JSON.parse(item) : fallback;
            } catch {
                return fallback;
            }
        },

        set: (key, value) => {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch {
                return false;
            }
        },

        remove: (key) => {
            try {
                localStorage.removeItem(key);
                return true;
            } catch {
                return false;
            }
        },

        clear: () => {
            try {
                localStorage.clear();
                return true;
            } catch {
                return false;
            }
        }
    },

    // URL utilities
    parseParams: (url = window.location.href) => {
        const params = new URLSearchParams(new URL(url).search);
        const result = {};
        for (const [key, value] of params) {
            result[key] = value;
        }
        return result;
    },

    buildURL: (base, params = {}) => {
        const url = new URL(base);
        Object.keys(params).forEach(key => {
            if (params[key] !== null && params[key] !== undefined) {
                url.searchParams.set(key, params[key]);
            }
        });
        return url.toString();
    },

    // Error handling
    createError: (message, type = 'Error', metadata = {}) => {
        const error = new Error(message);
        error.type = type;
        error.metadata = metadata;
        error.timestamp = Date.now();
        return error;
    },

    logError: (error, context = '') => {
        console.error(`[${context}] Error:`, {
            message: error.message,
            type: error.type || 'Unknown',
            stack: error.stack,
            metadata: error.metadata,
            timestamp: error.timestamp || Date.now()
        });
    },

    // Performance utilities
    measure: {
        start: (name) => {
            if (window.performance && performance.mark) {
                performance.mark(`${name}-start`);
            }
        },

        end: (name) => {
            if (window.performance && performance.mark && performance.measure) {
                performance.mark(`${name}-end`);
                performance.measure(name, `${name}-start`, `${name}-end`);

                const measure = performance.getEntriesByName(name)[0];
                return measure ? measure.duration : null;
            }
            return null;
        },

        clear: (name) => {
            if (window.performance) {
                if (performance.clearMarks) {
                    performance.clearMarks(`${name}-start`);
                    performance.clearMarks(`${name}-end`);
                }
                if (performance.clearMeasures) {
                    performance.clearMeasures(name);
                }
            }
        }
    },

    // DOM utilities
    waitForElement: (selector, timeout = 5000) => {
        return new Promise((resolve, reject) => {
            const element = document.querySelector(selector);
            if (element) {
                resolve(element);
                return;
            }

            const observer = new MutationObserver(() => {
                const element = document.querySelector(selector);
                if (element) {
                    observer.disconnect();
                    resolve(element);
                }
            });

            observer.observe(document.body, {
                childList: true,
                subtree: true
            });

            setTimeout(() => {
                observer.disconnect();
                reject(new Error(`Element not found: ${selector}`));
            }, timeout);
        });
    },

    createElement: (tag, props = {}, children = []) => {
        const element = document.createElement(tag);

        Object.keys(props).forEach(key => {
            if (key === 'className') {
                element.className = props[key];
            } else if (key === 'style' && typeof props[key] === 'object') {
                Object.assign(element.style, props[key]);
            } else if (key.startsWith('on') && typeof props[key] === 'function') {
                element.addEventListener(key.slice(2).toLowerCase(), props[key]);
            } else {
                element.setAttribute(key, props[key]);
            }
        });

        children.forEach(child => {
            if (typeof child === 'string') {
                element.appendChild(document.createTextNode(child));
            } else if (child instanceof Node) {
                element.appendChild(child);
            }
        });

        return element;
    },

    // Async utilities
    delay: (ms) => new Promise(resolve => setTimeout(resolve, ms)),

    timeout: (promise, ms) => {
        return Promise.race([
            promise,
            new Promise((_, reject) =>
                setTimeout(() => reject(new Error('Timeout')), ms)
            )
        ]);
    },

    retry: async (fn, attempts = 3, delay = 1000) => {
        for (let i = 0; i < attempts; i++) {
            try {
                return await fn();
            } catch (error) {
                if (i === attempts - 1) throw error;
                await Utils.delay(delay * Math.pow(2, i)); // Exponential backoff
            }
        }
    },

    // Math utilities
    clamp: (value, min, max) => Math.min(Math.max(value, min), max),

    lerp: (start, end, t) => start * (1 - t) + end * t,

    randomBetween: (min, max) => Math.random() * (max - min) + min,

    roundTo: (number, decimals) => {
        const factor = Math.pow(10, decimals);
        return Math.round(number * factor) / factor;
    }
};

// Export individual utilities for convenience
export const {
    generateId,
    sanitizeText,
    tokenizeBasic,
    formatDuration,
    getCurrentTimestamp,
    debounce,
    throttle,
    isValidJSON,
    safeJSONParse,
    chunkArray,
    unique,
    groupBy,
    deepClone,
    mergeDeep,
    isObject,
    storage,
    parseParams,
    buildURL,
    createError,
    logError,
    measure,
    waitForElement,
    createElement,
    delay,
    timeout,
    retry,
    clamp,
    lerp,
    randomBetween,
    roundTo
} = Utils;
