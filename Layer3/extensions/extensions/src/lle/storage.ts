import type { SU, Button } from './core';

export type Event = {
    type: 'event';
    session: string;
    seq: number;
    ts: string;
    button: string;
    buttonVid: string;
    inputCid: string;
    outputCid: string;
    params?: Record<string, unknown>;
};

export type Snapshot = {
    type: 'snapshot';
    session: string;
    index: number;
    su: SU;
    prev?: string;
};

// Minimal hash (not cryptographic)—for demo only. Swap for BLAKE3 in prod.
function hash(obj: unknown): string {
    return 'h' + (Math.abs([...JSON.stringify(obj)].reduce((a, c) => ((a << 5) - a) + c.charCodeAt(0), 0)) >>> 0).toString(16);
}

export class Catalog {
    private map = new Map<string, string>(); // key: lid@vid → cid

    set(lid: string, vid: string, cid: string) {
        this.map.set(`${lid}@${vid}`, cid);
    }

    get(lid: string, vid: string) {
        return this.map.get(`${lid}@${vid}`);
    }
}

export class InMemoryStore {
    objects = new Map<string, unknown>();

    put(o: unknown): string {
        const cid = hash(o);
        this.objects.set(cid, o);
        return cid;
    }

    get(cid: string) {
        return this.objects.get(cid);
    }
}

export class SessionLog {
    events: Event[] = [];
    snapshots: Snapshot[] = [];

    appendEvent(e: Event) {
        this.events.push(e);
    }

    addSnapshot(s: Snapshot) {
        this.snapshots.push(s);
    }
}
