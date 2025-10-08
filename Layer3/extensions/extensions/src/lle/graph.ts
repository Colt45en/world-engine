import { Event, Snapshot } from './storage';

export interface Node {
    id: string;
    label: string;
    type: 'SU' | 'Button' | 'Event' | 'Morpheme';
}

export interface Edge {
    from: string;
    to: string;
    label?: string;
}

export class GraphDecoder {
    private nodes: Map<string, Node>;
    private edges: Edge[];

    constructor() {
        this.nodes = new Map();
        this.edges = [];
    }

    addNode(node: Node) {
        this.nodes.set(node.id, node);
    }

    addEdge(edge: Edge) {
        this.edges.push(edge);
    }

    fromEvents(events: Event[], snapshots: Snapshot[]) {
        this.nodes = new Map();
        this.edges = [];

        events.forEach((event) => {
            const evId = `ev:${event.seq}`;
            this.addNode({ id: evId, label: `${event.button}@${event.seq}`, type: 'Event' });
            this.addEdge({ from: event.inputCid, to: evId, label: 'click' });
            this.addEdge({ from: evId, to: event.outputCid, label: event.button });
        });

        snapshots.forEach((snapshot) => {
            const sid = snapshot.index.toString();
            this.addNode({ id: sid, label: `SU#${snapshot.index}`, type: 'SU' });
        });

        return { nodes: Array.from(this.nodes.values()), edges: this.edges };
    }
}
