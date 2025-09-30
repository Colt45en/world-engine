/**
 * Synchronization Protocol Definition for World Engine V3.1
 * Defines message types, state representations, and protocol standards
 * for distributed multi-user sessions
 */

/**
 * Protocol version and constants
 */
export const SYNC_PROTOCOL = {
  VERSION: '1.0.0',
  MESSAGE_TYPES: {
    // Connection management
    REGISTER: 'register',
    HEARTBEAT: 'heartbeat',
    DISCONNECT: 'disconnect',

    // Peer management
    PEER_JOINED: 'peer-joined',
    PEER_LEFT: 'peer-left',

    // WebRTC signaling
    RTC_OFFER: 'rtc-offer',
    RTC_ANSWER: 'rtc-answer',
    RTC_CANDIDATE: 'rtc-candidate',

    // State synchronization
    STATE_UPDATE: 'state-update',
    SYNC_REQUEST: 'sync-request',
    SYNC_RESPONSE: 'sync-response',

    // Operations
    OPERATION: 'operation',
    BUTTON_PRESS: 'button-press',
    MORPHEME_APPLICATION: 'morpheme-application',

    // Conflict resolution
    CONFLICT_DETECTED: 'conflict-detected',
    CONFLICT_RESOLVED: 'conflict-resolved'
  },

  OPERATION_TYPES: {
    STATE_UPDATE: 'state-update',
    BUTTON_PRESS: 'button-press',
    MORPHEME_APPLICATION: 'morpheme-application',
    UNDO: 'undo',
    REDO: 'redo'
  },

  CONFLICT_TYPES: {
    CONCURRENT_MODIFICATION: 'concurrent-modification',
    VERSION_MISMATCH: 'version-mismatch',
    NETWORK_PARTITION: 'network-partition'
  },

  RESOLUTION_STRATEGIES: {
    TIMESTAMP_ORDER: 'timestamp-order',
    PEER_ID_ORDER: 'peer-id-order',
    LAST_WRITER_WINS: 'last-writer-wins',
    OPERATIONAL_TRANSFORM: 'operational-transform',
    MERGE_AVERAGE: 'merge-average'
  }
};

/**
 * Message schema validation and creation utilities
 */
export class SyncMessageFactory {
  /**
   * Create a registration message
   * @param {Object} params - Registration parameters
   * @returns {Object} Registration message
   */
  static createRegistration({ peerId, sessionId, isHost, clientInfo = {} }) {
    return {
      type: SYNC_PROTOCOL.MESSAGE_TYPES.REGISTER,
      peerId,
      sessionId,
      isHost,
      clientInfo: {
        version: SYNC_PROTOCOL.VERSION,
        timestamp: Date.now(),
        capabilities: ['webrtc', 'websocket', 'conflict-resolution'],
        ...clientInfo
      }
    };
  }

  /**
   * Create a heartbeat message
   * @param {Object} params - Heartbeat parameters
   * @returns {Object} Heartbeat message
   */
  static createHeartbeat({ peerId, sessionId, status = 'active' }) {
    return {
      type: SYNC_PROTOCOL.MESSAGE_TYPES.HEARTBEAT,
      peerId,
      sessionId,
      status,
      timestamp: Date.now()
    };
  }

  /**
   * Create an operation message
   * @param {Object} params - Operation parameters
   * @returns {Object} Operation message
   */
  static createOperation({
    operationType,
    peerId,
    sessionId,
    data,
    previousState = null,
    operationId = null
  }) {
    return {
      type: SYNC_PROTOCOL.MESSAGE_TYPES.OPERATION,
      operation: {
        id: operationId || `op_${peerId}_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
        type: operationType,
        peerId,
        sessionId,
        timestamp: Date.now(),
        data,
        previousState
      }
    };
  }

  /**
   * Create a state update message
   * @param {Object} params - State update parameters
   * @returns {Object} State update message
   */
  static createStateUpdate({ peerId, sessionId, state, version, checksum = null }) {
    return {
      type: SYNC_PROTOCOL.MESSAGE_TYPES.STATE_UPDATE,
      peerId,
      sessionId,
      state: {
        ...state,
        version,
        timestamp: Date.now(),
        checksum: checksum || SyncMessageFactory.calculateStateChecksum(state)
      }
    };
  }

  /**
   * Create WebRTC signaling messages
   * @param {Object} params - Signaling parameters
   * @returns {Object} Signaling message
   */
  static createRTCSignaling({ type, fromPeer, targetPeer, sessionId, payload }) {
    const validTypes = [
      SYNC_PROTOCOL.MESSAGE_TYPES.RTC_OFFER,
      SYNC_PROTOCOL.MESSAGE_TYPES.RTC_ANSWER,
      SYNC_PROTOCOL.MESSAGE_TYPES.RTC_CANDIDATE
    ];

    if (!validTypes.includes(type)) {
      throw new Error(`Invalid RTC signaling type: ${type}`);
    }

    return {
      type,
      fromPeer,
      targetPeer,
      sessionId,
      timestamp: Date.now(),
      ...payload
    };
  }

  /**
   * Create conflict detection/resolution messages
   * @param {Object} params - Conflict parameters
   * @returns {Object} Conflict message
   */
  static createConflictMessage({
    conflictType,
    peerId,
    sessionId,
    operation,
    conflictsWith,
    resolution = null
  }) {
    const messageType = resolution
      ? SYNC_PROTOCOL.MESSAGE_TYPES.CONFLICT_RESOLVED
      : SYNC_PROTOCOL.MESSAGE_TYPES.CONFLICT_DETECTED;

    return {
      type: messageType,
      peerId,
      sessionId,
      timestamp: Date.now(),
      conflict: {
        type: conflictType,
        operation,
        conflictsWith,
        resolution
      }
    };
  }

  /**
   * Calculate checksum for state verification
   * @param {Object} state - State object
   * @returns {string} State checksum
   */
  static calculateStateChecksum(state) {
    // Simple checksum based on state JSON
    const stateStr = JSON.stringify(state, Object.keys(state).sort());
    let hash = 0;

    for (let i = 0; i < stateStr.length; i++) {
      const char = stateStr.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }

    return hash.toString(36);
  }

  /**
   * Validate message format
   * @param {Object} message - Message to validate
   * @returns {Object} Validation result
   */
  static validateMessage(message) {
    const errors = [];

    if (!message.type) {
      errors.push('Missing message type');
    } else if (!Object.values(SYNC_PROTOCOL.MESSAGE_TYPES).includes(message.type)) {
      errors.push(`Invalid message type: ${message.type}`);
    }

    if (!message.timestamp) {
      errors.push('Missing timestamp');
    } else if (typeof message.timestamp !== 'number') {
      errors.push('Invalid timestamp format');
    }

    // Type-specific validation
    switch (message.type) {
    case SYNC_PROTOCOL.MESSAGE_TYPES.REGISTER:
      if (!message.peerId) errors.push('Missing peerId in registration');
      if (typeof message.isHost !== 'boolean') errors.push('Invalid isHost flag');
      break;

    case SYNC_PROTOCOL.MESSAGE_TYPES.OPERATION:
      if (!message.operation) errors.push('Missing operation data');
      if (!message.operation.id) errors.push('Missing operation ID');
      if (!message.operation.type) errors.push('Missing operation type');
      break;

    case SYNC_PROTOCOL.MESSAGE_TYPES.STATE_UPDATE:
      if (!message.state) errors.push('Missing state data');
      if (typeof message.state.version !== 'number') errors.push('Invalid state version');
      break;
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }
}

/**
 * State representation and serialization utilities
 */
export class StateSerializer {
  /**
   * Serialize World Engine state for transmission
   * @param {Object} state - World Engine state
   * @returns {Object} Serialized state
   */
  static serialize(state) {
    return {
      x: Array.isArray(state.x) ? [...state.x] : [0, 0, 0],
      kappa: typeof state.kappa === 'number' ? state.kappa : 1.0,
      level: typeof state.level === 'number' ? state.level : 0,
      timestamp: state.timestamp || Date.now(),
      version: state.version || 0,
      metadata: {
        dimensions: state.x ? state.x.length : 3,
        origin: state.origin || 'local',
        sessionId: state.sessionId || null
      }
    };
  }

  /**
   * Deserialize state from transmission format
   * @param {Object} serializedState - Serialized state
   * @returns {Object} World Engine state
   */
  static deserialize(serializedState) {
    const state = {
      x: serializedState.x || [0, 0, 0],
      kappa: serializedState.kappa || 1.0,
      level: serializedState.level || 0,
      timestamp: serializedState.timestamp || Date.now(),
      version: serializedState.version || 0
    };

    // Restore metadata if present
    if (serializedState.metadata) {
      state.sessionId = serializedState.metadata.sessionId;
      state.origin = serializedState.metadata.origin;
    }

    return state;
  }

  /**
   * Calculate state delta between two states
   * @param {Object} oldState - Previous state
   * @param {Object} newState - Current state
   * @returns {Object} State delta
   */
  static calculateDelta(oldState, newState) {
    const delta = {
      timestamp: newState.timestamp,
      version: newState.version,
      changes: {}
    };

    // Calculate vector difference
    if (oldState.x && newState.x) {
      const deltaX = newState.x.map((xi, i) => xi - (oldState.x[i] || 0));
      if (deltaX.some(dx => Math.abs(dx) > 1e-10)) {
        delta.changes.x = deltaX;
      }
    }

    // Check scalar changes
    if (Math.abs(newState.kappa - oldState.kappa) > 1e-10) {
      delta.changes.kappa = newState.kappa - oldState.kappa;
    }

    if (newState.level !== oldState.level) {
      delta.changes.level = newState.level - oldState.level;
    }

    return delta;
  }

  /**
   * Apply delta to state
   * @param {Object} state - Base state
   * @param {Object} delta - State delta
   * @returns {Object} Updated state
   */
  static applyDelta(state, delta) {
    const newState = { ...state };

    // Apply vector changes
    if (delta.changes.x) {
      newState.x = state.x.map((xi, i) => xi + (delta.changes.x[i] || 0));
    }

    // Apply scalar changes
    if (delta.changes.kappa !== undefined) {
      newState.kappa = state.kappa + delta.changes.kappa;
    }

    if (delta.changes.level !== undefined) {
      newState.level = state.level + delta.changes.level;
    }

    // Update metadata
    newState.timestamp = delta.timestamp;
    newState.version = delta.version;

    return newState;
  }
}

/**
 * Network topology and peer management utilities
 */
export class NetworkTopology {
  constructor() {
    this.peers = new Map(); // peerId -> peer info
    this.connections = new Map(); // peerId -> connection status
    this.sessionTopology = new Map(); // sessionId -> peer list
  }

  /**
   * Add peer to topology
   * @param {Object} peerInfo - Peer information
   */
  addPeer(peerInfo) {
    this.peers.set(peerInfo.peerId, {
      ...peerInfo,
      joinedAt: Date.now(),
      lastSeen: Date.now()
    });

    // Update session topology
    if (peerInfo.sessionId) {
      const sessionPeers = this.sessionTopology.get(peerInfo.sessionId) || new Set();
      sessionPeers.add(peerInfo.peerId);
      this.sessionTopology.set(peerInfo.sessionId, sessionPeers);
    }
  }

  /**
   * Remove peer from topology
   * @param {string} peerId - Peer ID to remove
   */
  removePeer(peerId) {
    const peerInfo = this.peers.get(peerId);

    if (peerInfo && peerInfo.sessionId) {
      const sessionPeers = this.sessionTopology.get(peerInfo.sessionId);
      if (sessionPeers) {
        sessionPeers.delete(peerId);
        if (sessionPeers.size === 0) {
          this.sessionTopology.delete(peerInfo.sessionId);
        }
      }
    }

    this.peers.delete(peerId);
    this.connections.delete(peerId);
  }

  /**
   * Update peer connection status
   * @param {string} peerId - Peer ID
   * @param {string} status - Connection status
   */
  updateConnectionStatus(peerId, status) {
    this.connections.set(peerId, {
      status,
      lastUpdate: Date.now()
    });

    // Update last seen
    const peerInfo = this.peers.get(peerId);
    if (peerInfo) {
      peerInfo.lastSeen = Date.now();
    }
  }

  /**
   * Get peers in a session
   * @param {string} sessionId - Session ID
   * @returns {Object[]} Array of peer info objects
   */
  getSessionPeers(sessionId) {
    const peerIds = this.sessionTopology.get(sessionId) || new Set();
    return Array.from(peerIds)
      .map(peerId => this.peers.get(peerId))
      .filter(peer => peer);
  }

  /**
   * Get connected peers
   * @returns {Object[]} Array of connected peer info objects
   */
  getConnectedPeers() {
    const connectedIds = Array.from(this.connections.entries())
      .filter(([, connection]) => connection.status === 'connected')
      .map(([peerId]) => peerId);

    return connectedIds
      .map(peerId => this.peers.get(peerId))
      .filter(peer => peer);
  }

  /**
   * Get topology statistics
   * @returns {Object} Topology statistics
   */
  getStatistics() {
    const now = Date.now();
    const peers = Array.from(this.peers.values());

    return {
      total_peers: peers.length,
      connected_peers: Array.from(this.connections.values())
        .filter(conn => conn.status === 'connected').length,
      active_sessions: this.sessionTopology.size,
      avg_session_size: this.sessionTopology.size > 0
        ? Array.from(this.sessionTopology.values()).reduce((sum, peers) => sum + peers.size, 0) / this.sessionTopology.size
        : 0,
      recent_activity: peers.filter(peer => now - peer.lastSeen < 60000).length,
      oldest_peer: Math.min(...peers.map(peer => peer.joinedAt)),
      newest_peer: Math.max(...peers.map(peer => peer.joinedAt))
    };
  }

  /**
   * Clean up inactive peers
   * @param {number} timeoutMs - Timeout in milliseconds
   */
  cleanupInactivePeers(timeoutMs = 300000) { // 5 minutes default
    const now = Date.now();
    const inactivePeers = Array.from(this.peers.entries())
      .filter(([, peer]) => now - peer.lastSeen > timeoutMs)
      .map(([peerId]) => peerId);

    inactivePeers.forEach(peerId => this.removePeer(peerId));

    return inactivePeers.length;
  }
}
