/**
 * Distributed Synchronization System for World Engine V3.1
 * Enables multi-user sessions with WebRTC/WebSocket communication
 * Includes conflict resolution and state merging capabilities
 */

export class DistributedSyncManager {
  constructor(options = {}) {
    this.peerId = options.peerId || this.generatePeerId();
    this.isHost = options.isHost || false;
    this.sessionId = options.sessionId || null;
    this.peers = new Map(); // peerId -> PeerConnection
    this.state = {
      x: [0, 0, 0],
      kappa: 1.0,
      level: 0,
      timestamp: Date.now(),
      version: 0
    };
    this.pendingOperations = new Map(); // operationId -> operation
    this.operationHistory = []; // Ordered history of applied operations
    this.conflictResolver = new ConflictResolver();
    this.eventEmitter = new Map();

    this.config = {
      maxHistorySize: options.maxHistorySize || 1000,
      syncInterval: options.syncInterval || 100,
      heartbeatInterval: options.heartbeatInterval || 5000,
      connectionTimeout: options.connectionTimeout || 30000,
      ...options
    };

    this.websocketUrl = options.websocketUrl || 'ws://localhost:8080/sync';
    this.websocket = null;
    this.isConnected = false;

    this.setupWebSocket();
    this.startHeartbeat();
  }

  /**
   * Generate unique peer ID
   * @returns {string} Unique peer identifier
   */
  generatePeerId() {
    return `peer_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Setup WebSocket connection for signaling
   */
  setupWebSocket() {
    try {
      this.websocket = new WebSocket(this.websocketUrl);

      this.websocket.onopen = () => {
        this.isConnected = true;
        console.log(`🌐 Connected to sync server as ${this.peerId}`);
        this.emit('connected', { peerId: this.peerId });

        // Register with server
        this.send({
          type: 'register',
          peerId: this.peerId,
          sessionId: this.sessionId,
          isHost: this.isHost
        });
      };

      this.websocket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this.handleSignalingMessage(message);
        } catch (error) {
          console.error('Failed to parse sync message:', error);
        }
      };

      this.websocket.onclose = () => {
        this.isConnected = false;
        console.log('🔌 Disconnected from sync server');
        this.emit('disconnected');

        // Attempt reconnection after delay
        setTimeout(() => this.setupWebSocket(), 3000);
      };

      this.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.emit('error', error);
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      this.emit('error', error);
    }
  }

  /**
   * Handle signaling messages from WebSocket server
   * @param {Object} message - Signaling message
   */
  handleSignalingMessage(message) {
    switch (message.type) {
    case 'peer-joined':
      this.handlePeerJoined(message);
      break;
    case 'peer-left':
      this.handlePeerLeft(message);
      break;
    case 'rtc-offer':
      this.handleRTCOffer(message);
      break;
    case 'rtc-answer':
      this.handleRTCAnswer(message);
      break;
    case 'rtc-candidate':
      this.handleRTCCandidate(message);
      break;
    case 'sync-state':
      this.handleStateSync(message);
      break;
    case 'operation':
      this.handleRemoteOperation(message);
      break;
    default:
      console.warn('Unknown signaling message type:', message.type);
    }
  }

  /**
   * Handle peer joining the session
   * @param {Object} message - Peer joined message
   */
  async handlePeerJoined(message) {
    const { peerId } = message;

    if (peerId === this.peerId) return; // Don't connect to self

    console.log(`👥 Peer joined: ${peerId}`);
    this.emit('peer-joined', { peerId });

    if (this.isHost) {
      // Host initiates WebRTC connection
      await this.createRTCConnection(peerId);
    }
  }

  /**
   * Handle peer leaving the session
   * @param {Object} message - Peer left message
   */
  handlePeerLeft(message) {
    const { peerId } = message;

    if (this.peers.has(peerId)) {
      this.peers.get(peerId).close();
      this.peers.delete(peerId);
    }

    console.log(`👋 Peer left: ${peerId}`);
    this.emit('peer-left', { peerId });
  }

  /**
   * Create WebRTC connection to peer
   * @param {string} peerId - Target peer ID
   */
  async createRTCConnection(peerId) {
    try {
      const peerConnection = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          { urls: 'stun:stun1.l.google.com:19302' }
        ]
      });

      // Setup data channel for direct peer communication
      const dataChannel = peerConnection.createDataChannel('worldengine', {
        ordered: true
      });

      dataChannel.onopen = () => {
        console.log(`🔗 Data channel open to ${peerId}`);
        this.emit('peer-connected', { peerId });
      };

      dataChannel.onmessage = (event) => {
        try {
          const operation = JSON.parse(event.data);
          this.handleRemoteOperation(operation);
        } catch (error) {
          console.error('Failed to parse P2P message:', error);
        }
      };

      // Handle ICE candidates
      peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
          this.send({
            type: 'rtc-candidate',
            targetPeer: peerId,
            candidate: event.candidate
          });
        }
      };

      // Store connection
      this.peers.set(peerId, { peerConnection, dataChannel });

      // Create and send offer
      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);

      this.send({
        type: 'rtc-offer',
        targetPeer: peerId,
        offer: offer
      });

    } catch (error) {
      console.error('Failed to create RTC connection:', error);
      this.emit('error', error);
    }
  }

  /**
   * Handle WebRTC offer from peer
   * @param {Object} message - RTC offer message
   */
  async handleRTCOffer(message) {
    const { fromPeer, offer } = message;

    try {
      const peerConnection = new RTCPeerConnection({
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          { urls: 'stun:stun1.l.google.com:19302' }
        ]
      });

      // Handle incoming data channel
      peerConnection.ondatachannel = (event) => {
        const dataChannel = event.channel;

        dataChannel.onopen = () => {
          console.log(`🔗 Data channel received from ${fromPeer}`);
          this.emit('peer-connected', { peerId: fromPeer });
        };

        dataChannel.onmessage = (event) => {
          try {
            const operation = JSON.parse(event.data);
            this.handleRemoteOperation(operation);
          } catch (error) {
            console.error('Failed to parse P2P message:', error);
          }
        };

        // Store connection
        this.peers.set(fromPeer, { peerConnection, dataChannel });
      };

      // Handle ICE candidates
      peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
          this.send({
            type: 'rtc-candidate',
            targetPeer: fromPeer,
            candidate: event.candidate
          });
        }
      };

      await peerConnection.setRemoteDescription(offer);

      const answer = await peerConnection.createAnswer();
      await peerConnection.setLocalDescription(answer);

      this.send({
        type: 'rtc-answer',
        targetPeer: fromPeer,
        answer: answer
      });

    } catch (error) {
      console.error('Failed to handle RTC offer:', error);
      this.emit('error', error);
    }
  }

  /**
   * Handle WebRTC answer from peer
   * @param {Object} message - RTC answer message
   */
  async handleRTCAnswer(message) {
    const { fromPeer, answer } = message;

    try {
      const peer = this.peers.get(fromPeer);
      if (peer && peer.peerConnection) {
        await peer.peerConnection.setRemoteDescription(answer);
      }
    } catch (error) {
      console.error('Failed to handle RTC answer:', error);
      this.emit('error', error);
    }
  }

  /**
   * Handle ICE candidate from peer
   * @param {Object} message - ICE candidate message
   */
  async handleRTCCandidate(message) {
    const { fromPeer, candidate } = message;

    try {
      const peer = this.peers.get(fromPeer);
      if (peer && peer.peerConnection) {
        await peer.peerConnection.addIceCandidate(candidate);
      }
    } catch (error) {
      console.error('Failed to handle ICE candidate:', error);
      this.emit('error', error);
    }
  }

  /**
   * Apply an operation locally and broadcast to peers
   * @param {Object} operation - World Engine operation
   */
  applyOperation(operation) {
    // Add metadata
    operation.id = this.generateOperationId();
    operation.peerId = this.peerId;
    operation.timestamp = Date.now();
    operation.version = this.state.version + 1;

    // Apply locally
    this.executeOperation(operation);

    // Broadcast to peers
    this.broadcastOperation(operation);

    // Store in history
    this.operationHistory.push(operation);
    this.maintainHistorySize();
  }

  /**
   * Generate unique operation ID
   * @returns {string} Unique operation identifier
   */
  generateOperationId() {
    return `op_${this.peerId}_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
  }

  /**
   * Execute operation on local state
   * @param {Object} operation - Operation to execute
   */
  executeOperation(operation) {
    switch (operation.type) {
    case 'state-update':
      this.state = { ...this.state, ...operation.data, version: operation.version };
      break;
    case 'button-press':
      this.handleButtonPress(operation.data);
      break;
    case 'morpheme-application':
      this.handleMorphemeApplication(operation.data);
      break;
    default:
      console.warn('Unknown operation type:', operation.type);
    }

    this.emit('operation-applied', operation);
  }

  /**
   * Handle button press operation
   * @param {Object} data - Button press data
   */
  handleButtonPress(data) {
    // Apply button transformation to state
    // This would integrate with the existing World Engine button system
    const { button, previousState } = data;

    // Simple state transformation (would use actual World Engine math)
    if (button.M && Array.isArray(button.M)) {
      const alpha = button.alpha || 1.0;
      this.state.x = this.state.x.map((xi, i) => {
        let newXi = 0;
        for (let j = 0; j < button.M[i].length; j++) {
          newXi += alpha * button.M[i][j] * this.state.x[j];
        }
        return newXi;
      });
    }

    this.state.version++;
    this.state.timestamp = Date.now();
  }

  /**
   * Handle morpheme application operation
   * @param {Object} data - Morpheme application data
   */
  handleMorphemeApplication(data) {
    // Apply sequence of buttons from morpheme
    const { morpheme } = data;

    if (morpheme.pattern && Array.isArray(morpheme.pattern)) {
      // Would integrate with morpheme discovery system
      console.log(`Applying morpheme: ${morpheme.pattern.join('→')}`);
      this.state.version++;
      this.state.timestamp = Date.now();
    }
  }

  /**
   * Broadcast operation to all connected peers
   * @param {Object} operation - Operation to broadcast
   */
  broadcastOperation(operation) {
    const message = JSON.stringify(operation);

    // Send via WebRTC data channels
    for (const [peerId, peer] of this.peers.entries()) {
      if (peer.dataChannel && peer.dataChannel.readyState === 'open') {
        try {
          peer.dataChannel.send(message);
        } catch (error) {
          console.error(`Failed to send to peer ${peerId}:`, error);
        }
      }
    }

    // Fallback via WebSocket
    if (this.isConnected) {
      this.send({
        type: 'operation',
        operation: operation
      });
    }
  }

  /**
   * Handle operation received from remote peer
   * @param {Object} operation - Remote operation
   */
  handleRemoteOperation(operation) {
    // Check for conflicts
    const conflict = this.conflictResolver.detectConflict(operation, this.operationHistory);

    if (conflict) {
      const resolved = this.conflictResolver.resolve(conflict, this.state);
      this.executeOperation(resolved);
      this.emit('conflict-resolved', { original: operation, resolved });
    } else {
      this.executeOperation(operation);
    }

    // Add to history
    this.operationHistory.push(operation);
    this.maintainHistorySize();
  }

  /**
   * Handle state synchronization message
   * @param {Object} message - State sync message
   */
  handleStateSync(message) {
    const { state, fromPeer } = message;

    // Merge state if remote version is newer
    if (state.version > this.state.version) {
      const mergedState = this.conflictResolver.mergeStates(this.state, state);
      this.state = mergedState;
      this.emit('state-synced', { fromPeer, mergedState });
    }
  }

  /**
   * Request full state sync from peers
   */
  requestStateSync() {
    this.send({
      type: 'sync-request',
      currentVersion: this.state.version,
      peerId: this.peerId
    });
  }

  /**
   * Send message via WebSocket
   * @param {Object} message - Message to send
   */
  send(message) {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify(message));
    }
  }

  /**
   * Start heartbeat mechanism
   */
  startHeartbeat() {
    setInterval(() => {
      if (this.isConnected) {
        this.send({
          type: 'heartbeat',
          peerId: this.peerId,
          timestamp: Date.now()
        });
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Maintain operation history size limit
   */
  maintainHistorySize() {
    if (this.operationHistory.length > this.config.maxHistorySize) {
      this.operationHistory.splice(0, this.operationHistory.length - this.config.maxHistorySize);
    }
  }

  /**
   * Get synchronization status
   * @returns {Object} Sync status information
   */
  getStatus() {
    return {
      peerId: this.peerId,
      isConnected: this.isConnected,
      isHost: this.isHost,
      sessionId: this.sessionId,
      connectedPeers: Array.from(this.peers.keys()),
      stateVersion: this.state.version,
      operationHistory: this.operationHistory.length,
      pendingOperations: this.pendingOperations.size
    };
  }

  /**
   * Add event listener
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   */
  on(event, callback) {
    if (!this.eventEmitter.has(event)) {
      this.eventEmitter.set(event, []);
    }
    this.eventEmitter.get(event).push(callback);
  }

  /**
   * Remove event listener
   * @param {string} event - Event name
   * @param {Function} callback - Callback function
   */
  off(event, callback) {
    const callbacks = this.eventEmitter.get(event);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  /**
   * Emit event to listeners
   * @param {string} event - Event name
   * @param {*} data - Event data
   */
  emit(event, data) {
    const callbacks = this.eventEmitter.get(event) || [];
    callbacks.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error(`Sync event error (${event}):`, error);
      }
    });
  }

  /**
   * Disconnect from all peers and server
   */
  disconnect() {
    // Close all peer connections
    for (const [peerId, peer] of this.peers.entries()) {
      peer.peerConnection.close();
    }
    this.peers.clear();

    // Close WebSocket
    if (this.websocket) {
      this.websocket.close();
    }

    this.isConnected = false;
    this.emit('disconnected');
  }
}

/**
 * Conflict Resolution System
 */
export class ConflictResolver {
  /**
   * Detect conflicts between operations
   * @param {Object} operation - New operation
   * @param {Object[]} history - Operation history
   * @returns {Object|null} Conflict information or null
   */
  detectConflict(operation, history) {
    // Find concurrent operations (same timestamp range)
    const timeWindow = 1000; // 1 second window
    const concurrent = history.filter(op =>
      Math.abs(op.timestamp - operation.timestamp) < timeWindow &&
      op.peerId !== operation.peerId
    );

    if (concurrent.length === 0) return null;

    // Check for state conflicts
    for (const concurrentOp of concurrent) {
      if (this.operationsConflict(operation, concurrentOp)) {
        return {
          type: 'concurrent-modification',
          operation: operation,
          conflictsWith: concurrentOp,
          timestamp: operation.timestamp
        };
      }
    }

    return null;
  }

  /**
   * Check if two operations conflict
   * @param {Object} op1 - First operation
   * @param {Object} op2 - Second operation
   * @returns {boolean} True if operations conflict
   */
  operationsConflict(op1, op2) {
    // Operations conflict if they modify the same state component
    if (op1.type === 'state-update' && op2.type === 'state-update') {
      return this.stateUpdatesConflict(op1.data, op2.data);
    }

    if (op1.type === 'button-press' && op2.type === 'button-press') {
      return this.buttonPressesConflict(op1.data, op2.data);
    }

    return false;
  }

  /**
   * Check if state updates conflict
   * @param {Object} data1 - First update data
   * @param {Object} data2 - Second update data
   * @returns {boolean} True if updates conflict
   */
  stateUpdatesConflict(data1, data2) {
    // Conflict if both update the same state property
    const keys1 = new Set(Object.keys(data1));
    const keys2 = new Set(Object.keys(data2));

    for (const key of keys1) {
      if (keys2.has(key)) {
        return true;
      }
    }

    return false;
  }

  /**
   * Check if button presses conflict
   * @param {Object} data1 - First button press
   * @param {Object} data2 - Second button press
   * @returns {boolean} True if presses conflict
   */
  buttonPressesConflict(data1, data2) {
    // Button presses always conflict if concurrent
    // as they both modify the same state vector
    return true;
  }

  /**
   * Resolve a detected conflict
   * @param {Object} conflict - Conflict information
   * @param {Object} currentState - Current system state
   * @returns {Object} Resolved operation
   */
  resolve(conflict, currentState) {
    switch (conflict.type) {
    case 'concurrent-modification':
      return this.resolveConcurrentModification(conflict, currentState);
    default:
      console.warn('Unknown conflict type:', conflict.type);
      return conflict.operation;
    }
  }

  /**
   * Resolve concurrent modification conflict
   * @param {Object} conflict - Conflict details
   * @param {Object} currentState - Current state
   * @returns {Object} Resolved operation
   */
  resolveConcurrentModification(conflict, currentState) {
    const { operation, conflictsWith } = conflict;

    // Use timestamp ordering as primary resolution strategy
    if (operation.timestamp < conflictsWith.timestamp) {
      // Operation came first, apply it
      return operation;
    } else if (operation.timestamp > conflictsWith.timestamp) {
      // Conflicting operation came first, transform this operation
      return this.transformOperation(operation, conflictsWith, currentState);
    } else {
      // Same timestamp, use peer ID ordering
      if (operation.peerId < conflictsWith.peerId) {
        return operation;
      } else {
        return this.transformOperation(operation, conflictsWith, currentState);
      }
    }
  }

  /**
   * Transform operation to account for conflicting operation
   * @param {Object} operation - Operation to transform
   * @param {Object} conflictsWith - Conflicting operation
   * @param {Object} currentState - Current state
   * @returns {Object} Transformed operation
   */
  transformOperation(operation, conflictsWith, currentState) {
    // Simple transformation: average the conflicting changes
    if (operation.type === 'state-update' && conflictsWith.type === 'state-update') {
      const transformedData = { ...operation.data };

      for (const key in conflictsWith.data) {
        if (transformedData.hasOwnProperty(key)) {
          // Average the values
          transformedData[key] = (transformedData[key] + conflictsWith.data[key]) / 2;
        }
      }

      return { ...operation, data: transformedData, transformed: true };
    }

    return operation;
  }

  /**
   * Merge two states intelligently
   * @param {Object} state1 - First state
   * @param {Object} state2 - Second state
   * @returns {Object} Merged state
   */
  mergeStates(state1, state2) {
    // Use the state with higher version
    const newerState = state2.version > state1.version ? state2 : state1;
    const olderState = state2.version > state1.version ? state1 : state2;

    // Merge vector components using weighted average based on timestamps
    const timeDiff = Math.abs(newerState.timestamp - olderState.timestamp);
    const weight = Math.min(1, timeDiff / 10000); // Decay over 10 seconds

    const mergedX = newerState.x.map((newXi, i) => {
      const oldXi = olderState.x[i] || 0;
      return weight * newXi + (1 - weight) * oldXi;
    });

    return {
      ...newerState,
      x: mergedX,
      timestamp: Math.max(state1.timestamp, state2.timestamp),
      version: Math.max(state1.version, state2.version) + 1
    };
  }
}
