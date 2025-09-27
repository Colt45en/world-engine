/**
 * World Engine V3.1 - AMQP-Style Robust Channel Communication
 * Inspired by AMQP channel.txt attachment
 *
 * Provides reliable messaging channels with error handling, retry logic,
 * and robust inter-component communication for studio controllers
 */

export class RobustChannel {
  constructor(name, options = {}) {
    this.name = name;
    this.options = {
      maxRetries: 3,
      retryDelay: 1000,
      timeout: 30000,
      persistent: true,
      enableLogging: true,
      ...options
    };

    // Channel state
    this.state = 'disconnected';
    this.messageQueue = [];
    this.subscribers = new Map();
    this.handlers = new Map();
    this.metrics = new ChannelMetrics();

    // Connection and error handling
    this.reconnectAttempts = 0;
    this.lastError = null;
    this.connectionId = null;

    // Message tracking
    this.messageId = 0;
    this.pendingMessages = new Map();
    this.acknowledgments = new Map();

    // Initialize subsystems
    this.initializeChannel();
  }

  initializeChannel() {
    this.state = 'connecting';
    this.connectionId = this.generateConnectionId();

    if (this.options.enableLogging) {
      console.log(`[${this.name}] Initializing robust channel ${this.connectionId}`);
    }

    // Set up core handlers
    this.registerHandler('system.ping', this.handlePing.bind(this));
    this.registerHandler('system.pong', this.handlePong.bind(this));
    this.registerHandler('system.error', this.handleError.bind(this));
    this.registerHandler('system.ack', this.handleAcknowledgment.bind(this));

    this.state = 'connected';
    this.emit('channel.ready', { channel: this.name, connectionId: this.connectionId });
  }

  // === CORE MESSAGING METHODS ===

  /**
   * Send message with reliability guarantees
   * @param {string} routingKey - Message route identifier
   * @param {Object} message - Message payload
   * @param {Object} options - Message options (persistent, priority, ttl)
   */
  async send(routingKey, message, options = {}) {
    const messageId = this.generateMessageId();
    const envelope = {
      id: messageId,
      routingKey,
      payload: message,
      timestamp: Date.now(),
      channel: this.name,
      options: {
        persistent: options.persistent ?? this.options.persistent,
        priority: options.priority || 0,
        ttl: options.ttl || this.options.timeout,
        requireAck: options.requireAck !== false,
        ...options
      }
    };

    // Queue message if channel not ready
    if (this.state !== 'connected') {
      if (this.options.persistent) {
        this.messageQueue.push(envelope);
        return { messageId, status: 'queued' };
      } else {
        throw new Error(`Channel ${this.name} not ready for non-persistent message`);
      }
    }

    try {
      return await this.deliverMessage(envelope);
    } catch (error) {
      this.metrics.recordError(error);

      if (envelope.options.persistent) {
        this.messageQueue.push(envelope);
        return { messageId, status: 'queued', error: error.message };
      } else {
        throw error;
      }
    }
  }

  /**
   * Subscribe to messages with pattern matching
   * @param {string} pattern - Routing pattern (supports wildcards)
   * @param {Function} handler - Message handler function
   * @param {Object} options - Subscription options
   */
  subscribe(pattern, handler, options = {}) {
    const subscriptionId = this.generateSubscriptionId();
    const subscription = {
      id: subscriptionId,
      pattern,
      handler,
      options: {
        autoAck: options.autoAck !== false,
        exclusive: options.exclusive || false,
        durable: options.durable !== false,
        ...options
      },
      metrics: {
        messagesReceived: 0,
        messagesProcessed: 0,
        errors: 0
      }
    };

    this.subscribers.set(subscriptionId, subscription);

    if (this.options.enableLogging) {
      console.log(`[${this.name}] Subscribed to pattern: ${pattern}`);
    }

    return subscriptionId;
  }

  /**
   * Unsubscribe from message pattern
   * @param {string} subscriptionId - Subscription identifier
   */
  unsubscribe(subscriptionId) {
    const subscription = this.subscribers.get(subscriptionId);
    if (subscription) {
      this.subscribers.delete(subscriptionId);

      if (this.options.enableLogging) {
        console.log(`[${this.name}] Unsubscribed from: ${subscription.pattern}`);
      }

      return true;
    }
    return false;
  }

  /**
   * Register message handler for specific routing key
   * @param {string} routingKey - Exact routing key
   * @param {Function} handler - Handler function
   */
  registerHandler(routingKey, handler) {
    this.handlers.set(routingKey, handler);
  }

  /**
   * Acknowledge message processing
   * @param {string} messageId - Message identifier
   * @param {boolean} success - Processing success status
   */
  acknowledge(messageId, success = true) {
    const ack = {
      messageId,
      success,
      timestamp: Date.now(),
      channel: this.name
    };

    this.acknowledgments.set(messageId, ack);

    // Send acknowledgment if required
    this.emit('system.ack', ack);

    // Clean up pending message
    this.pendingMessages.delete(messageId);

    this.metrics.recordAcknowledgment(success);
  }

  // === MESSAGE DELIVERY SYSTEM ===

  async deliverMessage(envelope) {
    this.metrics.recordMessageSent();

    // Add to pending messages for tracking
    this.pendingMessages.set(envelope.id, {
      envelope,
      timestamp: Date.now(),
      attempts: 0
    });

    try {
      // Find matching handlers and subscribers
      const targets = this.findMessageTargets(envelope.routingKey);

      if (targets.length === 0) {
        if (this.options.enableLogging) {
          console.warn(`[${this.name}] No handlers for routing key: ${envelope.routingKey}`);
        }
        return { messageId: envelope.id, status: 'no_handlers' };
      }

      // Deliver to all targets
      const deliveryPromises = targets.map(target =>
        this.deliverToTarget(envelope, target)
      );

      const results = await Promise.allSettled(deliveryPromises);

      // Process delivery results
      const successful = results.filter(r => r.status === 'fulfilled').length;
      const failed = results.filter(r => r.status === 'rejected').length;

      if (failed > 0 && envelope.options.requireAck) {
        throw new Error(`Delivery failed for ${failed}/${targets.length} targets`);
      }

      return {
        messageId: envelope.id,
        status: 'delivered',
        successful,
        failed,
        targets: targets.length
      };

    } catch (error) {
      return await this.handleDeliveryFailure(envelope, error);
    }
  }

  async deliverToTarget(envelope, target) {
    try {
      const context = {
        messageId: envelope.id,
        routingKey: envelope.routingKey,
        channel: this,
        timestamp: envelope.timestamp,
        acknowledge: (success) => this.acknowledge(envelope.id, success)
      };

      if (target.type === 'handler') {
        await target.handler(envelope.payload, context);
      } else if (target.type === 'subscriber') {
        target.subscription.metrics.messagesReceived++;
        await target.subscription.handler(envelope.payload, context);
        target.subscription.metrics.messagesProcessed++;

        // Auto-acknowledge if enabled
        if (target.subscription.options.autoAck) {
          this.acknowledge(envelope.id, true);
        }
      }

      this.metrics.recordDeliverySuccess();

    } catch (error) {
      if (target.type === 'subscriber') {
        target.subscription.metrics.errors++;
      }

      this.metrics.recordDeliveryFailure();
      throw error;
    }
  }

  async handleDeliveryFailure(envelope, error) {
    const pending = this.pendingMessages.get(envelope.id);
    if (!pending) return { messageId: envelope.id, status: 'lost' };

    pending.attempts++;

    if (pending.attempts >= this.options.maxRetries) {
      this.pendingMessages.delete(envelope.id);

      // Send to dead letter queue or handle failure
      await this.handleDeadLetter(envelope, error);

      return {
        messageId: envelope.id,
        status: 'failed',
        attempts: pending.attempts,
        error: error.message
      };
    }

    // Schedule retry
    setTimeout(() => {
      this.retryMessage(envelope.id);
    }, this.options.retryDelay * pending.attempts);

    return {
      messageId: envelope.id,
      status: 'retry_scheduled',
      attempts: pending.attempts,
      nextRetry: Date.now() + (this.options.retryDelay * pending.attempts)
    };
  }

  async retryMessage(messageId) {
    const pending = this.pendingMessages.get(messageId);
    if (!pending) return;

    if (this.options.enableLogging) {
      console.log(`[${this.name}] Retrying message ${messageId}, attempt ${pending.attempts + 1}`);
    }

    try {
      await this.deliverMessage(pending.envelope);
    } catch (error) {
      // handleDeliveryFailure will manage further retries
      await this.handleDeliveryFailure(pending.envelope, error);
    }
  }

  async handleDeadLetter(envelope, error) {
    const deadLetter = {
      originalEnvelope: envelope,
      error: error.message,
      timestamp: Date.now(),
      channel: this.name
    };

    // Emit dead letter event for external handling
    this.emit('message.dead_letter', deadLetter);

    if (this.options.enableLogging) {
      console.error(`[${this.name}] Message ${envelope.id} moved to dead letter: ${error.message}`);
    }
  }

  // === PATTERN MATCHING AND ROUTING ===

  findMessageTargets(routingKey) {
    const targets = [];

    // Exact handler matches
    const handler = this.handlers.get(routingKey);
    if (handler) {
      targets.push({ type: 'handler', handler });
    }

    // Pattern-based subscriber matches
    for (const [subscriptionId, subscription] of this.subscribers.entries()) {
      if (this.matchPattern(routingKey, subscription.pattern)) {
        targets.push({
          type: 'subscriber',
          subscription,
          subscriptionId
        });
      }
    }

    return targets;
  }

  matchPattern(routingKey, pattern) {
    // Convert AMQP-style pattern to regex
    const regex = pattern
      .replace(/\*/g, '[^.]*')  // * matches any single word
      .replace(/#/g, '.*');     // # matches zero or more words

    return new RegExp(`^${regex}$`).test(routingKey);
  }

  // === CONNECTION MANAGEMENT ===

  async reconnect() {
    if (this.state === 'connecting') return;

    this.state = 'connecting';
    this.reconnectAttempts++;

    if (this.options.enableLogging) {
      console.log(`[${this.name}] Attempting reconnection #${this.reconnectAttempts}`);
    }

    try {
      // Simulate reconnection logic
      await this.sleep(this.options.retryDelay);

      this.connectionId = this.generateConnectionId();
      this.state = 'connected';

      // Process queued messages
      await this.processQueuedMessages();

      this.emit('channel.reconnected', {
        channel: this.name,
        connectionId: this.connectionId,
        attempts: this.reconnectAttempts
      });

      this.reconnectAttempts = 0;

    } catch (error) {
      this.lastError = error;
      this.state = 'error';

      if (this.reconnectAttempts < this.options.maxRetries) {
        setTimeout(() => this.reconnect(), this.options.retryDelay * this.reconnectAttempts);
      } else {
        this.emit('channel.failed', {
          channel: this.name,
          error: error.message,
          attempts: this.reconnectAttempts
        });
      }
    }
  }

  async processQueuedMessages() {
    if (this.messageQueue.length === 0) return;

    if (this.options.enableLogging) {
      console.log(`[${this.name}] Processing ${this.messageQueue.length} queued messages`);
    }

    const queue = [...this.messageQueue];
    this.messageQueue = [];

    for (const envelope of queue) {
      try {
        await this.deliverMessage(envelope);
      } catch (error) {
        if (this.options.enableLogging) {
          console.error(`[${this.name}] Failed to process queued message ${envelope.id}:`, error);
        }
      }
    }
  }

  disconnect() {
    this.state = 'disconnected';
    this.messageQueue = [];
    this.pendingMessages.clear();
    this.acknowledgments.clear();

    this.emit('channel.disconnected', { channel: this.name });
  }

  // === EVENT SYSTEM ===

  emit(eventType, data) {
    // Simple event emission for internal use
    const event = {
      type: eventType,
      data,
      timestamp: Date.now(),
      channel: this.name
    };

    // Send to system event handler if exists
    const systemHandler = this.handlers.get(eventType);
    if (systemHandler) {
      try {
        systemHandler(data, { messageId: this.generateMessageId(), channel: this });
      } catch (error) {
        console.error(`[${this.name}] Error in system handler for ${eventType}:`, error);
      }
    }

    // Process through normal routing for subscribers
    if (this.state === 'connected') {
      this.send(eventType, data, { requireAck: false, persistent: false }).catch(err => {
        console.error(`[${this.name}] Error emitting event ${eventType}:`, err);
      });
    }
  }

  // === SYSTEM MESSAGE HANDLERS ===

  async handlePing(data, context) {
    const pong = {
      pingId: data.id,
      timestamp: Date.now(),
      channel: this.name,
      connectionId: this.connectionId
    };

    await this.send('system.pong', pong);
  }

  handlePong(data, context) {
    this.metrics.recordLatency(Date.now() - data.timestamp);
  }

  handleError(data, context) {
    this.lastError = data;
    this.metrics.recordError(data);

    if (this.options.enableLogging) {
      console.error(`[${this.name}] Received error:`, data);
    }
  }

  handleAcknowledgment(data, context) {
    const { messageId, success } = data;
    this.acknowledgments.set(messageId, data);

    if (!success && this.pendingMessages.has(messageId)) {
      // Handle negative acknowledgment
      const pending = this.pendingMessages.get(messageId);
      this.retryMessage(messageId);
    }
  }

  // === UTILITY METHODS ===

  generateConnectionId() {
    return `${this.name}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  generateMessageId() {
    return `msg_${++this.messageId}_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`;
  }

  generateSubscriptionId() {
    return `sub_${Date.now()}_${Math.random().toString(36).substr(2, 8)}`;
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // === METRICS AND MONITORING ===

  getMetrics() {
    return {
      channel: this.name,
      connectionId: this.connectionId,
      state: this.state,
      messagesSent: this.metrics.messagesSent,
      messagesReceived: this.metrics.messagesReceived,
      errors: this.metrics.errors,
      reconnectAttempts: this.reconnectAttempts,
      queuedMessages: this.messageQueue.length,
      pendingMessages: this.pendingMessages.size,
      subscribers: this.subscribers.size,
      handlers: this.handlers.size,
      uptime: Date.now() - (this.metrics.startTime || Date.now())
    };
  }

  getHealth() {
    const metrics = this.getMetrics();
    const errorRate = metrics.errors / Math.max(metrics.messagesSent, 1);

    return {
      status: this.state,
      healthy: this.state === 'connected' && errorRate < 0.1,
      errorRate,
      lastError: this.lastError,
      metrics
    };
  }
}

// === CHANNEL METRICS CLASS ===

class ChannelMetrics {
  constructor() {
    this.startTime = Date.now();
    this.messagesSent = 0;
    this.messagesReceived = 0;
    this.deliverySuccesses = 0;
    this.deliveryFailures = 0;
    this.errors = 0;
    this.acknowledgments = 0;
    this.negativeAcks = 0;
    this.latencies = [];
  }

  recordMessageSent() {
    this.messagesSent++;
  }

  recordMessageReceived() {
    this.messagesReceived++;
  }

  recordDeliverySuccess() {
    this.deliverySuccesses++;
  }

  recordDeliveryFailure() {
    this.deliveryFailures++;
  }

  recordError(error) {
    this.errors++;
  }

  recordAcknowledgment(success) {
    if (success) {
      this.acknowledgments++;
    } else {
      this.negativeAcks++;
    }
  }

  recordLatency(latency) {
    this.latencies.push(latency);

    // Keep only recent latencies (last 100)
    if (this.latencies.length > 100) {
      this.latencies = this.latencies.slice(-100);
    }
  }

  getAverageLatency() {
    if (this.latencies.length === 0) return 0;
    return this.latencies.reduce((sum, lat) => sum + lat, 0) / this.latencies.length;
  }

  getSuccessRate() {
    const total = this.deliverySuccesses + this.deliveryFailures;
    return total > 0 ? this.deliverySuccesses / total : 0;
  }
}

// === CHANNEL MANAGER ===

export class ChannelManager {
  constructor(options = {}) {
    this.channels = new Map();
    this.options = {
      defaultRetries: 3,
      defaultTimeout: 30000,
      enableGlobalLogging: true,
      ...options
    };
  }

  /**
   * Create or get a robust channel
   * @param {string} name - Channel name
   * @param {Object} options - Channel options
   */
  getChannel(name, options = {}) {
    if (this.channels.has(name)) {
      return this.channels.get(name);
    }

    const channelOptions = {
      ...this.options,
      ...options
    };

    const channel = new RobustChannel(name, channelOptions);
    this.channels.set(name, channel);

    if (this.options.enableGlobalLogging) {
      console.log(`[ChannelManager] Created channel: ${name}`);
    }

    return channel;
  }

  /**
   * Close and remove a channel
   * @param {string} name - Channel name
   */
  closeChannel(name) {
    const channel = this.channels.get(name);
    if (channel) {
      channel.disconnect();
      this.channels.delete(name);

      if (this.options.enableGlobalLogging) {
        console.log(`[ChannelManager] Closed channel: ${name}`);
      }

      return true;
    }
    return false;
  }

  /**
   * Get all channel names
   */
  getChannelNames() {
    return Array.from(this.channels.keys());
  }

  /**
   * Get metrics for all channels
   */
  getGlobalMetrics() {
    const metrics = {
      totalChannels: this.channels.size,
      channels: {},
      totals: {
        messagesSent: 0,
        messagesReceived: 0,
        errors: 0,
        connectedChannels: 0
      }
    };

    for (const [name, channel] of this.channels.entries()) {
      const channelMetrics = channel.getMetrics();
      metrics.channels[name] = channelMetrics;

      metrics.totals.messagesSent += channelMetrics.messagesSent;
      metrics.totals.messagesReceived += channelMetrics.messagesReceived;
      metrics.totals.errors += channelMetrics.errors;

      if (channelMetrics.state === 'connected') {
        metrics.totals.connectedChannels++;
      }
    }

    return metrics;
  }

  /**
   * Check health of all channels
   */
  getGlobalHealth() {
    const health = {
      healthy: true,
      channels: {},
      issues: []
    };

    for (const [name, channel] of this.channels.entries()) {
      const channelHealth = channel.getHealth();
      health.channels[name] = channelHealth;

      if (!channelHealth.healthy) {
        health.healthy = false;
        health.issues.push({
          channel: name,
          status: channelHealth.status,
          error: channelHealth.lastError
        });
      }
    }

    return health;
  }
}

// === WORLD ENGINE INTEGRATION ===

export function createStudioChannelSystem(options = {}) {
  const manager = new ChannelManager(options);

  // Create standard studio channels
  const chatChannel = manager.getChannel('chat', {
    maxRetries: 2,
    timeout: 10000
  });

  const engineChannel = manager.getChannel('engine', {
    maxRetries: 3,
    timeout: 30000
  });

  const recorderChannel = manager.getChannel('recorder', {
    maxRetries: 1,
    timeout: 5000
  });

  // Set up cross-channel communication
  setupStudioBridge(chatChannel, engineChannel, recorderChannel);

  return {
    manager,
    channels: {
      chat: chatChannel,
      engine: engineChannel,
      recorder: recorderChannel
    }
  };
}

function setupStudioBridge(chatChannel, engineChannel, recorderChannel) {
  // Chat to Engine bridge
  chatChannel.subscribe('engine.*', async (message, context) => {
    await engineChannel.send(context.routingKey.replace('engine.', ''), message);
    context.acknowledge(true);
  });

  // Engine to Recorder bridge
  engineChannel.subscribe('recorder.*', async (message, context) => {
    await recorderChannel.send(context.routingKey.replace('recorder.', ''), message);
    context.acknowledge(true);
  });

  // Recorder to Chat bridge (for status updates)
  recorderChannel.subscribe('chat.*', async (message, context) => {
    await chatChannel.send(context.routingKey.replace('chat.', ''), message);
    context.acknowledge(true);
  });
}

// Export factory function for World Engine V3.1 integration
export function createRobustChannelSystem(worldEngine, options = {}) {
  const system = createStudioChannelSystem(options);

  // Integrate with existing World Engine if available
  if (worldEngine) {
    // Hook into existing controller bridge system
    const { channels } = system;

    // Add World Engine specific channels
    const mathChannel = system.manager.getChannel('mathematics', {
      maxRetries: 5,
      timeout: 60000  // Math operations might take longer
    });

    channels.mathematics = mathChannel;
  }

  return system;
}
