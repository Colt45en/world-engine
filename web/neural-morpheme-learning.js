/**
 * World Engine V3.1 - Neural Morphological Learning System
 * Inspired by backpropagation.txt - XOR learning pattern applied to morphology
 *
 * This implements a simple neural network that learns to identify morphological patterns
 * using the same principles as XOR learning but applied to linguistic structures
 */

export class NeuralMorphologyLearner {
  constructor(config = {}) {
    this.config = {
      inputSize: 6,     // 6 morphological features
      hiddenSize: 4,    // 4 hidden units for representation learning
      outputSize: 3,    // 3 morphological categories (prefix, root, suffix)
      learningRate: 0.1,
      ...config
    };

    // Initialize weights randomly (like in the XOR example)
    this.initializeWeights();

    // Training data and results
    this.trainingHistory = [];
    this.patterns = new Map();
  }

  initializeWeights() {
    const { inputSize, hiddenSize, outputSize } = this.config;

    // Input to hidden layer weights
    this.weightsInputHidden = Array(inputSize).fill(0).map(() =>
      Array(hiddenSize).fill(0).map(() => (Math.random() - 0.5) * 2)
    );

    // Hidden to output layer weights
    this.weightsHiddenOutput = Array(hiddenSize).fill(0).map(() =>
      Array(outputSize).fill(0).map(() => (Math.random() - 0.5) * 2)
    );

    // Bias terms
    this.hiddenBias = Array(hiddenSize).fill(0).map(() => (Math.random() - 0.5) * 2);
    this.outputBias = Array(outputSize).fill(0).map(() => (Math.random() - 0.5) * 2);
  }

  // Sigmoid activation (as mentioned in backpropagation.txt)
  sigmoid(x) {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
  }

  sigmoidDerivative(x) {
    const s = this.sigmoid(x);
    return s * (1 - s);
  }

  // Forward pass through the network
  forwardPass(input) {
    const { hiddenSize, outputSize } = this.config;

    // Input to hidden layer
    const hiddenInput = Array(hiddenSize);
    const hiddenOutput = Array(hiddenSize);

    for (let h = 0; h < hiddenSize; h++) {
      hiddenInput[h] = this.hiddenBias[h];
      for (let i = 0; i < input.length; i++) {
        hiddenInput[h] += input[i] * this.weightsInputHidden[i][h];
      }
      hiddenOutput[h] = this.sigmoid(hiddenInput[h]);
    }

    // Hidden to output layer
    const outputInput = Array(outputSize);
    const output = Array(outputSize);

    for (let o = 0; o < outputSize; o++) {
      outputInput[o] = this.outputBias[o];
      for (let h = 0; h < hiddenSize; h++) {
        outputInput[o] += hiddenOutput[h] * this.weightsHiddenOutput[h][o];
      }
      output[o] = this.sigmoid(outputInput[o]);
    }

    return {
      hiddenInput,
      hiddenOutput,
      outputInput,
      output
    };
  }

  // Backpropagation implementation (following the attachment's explanation)
  backpropagate(input, target, forwardResult) {
    const { learningRate } = this.config;
    const { hiddenInput, hiddenOutput, outputInput, output } = forwardResult;

    // Calculate output layer error (δ_output = (ŷ - y) * σ'(z_out))
    const outputError = Array(output.length);
    for (let o = 0; o < output.length; o++) {
      outputError[o] = (output[o] - target[o]) * this.sigmoidDerivative(outputInput[o]);
    }

    // Calculate hidden layer error (δ_hidden = δ_output * w_out_hidden * σ'(z_hidden))
    const hiddenError = Array(hiddenOutput.length);
    for (let h = 0; h < hiddenOutput.length; h++) {
      hiddenError[h] = 0;
      for (let o = 0; o < output.length; o++) {
        hiddenError[h] += outputError[o] * this.weightsHiddenOutput[h][o];
      }
      hiddenError[h] *= this.sigmoidDerivative(hiddenInput[h]);
    }

    // Update weights and biases
    // Hidden to output weights
    for (let h = 0; h < hiddenOutput.length; h++) {
      for (let o = 0; o < output.length; o++) {
        this.weightsHiddenOutput[h][o] -= learningRate * outputError[o] * hiddenOutput[h];
      }
    }

    // Input to hidden weights
    for (let i = 0; i < input.length; i++) {
      for (let h = 0; h < hiddenOutput.length; h++) {
        this.weightsInputHidden[i][h] -= learningRate * hiddenError[h] * input[i];
      }
    }

    // Update biases
    for (let o = 0; o < output.length; o++) {
      this.outputBias[o] -= learningRate * outputError[o];
    }
    for (let h = 0; h < hiddenOutput.length; h++) {
      this.hiddenBias[h] -= learningRate * hiddenError[h];
    }

    return {
      outputError,
      hiddenError,
      loss: this.calculateLoss(output, target)
    };
  }

  calculateLoss(predicted, target) {
    // Mean squared error (like in the XOR example: L = 0.5 * (1 - 0.3)^2 = 0.245)
    let loss = 0;
    for (let i = 0; i < predicted.length; i++) {
      loss += 0.5 * Math.pow(predicted[i] - target[i], 2);
    }
    return loss;
  }

  // Convert morphological features to neural network input
  encodeMorphologicalFeatures(word) {
    const features = [
      word.startsWith('pre') ? 1 : 0,    // has prefix 'pre'
      word.startsWith('re') ? 1 : 0,     // has prefix 're'
      word.startsWith('un') ? 1 : 0,     // has prefix 'un'
      word.endsWith('tion') ? 1 : 0,     // has suffix 'tion'
      word.endsWith('ing') ? 1 : 0,      // has suffix 'ing'
      word.length > 6 ? 1 : 0            // is complex word
    ];
    return features;
  }

  // Create target output for morphological classification
  createMorphologicalTarget(word) {
    // Target: [isPrefixDominant, isRootDominant, isSuffixDominant]
    if (word.startsWith('pre') || word.startsWith('re') || word.startsWith('un')) {
      return [1, 0, 0]; // Prefix dominant
    } else if (word.endsWith('tion') || word.endsWith('ing') || word.endsWith('ed')) {
      return [0, 0, 1]; // Suffix dominant
    } else {
      return [0, 1, 0]; // Root dominant
    }
  }

  // Train on a single word (like XOR training)
  trainOnWord(word) {
    const input = this.encodeMorphologicalFeatures(word);
    const target = this.createMorphologicalTarget(word);

    const forwardResult = this.forwardPass(input);
    const backpropResult = this.backpropagate(input, target, forwardResult);

    // Record training step
    this.trainingHistory.push({
      word,
      input,
      target,
      predicted: forwardResult.output,
      loss: backpropResult.loss,
      timestamp: Date.now()
    });

    return {
      word,
      predicted: forwardResult.output,
      target,
      loss: backpropResult.loss
    };
  }

  // Train on multiple words (batch training)
  train(words, epochs = 100) {
    const results = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      const epochResults = [];

      for (const word of words) {
        const result = this.trainOnWord(word);
        epochResults.push(result);
        epochLoss += result.loss;
      }

      results.push({
        epoch,
        averageLoss: epochLoss / words.length,
        results: epochResults
      });

      // Early stopping if loss is very low
      if (epochLoss / words.length < 0.01) {
        console.log(`✅ Neural learning converged at epoch ${epoch}`);
        break;
      }
    }

    return results;
  }

  // Predict morphological pattern for new word
  predict(word) {
    const input = this.encodeMorphologicalFeatures(word);
    const result = this.forwardPass(input);

    // Interpret output
    const [prefixScore, rootScore, suffixScore] = result.output;
    const maxScore = Math.max(prefixScore, rootScore, suffixScore);

    let dominantType;
    if (prefixScore === maxScore) dominantType = 'prefix';
    else if (suffixScore === maxScore) dominantType = 'suffix';
    else dominantType = 'root';

    return {
      word,
      features: input,
      scores: {
        prefix: prefixScore,
        root: rootScore,
        suffix: suffixScore
      },
      dominantType,
      confidence: maxScore,
      hiddenRepresentation: result.hiddenOutput
    };
  }

  // Discover hidden representations (like the XOR example discovering OR and AND)
  analyzeHiddenRepresentations(testWords) {
    const representations = testWords.map(word => {
      const prediction = this.predict(word);
      return {
        word,
        hidden: prediction.hiddenRepresentation,
        type: prediction.dominantType
      };
    });

    // Group by hidden patterns
    const patterns = new Map();
    representations.forEach(rep => {
      const pattern = rep.hidden.map(h => h > 0.5 ? 1 : 0).join('');
      if (!patterns.has(pattern)) {
        patterns.set(pattern, []);
      }
      patterns.get(pattern).push(rep);
    });

    return {
      representations,
      patterns: Array.from(patterns.entries()).map(([pattern, words]) => ({
        pattern,
        words: words.map(w => w.word),
        count: words.length,
        dominantTypes: [...new Set(words.map(w => w.type))]
      }))
    };
  }

  // Get learning statistics
  getStats() {
    if (this.trainingHistory.length === 0) return null;

    const recentHistory = this.trainingHistory.slice(-100);
    const averageLoss = recentHistory.reduce((sum, h) => sum + h.loss, 0) / recentHistory.length;

    return {
      totalTrainingSteps: this.trainingHistory.length,
      averageRecentLoss: averageLoss,
      convergenceRate: this.trainingHistory.length > 10 ?
        (this.trainingHistory[9].loss - averageLoss) / this.trainingHistory[9].loss : 0,
      patternsDiscovered: this.patterns.size
    };
  }
}

// Integration with existing World Engine V3.1 systems
export class NeuralMorphemeIntegration {
  constructor(worldEngine) {
    this.engine = worldEngine;
    this.neuralLearner = new NeuralMorphologyLearner();
    this.cache = new Map();
  }

  // Train the neural system on existing morpheme patterns
  async trainOnExistingPatterns() {
    if (!this.engine.morphemeDiscovery) {
      throw new Error('Morpheme discovery system not available');
    }

    // Get existing vocabulary from morpheme system
    const vocabulary = this.extractVocabulary();

    console.log(`🧠 Training neural morphology learner on ${vocabulary.length} words...`);

    // Train in batches
    const batchSize = 20;
    const totalBatches = Math.ceil(vocabulary.length / batchSize);

    for (let i = 0; i < totalBatches; i++) {
      const batch = vocabulary.slice(i * batchSize, (i + 1) * batchSize);
      const results = this.neuralLearner.train(batch, 50);

      if (i % 5 === 0) {
        const finalLoss = results[results.length - 1]?.averageLoss || 0;
        console.log(`📊 Batch ${i + 1}/${totalBatches} - Loss: ${finalLoss.toFixed(4)}`);
      }
    }

    return this.neuralLearner.getStats();
  }

  extractVocabulary() {
    // Extract vocabulary from various sources in the World Engine
    const words = new Set();

    // Add common morphological examples
    const examples = [
      'transform', 'restructure', 'preprocessing', 'antipattern',
      'reconstruction', 'formation', 'movement', 'static', 'dynamic',
      'creation', 'destruction', 'interaction', 'transaction',
      'prediction', 'reaction', 'action', 'fraction', 'attraction'
    ];

    examples.forEach(word => words.add(word));

    return Array.from(words);
  }

  // Enhanced morphological analysis using neural predictions
  analyzeMorphologyWithNeural(word) {
    // Get traditional analysis
    const traditional = this.engine.morphemeDiscovery?.analyze?.(word) || {
      morphemes: [],
      structure: 'unknown'
    };

    // Get neural prediction
    const neural = this.neuralLearner.predict(word);

    // Combine insights
    return {
      word,
      traditional,
      neural: {
        dominantType: neural.dominantType,
        confidence: neural.confidence,
        scores: neural.scores,
        hiddenRepresentation: neural.hiddenRepresentation
      },
      combined: {
        morphologicalComplexity: neural.features.reduce((sum, f) => sum + f, 0),
        structuralType: neural.dominantType,
        confidence: neural.confidence,
        isLearned: this.cache.has(word)
      }
    };
  }
}

// Export for use in World Engine V3.1 system
export function createNeuralMorphemeSystem(worldEngine) {
  return new NeuralMorphemeIntegration(worldEngine);
}
