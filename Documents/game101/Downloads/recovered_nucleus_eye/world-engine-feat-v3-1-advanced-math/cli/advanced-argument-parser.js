/**
 * World Engine V3.1 - Advanced CLI Argument Parser
 * Inspired by argument-parser-improvements.txt attachment
 *
 * Provides limit-long-syntax validation, better error handling, type checking,
 * and improved CLI experience for World Engine tools
 */

export class AdvancedArgumentParser {
  constructor(options = {}) {
    this.options = {
      caseSensitive: false,
      allowUnknown: false,
      enableHelp: true,
      enableValidation: true,
      enableTypeCoercion: true,
      limitLongSyntax: true,
      strictMode: false,
      ...options
    };

    // Parser state
    this.arguments = new Map();
    this.commands = new Map();
    this.validators = new Map();
    this.typeCoercers = new Map();
    this.middleware = [];

    // Parsed results
    this.parsed = null;
    this.errors = [];
    this.warnings = [];

    // Initialize built-in types and validators
    this.initializeBuiltins();
  }

  initializeBuiltins() {
    // Built-in type coercers
    this.addTypeCoercer('string', (value) => String(value));
    this.addTypeCoercer('number', (value) => {
      const num = Number(value);
      if (isNaN(num)) throw new Error(`Invalid number: ${value}`);
      return num;
    });
    this.addTypeCoercer('integer', (value) => {
      const num = parseInt(value, 10);
      if (isNaN(num)) throw new Error(`Invalid integer: ${value}`);
      return num;
    });
    this.addTypeCoercer('float', (value) => {
      const num = parseFloat(value);
      if (isNaN(num)) throw new Error(`Invalid float: ${value}`);
      return num;
    });
    this.addTypeCoercer('boolean', (value) => {
      if (typeof value === 'boolean') return value;
      const str = String(value).toLowerCase();
      if (['true', '1', 'yes', 'on'].includes(str)) return true;
      if (['false', '0', 'no', 'off'].includes(str)) return false;
      throw new Error(`Invalid boolean: ${value}`);
    });
    this.addTypeCoercer('array', (value) => {
      if (Array.isArray(value)) return value;
      return String(value).split(',').map(s => s.trim());
    });

    // Built-in validators
    this.addValidator('required', (value, option) => {
      if (value === undefined || value === null || value === '') {
        throw new Error(`${option.name} is required`);
      }
      return true;
    });
    this.addValidator('min', (value, option, param) => {
      if (typeof value === 'number' && value < param) {
        throw new Error(`${option.name} must be at least ${param}`);
      }
      if (typeof value === 'string' && value.length < param) {
        throw new Error(`${option.name} must be at least ${param} characters`);
      }
      return true;
    });
    this.addValidator('max', (value, option, param) => {
      if (typeof value === 'number' && value > param) {
        throw new Error(`${option.name} must be at most ${param}`);
      }
      if (typeof value === 'string' && value.length > param) {
        throw new Error(`${option.name} must be at most ${param} characters`);
      }
      return true;
    });
    this.addValidator('pattern', (value, option, pattern) => {
      const regex = new RegExp(pattern);
      if (!regex.test(String(value))) {
        throw new Error(`${option.name} must match pattern: ${pattern}`);
      }
      return true;
    });
    this.addValidator('choices', (value, option, choices) => {
      if (!choices.includes(value)) {
        throw new Error(`${option.name} must be one of: ${choices.join(', ')}`);
      }
      return true;
    });
  }

  // === ARGUMENT DEFINITION METHODS ===

  /**
   * Add a command-line argument
   * @param {string} name - Argument name
   * @param {Object} options - Argument configuration
   */
  addArgument(name, options = {}) {
    const argument = {
      name: this.normalizeName(name),
      originalName: name,
      type: options.type || 'string',
      default: options.default,
      description: options.description || '',
      required: options.required || false,
      multiple: options.multiple || false,
      choices: options.choices,
      validators: options.validators || [],
      alias: options.alias ? this.normalizeAlias(options.alias) : null,
      group: options.group || 'General',
      hidden: options.hidden || false,
      deprecated: options.deprecated || false,

      // Advanced options from attachment
      limitLongSyntax: options.limitLongSyntax !== false,
      strictValidation: options.strictValidation || false,
      transformValue: options.transformValue,
      conflictsWith: options.conflictsWith || [],
      dependsOn: options.dependsOn || [],

      // Help and documentation
      examples: options.examples || [],
      since: options.since,

      // Internal tracking
      parsed: false,
      value: undefined
    };

    // Validate argument configuration
    this.validateArgumentConfig(argument);

    this.arguments.set(argument.name, argument);

    // Also register alias if provided
    if (argument.alias) {
      this.arguments.set(argument.alias, argument);
    }

    return this;
  }

  /**
   * Add a command
   * @param {string} name - Command name
   * @param {Object} options - Command configuration
   */
  addCommand(name, options = {}) {
    const command = {
      name: this.normalizeName(name),
      description: options.description || '',
      handler: options.handler,
      arguments: new Map(),
      subcommands: new Map(),
      parent: options.parent || null,
      hidden: options.hidden || false,
      examples: options.examples || []
    };

    this.commands.set(command.name, command);
    return new CommandBuilder(this, command);
  }

  /**
   * Add custom type coercer
   * @param {string} typeName - Type identifier
   * @param {Function} coercer - Coercion function
   */
  addTypeCoercer(typeName, coercer) {
    this.typeCoercers.set(typeName, coercer);
    return this;
  }

  /**
   * Add custom validator
   * @param {string} name - Validator name
   * @param {Function} validator - Validation function
   */
  addValidator(name, validator) {
    this.validators.set(name, validator);
    return this;
  }

  /**
   * Add middleware function
   * @param {Function} middleware - Middleware function
   */
  use(middleware) {
    this.middleware.push(middleware);
    return this;
  }

  // === PARSING METHODS ===

  /**
   * Parse command line arguments
   * @param {Array} args - Arguments to parse (defaults to process.argv.slice(2))
   */
  parse(args = null) {
    if (args === null) {
      args = typeof process !== 'undefined' && process.argv ? process.argv.slice(2) : [];
    }

    this.errors = [];
    this.warnings = [];
    this.parsed = {
      command: null,
      arguments: {},
      positional: [],
      unknown: [],
      raw: args.slice()
    };

    try {
      // Apply limit-long-syntax validation if enabled
      if (this.options.limitLongSyntax) {
        this.validateLongSyntax(args);
      }

      // Parse arguments
      this.parseArguments(args);

      // Run middleware
      for (const middleware of this.middleware) {
        middleware(this.parsed, this);
      }

      // Validate parsed results
      if (this.options.enableValidation) {
        this.validateParsedArguments();
      }

      // Check for required arguments
      this.checkRequiredArguments();

      // Apply type coercion
      if (this.options.enableTypeCoercion) {
        this.applyTypeCoercion();
      }

      // Check argument dependencies and conflicts
      this.checkArgumentDependencies();
      this.checkArgumentConflicts();

      return this.parsed;

    } catch (error) {
      this.errors.push(error.message);

      if (this.options.strictMode) {
        throw new ParseError(this.errors, this.warnings);
      }

      return null;
    }
  }

  parseArguments(args) {
    let i = 0;

    while (i < args.length) {
      const arg = args[i];

      if (arg.startsWith('--')) {
        // Long option
        const result = this.parseLongOption(arg, args, i);
        i = result.nextIndex;
      } else if (arg.startsWith('-') && arg.length > 1) {
        // Short option(s)
        const result = this.parseShortOption(arg, args, i);
        i = result.nextIndex;
      } else if (this.commands.has(this.normalizeName(arg))) {
        // Command
        this.parsed.command = this.normalizeName(arg);
        i++;
      } else {
        // Positional argument
        this.parsed.positional.push(arg);
        i++;
      }
    }
  }

  parseLongOption(arg, args, index) {
    let [name, value] = arg.slice(2).split('=', 2);
    name = this.normalizeName(name);

    const argument = this.arguments.get(name);

    if (!argument) {
      if (!this.options.allowUnknown) {
        throw new Error(`Unknown argument: --${name}`);
      }
      this.parsed.unknown.push(arg);
      return { nextIndex: index + 1 };
    }

    // Check deprecated status
    if (argument.deprecated) {
      this.warnings.push(`Argument --${name} is deprecated`);
    }

    // Get value
    let nextIndex = index + 1;

    if (value === undefined) {
      if (argument.type === 'boolean') {
        value = true;
      } else if (nextIndex < args.length && !args[nextIndex].startsWith('-')) {
        value = args[nextIndex];
        nextIndex++;
      } else {
        throw new Error(`Argument --${name} requires a value`);
      }
    }

    this.setArgumentValue(argument, value);
    return { nextIndex };
  }

  parseShortOption(arg, args, index) {
    const flags = arg.slice(1);
    let nextIndex = index + 1;

    for (let i = 0; i < flags.length; i++) {
      const flag = flags[i];
      const argument = this.findArgumentByAlias(flag);

      if (!argument) {
        if (!this.options.allowUnknown) {
          throw new Error(`Unknown flag: -${flag}`);
        }
        this.parsed.unknown.push(`-${flag}`);
        continue;
      }

      // Check deprecated status
      if (argument.deprecated) {
        this.warnings.push(`Flag -${flag} is deprecated`);
      }

      let value;

      if (argument.type === 'boolean') {
        value = true;
      } else if (i === flags.length - 1) {
        // Last flag can take a value
        if (nextIndex < args.length && !args[nextIndex].startsWith('-')) {
          value = args[nextIndex];
          nextIndex++;
        } else {
          throw new Error(`Flag -${flag} requires a value`);
        }
      } else {
        throw new Error(`Flag -${flag} requires a value but is not the last flag`);
      }

      this.setArgumentValue(argument, value);
    }

    return { nextIndex };
  }

  setArgumentValue(argument, value) {
    if (argument.multiple) {
      if (!this.parsed.arguments[argument.name]) {
        this.parsed.arguments[argument.name] = [];
      }
      this.parsed.arguments[argument.name].push(value);
    } else {
      this.parsed.arguments[argument.name] = value;
    }

    argument.parsed = true;
    argument.value = this.parsed.arguments[argument.name];
  }

  // === VALIDATION METHODS ===

  validateLongSyntax(args) {
    if (!this.options.limitLongSyntax) return;

    const violations = [];

    for (const arg of args) {
      if (arg.startsWith('--')) {
        const name = arg.slice(2).split('=')[0];

        // Check for invalid characters
        if (!/^[a-zA-Z0-9-]+$/.test(name)) {
          violations.push(`Invalid characters in argument: ${arg}`);
        }

        // Check for consecutive hyphens
        if (name.includes('--')) {
          violations.push(`Consecutive hyphens not allowed: ${arg}`);
        }

        // Check for leading/trailing hyphens
        if (name.startsWith('-') || name.endsWith('-')) {
          violations.push(`Invalid hyphen placement: ${arg}`);
        }

        // Check length limits
        if (name.length > 50) {
          violations.push(`Argument name too long: ${arg}`);
        }

        if (name.length < 2) {
          violations.push(`Argument name too short: ${arg}`);
        }
      }
    }

    if (violations.length > 0) {
      throw new Error(`Long syntax violations:\n${violations.join('\n')}`);
    }
  }

  validateArgumentConfig(argument) {
    // Validate argument name
    if (!argument.name || typeof argument.name !== 'string') {
      throw new Error('Argument name must be a non-empty string');
    }

    // Validate type
    if (!this.typeCoercers.has(argument.type)) {
      throw new Error(`Unknown type: ${argument.type}`);
    }

    // Validate choices
    if (argument.choices && !Array.isArray(argument.choices)) {
      throw new Error('Choices must be an array');
    }

    // Validate validators
    if (argument.validators) {
      for (const validator of argument.validators) {
        if (typeof validator === 'string') {
          if (!this.validators.has(validator)) {
            throw new Error(`Unknown validator: ${validator}`);
          }
        } else if (typeof validator !== 'function') {
          throw new Error('Validator must be a string or function');
        }
      }
    }
  }

  validateParsedArguments() {
    for (const [name, argument] of this.arguments.entries()) {
      // Skip aliases
      if (name !== argument.name) continue;

      const value = this.parsed.arguments[argument.name];

      if (value !== undefined) {
        // Validate choices
        if (argument.choices) {
          if (argument.multiple) {
            for (const val of value) {
              if (!argument.choices.includes(val)) {
                throw new Error(`Invalid choice for ${argument.name}: ${val}`);
              }
            }
          } else {
            if (!argument.choices.includes(value)) {
              throw new Error(`Invalid choice for ${argument.name}: ${value}`);
            }
          }
        }

        // Run custom validators
        for (const validator of argument.validators) {
          if (typeof validator === 'string') {
            const validatorFn = this.validators.get(validator);
            validatorFn(value, argument);
          } else if (typeof validator === 'function') {
            validator(value, argument, this.parsed);
          } else if (typeof validator === 'object') {
            const validatorFn = this.validators.get(validator.name);
            validatorFn(value, argument, validator.param);
          }
        }
      }
    }
  }

  checkRequiredArguments() {
    for (const [name, argument] of this.arguments.entries()) {
      // Skip aliases
      if (name !== argument.name) continue;

      if (argument.required && !argument.parsed) {
        throw new Error(`Required argument missing: ${argument.name}`);
      }
    }
  }

  applyTypeCoercion() {
    for (const [name, argument] of this.arguments.entries()) {
      // Skip aliases
      if (name !== argument.name) continue;

      const value = this.parsed.arguments[argument.name];

      if (value !== undefined) {
        const coercer = this.typeCoercers.get(argument.type);

        try {
          if (argument.multiple && Array.isArray(value)) {
            this.parsed.arguments[argument.name] = value.map(v => coercer(v));
          } else {
            this.parsed.arguments[argument.name] = coercer(value);
          }

          // Apply custom transformation if provided
          if (argument.transformValue) {
            this.parsed.arguments[argument.name] = argument.transformValue(
              this.parsed.arguments[argument.name],
              argument,
              this.parsed
            );
          }

        } catch (error) {
          throw new Error(`Type coercion failed for ${argument.name}: ${error.message}`);
        }
      }
    }
  }

  checkArgumentDependencies() {
    for (const [name, argument] of this.arguments.entries()) {
      // Skip aliases
      if (name !== argument.name) continue;

      if (argument.parsed && argument.dependsOn.length > 0) {
        for (const dependency of argument.dependsOn) {
          const depArg = this.arguments.get(this.normalizeName(dependency));
          if (!depArg || !depArg.parsed) {
            throw new Error(`${argument.name} requires ${dependency} to be specified`);
          }
        }
      }
    }
  }

  checkArgumentConflicts() {
    for (const [name, argument] of this.arguments.entries()) {
      // Skip aliases
      if (name !== argument.name) continue;

      if (argument.parsed && argument.conflictsWith.length > 0) {
        for (const conflict of argument.conflictsWith) {
          const conflictArg = this.arguments.get(this.normalizeName(conflict));
          if (conflictArg && conflictArg.parsed) {
            throw new Error(`${argument.name} conflicts with ${conflict}`);
          }
        }
      }
    }
  }

  // === UTILITY METHODS ===

  normalizeName(name) {
    return this.options.caseSensitive ? name : name.toLowerCase();
  }

  normalizeAlias(alias) {
    // Aliases are single characters
    return alias.charAt(0);
  }

  findArgumentByAlias(alias) {
    for (const argument of this.arguments.values()) {
      if (argument.alias === alias) {
        return argument;
      }
    }
    return null;
  }

  // === HELP GENERATION ===

  generateHelp(commandName = null) {
    const command = commandName ? this.commands.get(commandName) : null;
    const args = command ? command.arguments : this.arguments;

    let help = '';

    if (command) {
      help += `Command: ${command.name}\n`;
      help += `${command.description}\n\n`;
    }

    // Group arguments
    const groups = new Map();

    for (const [name, argument] of args.entries()) {
      // Skip aliases and hidden arguments
      if (name !== argument.name || argument.hidden) continue;

      if (!groups.has(argument.group)) {
        groups.set(argument.group, []);
      }
      groups.get(argument.group).push(argument);
    }

    // Generate help for each group
    for (const [groupName, groupArgs] of groups.entries()) {
      help += `${groupName} Options:\n`;

      for (const argument of groupArgs) {
        help += this.generateArgumentHelp(argument);
      }

      help += '\n';
    }

    return help;
  }

  generateArgumentHelp(argument) {
    let line = '  ';

    // Add flags
    if (argument.alias) {
      line += `-${argument.alias}, `;
    }

    line += `--${argument.originalName}`;

    // Add value placeholder
    if (argument.type !== 'boolean') {
      const placeholder = argument.choices
        ? `{${argument.choices.join('|')}}`
        : `<${argument.type}>`;
      line += ` ${placeholder}`;
    }

    // Pad to align descriptions
    line = line.padEnd(30);

    // Add description
    line += argument.description;

    // Add additional info
    if (argument.required) {
      line += ' (required)';
    }

    if (argument.default !== undefined) {
      line += ` (default: ${argument.default})`;
    }

    if (argument.deprecated) {
      line += ' [DEPRECATED]';
    }

    line += '\n';

    // Add examples if available
    if (argument.examples.length > 0) {
      for (const example of argument.examples) {
        line += `      Example: ${example}\n`;
      }
    }

    return line;
  }

  // === ERROR HANDLING ===

  getErrors() {
    return this.errors;
  }

  getWarnings() {
    return this.warnings;
  }

  hasErrors() {
    return this.errors.length > 0;
  }

  hasWarnings() {
    return this.warnings.length > 0;
  }
}

// === COMMAND BUILDER CLASS ===

class CommandBuilder {
  constructor(parser, command) {
    this.parser = parser;
    this.command = command;
  }

  addArgument(name, options = {}) {
    const argument = {
      name: this.parser.normalizeName(name),
      originalName: name,
      type: options.type || 'string',
      default: options.default,
      description: options.description || '',
      required: options.required || false,
      multiple: options.multiple || false,
      choices: options.choices,
      validators: options.validators || [],
      alias: options.alias ? this.parser.normalizeAlias(options.alias) : null,
      group: options.group || 'General',
      hidden: options.hidden || false,
      deprecated: options.deprecated || false,
      parsed: false,
      value: undefined
    };

    this.command.arguments.set(argument.name, argument);
    return this;
  }

  handler(fn) {
    this.command.handler = fn;
    return this;
  }

  build() {
    return this.parser;
  }
}

// === ERROR CLASSES ===

export class ParseError extends Error {
  constructor(errors, warnings = []) {
    const message = `Parse errors:\n${errors.join('\n')}`;
    super(message);
    this.name = 'ParseError';
    this.errors = errors;
    this.warnings = warnings;
  }
}

export class ValidationError extends Error {
  constructor(message, argument, value) {
    super(message);
    this.name = 'ValidationError';
    this.argument = argument;
    this.value = value;
  }
}

// === WORLD ENGINE CLI FACTORY ===

export function createWorldEngineCLI(options = {}) {
  const parser = new AdvancedArgumentParser({
    caseSensitive: false,
    allowUnknown: false,
    enableHelp: true,
    enableValidation: true,
    enableTypeCoercion: true,
    limitLongSyntax: true,
    strictMode: false,
    ...options
  });

  // Add common World Engine arguments
  parser
    .addArgument('input', {
      type: 'string',
      description: 'Input text or file path',
      alias: 'i',
      required: false,
      examples: ['--input "hello world"', '--input ./data.txt']
    })
    .addArgument('output', {
      type: 'string',
      description: 'Output file path',
      alias: 'o',
      examples: ['--output results.json']
    })
    .addArgument('format', {
      type: 'string',
      description: 'Output format',
      alias: 'f',
      choices: ['json', 'csv', 'text', 'xml'],
      default: 'json'
    })
    .addArgument('verbose', {
      type: 'boolean',
      description: 'Enable verbose output',
      alias: 'v'
    })
    .addArgument('quiet', {
      type: 'boolean',
      description: 'Suppress output',
      alias: 'q',
      conflictsWith: ['verbose']
    })
    .addArgument('config', {
      type: 'string',
      description: 'Configuration file path',
      alias: 'c',
      validators: ['required'],
      transformValue: (value) => {
        // Expand relative paths
        if (value.startsWith('./') || value.startsWith('../')) {
          return require('path').resolve(value);
        }
        return value;
      }
    })
    .addArgument('threads', {
      type: 'integer',
      description: 'Number of processing threads',
      alias: 't',
      default: 1,
      validators: [
        { name: 'min', param: 1 },
        { name: 'max', param: 16 }
      ]
    })
    .addArgument('timeout', {
      type: 'integer',
      description: 'Processing timeout in seconds',
      default: 30,
      validators: [{ name: 'min', param: 1 }]
    });

  // Add World Engine specific commands
  parser
    .addCommand('analyze', {
      description: 'Analyze text using World Engine',
      examples: ['analyze --input "text to analyze"']
    })
    .addArgument('algorithm', {
      type: 'string',
      description: 'Analysis algorithm to use',
      choices: ['morpheme', 'semantic', 'neural', 'combined'],
      default: 'combined'
    })
    .addArgument('depth', {
      type: 'integer',
      description: 'Analysis depth level',
      default: 3,
      validators: [
        { name: 'min', param: 1 },
        { name: 'max', param: 10 }
      ]
    })
    .build();

  parser
    .addCommand('train', {
      description: 'Train neural components',
      examples: ['train --dataset ./training.json --epochs 100']
    })
    .addArgument('dataset', {
      type: 'string',
      description: 'Training dataset path',
      required: true,
      validators: ['required']
    })
    .addArgument('epochs', {
      type: 'integer',
      description: 'Number of training epochs',
      default: 50,
      validators: [{ name: 'min', param: 1 }]
    })
    .addArgument('learning-rate', {
      type: 'float',
      description: 'Learning rate for training',
      default: 0.01,
      validators: [
        { name: 'min', param: 0.0001 },
        { name: 'max', param: 1.0 }
      ]
    })
    .build();

  parser
    .addCommand('export', {
      description: 'Export analysis results or models',
      examples: ['export --type model --output ./model.json']
    })
    .addArgument('type', {
      type: 'string',
      description: 'Export type',
      choices: ['results', 'model', 'lexicon', 'config'],
      required: true
    })
    .addArgument('compress', {
      type: 'boolean',
      description: 'Compress exported data'
    })
    .build();

  return parser;
}

// === MIDDLEWARE FUNCTIONS ===

export const limitLongSyntaxMiddleware = (parsed, parser) => {
  // Additional limit-long-syntax validation during parsing
  for (const arg in parsed.arguments) {
    if (arg.length > 50) {
      throw new Error(`Argument name too long: ${arg}`);
    }
  }
};

export const deprecationWarningMiddleware = (parsed, parser) => {
  // Check for deprecated argument patterns
  const deprecatedPatterns = [
    { pattern: /^old-/, message: 'Arguments starting with "old-" are deprecated' },
    { pattern: /^legacy/, message: 'Legacy arguments are deprecated' }
  ];

  for (const arg in parsed.arguments) {
    for (const { pattern, message } of deprecatedPatterns) {
      if (pattern.test(arg)) {
        parser.warnings.push(`${message}: ${arg}`);
      }
    }
  }
};

export const configFileMiddleware = (parsed, parser) => {
  // Auto-load config file if specified
  if (parsed.arguments.config) {
    try {
      const fs = require('fs');
      const config = JSON.parse(fs.readFileSync(parsed.arguments.config, 'utf8'));

      // Merge config file options with parsed arguments
      for (const [key, value] of Object.entries(config)) {
        if (!(key in parsed.arguments)) {
          parsed.arguments[key] = value;
        }
      }
    } catch (error) {
      throw new Error(`Failed to load config file: ${error.message}`);
    }
  }
};

// Export factory function for World Engine V3.1 integration
export function createAdvancedCLI(worldEngine, options = {}) {
  const parser = createWorldEngineCLI(options);

  // Add middleware
  parser
    .use(limitLongSyntaxMiddleware)
    .use(deprecationWarningMiddleware)
    .use(configFileMiddleware);

  // Integrate with World Engine if available
  if (worldEngine) {
    // Add World Engine specific validation
    parser.addValidator('morpheme', (value, option) => {
      // Validate morpheme patterns using World Engine
      if (worldEngine.morphemeDiscovery) {
        const isValid = worldEngine.morphemeDiscovery.validatePattern(value);
        if (!isValid) {
          throw new ValidationError(`Invalid morpheme pattern: ${value}`, option, value);
        }
      }
      return true;
    });

    parser.addValidator('semantic-range', (value, option, range) => {
      // Validate semantic range using World Engine
      if (typeof value === 'number' && (value < range.min || value > range.max)) {
        throw new ValidationError(
          `Semantic value must be between ${range.min} and ${range.max}`,
          option,
          value
        );
      }
      return true;
    });
  }

  return parser;
}
