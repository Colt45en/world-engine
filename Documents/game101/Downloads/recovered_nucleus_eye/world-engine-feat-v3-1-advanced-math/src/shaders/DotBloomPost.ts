// Pattern dots post-processing effect (hardware-accelerated)
export const DotBloomPost = {
    vertex: `
    attribute vec2 a_position;
    varying vec2 v_uv;

    void main() {
      v_uv = (a_position + 1.0) * 0.5;
      gl_Position = vec4(a_position, 0.0, 1.0);
    }
  `,

    fragment: `
    precision highp float;

    uniform sampler2D u_texture;
    uniform vec2 u_resolution;
    uniform float u_time;
    uniform float u_intensity;
    uniform float u_pattern_scale;
    uniform vec3 u_bloom_color;

    varying vec2 v_uv;

    float pattern(vec2 p) {
      vec2 grid = floor(p * u_pattern_scale);
      vec2 cell = fract(p * u_pattern_scale) - 0.5;

      // Hexagonal pattern
      float d = length(cell);
      float hex = smoothstep(0.3, 0.25, d);

      // Add rotation based on grid position
      float angle = (grid.x + grid.y * 3.0) * 0.5 + u_time * 0.1;
      mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
      cell = rot * cell;

      // Secondary pattern
      float d2 = max(abs(cell.x), abs(cell.y));
      float square = smoothstep(0.4, 0.35, d2);

      return max(hex, square * 0.7);
    }

    void main() {
      vec4 original = texture2D(u_texture, v_uv);

      // Calculate luminance
      float luma = dot(original.rgb, vec3(0.299, 0.587, 0.114));

      // Generate pattern
      float pat = pattern(v_uv + u_time * 0.05);

      // Bloom effect based on luminance
      float bloom = pow(luma, 2.0) * u_intensity;
      vec3 bloom_color = u_bloom_color * bloom * pat;

      // Composite
      vec3 result = original.rgb + bloom_color;

      // Tone mapping (simple Reinhard)
      result = result / (1.0 + result);

      gl_FragColor = vec4(result, original.a);
    }
  `,

    uniforms: {
        u_texture: { type: "sampler2D" },
        u_resolution: { type: "vec2", value: [1920, 1080] },
        u_time: { type: "float", value: 0 },
        u_intensity: { type: "float", value: 0.8 },
        u_pattern_scale: { type: "float", value: 32.0 },
        u_bloom_color: { type: "vec3", value: [0.2, 0.6, 1.0] }
    },

    // WebGL setup helper
    create(gl: WebGLRenderingContext): {
        program: WebGLProgram;
        uniforms: Record<string, WebGLUniformLocation>;
        attributes: Record<string, number>;
    } {
        const vs = gl.createShader(gl.VERTEX_SHADER)!;
        gl.shaderSource(vs, this.vertex);
        gl.compileShader(vs);

        const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
        gl.shaderSource(fs, this.fragment);
        gl.compileShader(fs);

        const program = gl.createProgram()!;
        gl.attachShader(program, vs);
        gl.attachShader(program, fs);
        gl.linkProgram(program);

        // Extract uniforms and attributes
        const uniforms: Record<string, WebGLUniformLocation> = {};
        for (const name in this.uniforms) {
            const loc = gl.getUniformLocation(program, name);
            if (loc) uniforms[name] = loc;
        }

        const attributes = {
            a_position: gl.getAttribLocation(program, "a_position")
        };

        return { program, uniforms, attributes };
    }
};
