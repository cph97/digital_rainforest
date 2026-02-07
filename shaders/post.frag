#version 330

uniform sampler2D current_frame;
uniform sampler2D previous_frame;
uniform float time;
uniform float temporal_feedback;
uniform float smear_strength;
uniform float tear_rate;
uniform float column_shift_scale;
uniform float chroma_offset;
uniform float glitch_intensity;
uniform float audio_onset;
uniform float audio_band_low;
uniform float audio_band_mid;
uniform float audio_band_high;
uniform vec2 resolution;
uniform bool glitch_enabled;

in vec2 v_texcoord;
out vec4 fragColor;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

void main() {
    vec2 uv = v_texcoord;
    
    // Databending-style vertical smear effect
    float column_x = floor(uv.x * resolution.x);
    float column_noise = hash(vec2(column_x, floor(time * 0.5)));
    float column_noise2 = hash(vec2(column_x * 0.5, floor(time * 0.3)));
    
    // Audio-reactive parameters - stronger response
    float onset_boost = audio_onset * 3.0;
    float band_energy = (audio_band_low * 1.2 + audio_band_mid * 2.0 + audio_band_high * 0.8) / 4.0;
    
    // Vertical smear displacement (databending effect) - audio-modulated
    float smear_offset = 0.0;
    float smear_intensity = smear_strength * (1.0 + onset_boost + band_energy * 2.0);
    
    // Random vertical displacement per column - more frequent with audio
    if (column_noise < smear_intensity) {
        float displacement = (column_noise2 - 0.5) * column_shift_scale * (1.0 + band_energy);
        smear_offset = displacement * (1.0 + onset_boost * 0.5);
    }
    
    // Additional glitch-based vertical tears
    if (glitch_enabled) {
        float tear_threshold = tear_rate * glitch_intensity;
        float tear_noise = noise(vec2(uv.x * 10.0, time * 5.0));
        
        if (tear_noise > (1.0 - tear_threshold)) {
            float hard_tear = (tear_noise - (1.0 - tear_threshold)) / tear_threshold;
            smear_offset += hard_tear * 0.15 * sign(column_noise - 0.5);
        }
    }
    
    // Scanline drift for analog feel
    float scanline_drift = sin(uv.y * 100.0 + time * 3.0) * 0.002 * band_energy;
    
    // Apply vertical smear
    vec2 smear_uv = vec2(uv.x + scanline_drift, uv.y + smear_offset);
    smear_uv.y = fract(smear_uv.y); // Wrap vertically for databending effect
    smear_uv.x = clamp(smear_uv.x, 0.0, 1.0);
    
    // RGB channel separation (databending aesthetic) - audio-reactive
    float chroma_amount = chroma_offset * (0.3 + onset_boost * 0.8 + band_energy * 0.5);
    vec2 chroma_r_uv = vec2(smear_uv.x + chroma_amount * 0.01, smear_uv.y);
    vec2 chroma_g_uv = smear_uv;
    vec2 chroma_b_uv = vec2(smear_uv.x - chroma_amount * 0.01, smear_uv.y);
    
    // Wrap channel UVs vertically
    chroma_r_uv.y = fract(chroma_r_uv.y);
    chroma_b_uv.y = fract(chroma_b_uv.y);
    chroma_r_uv.x = clamp(chroma_r_uv.x, 0.0, 1.0);
    chroma_b_uv.x = clamp(chroma_b_uv.x, 0.0, 1.0);
    
    float r = texture(current_frame, chroma_r_uv).r;
    float g = texture(current_frame, chroma_g_uv).g;
    float b = texture(current_frame, chroma_b_uv).b;
    
    vec3 current = vec3(r, g, b);
    
    // Get previous frame
    vec3 previous = texture(previous_frame, uv).rgb;
    
    // Simple temporal accumulation with decay for trails
    // This creates persistent trails as objects move
    float decay = 1.0 - temporal_feedback;
    vec3 accumulated = previous * (1.0 - decay);
    
    // Add current frame with full brightness
    vec3 blended = max(current, accumulated);
    
    // Audio-reactive vertical blur for downward trails
    float trail_strength = 0.003 + band_energy * 0.008 + onset_boost * 0.005;
    vec3 trail_sample1 = texture(previous_frame, vec2(uv.x, clamp(uv.y + trail_strength, 0.0, 1.0))).rgb;
    vec3 trail_sample2 = texture(previous_frame, vec2(uv.x, clamp(uv.y + trail_strength * 2.0, 0.0, 1.0))).rgb;
    vec3 trail_sample3 = texture(previous_frame, vec2(uv.x, clamp(uv.y + trail_strength * 3.0, 0.0, 1.0))).rgb;
    
    vec3 vertical_trail = (previous + trail_sample1 + trail_sample2 + trail_sample3) * 0.25;
    float trail_intensity = temporal_feedback * (0.5 + band_energy * 0.3);
    blended = max(blended, vertical_trail * trail_intensity);
    
    // Add subtle noise for texture
    float grain = (hash(uv * resolution + time) - 0.5) * 0.015;
    blended += grain;
    
    fragColor = vec4(blended, 1.0);
}
