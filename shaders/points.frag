#version 330

uniform float fog_density;
uniform float fog_start;
uniform vec3 fog_color;

in vec3 v_color;
in float v_depth;

out vec4 fragColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    if (dist > 0.5) {
        discard;
    }
    
    float falloff = 1.0 - smoothstep(0.3, 0.5, dist);
    
    float fog_factor = exp(-fog_density * max(0.0, v_depth - fog_start));
    fog_factor = clamp(fog_factor, 0.0, 1.0);
    
    vec3 color = mix(fog_color, v_color, fog_factor);
    
    fragColor = vec4(color, falloff);
}
