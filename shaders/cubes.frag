#version 330

uniform vec3 fog_color;
uniform float fog_density;
uniform float fog_start;

in vec3 v_color;
in vec3 v_normal;
in vec3 v_position;

out vec4 fragColor;

void main() {
    // Simple lighting
    vec3 light_dir = normalize(vec3(0.5, 1.0, 0.3));
    float diffuse = max(dot(normalize(v_normal), light_dir), 0.0);
    float ambient = 0.3;
    float lighting = ambient + diffuse * 0.7;
    
    vec3 lit_color = v_color * lighting;
    
    // Fog
    float dist = length(v_position);
    float fog_factor = 1.0 - exp(-fog_density * max(0.0, dist - fog_start));
    fog_factor = clamp(fog_factor, 0.0, 1.0);
    
    vec3 final_color = mix(lit_color, fog_color, fog_factor);
    
    fragColor = vec4(final_color, 1.0);
}
