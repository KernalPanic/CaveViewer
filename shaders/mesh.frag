#version 330

uniform sampler2D u_texture;
uniform vec3 u_camera_pos;     // diver's headlamp lives at the camera
uniform vec3 u_light_color;
uniform float u_light_intensity;
uniform float u_ambient;       // tiny fill light so unlit areas aren't pure black
uniform bool u_texture_enabled; // texture display toggle, controlled by the Texture button in the UI

in vec2 v_uv;
in vec3 v_normal;
in vec3 v_world_pos;

out vec4 f_color;

void main() {
    // When texture display is toggled off, fall back to a neutral gray
    // instead of sampling the bound texture at all -- this lets the
    // person inspect pure geometry/shape without the photo detail, and
    // avoids a texture lookup entirely when it's not wanted (a small but
    // free performance win when flying with texture off).
    //
    // Written as an explicit if/else rather than a ternary: GLSL ternaries
    // involving a texture() sampler call aren't guaranteed to be true
    // short-circuit branches on every GPU/driver (some older/stricter
    // implementations can still evaluate both sides) -- an if/else avoids
    // any risk of that on older hardware.
    vec3 tex_color;
    if (u_texture_enabled) {
        tex_color = texture(u_texture, v_uv).rgb;
    } else {
        tex_color = vec3(0.65, 0.65, 0.68);
    }

    vec3 N = normalize(v_normal);
    vec3 to_light = u_camera_pos - v_world_pos;
    float dist = length(to_light);
    vec3 L = to_light / max(dist, 0.0001);

    float diffuse = max(dot(N, L), 0.0);

    // headlamp falloff: caves have no ambient light, so attenuation should
    // feel like a real light source, not a flat shade -- inverse-square
    // would be too harsh up close, so we use a softened inverse falloff.
    float attenuation = 1.0 / (1.0 + 0.05 * dist + 0.01 * dist * dist);

    vec3 lit = tex_color * (u_ambient + diffuse * attenuation * u_light_intensity * u_light_color);
    f_color = vec4(lit, 1.0);
}
