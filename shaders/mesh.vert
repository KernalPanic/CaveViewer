#version 330

uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat4 u_model;

in vec3 in_position;
in vec2 in_uv;
in vec3 in_normal;

out vec2 v_uv;
out vec3 v_normal;
out vec3 v_world_pos;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_world_pos = world_pos.xyz;
    v_uv = in_uv;
    v_normal = mat3(u_model) * in_normal;
    gl_Position = u_projection * u_view * world_pos;
}
