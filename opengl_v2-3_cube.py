import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
import time

# Vertex shader source code with lighting calculations
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normalMatrix;

out vec3 vertexColor;
out vec3 FragPos;
out vec3 Normal;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = normalMatrix * aNormal;
    vertexColor = aColor;
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

# Fragment shader with Blinn-Phong lighting
fragment_shader_source = """
#version 330 core
in vec3 vertexColor;
in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;

// Material properties
uniform float ambientStrength;
uniform float specularStrength;
uniform int shininess;

void main()
{
    // Normalize the normal vector
    vec3 norm = normalize(Normal);
    
    // Ambient lighting
    vec3 ambient = ambientStrength * lightColor;
    
    // Diffuse lighting
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular lighting (Blinn-Phong)
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * lightColor;
    
    // Combine all lighting components
    vec3 result = (ambient + diffuse + specular) * vertexColor;
    FragColor = vec4(result, 1.0);
}
"""

def create_cube_data():
    # Cube vertices with positions, colors, and normals
    # Each face needs separate vertices for proper normal vectors
    vertices = np.array([
        # Front face (red) - normal: (0, 0, 1)
        -1.0, -1.0,  1.0,  1.0, 0.0, 0.0,  0.0, 0.0, 1.0,  # Bottom-left
         1.0, -1.0,  1.0,  1.0, 0.0, 0.0,  0.0, 0.0, 1.0,  # Bottom-right
         1.0,  1.0,  1.0,  1.0, 0.0, 0.0,  0.0, 0.0, 1.0,  # Top-right
        -1.0,  1.0,  1.0,  1.0, 0.0, 0.0,  0.0, 0.0, 1.0,  # Top-left
        
        # Back face (green) - normal: (0, 0, -1)
         1.0, -1.0, -1.0,  0.0, 1.0, 0.0,  0.0, 0.0, -1.0,  # Bottom-left
        -1.0, -1.0, -1.0,  0.0, 1.0, 0.0,  0.0, 0.0, -1.0,  # Bottom-right
        -1.0,  1.0, -1.0,  0.0, 1.0, 0.0,  0.0, 0.0, -1.0,  # Top-right
         1.0,  1.0, -1.0,  0.0, 1.0, 0.0,  0.0, 0.0, -1.0,  # Top-left
        
        # Left face (blue) - normal: (-1, 0, 0)
        -1.0, -1.0, -1.0,  0.0, 0.0, 1.0,  -1.0, 0.0, 0.0,  # Bottom-left
        -1.0, -1.0,  1.0,  0.0, 0.0, 1.0,  -1.0, 0.0, 0.0,  # Bottom-right
        -1.0,  1.0,  1.0,  0.0, 0.0, 1.0,  -1.0, 0.0, 0.0,  # Top-right
        -1.0,  1.0, -1.0,  0.0, 0.0, 1.0,  -1.0, 0.0, 0.0,  # Top-left
        
        # Right face (yellow) - normal: (1, 0, 0)
         1.0, -1.0,  1.0,  1.0, 1.0, 0.0,  1.0, 0.0, 0.0,  # Bottom-left
         1.0, -1.0, -1.0,  1.0, 1.0, 0.0,  1.0, 0.0, 0.0,  # Bottom-right
         1.0,  1.0, -1.0,  1.0, 1.0, 0.0,  1.0, 0.0, 0.0,  # Top-right
         1.0,  1.0,  1.0,  1.0, 1.0, 0.0,  1.0, 0.0, 0.0,  # Top-left
        
        # Top face (magenta) - normal: (0, 1, 0)
        -1.0,  1.0,  1.0,  1.0, 0.0, 1.0,  0.0, 1.0, 0.0,  # Bottom-left
         1.0,  1.0,  1.0,  1.0, 0.0, 1.0,  0.0, 1.0, 0.0,  # Bottom-right
         1.0,  1.0, -1.0,  1.0, 0.0, 1.0,  0.0, 1.0, 0.0,  # Top-right
        -1.0,  1.0, -1.0,  1.0, 0.0, 1.0,  0.0, 1.0, 0.0,  # Top-left
        
        # Bottom face (cyan) - normal: (0, -1, 0)
        -1.0, -1.0, -1.0,  0.0, 1.0, 1.0,  0.0, -1.0, 0.0,  # Bottom-left
         1.0, -1.0, -1.0,  0.0, 1.0, 1.0,  0.0, -1.0, 0.0,  # Bottom-right
         1.0, -1.0,  1.0,  0.0, 1.0, 1.0,  0.0, -1.0, 0.0,  # Top-right
        -1.0, -1.0,  1.0,  0.0, 1.0, 1.0,  0.0, -1.0, 0.0,  # Top-left
    ], dtype=np.float32)
    
    # Define indices for each face (two triangles per face)
    indices = np.array([
        # Front face
        0, 1, 2, 2, 3, 0,
        # Back face
        4, 5, 6, 6, 7, 4,
        # Left face
        8, 9, 10, 10, 11, 8,
        # Right face
        12, 13, 14, 14, 15, 12,
        # Top face
        16, 17, 18, 18, 19, 16,
        # Bottom face
        20, 21, 22, 22, 23, 20
    ], dtype=np.uint32)
    
    return vertices, indices

def apply_top_scale(base_vertices, scale):
    """
    Return a modified copy of the vertex array where every vertex with
    y == 1.0 (i.e. anything on the top face, including the top edges of
    the front/back/left/right faces) has its x and z coordinates scaled.

    This pulls the top face inward/outward, turning the 4 side faces
    into trapezoids while leaving the bottom face untouched.
    """
    # View as (24 vertices, 9 floats each) without copying underlying data yet
    verts = base_vertices.reshape(-1, 9).copy()

    # Column 1 of each row is the y position (columns 0,1,2 = x,y,z)
    top_mask = verts[:, 1] > 0.5  # selects the y == 1.0 vertices

    verts[top_mask, 0] *= scale  # scale x
    verts[top_mask, 2] *= scale  # scale z

    return verts.reshape(-1)

def recalculate_normals(vertices):
    """
    Recompute a flat shading normal for each face from its current
    (possibly deformed) vertex positions, and write that normal into
    all 4 of that face's duplicated vertices.

    Each face's 4 vertices are laid out in order: bottom-left (0),
    bottom-right (1), top-right (2), top-left (3) - see create_cube_data().
    For a planar quad, the face normal is just the cross product of two
    edges from a shared corner:

        edge1 = bottom-right - bottom-left
        edge2 = top-left     - bottom-left
        normal = normalize(cross(edge1, edge2))

    This has to be redone every frame that the geometry changes, because
    a normal describes the orientation of the *current* surface - if the
    top face shrinks and the side faces slant inward, their normals are
    no longer purely horizontal/vertical like they were on the resting cube.
    """
    verts = vertices.reshape(-1, 9).copy()
    num_faces = verts.shape[0] // 4

    for face in range(num_faces):
        i0 = face * 4  # index of this face's bottom-left vertex
        p_bl = verts[i0, 0:3]
        p_br = verts[i0 + 1, 0:3]
        p_tl = verts[i0 + 3, 0:3]

        edge1 = p_br - p_bl
        edge2 = p_tl - p_bl
        normal = np.cross(edge1, edge2)

        length = np.linalg.norm(normal)
        if length > 1e-8:
            normal = normal / length

        # Same normal for all 4 vertices of this face (flat shading)
        verts[i0:i0 + 4, 6:9] = normal

    return verts.reshape(-1)

def create_shader_program():
    return compileProgram(
        compileShader(vertex_shader_source, GL_VERTEX_SHADER),
        compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
    )

def main():
    # Initialize GLFW
    if not glfw.init():
        return
    
    # Configure GLFW for OpenGL 3.3 Core Profile
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    
    # Create window
    window = glfw.create_window(800, 600, "OpenGL Core Profile Cube with Blinn-Phong Lighting", None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)

    # Generate and bind a Vertex Array Object ID required for macOS
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # Enable depth testing
    glEnable(GL_DEPTH_TEST)
    
    # Create shader program
    shader_program = create_shader_program()
    
    # Create cube data. `base_vertices` is kept around as the untouched
    # "resting" shape; each frame we derive a modified copy from it rather
    # than mutating it directly, so we always have a clean reference.
    base_vertices, indices = create_cube_data()
    vertices = base_vertices.copy()
    
    # Create and bind VAO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    
    # Create and bind VBO
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    # GL_DYNAMIC_DRAW hints to the driver that this buffer's contents will
    # be updated frequently (every frame), unlike GL_STATIC_DRAW which is
    # meant for data that's uploaded once and never touched again.
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
    
    # Create and bind EBO (Element Buffer Object)
    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    
    # Configure vertex attributes
    stride = 9 * 4  # 9 floats per vertex (3 pos + 3 color + 3 normal)
    
    # Position attribute (location = 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    
    # Color attribute (location = 1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)
    
    # Normal attribute (location = 2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))
    glEnableVertexAttribArray(2)
    
    # Unbind VAO
    glBindVertexArray(0)
    
    # Get uniform locations
    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    projection_loc = glGetUniformLocation(shader_program, "projection")
    normal_matrix_loc = glGetUniformLocation(shader_program, "normalMatrix")
    light_pos_loc = glGetUniformLocation(shader_program, "lightPos")
    view_pos_loc = glGetUniformLocation(shader_program, "viewPos")
    light_color_loc = glGetUniformLocation(shader_program, "lightColor")
    ambient_strength_loc = glGetUniformLocation(shader_program, "ambientStrength")
    specular_strength_loc = glGetUniformLocation(shader_program, "specularStrength")
    shininess_loc = glGetUniformLocation(shader_program, "shininess")
    
    # Create transformation matrices
    projection = pyrr.matrix44.create_perspective_projection_matrix(
        45.0, 800/600, 0.1, 100.0
    )
    
    # Camera position
    camera_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    view = pyrr.matrix44.create_look_at(
        camera_pos,            # Camera position
        [0.0, 0.0, 0.0],       # Target position
        [0.0, 1.0, 0.0]        # Up vector
    )
    
    # Lighting properties
    light_pos = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # White light
    ambient_strength = 0.1
    specular_strength = 0.5
    shininess = 32
    
    rotation_x = 0.0
    rotation_y = 0.0
    
    # Main loop
    while not glfw.window_should_close(window):
        # Handle events
        glfw.poll_events()
        
        # Clear the screen
        glClearColor(0.1, 0.1, 0.1, 1.0)  # Darker background to show lighting better
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Use shader program
        glUseProgram(shader_program)
        
        # Update rotation
        rotation_x += 0.5
        rotation_y += 0.3
    
        
        # --- Animate the top face size (trapezoid effect) ---
        # Oscillate the top face's x/z scale between 0.3x and 1.0x over time,
        # using glfw's clock so the speed doesn't depend on frame rate.
        elapsed = glfw.get_time()
        top_scale = 0.65 + 0.35 * np.sin(elapsed * 1.5)
        
        # Recompute the vertex array from the untouched base shape, then
        # push it to the existing GPU buffer with glBufferSubData. This
        # overwrites the buffer's contents in place rather than allocating
        # a new buffer (which is what glBufferData would do).
        vertices = apply_top_scale(base_vertices, top_scale)
        vertices = recalculate_normals(vertices)
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
        
        # Create model matrix with rotation
        model = pyrr.matrix44.create_from_x_rotation(np.radians(rotation_x))
        model = pyrr.matrix44.multiply(model, pyrr.matrix44.create_from_y_rotation(np.radians(rotation_y)))
        
        # Calculate normal matrix (inverse transpose of model matrix upper-left 3x3)
        normal_matrix = np.linalg.inv(model[:3, :3]).T
        
        # Set uniform matrices
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)
        glUniformMatrix3fv(normal_matrix_loc, 1, GL_FALSE, normal_matrix)
        
        # Set lighting uniforms
        glUniform3fv(light_pos_loc, 1, light_pos)
        glUniform3fv(view_pos_loc, 1, camera_pos)
        glUniform3fv(light_color_loc, 1, light_color)
        glUniform1f(ambient_strength_loc, ambient_strength)
        glUniform1f(specular_strength_loc, specular_strength)
        glUniform1i(shininess_loc, shininess)
        
        # Bind VAO and draw
        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        
        # Swap buffers
        glfw.swap_buffers(window)
        
        # Small delay to control rotation speed
        time.sleep(0.016)  # ~60 FPS
    
    # Cleanup
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteBuffers(1, [EBO])
    glDeleteProgram(shader_program)
    
    glfw.terminate()

if __name__ == "__main__":
    main()
