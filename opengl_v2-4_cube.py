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
    """
    Build the cube as 6 faces * 2 triangles * 3 vertices = 36 vertices,
    with NO shared vertices between triangles (not even within the same
    face). Each face is defined by its 4 corners (BL, BR, TR, TL), a
    color, and a resting-state normal; it's then split into two
    triangles: (BL, BR, TR) and (TR, TL, BL).

    Why no sharing: once we start deforming vertex positions frame to
    frame, a quad's two triangles can end up in different planes (the
    quad becomes non-planar). If the two triangles shared vertices, each
    shared vertex could only store ONE normal, which can't correctly
    represent two differently-oriented triangles at once. Duplicating
    every vertex per-triangle means each triangle owns its own normal,
    independent of its neighbor.

    Because nothing is shared, we no longer need an index buffer -
    the vertex array is drawn directly with glDrawArrays.
    """
    # Each entry: (bottom-left, bottom-right, top-right, top-left, color, normal)
    faces = [
        ((-1, -1,  1), ( 1, -1,  1), ( 1,  1,  1), (-1,  1,  1), (1, 0, 0), ( 0,  0,  1)),  # Front  - red
        (( 1, -1, -1), (-1, -1, -1), (-1,  1, -1), ( 1,  1, -1), (0, 1, 0), ( 0,  0, -1)),  # Back   - green
        ((-1, -1, -1), (-1, -1,  1), (-1,  1,  1), (-1,  1, -1), (0, 0, 1), (-1,  0,  0)),  # Left   - blue
        (( 1, -1,  1), ( 1, -1, -1), ( 1,  1, -1), ( 1,  1,  1), (1, 1, 0), ( 1,  0,  0)),  # Right  - yellow
        ((-1,  1,  1), ( 1,  1,  1), ( 1,  1, -1), (-1,  1, -1), (1, 0, 1), ( 0,  1,  0)),  # Top    - magenta
        ((-1, -1, -1), ( 1, -1, -1), ( 1, -1,  1), (-1, -1,  1), (0, 1, 1), ( 0, -1,  0)),  # Bottom - cyan
    ]

    vertex_list = []
    for bl, br, tr, tl, color, normal in faces:
        # Triangle 1: BL, BR, TR   |   Triangle 2: TR, TL, BL
        for corner in (bl, br, tr, tr, tl, bl):
            vertex_list.extend(corner)
            vertex_list.extend(color)
            vertex_list.extend(normal)

    vertices = np.array(vertex_list, dtype=np.float32)
    return vertices

def apply_top_scale(base_vertices, scale):
    """
    Return a modified copy of the vertex array where every vertex with
    y == 1.0 (i.e. anything on the top face, including the top edges of
    the front/back/left/right faces) has its x and z coordinates scaled.

    This pulls the top face inward/outward, turning the 4 side faces
    into trapezoids while leaving the bottom face untouched. Because
    triangles are no longer shared across a face's diagonal, this can
    now genuinely bend a face into two differently-angled triangles.
    """
    verts = base_vertices.reshape(-1, 9).copy()

    # Column 1 of each row is the y position (columns 0,1,2 = x,y,z)
    top_mask = verts[:, 1] > 0.5  # selects the y == 1.0 vertices

    verts[top_mask, 0] *= scale  # scale x
    verts[top_mask, 2] *= scale  # scale z

    return verts.reshape(-1)

def recalculate_normals(vertices):
    """
    Recompute a flat shading normal for each TRIANGLE (not each face)
    from its current, possibly-deformed vertex positions.

    Every 3 consecutive vertices in the array form one triangle. For a
    triangle with corners p0, p1, p2 (in the same winding order used
    when the vertex list was built), the normal is:

        edge1 = p1 - p0
        edge2 = p2 - p0
        normal = normalize(cross(edge1, edge2))

    That normal gets written into all 3 of that triangle's vertices.
    Since the two triangles making up a face are no longer forced to
    share vertices, they can end up with different normals if the face
    has bent into a non-planar shape - you'll see a subtle crease along
    the diagonal in that case, which is the correct, expected result of
    faceted (flat) shading on non-planar geometry.
    """
    verts = vertices.reshape(-1, 9).copy()
    num_triangles = verts.shape[0] // 3

    for tri in range(num_triangles):
        i0 = tri * 3
        p0 = verts[i0, 0:3]
        p1 = verts[i0 + 1, 0:3]
        p2 = verts[i0 + 2, 0:3]

        edge1 = p1 - p0
        edge2 = p2 - p0
        normal = np.cross(edge1, edge2)

        length = np.linalg.norm(normal)
        if length > 1e-8:
            normal = normal / length

        verts[i0:i0 + 3, 6:9] = normal

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
    base_vertices = create_cube_data()
    vertices = base_vertices.copy()
    vertex_count = len(base_vertices) // 9  # 9 floats per vertex; used by glDrawArrays
    
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
    
    # No EBO / index buffer: every triangle now owns its own 3 vertices,
    # so there's nothing left to share via indices.
    
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
        
        # Recompute the vertex array from the untouched base shape, fix up
        # the normals to match the new (possibly non-planar) triangles,
        # then push it to the existing GPU buffer with glBufferSubData.
        # glBufferSubData overwrites the buffer's contents in place rather
        # than allocating a new buffer, which is what glBufferData would do.
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
        
        # Bind VAO and draw. glDrawArrays reads `vertex_count` vertices
        # straight from the VBO in order - no index buffer needed since
        # every triangle owns its own 3 vertices.
        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, vertex_count)
        glBindVertexArray(0)
        
        # Swap buffers
        glfw.swap_buffers(window)
        
        # Small delay to control rotation speed
        time.sleep(0.016)  # ~60 FPS
    
    # Cleanup
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteProgram(shader_program)
    
    glfw.terminate()

if __name__ == "__main__":
    main()
