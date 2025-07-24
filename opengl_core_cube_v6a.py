import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
import time
import save_image

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

def create_torus_data():
    major_radius = 10.0
    minor_radius = 3.0
    major_segments = 100  # segments around the major radius
    minor_segments = 16   # segments around the tube
    
    # Define 4 colors for the 4 segments
    colors = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 1.0, 0.0],  # Green
        [0.0, 0.0, 1.0],  # Blue
        [1.0, 1.0, 0.0],  # Yellow
    ]

##    colors = [
##        [1.0, 1.0, 1.0],  # White
##        [0.0, 0.0, 0.0],  # Black
##        [1.0, 1.0, 1.0],  # White
##        [0.0, 0.0, 0.0],  # Black
##    ]    
    vertices = []
    indices = []
    
    # Generate vertices
    for i in range(major_segments):
        # Angle around the major radius
        theta = 2.0 * np.pi * i / major_segments
        
        # Determine color based on which quarter of the major radius we're in
        color_index = int(i / (major_segments / 4)) % 4
        color = colors[color_index]
        
        for j in range(minor_segments):
            # Angle around the minor radius (tube)
            phi = 2.0 * np.pi * j / minor_segments
            
            # Calculate position
            x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
            y = minor_radius * np.sin(phi)
            z = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
            
            # Calculate normal vector
            # Normal points outward from the tube surface
            center_x = major_radius * np.cos(theta)
            center_y = 0.0
            center_z = major_radius * np.sin(theta)
            
            normal_x = x - center_x
            normal_y = y - center_y
            normal_z = z - center_z
            
            # Normalize the normal vector
            normal_length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
            normal_x /= normal_length
            normal_y /= normal_length
            normal_z /= normal_length
            
            # Add vertex: position (3) + color (3) + normal (3)
            vertices.extend([
                x, y, z,
                color[0], color[1], color[2],
                normal_x, normal_y, normal_z
            ])
    
    # Generate indices for triangles
    for i in range(major_segments):
        for j in range(minor_segments):
            # Current vertex indices
            current = i * minor_segments + j
            next_major = ((i + 1) % major_segments) * minor_segments + j
            next_minor = i * minor_segments + ((j + 1) % minor_segments)
            next_both = ((i + 1) % major_segments) * minor_segments + ((j + 1) % minor_segments)
            
            # Two triangles per quad
            indices.extend([
                current, next_major, next_minor,
                next_major, next_both, next_minor
            ])
    
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

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
    window = glfw.create_window(800, 600, "OpenGL Core Profile Torus with Blinn-Phong Lighting", None, None)
    if not window:
        glfw.terminate()
        return
    
    glfw.make_context_current(window)
    
    # Enable depth testing
    glEnable(GL_DEPTH_TEST)
    
    # Create shader program
    shader_program = create_shader_program()
    
    # Create torus data
    vertices, indices = create_torus_data()
    
    # Create and bind VAO
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    
    # Create and bind VBO
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    
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
    
    # Camera position - moved back to accommodate larger torus
    camera_pos = np.array([0.0, 0.0, 40.0], dtype=np.float32)
    view = pyrr.matrix44.create_look_at(
        camera_pos,            # Camera position
        [0.0, 0.0, 0.0],       # Target position
        [0.0, 1.0, 0.0]        # Up vector
    )
    
    # Lighting properties - adjusted for larger scene
    light_pos = np.array([30.0, 30.0, 30.0], dtype=np.float32)
    light_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # White light
    ambient_strength = 0.1
    specular_strength = 0.5
    shininess = 32
    
    rotation_x = 90.0
    rotation_y = 0.0
    
    # Main loop
    save = True
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

        if save:
            save_image.save(window, "screenshot.png")
            save = False
        
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
