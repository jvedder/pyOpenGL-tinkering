import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
import time

# Vertex shader source code
vertex_shader_source = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 vertexColor;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    vertexColor = aColor;
}
"""

# Fragment shader source code
fragment_shader_source = """
#version 330 core
in vec3 vertexColor;
out vec4 FragColor;

void main()
{
    FragColor = vec4(vertexColor, 1.0);
}
"""

def create_cube_data():
    # Cube vertices (each face needs separate vertices for different colors)
    vertices = np.array([
        # Front face (red)
        -1.0, -1.0,  1.0,  1.0, 0.0, 0.0,  # Bottom-left
         1.0, -1.0,  1.0,  1.0, 0.0, 0.0,  # Bottom-right
         1.0,  1.0,  1.0,  1.0, 0.0, 0.0,  # Top-right
        -1.0,  1.0,  1.0,  1.0, 0.0, 0.0,  # Top-left
        
        # Back face (green)
         1.0, -1.0, -1.0,  0.0, 1.0, 0.0,  # Bottom-left
        -1.0, -1.0, -1.0,  0.0, 1.0, 0.0,  # Bottom-right
        -1.0,  1.0, -1.0,  0.0, 1.0, 0.0,  # Top-right
         1.0,  1.0, -1.0,  0.0, 1.0, 0.0,  # Top-left
        
        # Left face (blue)
        -1.0, -1.0, -1.0,  0.0, 0.0, 1.0,  # Bottom-left
        -1.0, -1.0,  1.0,  0.0, 0.0, 1.0,  # Bottom-right
        -1.0,  1.0,  1.0,  0.0, 0.0, 1.0,  # Top-right
        -1.0,  1.0, -1.0,  0.0, 0.0, 1.0,  # Top-left
        
        # Right face (yellow)
         1.0, -1.0,  1.0,  1.0, 1.0, 0.0,  # Bottom-left
         1.0, -1.0, -1.0,  1.0, 1.0, 0.0,  # Bottom-right
         1.0,  1.0, -1.0,  1.0, 1.0, 0.0,  # Top-right
         1.0,  1.0,  1.0,  1.0, 1.0, 0.0,  # Top-left
        
        # Top face (magenta)
        -1.0,  1.0,  1.0,  1.0, 0.0, 1.0,  # Bottom-left
         1.0,  1.0,  1.0,  1.0, 0.0, 1.0,  # Bottom-right
         1.0,  1.0, -1.0,  1.0, 0.0, 1.0,  # Top-right
        -1.0,  1.0, -1.0,  1.0, 0.0, 1.0,  # Top-left
        
        # Bottom face (cyan)
        -1.0, -1.0, -1.0,  0.0, 1.0, 1.0,  # Bottom-left
         1.0, -1.0, -1.0,  0.0, 1.0, 1.0,  # Bottom-right
         1.0, -1.0,  1.0,  0.0, 1.0, 1.0,  # Top-right
        -1.0, -1.0,  1.0,  0.0, 1.0, 1.0,  # Top-left
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
    window = glfw.create_window(800, 600, "OpenGL Core Profile Cube", None, None)
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
    
    # Create cube data
    vertices, indices = create_cube_data()
    
    # Create and bind VAO (Vertex Array Objects)
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    
    # Create and bind VBO (Vertex Buffer Objects)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    
    # Create and bind EBO (Element Buffer Object)
    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
    
    # Configure vertex attributes
    # Position attribute (location = 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    
    # Color attribute (location = 1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)
    
    # Unbind VAO
    glBindVertexArray(0)
    
    # Get uniform locations
    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    projection_loc = glGetUniformLocation(shader_program, "projection")
    
    # Create transformation matrices
    projection = pyrr.matrix44.create_perspective_projection_matrix(
        45.0, 800/600, 0.1, 100.0
    )
    view = pyrr.matrix44.create_look_at(
        [0.0, 0.0, 5.0],   # Camera position
        [0.0, 0.0, 0.0],   # Target position
        [0.0, 1.0, 0.0]    # Up vector
    )
    
    rotation_x = 0.0
    rotation_y = 0.0
    
    # Main loop
    while not glfw.window_should_close(window):
        # Handle events
        glfw.poll_events()
        
        # Clear the screen
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Use shader program
        glUseProgram(shader_program)
        
        # Update rotation
        rotation_x += 0.5
        rotation_y += 0.3
        
        # Create model matrix with rotation
        model = pyrr.matrix44.create_from_x_rotation(np.radians(rotation_x))
        model = pyrr.matrix44.multiply(model, pyrr.matrix44.create_from_y_rotation(np.radians(rotation_y)))
        
        # Set uniform matrices
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)
        
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
