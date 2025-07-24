import glfw
from OpenGL.GL import *
from PIL import Image

def save(window, filename="screenshot.png"):
    """
    Captures the content of a GLFW window and saves it as an image.
    """
    width, height = glfw.get_framebuffer_size(window)

    # Read pixels from the front buffer
    glReadBuffer(GL_FRONT) 
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

    # Create a Pillow Image and flip it vertically
    image = Image.frombytes("RGB", (width, height), pixels)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # Save the image
    image.save(filename)
    print(f"Image saved to {filename}")

# Example usage (assuming you have a GLFW window 'window' and rendered content)
# save_glfw_image(window, "opengl_torus.png")
