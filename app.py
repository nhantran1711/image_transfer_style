import os
import re
import subprocess
import sys
import tempfile

import streamlit as st

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(BASE_DIR, "nst", "main.py")
ITERATION_RE = re.compile(r"Iteration (\d+):")

# Streamlit app
st.set_page_config(page_title="Neural Style Transfer Project", layout = "centered")
st.title("Neural Style Transfer")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    steps = st.slider("Steps", 50, 1000, 300, step = 50) # Number of optimization steps
    style_weight = st.number_input("Style weight", value = 1e6, format = "%.0f") # Weight for style loss
    content_weight = st.number_input("Content weight", value = 1.0) # Weight for content loss
    lr = st.number_input("Learning rate", value = 0.01, format = "%.4f") # Learning rate for the optimizer
    max_size = st.slider("Max image size", 128, 1024, 512, step = 64) # Max image dimension

# File uploaders for content and style images
col1, col2 = st.columns(2)
with col1:
    # File uploader for content image
    content_file = st.file_uploader("Content image", type=["jpg", "jpeg", "png"])
    if content_file:
        # Display the uploaded content image
        st.image(content_file, use_container_width=True)

with col2:
    # File uploader for style image
    style_file = st.file_uploader("Style image", type=["jpg", "jpeg", "png"])
    if style_file:
        # Display the uploaded style image
        st.image(style_file, use_container_width=True)

# Run button
run_clicked = st.button("Run style transfer", disabled = not (content_file and style_file))

# Run the style transfer script when the button is clicked
if run_clicked:
    # Create a temporary directory to save the uploaded images and the generated image
    with tempfile.TemporaryDirectory() as tmp_dir:
        content_path = os.path.join(tmp_dir, "content" + os.path.splitext(content_file.name)[1])
        style_path = os.path.join(tmp_dir, "style" + os.path.splitext(style_file.name)[1])
        output_path = os.path.join(tmp_dir, "generated_image.jpg")

        # Save the uploaded images to the temporary directory
        with open(content_path, "wb") as f:
            f.write(content_file.getbuffer())
        with open(style_path, "wb") as f:
            f.write(style_file.getbuffer())

        # Report progress roughly every 5% of steps, regardless of step count
        print_every = max(1, steps // 20)

        # Build the command to run the style transfer script with the specified parameters
        cmd = [
            sys.executable, "-u", MAIN_SCRIPT,
            "--content", content_path,
            "--style", style_path,
            "--output", output_path,
            "--steps", str(steps),
            "--style-weight", str(style_weight),
            "--content-weight", str(content_weight),
            "--lr", str(lr),
            "--max-size", str(max_size),
            "--print-every", str(print_every),
        ]

        # Run the command, streaming its output to drive a live progress bar
        progress_bar = st.progress(0, text = f"Running style transfer for {steps} steps...")
        output_lines = []
        process = subprocess.Popen(
            cmd, cwd = os.path.join(BASE_DIR, "nst"),
            stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True, bufsize = 1,
        )
        for line in process.stdout:
            output_lines.append(line)
            match = ITERATION_RE.search(line)
            if match:
                iteration = int(match.group(1))
                fraction = min(iteration / steps, 1.0)
                progress_bar.progress(fraction, text = f"Step {iteration}/{steps}")
        process.wait()
        progress_bar.empty()

        # Check the result of the subprocess and display appropriate messages
        if process.returncode != 0:
            st.error("Style transfer failed:")
            st.code("".join(output_lines))
        else:
            st.success("Done!")
            st.image(output_path, caption = "Generated image", use_container_width = True)
            # Provide a download button for the generated image
            with open(output_path, "rb") as f:
                st.download_button("Download result", f, file_name="generated_image.jpg", mime = "image/jpeg")
