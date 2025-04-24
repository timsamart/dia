import argparse
import tempfile
import time
import re
from pathlib import Path
from typing import Optional, Tuple, List
import concurrent.futures
import os

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import random

from dia.model import Dia


# --- Global Setup ---
parser = argparse.ArgumentParser(description="Gradio interface for Nari TTS")
parser.add_argument("--device", type=str, default=None, help="Force device (e.g., 'cuda', 'mps', 'cpu')")
parser.add_argument("--share", action="store_true", help="Enable Gradio sharing")

args = parser.parse_args()


# Determine device
if args.device:
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
# Simplified MPS check for broader compatibility
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Basic check is usually sufficient, detailed check can be problematic
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load Nari model and config
print("Loading Nari model...")
try:
    # Use the function from inference.py
    model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", device=device)
except Exception as e:
    print(f"Error loading Nari model: {e}")
    raise


def split_text_into_chunks(text: str, max_chunk_length: int = 300) -> List[str]:
    """
    Split text into manageable chunks at meaningful boundaries (sentence/dialogue breaks).
    
    Args:
        text: Input text to be split
        max_chunk_length: Maximum length of each chunk (in characters)
        
    Returns:
        List of text chunks
    """
    # Define meaningful boundaries where we can split (in order of priority)
    boundaries = [
        r'\n\n',          # Double line breaks - paragraph boundaries
        r'\[S[12]\]',     # Speaker changes
        r'\.\s',          # End of sentences
        r'[?!]\s',        # Question or exclamation marks
        r',\s',           # Commas
        r'\s'             # Any whitespace as last resort
    ]
    
    chunks = []
    current_text = text.strip()
    
    while current_text:
        # If the current text is already below the maximum length, add it as the final chunk
        if len(current_text) <= max_chunk_length:
            chunks.append(current_text)
            break
        
        # Try to find a good split point within the max_chunk_length
        split_index = None
        
        # Try each boundary pattern in order of priority
        for pattern in boundaries:
            # Look for the pattern near the maximum chunk length
            # Search backwards from max_chunk_length
            matches = list(re.finditer(pattern, current_text[:max_chunk_length]))
            if matches:
                # Get the last match (closest to max_chunk_length without exceeding it)
                match = matches[-1]
                # For speaker changes, we want to include the speaker tag in the next chunk
                if pattern == r'\[S[12]\]':
                    split_index = match.start()
                else:
                    split_index = match.end()
                break
        
        # If no good boundary found, just split at max_chunk_length
        if split_index is None or split_index == 0:
            split_index = max_chunk_length
        
        # Add the chunk and continue with the rest of the text
        chunks.append(current_text[:split_index].strip())
        current_text = current_text[split_index:].strip()
        
        # Ensure the next chunk starts with a speaker tag if possible
        if chunks and not re.match(r'^\[S[12]\]', current_text) and re.search(r'\[S[12]\]', current_text):
            # Find the first speaker tag in the remaining text
            match = re.search(r'\[S[12]\]', current_text)
            if match:
                # Add text before the speaker tag to the previous chunk if it's not too long
                previous_chunk = chunks[-1]
                if len(previous_chunk) + match.start() <= max_chunk_length * 1.5:
                    chunks[-1] = previous_chunk + " " + current_text[:match.start()].strip()
                    current_text = current_text[match.start():].strip()
    
    return chunks


def run_inference_on_chunk(
    chunk_text: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_new_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    speed_factor: float,
    seed: Optional[int] = None,
) -> tuple:
    """Run inference on a single text chunk"""
    global model, device

    if not chunk_text or chunk_text.isspace():
        return None

    temp_audio_prompt_path = None
    output_audio = (44100, np.zeros(1, dtype=np.float32))

    try:
        prompt_path_for_generate = None
        if audio_prompt_input is not None:
            sr, audio_data = audio_prompt_input
            # Check if audio_data is valid
            if audio_data is None or audio_data.size == 0 or audio_data.max() == 0:  # Check for silence/empty
                print("Audio prompt seems empty or silent, ignoring prompt.")
            else:
                # Save prompt audio to a temporary WAV file
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f_audio:
                    temp_audio_prompt_path = f_audio.name  # Store path for cleanup

                    # Basic audio preprocessing for consistency
                    # Convert to float32 in [-1, 1] range if integer type
                    if np.issubdtype(audio_data.dtype, np.integer):
                        max_val = np.iinfo(audio_data.dtype).max
                        audio_data = audio_data.astype(np.float32) / max_val
                    elif not np.issubdtype(audio_data.dtype, np.floating):
                        print(f"Unsupported audio prompt dtype {audio_data.dtype}, attempting conversion.")
                        # Attempt conversion, might fail for complex types
                        try:
                            audio_data = audio_data.astype(np.float32)
                        except Exception as conv_e:
                            raise Exception(f"Failed to convert audio prompt to float32: {conv_e}")

                    # Ensure mono (average channels if stereo)
                    if audio_data.ndim > 1:
                        if audio_data.shape[0] == 2:  # Assume (2, N)
                            audio_data = np.mean(audio_data, axis=0)
                        elif audio_data.shape[1] == 2:  # Assume (N, 2)
                            audio_data = np.mean(audio_data, axis=1)
                        else:
                            print(f"Audio prompt has unexpected shape {audio_data.shape}, taking first channel/axis.")
                            audio_data = (
                                audio_data[0] if audio_data.shape[0] < audio_data.shape[1] else audio_data[:, 0]
                            )
                        audio_data = np.ascontiguousarray(audio_data)  # Ensure contiguous after slicing/mean

                    # Write using soundfile
                    try:
                        sf.write(
                            temp_audio_prompt_path, audio_data, sr, subtype="FLOAT"
                        )  # Explicitly use FLOAT subtype
                        prompt_path_for_generate = temp_audio_prompt_path
                        print(f"Created temporary audio prompt file: {temp_audio_prompt_path} (orig sr: {sr})")
                    except Exception as write_e:
                        print(f"Error writing temporary audio file: {write_e}")
                        raise Exception(f"Failed to save audio prompt: {write_e}")

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            print(f"Using seed: {seed}")
            random.seed(seed)
            np.random.seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Run Generation
        with torch.inference_mode():
            output_audio_np = model.generate(
                chunk_text,
                max_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,  # Pass the value here
                use_torch_compile=True,  # Keep False for Gradio stability
                audio_prompt=prompt_path_for_generate,
            )

        # Process Audio
        if output_audio_np is not None:
            # Get sample rate from the loaded DAC model
            output_sr = 44100

            # --- Slow down audio ---
            original_len = len(output_audio_np)
            # Ensure speed_factor is positive and not excessively small/large to avoid issues
            speed_factor = max(0.1, min(speed_factor, 5.0))
            target_len = int(original_len / speed_factor)  # Target length based on speed_factor
            if target_len != original_len and target_len > 0:  # Only interpolate if length changes and is valid
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                resampled_audio_np = np.interp(x_resampled, x_original, output_audio_np)
                output_audio = (
                    output_sr,
                    resampled_audio_np.astype(np.float32),
                )  # Use resampled audio
                print(f"Resampled audio from {original_len} to {target_len} samples for {speed_factor:.2f}x speed.")
            else:
                output_audio = (
                    output_sr,
                    output_audio_np,
                )  # Keep original if calculation fails or no change
                print(f"Skipping audio speed adjustment (factor: {speed_factor:.2f}).")
            # --- End slowdown ---

        else:
            print("\nGeneration for chunk finished, but no valid tokens were produced.")
            output_audio = (44100, np.zeros(1, dtype=np.float32))

    except Exception as e:
        print(f"Error during inference for chunk: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Cleanup
        if temp_audio_prompt_path and Path(temp_audio_prompt_path).exists():
            try:
                Path(temp_audio_prompt_path).unlink()
                print(f"Deleted temporary audio prompt file: {temp_audio_prompt_path}")
            except OSError as e:
                print(f"Warning: Error deleting temporary audio prompt file {temp_audio_prompt_path}: {e}")

    return output_audio


def run_inference(
    text_input: str,
    audio_prompt_input: Optional[Tuple[int, np.ndarray]],
    max_new_tokens: int,
    cfg_scale: float,
    temperature: float,
    top_p: float,
    cfg_filter_top_k: int,
    speed_factor: float,
    seed: Optional[int] = None,
    enable_chunking: bool = False,
    max_chunk_length: int = 300,
    progress=gr.Progress(),
):
    """
    Runs Nari inference using the globally loaded model and provided inputs.
    Supports processing long texts by splitting into chunks and processing in sequence.
    """
    global model, device

    if not text_input or text_input.isspace():
        raise gr.Error("Text input cannot be empty.")
    
    start_time = time.time()
    
    # Process text as a single unit or split into chunks
    if enable_chunking:
        chunks = split_text_into_chunks(text_input, max_chunk_length)
        total_chunks = len(chunks)
        print(f"Split text into {total_chunks} chunks for processing")
        
        # Create progress tracker
        progress_text = gr.Markdown(f"Processing 0/{total_chunks} chunks...")
        progress(0, desc="Starting batch processing...")
        
        # Process chunks with same seed to maintain speaker consistency
        chunk_results = []
        chunk_audios = []
        
        for i, chunk in enumerate(chunks):
            progress((i + 0.5) / total_chunks, desc=f"Processing chunk {i+1}/{total_chunks}...")
            print(f"Processing chunk {i+1}/{total_chunks}: {chunk[:50]}...")
            
            # Process this chunk
            result = run_inference_on_chunk(
                chunk,
                audio_prompt_input if i == 0 else None,  # Only use audio prompt for first chunk
                max_new_tokens,
                cfg_scale,
                temperature,
                top_p,
                cfg_filter_top_k,
                speed_factor,
                seed if seed is not None else None  # Use the same seed for all chunks
            )
            
            # Check if generation succeeded
            if result is not None:
                chunk_audios.append(result[1])
                chunk_results.append(result)
            else:
                print(f"Warning: Chunk {i+1} failed to generate, skipping.")
            
            progress((i + 1) / total_chunks, desc=f"Processed {i+1}/{total_chunks} chunks")
        
        # Combine all audio chunks
        if chunk_audios:
            sample_rate = 44100  # Assuming all chunks have the same sample rate
            combined_audio = np.concatenate(chunk_audios)
            output_audio = (sample_rate, combined_audio)
            
            # Explicitly convert to int16 to prevent Gradio warning
            if output_audio[1].dtype == np.float32 or output_audio[1].dtype == np.float64:
                audio_for_gradio = np.clip(output_audio[1], -1.0, 1.0)
                audio_for_gradio = (audio_for_gradio * 32767).astype(np.int16)
                output_audio = (output_audio[0], audio_for_gradio)
                print("Converted audio to int16 for Gradio output.")
        else:
            print("No valid audio was generated from any chunk.")
            output_audio = (44100, np.zeros(1, dtype=np.float32))
    else:
        # Process as a single unit, original method
        progress(0, desc="Starting processing...")
        output_audio = run_inference_on_chunk(
            text_input,
            audio_prompt_input,
            max_new_tokens,
            cfg_scale,
            temperature,
            top_p,
            cfg_filter_top_k,
            speed_factor,
            seed
        )
        progress(1, desc="Processing complete")
        
        # Ensure output is in the right format
        if output_audio is None:
            output_audio = (44100, np.zeros(1, dtype=np.float32))
            print("Generation failed to produce valid output.")
        else:
            # Explicitly convert to int16 to prevent Gradio warning
            if output_audio[1].dtype == np.float32 or output_audio[1].dtype == np.float64:
                audio_for_gradio = np.clip(output_audio[1], -1.0, 1.0)
                audio_for_gradio = (audio_for_gradio * 32767).astype(np.int16)
                output_audio = (output_audio[0], audio_for_gradio)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total generation time: {total_time:.2f} seconds")
    
    return output_audio, f"Generation completed in {total_time:.2f} seconds"


# --- Create Gradio Interface ---
css = """
#col-container {max-width: 90%; margin-left: auto; margin-right: auto;}
"""
# Attempt to load default text from example.txt
default_text = "[S1] Dia is an open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] Wow. Amazing. (laughs) \n[S2] Try it now on Git hub or Hugging Face."
example_txt_path = Path("./example.txt")
if example_txt_path.exists():
    try:
        default_text = example_txt_path.read_text(encoding="utf-8").strip()
        if not default_text:  # Handle empty example file
            default_text = "Example text file was empty."
    except Exception as e:
        print(f"Warning: Could not read example.txt: {e}")


# Build Gradio UI
with gr.Blocks(css=css) as demo:
    gr.Markdown("# Nari Text-to-Speech Synthesis")
    status_output = gr.Markdown("Ready to generate audio")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter text here...",
                value=default_text,
                lines=7,  # Increased lines for longer text
            )
            audio_prompt_input = gr.Audio(
                label="Audio Prompt (Optional)",
                show_label=True,
                sources=["upload", "microphone"],
                type="numpy",
            )
            
            with gr.Accordion("Chunking and Batch Settings", open=True):
                enable_chunking = gr.Checkbox(
                    label="Enable Text Chunking for Long Inputs",
                    value=True,
                    info="Process text in meaningful chunks for better handling of long inputs."
                )
                max_chunk_length = gr.Slider(
                    label="Maximum Chunk Length (characters)",
                    minimum=100,
                    maximum=800,
                    value=300,
                    step=50,
                    info="Maximum character length for each text chunk when processing long inputs."
                )
                seed_value = gr.Number(
                    label="Random Seed",
                    value=None,
                    precision=0,
                    info="Set a seed value to maintain consistent voices across generations. Leave empty for random results."
                )
            
            with gr.Accordion("Generation Parameters", open=False):
                max_new_tokens = gr.Slider(
                    label="Max New Tokens (Audio Length)",
                    minimum=860,
                    maximum=3072,
                    value=model.config.data.audio_length,  # Use config default if available, else fallback
                    step=50,
                    info="Controls the maximum length of the generated audio (more tokens = longer audio).",
                )
                cfg_scale = gr.Slider(
                    label="CFG Scale (Guidance Strength)",
                    minimum=1.0,
                    maximum=5.0,
                    value=3.0,  # Default from inference.py
                    step=0.1,
                    info="Higher values increase adherence to the text prompt.",
                )
                temperature = gr.Slider(
                    label="Temperature (Randomness)",
                    minimum=1.0,
                    maximum=1.5,
                    value=1.3,  # Default from inference.py
                    step=0.05,
                    info="Lower values make the output more deterministic, higher values increase randomness.",
                )
                top_p = gr.Slider(
                    label="Top P (Nucleus Sampling)",
                    minimum=0.80,
                    maximum=1.0,
                    value=0.95,  # Default from inference.py
                    step=0.01,
                    info="Filters vocabulary to the most likely tokens cumulatively reaching probability P.",
                )
                cfg_filter_top_k = gr.Slider(
                    label="CFG Filter Top K",
                    minimum=15,
                    maximum=50,
                    value=30,
                    step=1,
                    info="Top k filter for CFG guidance.",
                )
                speed_factor_slider = gr.Slider(
                    label="Speed Factor",
                    minimum=0.8,
                    maximum=1.0,
                    value=0.94,
                    step=0.02,
                    info="Adjusts the speed of the generated audio (1.0 = original speed).",
                )

            run_button = gr.Button("Generate Audio", variant="primary")

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Audio",
                type="numpy",
                autoplay=False,
            )
            generation_time = gr.Markdown("")

    # Link button click to function
    run_button.click(
        fn=run_inference,
        inputs=[
            text_input,
            audio_prompt_input,
            max_new_tokens,
            cfg_scale,
            temperature,
            top_p,
            cfg_filter_top_k,
            speed_factor_slider,
            seed_value,
            enable_chunking,
            max_chunk_length,
        ],
        outputs=[audio_output, generation_time],
        api_name="generate_audio",
    )

    # Add examples (ensure the prompt path is correct or remove it if example file doesn't exist)
    example_prompt_path = "./example_prompt.mp3"  # Adjust if needed
    examples_list = [
        [
            "[S1] Oh fire! Oh my goodness! What's the procedure? What to we do people? The smoke could be coming through an air duct! \n[S2] Oh my god! Okay.. it's happening. Everybody stay calm! \n[S1] What's the procedure... \n[S2] Everybody stay fucking calm!!!... Everybody fucking calm down!!!!! \n[S1] No! No! If you touch the handle, if its hot there might be a fire down the hallway! ",
            None,
            3072,
            3.0,
            1.3,
            0.95,
            35,
            0.94,
            42,  # Seed value
            True,  # Enable chunking
            300,   # Max chunk length
        ],
        [
            "[S1] Open weights text to dialogue model. \n[S2] You get full control over scripts and voices. \n[S1] I'm biased, but I think we clearly won. \n[S2] Hard to disagree. (laughs) \n[S1] Thanks for listening to this demo. \n[S2] Try it now on Git hub and Hugging Face. \n[S1] If you liked our model, please give us a star and share to your friends. \n[S2] This was Nari Labs.",
            example_prompt_path if Path(example_prompt_path).exists() else None,
            3072,
            3.0,
            1.3,
            0.95,
            35,
            0.94,
            42,  # Seed value
            True,  # Enable chunking
            300,   # Max chunk length
        ],
    ]

    if examples_list:
        gr.Examples(
            examples=examples_list,
            inputs=[
                text_input,
                audio_prompt_input,
                max_new_tokens,
                cfg_scale,
                temperature,
                top_p,
                cfg_filter_top_k,
                speed_factor_slider,
                seed_value,
                enable_chunking,
                max_chunk_length,
            ],
            outputs=[audio_output, generation_time],
            fn=run_inference,
            cache_examples=False,
            label="Examples (Click to Run)",
        )
    else:
        gr.Markdown("_(No examples configured or example prompt file missing)_")

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching Gradio interface...")

    # set `GRADIO_SERVER_NAME`, `GRADIO_SERVER_PORT` env vars to override default values
    # use `GRADIO_SERVER_NAME=0.0.0.0` for Docker
    demo.launch(share=args.share)
