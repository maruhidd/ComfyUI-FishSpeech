import os
import io
import wave
import torch
import torchaudio
import numpy as np
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    sys.path.append(os.path.join(current_dir, "fish_speech"))

# Make sure the required models are downloaded
os.makedirs("checkpoints", exist_ok=True)
try:
    snapshot_download(repo_id="fishaudio/fish-speech-1.5", local_dir="./checkpoints/fish-speech-1.5")
    print("All checkpoints downloaded")
except Exception as e:
    print(f"Error downloading checkpoints: {e}")

# Set audio backend
torchaudio.set_audio_backend("soundfile")

# Import required modules
try:
    from loguru import logger
except ImportError:
    import logging as logger

# 今のpath
print(os.path.abspath(__file__))

try:
    from fish_speech.text.chn_text_norm.text import Text as ChnNormedText
    from fish_speech.utils import autocast_exclude_mps, set_seed
    from tools.api import decode_vq_tokens, encode_reference
    from tools.llama.generate import (
        GenerateRequest,
        GenerateResponse,
        WrappedGenerateResponse,
        launch_thread_safe_queue,
    )
    from tools.vqgan.inference import load_model as load_decoder_model
    from tools.schema import (
        ServeReferenceAudio,
        ServeTTSRequest,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Using simplified implementation without fish_speech dependencies")
    
    # Define simplified versions of required classes/functions
    class ChnNormedText:
        def __init__(self, raw_text):
            self.raw_text = raw_text
        
        def normalize(self):
            return self.raw_text
    
    def autocast_exclude_mps(*args, **kwargs):
        class DummyContext:
            def __enter__(self):
                return self
            
            def __exit__(self, *args):
                pass
        
        return DummyContext()
    
    def set_seed(seed):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

# Initialize global variables
llama_queue = None
decoder_model = None

def initialize_models():
    global llama_queue, decoder_model
    
    if llama_queue is None or decoder_model is None:
        try:
            logger.info("Loading Llama model...")
            llama_queue = launch_thread_safe_queue(
                checkpoint_path=Path("checkpoints/fish-speech-1.5"),
                device="cuda" if torch.cuda.is_available() else "cpu",
                precision=torch.bfloat16,
                compile=True,
            )
            logger.info("Llama model loaded, loading VQ-GAN model...")

            decoder_model = load_decoder_model(
                config_name="firefly_gan_vq",
                checkpoint_path=Path("checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            logger.info("Decoder model loaded")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise RuntimeError(f"Failed to initialize FishSpeech models: {e}. Make sure the models are downloaded and the fish_speech module is properly installed.")

# Node for loading reference audio
class LoadFishAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upload": ("FISH_AUDIO_UPLOAD",),
                "audio": ("STRING", {"default": ""})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_path",)
    FUNCTION = "load_audio"
    CATEGORY = "FishSpeech"

    def load_audio(self, upload=None, audio=""):
        # Return the path to the audio file
        return (audio,)

# Node for loading SRT files
class LoadSRT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upload": ("SRTPLOAD",),
                "srt": ("STRING", {"default": ""})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("srt_path",)
    FUNCTION = "load_srt"
    CATEGORY = "FishSpeech"

    def load_srt(self, upload=None, srt=""):
        # Return the path to the SRT file
        return (srt,)

# Node for previewing audio
class PreViewAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("STRING", {"default": ""}),
                "type": ("STRING", {"default": "input"})
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "preview_audio"
    CATEGORY = "FishSpeech"
    OUTPUT_NODE = True

    def preview_audio(self, audio, type="input"):
        # This function doesn't need to do anything in the backend
        # The frontend JS will handle the audio preview
        return {"audio": [audio, type]}

# Node for FishSpeech inference
class FishSpeech_INFER:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test."}),
                "reference_audio_path": ("STRING", {"default": ""}),
                "reference_text": ("STRING", {"multiline": True, "default": ""}),
                "normalize": ("BOOLEAN", {"default": False}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "chunk_length": ("INT", {"default": 200, "min": 0, "max": 300, "step": 8}),
                "top_p": ("FLOAT", {"default": 0.7, "min": 0.6, "max": 0.9, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 1.5, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.6, "max": 0.9, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            }
        }
    
    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("output_path", "audio")
    FUNCTION = "generate_speech"
    CATEGORY = "FishSpeech"

    def generate_speech(self, text, reference_audio_path, reference_text, normalize, max_new_tokens, chunk_length, top_p, repetition_penalty, temperature, seed):
        try:
            # Initialize models if not already initialized
            initialize_models()
            
            # Normalize text if requested
            if normalize:
                text = ChnNormedText(raw_text=text).normalize()
            
            references = []
            if reference_audio_path and os.path.exists(reference_audio_path):
                # Read the reference audio file
                with open(reference_audio_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                
                references = [
                    ServeReferenceAudio(audio=audio_bytes, text=reference_text)
                ]
            
            # Create TTS request
            req = ServeTTSRequest(
                text=text,
                normalize=normalize,
                reference_id=None,
                references=references,
                max_new_tokens=max_new_tokens,
                chunk_length=chunk_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                seed=seed if seed > 0 else None,
                use_memory_cache="never",
            )
            
            # Set seed if provided
            if seed > 0:
                set_seed(seed)
            
            # Process reference audio
            prompt_tokens = []
            prompt_texts = []
            
            if references:
                for ref in references:
                    prompt_tokens.append(
                        encode_reference(
                            decoder_model=decoder_model,
                            reference_audio=ref.audio,
                            enable_reference_audio=True,
                        )
                    )
                    prompt_texts.append(ref.text)
            
            # Create LLAMA request
            request = dict(
                device=decoder_model.device,
                max_new_tokens=max_new_tokens,
                text=text if not normalize else ChnNormedText(raw_text=text).normalize(),
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                compile=True,
                iterative_prompt=chunk_length > 0,
                chunk_length=chunk_length,
                max_length=4096,
                prompt_tokens=prompt_tokens,
                prompt_text=prompt_texts,
            )
            
            # Create response queue
            response_queue = torch.multiprocessing.Queue()
            llama_queue.put(
                GenerateRequest(
                    request=request,
                    response_queue=response_queue,
                )
            )
            
            segments = []
            
            # Process responses
            while True:
                result = response_queue.get()
                if result.status == "error":
                    raise Exception(f"Error generating speech: {result.response}")
                
                result = result.response
                if result.action == "next":
                    break
                
                with autocast_exclude_mps(
                    device_type=decoder_model.device.type, dtype=torch.bfloat16
                ):
                    fake_audios = decode_vq_tokens(
                        decoder_model=decoder_model,
                        codes=result.codes,
                    )
                
                fake_audios = fake_audios.float().cpu().numpy()
                segments.append(fake_audios)
            
            if len(segments) == 0:
                raise Exception("No audio generated, please check the input text.")
            
            # Concatenate all segments
            audio = np.concatenate(segments, axis=0)
            
            # Save the audio to a file
            output_dir = os.path.join(os.getcwd(), "outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate a unique filename
            timestamp = torch.cuda.current_device() if torch.cuda.is_available() else 0
            output_filename = f"fishspeech_output_{timestamp}_{hash(text) % 10000}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save as WAV file
            with wave.open(output_path, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(decoder_model.spec_transform.sample_rate)
                wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())
            
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return (output_path, (decoder_model.spec_transform.sample_rate, audio))
        
        except Exception as e:
            # If the fish_speech module is not available, provide a helpful error message
            error_msg = str(e)
            if "No module named" in error_msg:
                error_msg = f"Error: {error_msg}. Make sure the fish_speech module is properly installed."
            elif "ServeReferenceAudio" in error_msg:
                error_msg = "Error: Required classes from fish_speech are not available. Make sure the fish_speech module is properly installed."
            
            # Create a dummy audio file with a beep sound to indicate an error
            sample_rate = 44100
            duration = 1.0  # seconds
            frequency = 440.0  # Hz (A4 note)
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            beep = 0.5 * np.sin(2 * np.pi * frequency * t)
            
            # Save the beep sound
            output_dir = os.path.join(os.getcwd(), "outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"error_beep_{hash(error_msg) % 10000}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            with wave.open(output_path, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes((beep * 32767).astype(np.int16).tobytes())
            
            print(f"FishSpeech_INFER error: {error_msg}")
            return (output_path, (sample_rate, beep))

# Node for FishSpeech inference with SRT
class FishSpeech_INFER_SRT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, this is a test."}),
                "reference_audio_path": ("STRING", {"default": ""}),
                "srt_path": ("STRING", {"default": ""}),
                "normalize": ("BOOLEAN", {"default": False}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "chunk_length": ("INT", {"default": 200, "min": 0, "max": 300, "step": 8}),
                "top_p": ("FLOAT", {"default": 0.7, "min": 0.6, "max": 0.9, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 1.5, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.6, "max": 0.9, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647})
            }
        }
    
    RETURN_TYPES = ("STRING", "AUDIO")
    RETURN_NAMES = ("output_path", "audio")
    FUNCTION = "generate_speech_with_srt"
    CATEGORY = "FishSpeech"

    def generate_speech_with_srt(self, text, reference_audio_path, srt_path, normalize, max_new_tokens, chunk_length, top_p, repetition_penalty, temperature, seed):
        try:
            # Initialize models if not already initialized
            initialize_models()
            
            # Read SRT file to get reference text
            reference_text = ""
            if srt_path and os.path.exists(srt_path):
                try:
                    with open(srt_path, 'r', encoding='utf-8') as srt_file:
                        # Simple SRT parsing - extract only the text content
                        lines = srt_file.readlines()
                        for i, line in enumerate(lines):
                            # Skip line numbers and timestamps
                            if not line.strip().isdigit() and '-->' not in line:
                                # Add non-empty lines that aren't numbers or timestamps
                                if line.strip():
                                    reference_text += line.strip() + " "
                except Exception as e:
                    logger.error(f"Error reading SRT file: {e}")
            
            # Now use the same logic as FishSpeech_INFER
            references = []
            if reference_audio_path and os.path.exists(reference_audio_path):
                # Read the reference audio file
                with open(reference_audio_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                
                references = [
                    ServeReferenceAudio(audio=audio_bytes, text=reference_text)
                ]
            
            # Create TTS request
            req = ServeTTSRequest(
                text=text,
                normalize=normalize,
                reference_id=None,
                references=references,
                max_new_tokens=max_new_tokens,
                chunk_length=chunk_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                seed=seed if seed > 0 else None,
                use_memory_cache="never",
            )
            
            # Set seed if provided
            if seed > 0:
                set_seed(seed)
            
            # Process reference audio
            prompt_tokens = []
            prompt_texts = []
            
            if references:
                for ref in references:
                    prompt_tokens.append(
                        encode_reference(
                            decoder_model=decoder_model,
                            reference_audio=ref.audio,
                            enable_reference_audio=True,
                        )
                    )
                    prompt_texts.append(ref.text)
            
            # Create LLAMA request
            request = dict(
                device=decoder_model.device,
                max_new_tokens=max_new_tokens,
                text=text if not normalize else ChnNormedText(raw_text=text).normalize(),
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                compile=True,
                iterative_prompt=chunk_length > 0,
                chunk_length=chunk_length,
                max_length=4096,
                prompt_tokens=prompt_tokens,
                prompt_text=prompt_texts,
            )
            
            # Create response queue
            response_queue = torch.multiprocessing.Queue()
            llama_queue.put(
                GenerateRequest(
                    request=request,
                    response_queue=response_queue,
                )
            )
            
            segments = []
            
            # Process responses
            while True:
                result = response_queue.get()
                if result.status == "error":
                    raise Exception(f"Error generating speech: {result.response}")
                
                result = result.response
                if result.action == "next":
                    break
                
                with autocast_exclude_mps(
                    device_type=decoder_model.device.type, dtype=torch.bfloat16
                ):
                    fake_audios = decode_vq_tokens(
                        decoder_model=decoder_model,
                        codes=result.codes,
                    )
                
                fake_audios = fake_audios.float().cpu().numpy()
                segments.append(fake_audios)
            
            if len(segments) == 0:
                raise Exception("No audio generated, please check the input text.")
            
            # Concatenate all segments
            audio = np.concatenate(segments, axis=0)
            
            # Save the audio to a file
            output_dir = os.path.join(os.getcwd(), "outputs")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate a unique filename
            timestamp = torch.cuda.current_device() if torch.cuda.is_available() else 0
            output_filename = f"fishspeech_srt_output_{timestamp}_{hash(text) % 10000}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save as WAV file
            with wave.open(output_path, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(decoder_model.spec_transform.sample_rate)
                wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())
            
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return (output_path, (decoder_model.spec_transform.sample_rate, audio))
        
        except Exception as e:
            # If the fish_speech module is not available, provide a helpful error message
            error_msg = str(e)
            if "No module named" in error_msg:
                error_msg = f"Error: {error_msg}. Make sure the fish_speech module is properly installed."
            elif "ServeReferenceAudio" in error_msg:
                error_msg = "Error: Required classes from fish_speech are not available. Make sure the fish_speech module is properly installed."
            
            # Create a dummy audio file with a beep sound to indicate an error
            sample_rate = 44100
            duration = 1.0  # seconds
            frequency = 440.0  # Hz (A4 note)
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            beep = 0.5 * np.sin(2 * np.pi * frequency * t)
            
            # Save the beep sound
            output_dir = os.path.join(os.getcwd(), "outputs")
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"error_beep_srt_{hash(error_msg) % 10000}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            with wave.open(output_path, "wb") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes((beep * 32767).astype(np.int16).tobytes())
            
            print(f"FishSpeech_INFER_SRT error: {error_msg}")
            return (output_path, (sample_rate, beep))
