import os
import sys
import time
import io
import wave
import numpy as np
import gc
import torch
from subprocess import Popen, PIPE
import folder_paths
import cuda_malloc
import audiotsm
import audiotsm.io.wav
from pydub import AudioSegment
from srt import parse as SrtPare
from huggingface_hub import hf_hub_download, snapshot_download

input_path = folder_paths.get_input_directory()
output_path = folder_paths.get_output_directory()
device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"
fish_tmp_out = os.path.join(output_path, "fish_speech")
os.makedirs(fish_tmp_out, exist_ok=True)
parent_directory = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(parent_directory, "checkpoints")
os.makedirs(checkpoint_path, exist_ok=True)

# Create examples directory if it doesn't exist
examples_dir = os.path.join(parent_directory, "examples")
os.makedirs(examples_dir, exist_ok=True)

def download_model_files(hf_token=None):
    """Download model files if they don't exist"""
    # Download all model files using snapshot_download
    try:
        snapshot_download(repo_id="fishaudio/fish-speech-1.5", local_dir=checkpoint_path, token=hf_token)
        print("All checkpoints downloaded successfully")
    except Exception as e:
        print(f"Error downloading checkpoints: {e}")
        # Fallback to individual file downloads
        files_to_download = [
            "model.pth",
            "firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
            "config.json",
            "special_tokens.json",
            "tokenizer.json"
        ]
        
        for filename in files_to_download:
            file_path = os.path.join(checkpoint_path, filename)
            if not os.path.isfile(file_path):
                try:
                    hf_hub_download(
                        repo_id="fishaudio/fish-speech-1.5",
                        filename=filename,
                        local_dir=checkpoint_path,
                        token=hf_token
                    )
                    print(f"Downloaded {filename}")
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")

class FishSpeech_INFER_SRT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "text":("SRT",),
            "prompt_audio": ("STRING",),
            "prompt_text":("SRT",),
            "if_mutiple_speaker":("BOOLEAN",{
                "default": False
            }),
            "text2semantic_type":(["medium","large"],{
                "default": "medium"
            }),
            "hf_token":("STRING",{
                "default": "your token to download weights"
            }),
            "normalize_text":("BOOLEAN",{
                "default": False
            }),
            "num_samples":("INT", {
                "default":1
            }),
            "max_new_tokens": ("INT", {
                "default":1024
            }),
            "top_p":("FLOAT",{
                "default": 0.7
            }),
            "repetition_penalty":("FLOAT",{
                "default": 1.2
            }),
            "temperature":("FLOAT",{
                "default": 0.7
            }),
            "compile":("BOOLEAN",{
                "default": True
            }),
            "seed":("INT",{
                "default": 0
            }),
            "half":("BOOLEAN",{
                "default": False
            }),
            "chunk_length":("INT",{
                "default": 200
            }),
        }}
    
    CATEGORY = "AIFSH_FishSpeech"
    RETURN_TYPES = ('STRING',)
    OUTPUT_NODE = False

    FUNCTION = "get_tts_wav"
    
    def get_tts_wav(self, text, prompt_audio, prompt_text, if_mutiple_speaker,
                    text2semantic_type, hf_token, normalize_text,
                    num_samples, max_new_tokens, top_p, repetition_penalty,
                    temperature, compile, seed, half, chunk_length):
        
        # Download model files if needed
        download_model_files(hf_token)
        
        t2s_model_path = os.path.join(checkpoint_path, "model.pth")
        vq_model_path = os.path.join(checkpoint_path, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
        
        with open(text, 'r', encoding="utf-8") as file:
            text_file_content = file.read()
        with open(prompt_text, 'r', encoding="utf-8") as file:
            prompt_text_file_content = file.read()
            
        audio_seg = AudioSegment.from_file(prompt_audio)
        prompt_subtitles = list(SrtPare(prompt_text_file_content))
        python_exec = sys.executable or "python"
        spk_audio_dict = {}
        
        if if_mutiple_speaker:
            for i, prompt_sub in enumerate(prompt_subtitles):
                start_time = prompt_sub.start.total_seconds() * 1000
                end_time = prompt_sub.end.total_seconds() * 1000
                speaker = 'SPK'+prompt_sub.content[0]
                if speaker in spk_audio_dict:
                    spk_audio_dict[speaker] += audio_seg[start_time:end_time]
                else:
                    spk_audio_dict[speaker] = audio_seg[start_time:end_time]
                    
            for speaker in spk_audio_dict.keys():
                speaker_audio_seg = spk_audio_dict[speaker]
                speaker_audio = os.path.join(input_path, f"{speaker}.wav")
                speaker_audio_seg.export(speaker_audio, format='wav')
                npy_path = os.path.join(fish_tmp_out, os.path.basename(speaker_audio))
                step_1 = f"{python_exec} {parent_directory}/tools/vqgan/inference.py -i {speaker_audio} -o {npy_path} -ckpt {vq_model_path} -d {device}"
                print("step 1 ", step_1)
                p = Popen(step_1, shell=True)
                p.wait()
        else:
            npy_path = os.path.join(fish_tmp_out, "SPK0.wav")
            step_1 = f"{python_exec} {parent_directory}/tools/vqgan/inference.py -i {prompt_audio} -o {npy_path} -ckpt {vq_model_path} -d {device}"
            print("step 1 ", step_1)
            p = Popen(step_1, shell=True)
            p.wait()
            
        config_name = f"dual_ar_2_codebook_{text2semantic_type}"
        new_audio_seg = AudioSegment.silent(0)
        for i, (prompt_sub, text_sub) in enumerate(zip(prompt_subtitles, list(SrtPare(text_file_content)))):
            start_time = prompt_sub.start.total_seconds() * 1000
            end_time = prompt_sub.end.total_seconds() * 1000
            if i == 0:
                new_audio_seg += audio_seg[:start_time]
            
            refer_wav_seg = audio_seg[start_time:end_time]
            refer_wav = os.path.join(fish_tmp_out, f"{i}_fishspeech_refer.wav")
            refer_wav_seg.export(refer_wav, format='wav')
            
            new_text = text_sub.content
            new_prompt_text = prompt_sub.content
            if if_mutiple_speaker:
                new_text = new_text[1:]
                new_prompt_text = new_prompt_text[1:]
                npy_path = os.path.join(fish_tmp_out, f"SPK{new_text[0]}.npy")
                out_put_path = os.path.join(fish_tmp_out, f"SPK{new_text[0]}")
            else:
                npy_path = os.path.join(fish_tmp_out, "SPK0.npy")
                out_put_path = os.path.join(fish_tmp_out, "SPK0")
            os.makedirs(out_put_path, exist_ok=True)
            
            # Handle normalize flag correctly
            normalize_flag = ""
            if normalize_text:
                normalize_flag = "--normalize"
            
            step_2 = f'{python_exec} {parent_directory}/tools/llama/generate.py --text "{new_text}" --prompt-text "{new_prompt_text}" \
            --prompt-tokens {npy_path} --config-name {config_name} --num-samples {num_samples} --max-new-tokens {max_new_tokens} \
                --top-p {top_p} --repetition-penalty {repetition_penalty} --temperature {temperature} \
                    --checkpoint-path {t2s_model_path} --tokenizer {checkpoint_path} {"--compile" if compile else "--no-compile"} \
                        --seed {seed} {"--half" if half else "--no-half"} {normalize_flag} \
                            --chunk-length {chunk_length} --output-path {out_put_path}'
            print("step 2 ", step_2)
            p2 = Popen(step_2, shell=True, stdout=PIPE, stderr=PIPE)
            stdout, stderr = p2.communicate()
            if p2.returncode != 0:
                print(f"Error in step 2: {stderr.decode()}")
                continue
            
            step_2_npy = os.path.join(out_put_path, "codes_0.npy")
            if not os.path.exists(step_2_npy):
                print(f"Error: Output file {step_2_npy} not found")
                continue
                
            out_wav_path = os.path.join(out_put_path, f"{i}_fish_speech.wav")
            step_3 = f"{python_exec} {parent_directory}/tools/vqgan/inference.py -i {step_2_npy} -o {out_wav_path} -ckpt {vq_model_path} -d {device}"
            print("step 3 ", step_3)
            p3 = Popen(step_3, shell=True, stdout=PIPE, stderr=PIPE)
            stdout, stderr = p3.communicate()
            if p3.returncode != 0:
                print(f"Error in step 3: {stderr.decode()}")
                continue
                
            if not os.path.exists(out_wav_path):
                print(f"Error: Output file {out_wav_path} not found")
                continue
            
            text_audio = AudioSegment.from_file(out_wav_path)
            text_audio_dur_time = text_audio.duration_seconds * 1000
            
            if i < len(prompt_subtitles) - 1:
                nxt_start = prompt_subtitles[i+1].start.total_seconds() * 1000
                dur_time = nxt_start - start_time
            else:
                org_dur_time = audio_seg.duration_seconds * 1000
                dur_time = org_dur_time - start_time
            
            ratio = text_audio_dur_time / dur_time

            if text_audio_dur_time > dur_time:
                tmp_audio = self.map_vocal(audio=text_audio, ratio=ratio, dur_time=dur_time,
                                          wav_name=f"map_{i}_refer.wav", temp_folder=out_put_path)
                tmp_audio += AudioSegment.silent(dur_time - tmp_audio.duration_seconds*1000)
            else:
                tmp_audio = text_audio + AudioSegment.silent(dur_time - text_audio_dur_time)
          
            new_audio_seg += tmp_audio

        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        infer_audio = os.path.join(output_path, f"{int(time.time())}_fishspeech_refer.wav")
        new_audio_seg.export(infer_audio, format="wav")
        return (infer_audio, )

    def map_vocal(self, audio:AudioSegment, ratio:float, dur_time:float, wav_name:str, temp_folder:str):
        tmp_path = f"{temp_folder}/map_{wav_name}"
        audio.export(tmp_path, format="wav")
        
        clone_path = f"{temp_folder}/cloned_{wav_name}"
        reader = audiotsm.io.wav.WavReader(tmp_path)
        
        writer = audiotsm.io.wav.WavWriter(clone_path, channels=reader.channels,
                                          samplerate=reader.samplerate)
        wsloa = audiotsm.wsola(channels=reader.channels, speed=ratio)
        wsloa.run(reader=reader, writer=writer)
        audio_extended = AudioSegment.from_file(clone_path)
        return audio_extended[:dur_time]


class FishSpeech_INFER:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
            "prompt_audio": ("STRING",),
            "text":("STRING",{
                "multiline": True,
                "default": "你好啊，世界！"
            }),
            "prompt_text_by_srt":("SRT",{
                "multiline": True,
                "default": "a man voice"
            }),
            "text2semantic_type":(["medium","large"],{
                "default": "medium"
            }),
            "hf_token":("STRING",{
                "default": "your token"
            }),
            "normalize_text":("BOOLEAN",{
                "default": False
            }),
            "num_samples":("INT", {
                "default":1
            }),
            "max_new_tokens": ("INT", {
                "default":1024
            }),
            "top_p":("FLOAT",{
                "default": 0.7
            }),
            "repetition_penalty":("FLOAT",{
                "default": 1.2
            }),
            "temperature":("FLOAT",{
                "default": 0.7
            }),
            "compile":("BOOLEAN",{
                "default": True
            }),
            "seed":("INT",{
                "default": 0
            }),
            "half":("BOOLEAN",{
                "default": False
            }),
            "chunk_length":("INT",{
                "default": 200
            }),
        }}
    
    CATEGORY = "AIFSH_FishSpeech"
    RETURN_TYPES = ('STRING',)
    OUTPUT_NODE = False

    FUNCTION = "get_tts_wav"
    
    def get_tts_wav(self, prompt_audio, text, prompt_text_by_srt, text2semantic_type, hf_token,
                    normalize_text, num_samples, max_new_tokens, top_p, repetition_penalty,
                    temperature, compile, seed, half, chunk_length):
        
        with open(prompt_text_by_srt, 'r', encoding="utf-8") as file:
            file_content = file.read()
        prompt_text = ' '.join([sub.content for sub in list(SrtPare(file_content))])
        
        # Download model files if needed
        download_model_files(hf_token)
        
        t2s_model_path = os.path.join(checkpoint_path, "model.pth")
        vq_model_path = os.path.join(checkpoint_path, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
        
        python_exec = sys.executable or "python"
        
        npy_path = os.path.join(fish_tmp_out, os.path.basename(prompt_audio))
        step_1 = f"{python_exec} {parent_directory}/tools/vqgan/inference.py -i {prompt_audio} -o {npy_path} -ckpt {vq_model_path} -d {device}"
        print("step 1 ", step_1)
        p = Popen(step_1, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            print(f"Error in step 1: {stderr.decode()}")
            return ("Error processing audio", )
        
        config_name = f"dual_ar_2_codebook_{text2semantic_type}"
        npy_path = os.path.join(fish_tmp_out, os.path.basename(prompt_audio)[:-4]+".npy")
        
        if not os.path.exists(npy_path):
            print(f"Error: NPY file {npy_path} not found")
            return ("Error processing audio", )
        
        # Handle normalize flag correctly
        normalize_flag = ""
        if normalize_text:
            normalize_flag = "--normalize"
        
        step_2 = f'{python_exec} {parent_directory}/tools/llama/generate.py --text "{text}" --prompt-text "{prompt_text}" \
            --prompt-tokens {npy_path} --config-name {config_name} --num-samples {num_samples} --max-new-tokens {max_new_tokens} \
                --top-p {top_p} --repetition-penalty {repetition_penalty} --temperature {temperature} \
                    --checkpoint-path {t2s_model_path} --tokenizer {checkpoint_path} {"--compile" if compile else "--no-compile"} \
                        --seed {seed} {"--half" if half else "--no-half"} {normalize_flag} \
                            --chunk-length {chunk_length} --output-path {fish_tmp_out}'
        print("step 2 ", step_2)
        p2 = Popen(step_2, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p2.communicate()
        if p2.returncode != 0:
            print(f"Error in step 2: {stderr.decode()}")
            return ("Error generating speech", )
        
        step_2_npy = os.path.join(fish_tmp_out, "codes_0.npy")
        if not os.path.exists(step_2_npy):
            print(f"Error: Output file {step_2_npy} not found")
            return ("Error generating speech", )
            
        out_wav_path = os.path.join(output_path, f"{int(time.time())}_fish_speech.wav")
        step_3 = f"{python_exec} {parent_directory}/tools/vqgan/inference.py -i {step_2_npy} -o {out_wav_path} -ckpt {vq_model_path} -d {device}"
        print("step 3 ", step_3)
        p3 = Popen(step_3, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p3.communicate()
        if p3.returncode != 0:
            print(f"Error in step 3: {stderr.decode()}")
            return ("Error generating audio", )
        
        if not os.path.exists(out_wav_path):
            print(f"Error: Output file {out_wav_path} not found")
            return ("Error generating audio", )
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        return (out_wav_path, )


class FishSpeech_INFER_V15:
    @classmethod
    def INPUT_TYPES(s):
        # Get example audio files if they exist
        example_files = []
        if os.path.exists(examples_dir):
            example_files = [f for f in os.listdir(examples_dir) if f.endswith(".wav")]
        
        return {"required":{
            "text":("STRING",{
                "multiline": True,
                "default": "你好啊，世界！"
            }),
            "text2semantic_type":(["medium","large"],{
                "default": "medium"
            }),
            "hf_token":("STRING",{
                "default": "your token"
            }),
            "normalize_text":("BOOLEAN",{
                "default": False
            }),
            "reference_audio": ("STRING", {
                "default": ""
            }),
            "reference_text":("STRING",{
                "multiline": True,
                "default": ""
            }),
            "example_audio": ([""] + example_files, {
                "default": ""
            }),
            "num_samples":("INT", {
                "default":1
            }),
            "max_new_tokens": ("INT", {
                "default":1024
            }),
            "top_p":("FLOAT",{
                "default": 0.7
            }),
            "repetition_penalty":("FLOAT",{
                "default": 1.2
            }),
            "temperature":("FLOAT",{
                "default": 0.7
            }),
            "compile":("BOOLEAN",{
                "default": True
            }),
            "seed":("INT",{
                "default": 0
            }),
            "half":("BOOLEAN",{
                "default": False
            }),
            "chunk_length":("INT",{
                "default": 200
            }),
        }}
    
    CATEGORY = "AIFSH_FishSpeech"
    RETURN_TYPES = ('STRING',)
    OUTPUT_NODE = False

    FUNCTION = "get_tts_wav"
    
    def get_tts_wav(self, text, text2semantic_type, hf_token, normalize_text,
                    reference_audio, reference_text, example_audio,
                    num_samples, max_new_tokens, top_p, repetition_penalty,
                    temperature, compile, seed, half, chunk_length):
        
        # Download model files if needed
        download_model_files(hf_token)
        
        t2s_model_path = os.path.join(checkpoint_path, "model.pth")
        vq_model_path = os.path.join(checkpoint_path, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
        
        python_exec = sys.executable or "python"
        
        # Handle reference audio
        if example_audio and not reference_audio:
            reference_audio = os.path.join(examples_dir, example_audio)
            # Try to load reference text from .lab file if it exists
            lab_file = os.path.splitext(example_audio)[0] + ".lab"
            lab_path = os.path.join(examples_dir, lab_file)
            if os.path.exists(lab_path) and not reference_text:
                with open(lab_path, "r", encoding="utf-8") as f:
                    reference_text = f.read().strip()
        
        # Process reference audio if provided
        references_arg = ""
        if reference_audio:
            # Convert reference audio to bytes and encode for command line
            npy_path = os.path.join(fish_tmp_out, "reference.npy")
            step_1 = f"{python_exec} {parent_directory}/tools/vqgan/inference.py -i {reference_audio} -o {npy_path} -ckpt {vq_model_path} -d {device}"
            print("step 1 ", step_1)
            p = Popen(step_1, shell=True, stdout=PIPE, stderr=PIPE)
            stdout, stderr = p.communicate()
            if p.returncode != 0:
                print(f"Error in step 1: {stderr.decode()}")
                return ("Error processing reference audio", )
                
            if not os.path.exists(npy_path):
                print(f"Error: NPY file {npy_path} not found")
                return ("Error processing reference audio", )
                
            references_arg = f"--prompt-tokens {npy_path} --prompt-text \"{reference_text}\""
        
        # Handle normalize flag correctly
        normalize_flag = ""
        if normalize_text:
            normalize_flag = "--normalize"
        
        # Create output directory
        out_put_path = os.path.join(fish_tmp_out, "v15_output")
        os.makedirs(out_put_path, exist_ok=True)
        
        step_2 = f'{python_exec} {parent_directory}/tools/llama/generate.py --text "{text}" {references_arg} \
            --config-name dual_ar_2_codebook_{text2semantic_type} --num-samples {num_samples} --max-new-tokens {max_new_tokens} \
                --top-p {top_p} --repetition-penalty {repetition_penalty} --temperature {temperature} \
                    --checkpoint-path {t2s_model_path} --tokenizer {checkpoint_path} {"--compile" if compile else "--no-compile"} \
                        --seed {seed} {"--half" if half else "--no-half"} {normalize_flag} \
                            --chunk-length {chunk_length} --output-path {out_put_path}'
        print("step 2 ", step_2)
        p2 = Popen(step_2, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p2.communicate()
        if p2.returncode != 0:
            print(f"Error in step 2: {stderr.decode()}")
            return ("Error generating speech", )
        
        step_2_npy = os.path.join(out_put_path, "codes_0.npy")
        if not os.path.exists(step_2_npy):
            print(f"Error: Output file {step_2_npy} not found")
            return ("Error generating speech", )
            
        out_wav_path = os.path.join(output_path, f"{int(time.time())}_fish_speech_v15.wav")
        step_3 = f"{python_exec} {parent_directory}/tools/vqgan/inference.py -i {step_2_npy} -o {out_wav_path} -ckpt {vq_model_path} -d {device}"
        print("step 3 ", step_3)
        p3 = Popen(step_3, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p3.communicate()
        if p3.returncode != 0:
            print(f"Error in step 3: {stderr.decode()}")
            return ("Error generating audio", )
            
        if not os.path.exists(out_wav_path):
            print(f"Error: Output file {out_wav_path} not found")
            return ("Error generating audio", )
        
        # Clean up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        return (out_wav_path, )


class PreViewAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"audio": ("STRING",),}
                }

    CATEGORY = "AIFSH_FishSpeech"
    DESCRIPTION = "hello world!"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_name = os.path.basename(audio)
        tmp_path = os.path.dirname(audio)
        audio_root = os.path.basename(tmp_path)
        return {"ui": {"audio":[audio_name,audio_root]}}


class LoadFishAudio:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["wav", "mp3","WAV","flac","m4a"]]
        return {"required":
                    {"audio": (sorted(files),)},
                }

    CATEGORY = "AIFSH_FishSpeech"

    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        return (audio_path,)


class LoadSRT:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["srt", "txt"]]
        return {"required":
                    {"srt": (sorted(files),)},
                }

    CATEGORY = "AIFSH_FishSpeech"

    RETURN_TYPES = ("SRT",)
    FUNCTION = "load_srt"

    def load_srt(self, srt):
        srt_path = folder_paths.get_annotated_filepath(srt)
        return (srt_path,)


class LoadExampleAudio:
    @classmethod
    def INPUT_TYPES(s):
        files = []
        if os.path.exists(examples_dir):
            files = [f for f in os.listdir(examples_dir) if os.path.isfile(os.path.join(examples_dir, f)) and f.split('.')[-1] in ["wav", "mp3","WAV","flac","m4a"]]
        return {"required":
                    {"audio": (sorted(files),)},
                }

    CATEGORY = "AIFSH_FishSpeech"

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("audio_path", "reference_text")
    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_path = os.path.join(examples_dir, audio)
        reference_text = ""
        
        # Try to load reference text from .lab file if it exists
        lab_file = os.path.splitext(audio)[0] + ".lab"
        lab_path = os.path.join(examples_dir, lab_file)
        if os.path.exists(lab_path):
            with open(lab_path, "r", encoding="utf-8") as f:
                reference_text = f.read().strip()
                
        return (audio_path, reference_text)
