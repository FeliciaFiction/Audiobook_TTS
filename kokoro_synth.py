"""
Kokoro Synthesis for Audiobook TTS

Handles Kokoro-82M model loading, voice management, and audio synthesis on XPU.
"""

import os
import sys
import tempfile
import importlib
import subprocess
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass


@dataclass
class VoiceInfo:
    """Information about a Kokoro voice."""
    name: str  # e.g., "af_heart"
    lang_code: str  # e.g., "a" for American English
    language: str  # e.g., "American English"
    gender: str  # "f" or "m"
    display_name: str  # e.g., "Heart (Female, American English)"


class KokoroSynthesizer:
    """
    Synthesizes audio using Kokoro-82M on Intel XPU.
    
    Handles model loading, voice enumeration, and audio generation.
    """
    
    # Kokoro language codes to full names
    LANG_CODES = {
        'a': 'American English',
        'b': 'British English',
        'e': 'Spanish',
        'f': 'French',
        'h': 'Hindi',
        'i': 'Italian',
        'j': 'Japanese',
        'p': 'Portuguese (pt-BR)',
        'z': 'Mandarin Chinese',
    }
    
    # Gender codes
    GENDER_CODES = {
        'f': 'Female',
        'm': 'Male',
    }
    
    # Default sample text for voice preview
    DEFAULT_PREVIEW_TEXT = "The quick brown fox jumps over the lazy dog."
    
    # Kokoro output sample rate
    SAMPLE_RATE = 24000
    
    DEFAULT_REPO_ID = "hexgrad/Kokoro-82M"

    def __init__(self, device: str = "xpu", lang_code: str = "a"):
        """
        Initialize synthesizer.

        Args:
            device: Device to use ("xpu", "cpu", "cuda"). Falls back to CPU
                    at load time if the requested device is unavailable.
            lang_code: Default language code
        """
        self.device = device
        self.lang_code = lang_code
        self.pipeline = None
        self._loaded_lang = None
        self._available_voices = None
    
    def load_pipeline(self, lang_code: str = None) -> None:
        """
        Load Kokoro pipeline for the specified language.
        
        Args:
            lang_code: Language code to load (defaults to self.lang_code)
        """
        if lang_code is None:
            lang_code = self.lang_code
        
        # Only reload if language changed
        if self.pipeline is not None and self._loaded_lang == lang_code:
            return
        
        try:
            # Ensure required spaCy models are installed before loading pipeline
            # Kokoro's misaki G2P uses spacy for English text processing
            self._ensure_spacy_models(lang_code)
            
            from kokoro import KPipeline
            
            # Load pipeline with specified language
            self.pipeline = KPipeline(lang_code=lang_code)
            
            # Move to device (fall back to CPU if the device is unavailable)
            if self.device == "xpu":
                if torch.xpu.is_available():
                    self.pipeline.model.to("xpu")
                else:
                    print("XPU not available, falling back to CPU")
                    self.device = "cpu"
            elif self.device == "cuda":
                if torch.cuda.is_available():
                    self.pipeline.model.to("cuda")
                else:
                    print("CUDA not available, falling back to CPU")
                    self.device = "cpu"
            
            self._loaded_lang = lang_code
            self._available_voices = None  # Reset cached voices
            
            print(f"Kokoro pipeline loaded for language '{lang_code}' on device '{self.device}'")
            
        except ImportError as e:
            print(f"Error loading Kokoro: {e}")
            print("Make sure 'kokoro' package is installed and models are cached.")
            raise
    
    def _ensure_spacy_models(self, lang_code: str) -> None:
        """
        Ensure required spaCy models are installed and loadable.
        
        Kokoro's misaki G2P uses spaCy for English text processing.
        For English languages ('a' and 'b'), we need en_core_web_sm.
        
        IMPORTANT: If the model is not pre-installed, there's a known issue where
        spacy.cli.download() installs the model but the parent Python process
        can't see it immediately. To avoid this, ensure the model is installed via
        `python -m spacy download en_core_web_sm` before running the application.
        
        Args:
            lang_code: Kokoro language code
        """
        try:
            import spacy
            
            # For English (American 'a' or British 'b'), ensure en_core_web_sm is installed
            if lang_code in ['a', 'b']:
                model_name = 'en_core_web_sm'
                
                # Check if we can load the model
                try:
                    nlp = spacy.load(model_name, enable=['tok2vec', 'tagger'])
                    print(f"spaCy model '{model_name}' is installed and loadable")
                    return
                except Exception as e:
                    # Model not loadable, provide helpful error message
                    error_msg = f"""
                    spaCy model '{model_name}' is required but could not be loaded.
                    
                    This is likely because the model was installed on-the-fly but Python
                    couldn't see it immediately. To fix this:
                    
                    1. Run: python -m spacy download {model_name}
                    2. Restart your application
                    
                    Or if using the setup script, ensure setup-xpu.ps1 completed successfully.
                    """
                    print(error_msg)
                    raise RuntimeError(f"Required spaCy model '{model_name}' is not available. {error_msg.strip()}")
        except ImportError:
            # spaCy not installed, will be handled by kokoro/misaki
            pass
    
    def get_available_voices(self, force_reload: bool = False) -> List[VoiceInfo]:
        """
        Get list of available voices across all languages.

        Voices are enumerated from the locally cached Kokoro voice pack
        (<lang><gender>_<name>.pt files), so this does not require the
        model pipeline to be loaded.

        Args:
            force_reload: Force reload voice list from disk

        Returns:
            List of VoiceInfo objects
        """
        if self._available_voices is None or force_reload:
            self._available_voices = self._enumerate_voices()

        return self._available_voices

    def _get_voice_pack_dirs(self) -> List[Path]:
        """Return candidate directories containing the Kokoro voice pack."""
        repo_id = getattr(self.pipeline, "repo_id", self.DEFAULT_REPO_ID)
        repo_dir_name = "models--" + repo_id.replace("/", "--")

        cache_roots = []
        try:
            from huggingface_hub import constants
            cache_roots.append(Path(constants.HF_HUB_CACHE))
        except ImportError:
            pass
        cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")
        if os.environ.get("HF_HOME"):
            cache_roots.append(Path(os.environ["HF_HOME"]) / "hub")

        dirs = []
        for cache_root in cache_roots:
            repo_dir = cache_root / repo_dir_name
            dirs.append(repo_dir / "voices")
            dirs.extend(sorted(repo_dir.glob("snapshots/*/voices")))

        return dirs

    def _enumerate_voices(self) -> List[VoiceInfo]:
        """Enumerate voices from the locally cached voice pack."""
        voices: Dict[str, VoiceInfo] = {}

        for voices_dir in self._get_voice_pack_dirs():
            if not voices_dir.is_dir():
                continue
            for voice_file in sorted(voices_dir.glob("*.pt")):
                voice_name = voice_file.stem

                # Parse voice info from name: <lang><gender>_<name>
                if "_" not in voice_name or len(voice_name.split("_", 1)[0]) < 2:
                    continue
                prefix, name = voice_name.split("_", 1)
                lang_code = prefix[0]
                gender_code = prefix[1]

                lang_name = self.LANG_CODES.get(lang_code, f"Unknown ({lang_code})")
                gender_name = self.GENDER_CODES.get(gender_code, f"Unknown ({gender_code})")

                voices[voice_name] = VoiceInfo(
                    name=voice_name,
                    lang_code=lang_code,
                    language=lang_name,
                    gender=gender_name,
                    display_name=f"{name.capitalize()} ({gender_name}, {lang_name})",
                )

        if voices:
            return sorted(voices.values(), key=lambda v: (v.lang_code, v.name.lower()))

        print("Warning: voice pack not found in Hugging Face cache. Using known voice list.")
        all_known = []
        for lang in self.LANG_CODES:
            all_known.extend(self._get_hardcoded_voices(lang))
        return all_known
    
    def _get_hardcoded_voices(self, lang_code: str) -> List[VoiceInfo]:
        """
        Return hardcoded voice list for known languages.
        This is a fallback when voice auto-detection fails.
        """
        # Known Kokoro v1.0 voices by language
        voice_map = {
            'a': [  # American English
                ('af_heart', 'Female'),
                ('af_hope', 'Female'),
                ('af_lilly', 'Female'),
                ('af_lily', 'Female'),
                ('af_rose', 'Female'),
                ('am_ado', 'Male'),
                ('am_brad', 'Male'),
                ('am_don', 'Male'),
                ('am_joe', 'Male'),
                ('am_kevin', 'Male'),
                ('am_lee', 'Male'),
                ('am_parker', 'Male'),
                ('am_ravi', 'Male'),
                ('am_ray', 'Male'),
            ],
            'b': [  # British English
                ('bf_ada', 'Female'),
                ('bf_claire', 'Female'),
                ('bf_ella', 'Female'),
                ('bf_emily', 'Female'),
                ('bf_lucy', 'Female'),
                ('bf_maja', 'Female'),
                ('bm_alex', 'Male'),
                ('bm_arthur', 'Male'),
                ('bm_david', 'Male'),
                ('bm_george', 'Male'),
                ('bm_james', 'Male'),
                ('bm_liam', 'Male'),
                ('bm_oliver', 'Male'),
            ],
            'e': [('ef_sofia', 'Female'), ('em_carlos', 'Male')],  # Spanish
            'f': [('ff_amie', 'Female'), ('fm_pierre', 'Male')],  # French
            'h': [('hf_rashi', 'Female'), ('hm_arav', 'Male')],  # Hindi
            'i': [('if_rosa', 'Female'), ('im_luca', 'Male')],  # Italian
            'p': [('pf_maria', 'Female'), ('pm_joao', 'Male')],  # Portuguese
            'j': [('jf_hanako', 'Female'), ('jm_taro', 'Male')],  # Japanese
            'z': [('zf_li', 'Female'), ('zm_wei', 'Male')],  # Chinese
        }
        
        voices = []
        voice_list = voice_map.get(lang_code, [])
        
        lang_name = self.LANG_CODES.get(lang_code, f"Unknown ({lang_code})")
        
        for voice_name, gender_name in voice_list:
            gender_code = 'f' if gender_name.lower() == 'female' else 'm'
            voices.append(VoiceInfo(
                name=voice_name,
                lang_code=lang_code,
                language=lang_name,
                gender=gender_name,
                display_name=f"{voice_name.split('_')[1].capitalize()} ({gender_name}, {lang_name})"
            ))
        
        return voices
    
    def filter_voices_by_language(self, lang_code: str) -> List[VoiceInfo]:
        """
        Get voices filtered by language code.
        
        Args:
            lang_code: Kokoro language code to filter by
            
        Returns:
            List of VoiceInfo objects for the specified language
        """
        all_voices = self.get_available_voices()
        return [v for v in all_voices if v.lang_code == lang_code]
    
    def filter_voices_by_iso_language(self, iso_lang: str) -> List[VoiceInfo]:
        """
        Filter voices by ISO 639-1 language code.
        
        Args:
            iso_lang: ISO 639-1 language code (e.g., "en", "es", "fr")
            
        Returns:
            List of VoiceInfo objects for languages matching the ISO code
        """
        # Map ISO 639-1 to Kokoro language codes
        iso_to_kokoro = {
            'en': ['a', 'b'],  # English: American or British
            'es': ['e'],       # Spanish
            'fr': ['f'],       # French
            'hi': ['h'],       # Hindi
            'it': ['i'],       # Italian
            'pt': ['p'],       # Portuguese
            'ja': ['j'],       # Japanese
            'zh': ['z'],       # Chinese
        }
        
        kokoro_codes = iso_to_kokoro.get(iso_lang.lower(), [])
        
        all_voices = []
        for code in kokoro_codes:
            try:
                voices = self.filter_voices_by_language(code)
                all_voices.extend(voices)
            except:
                pass
        
        return all_voices
    
    def synthesize_text(self, text: str, voice: str = None, speed: float = 1.0, 
                        lang_code: str = None) -> np.ndarray:
        """
        Synthesize text to audio using Kokoro.
        
        Args:
            text: Text to synthesize
            voice: Voice name (e.g., "af_heart")
            speed: Speed multiplier (default 1.0)
            lang_code: Language code (defaults to self.lang_code)
            
        Returns:
            Numpy array of float32 audio at 24 kHz
        """
        if self.pipeline is None:
            self.load_pipeline(lang_code or self.lang_code)
        
        if lang_code and lang_code != self._loaded_lang:
            self.load_pipeline(lang_code)
        
        # Extract voice name from dict if needed (Gradio dropdown returns dict)
        if isinstance(voice, dict):
            voice_name = voice.get("value", voice.get("name", "af_heart"))
        else:
            voice_name = voice
        
        if not voice_name:
            voices = self.get_available_voices()
            if voices:
                voice_name = voices[0].name
            else:
                voice_name = "af_heart"  # Default
        
        # Generate audio using pipeline
        audio_chunks = []
        
        with torch.inference_mode():
            for graphemes, phonemes, audio in self.pipeline(
                text, 
                voice=voice_name, 
                speed=speed
            ):
                # audio is a tensor on the device, possibly shape (1, N); flatten to mono
                if isinstance(audio, torch.Tensor):
                    audio_cpu = audio.detach().cpu().numpy().reshape(-1)
                else:
                    audio_cpu = np.asarray(audio, dtype=np.float32).reshape(-1)
                
                audio_chunks.append(audio_cpu)
        
        # Concatenate all audio chunks
        if audio_chunks:
            result = np.concatenate(audio_chunks).astype(np.float32)
        else:
            result = np.array([], dtype=np.float32)
        
        return result
    
    def synthesize_to_wav(self, text: str, output_path: str, voice: str = None, 
                         speed: float = 1.0, lang_code: str = None) -> None:
        """
        Synthesize text to WAV file.
        
        Args:
            text: Text to synthesize
            output_path: Output WAV file path
            voice: Voice name
            speed: Speed multiplier
            lang_code: Language code
        """
        audio = self.synthesize_text(text, voice, speed, lang_code)
        
        if len(audio) > 0:
            # Write atomically (tmp + rename)
            tmp_path = output_path + ".tmp"
            sf.write(tmp_path, audio, self.SAMPLE_RATE, format='WAV')
            Path(tmp_path).rename(output_path)
    
    def synthesize_chunks(self, text: str, voice: str = None, speed: float = 1.0, 
                          lang_code: str = None) -> Generator[np.ndarray, None, None]:
        """
        Synthesize text chunk by chunk, yielding audio as it's generated.
        
        Args:
            text: Text to synthesize
            voice: Voice name
            speed: Speed multiplier
            lang_code: Language code
            
        Yields:
            Numpy arrays of audio chunks
        """
        if self.pipeline is None:
            self.load_pipeline(lang_code or self.lang_code)
        
        if lang_code and lang_code != self._loaded_lang:
            self.load_pipeline(lang_code)
        
        if not voice:
            voices = self.get_available_voices()
            voice = voices[0].name if voices else "af_heart"
        
        with torch.inference_mode():
            for graphemes, phonemes, audio in self.pipeline(
                text, 
                voice=voice, 
                speed=speed
            ):
                if isinstance(audio, torch.Tensor):
                    audio_cpu = audio.detach().cpu().numpy().reshape(-1)
                else:
                    audio_cpu = np.asarray(audio, dtype=np.float32).reshape(-1)
                
                yield audio_cpu
    
    def synthesize_chapter(self, text: str, voice: str = None, speed: float = 1.0,
                           lang_code: str = None) -> Generator[np.ndarray, None, None]:
        """
        Synthesize a whole chapter with a single pipeline call.

        KPipeline performs its own sentence-level splitting and yields one
        audio chunk per sentence group. Feeding the whole chapter at once
        avoids re-running tokenization/G2P and per-call setup for every small
        chunk, roughly halving wall time versus per-chunk pipeline calls.

        Yields:
            Numpy arrays (float32, mono) of audio chunks in order
        """
        if self.pipeline is None:
            self.load_pipeline(lang_code or self.lang_code)

        if lang_code and lang_code != self._loaded_lang:
            self.load_pipeline(lang_code)

        if isinstance(voice, dict):
            voice = voice.get("value") or voice.get("name")

        if not voice:
            voices = self.get_available_voices()
            voice = voices[0].name if voices else "af_heart"

        with torch.inference_mode():
            for graphemes, phonemes, audio in self.pipeline(
                text, voice=voice, speed=speed
            ):
                if audio is None:
                    continue
                if isinstance(audio, torch.Tensor):
                    yield audio.detach().cpu().numpy().reshape(-1)
                else:
                    yield np.asarray(audio, dtype=np.float32).reshape(-1)

    def warm_up(self, lang_code: str = None, voice: str = None) -> None:
        """
        Run a few short dummy syntheses to trigger model load and SYCL/oneDNN
        JIT kernel compilation. With SYCL_CACHE_PERSISTENT=1 the compiled
        kernels persist on disk, so this cost is paid once per shape family,
        and later real generations start fast.
        """
        self.load_pipeline(lang_code or self.lang_code)
        voice = voice or "af_heart"
        try:
            with torch.inference_mode():
                for text in ("Warm up.", "This second sentence is a little "
                              "longer, to compile a few more kernels."):
                    for _ in self.pipeline(text, voice=voice, speed=1.0):
                        pass
            print("Kokoro warm-up complete")
        except Exception as e:
            print(f"Warm-up skipped: {e}")

    def preview_voice(self, voice: str, text: str = None, speed: float = 1.0) -> np.ndarray:
        """
        Generate a short preview audio for a voice.
        
        Args:
            voice: Voice name or dict with 'value' key (from Gradio dropdown)
            text: Text to use for preview (defaults to DEFAULT_PREVIEW_TEXT)
            speed: Speed multiplier
            
        Returns:
            Numpy array of preview audio
        """
        if text is None:
            text = self.DEFAULT_PREVIEW_TEXT
        
        # Extract voice name from dict if needed (Gradio dropdown returns dict)
        if isinstance(voice, dict):
            voice_name = voice.get("value", voice.get("name", "af_heart"))
        else:
            voice_name = voice
        
        # Extract language code from voice name
        if '_' in voice_name:
            lang_code = voice_name.split('_')[0][0]
        else:
            lang_code = self.lang_code
        
        return self.synthesize_text(text, voice_name, speed, lang_code)
    
    def add_silence(self, duration_seconds: float = 0.75) -> np.ndarray:
        """
        Generate silence audio for inter-chapter gaps.
        
        Args:
            duration_seconds: Duration of silence
            
        Returns:
            Numpy array of silence
        """
        samples = int(duration_seconds * self.SAMPLE_RATE)
        return np.zeros(samples, dtype=np.float32)


# Global synthesizer instance
synthesizer = KokoroSynthesizer()


# Convenience functions
def get_voices(lang_code: str = None) -> List[VoiceInfo]:
    """Get available voices."""
    if lang_code:
        return synthesizer.filter_voices_by_language(lang_code)
    return synthesizer.get_available_voices()


def synthesize_text(text: str, voice: str = None, speed: float = 1.0) -> np.ndarray:
    """Synthesize text to audio."""
    return synthesizer.synthesize_text(text, voice, speed)


def preview_voice(voice: str, text: str = None) -> np.ndarray:
    """Generate voice preview."""
    return synthesizer.preview_voice(voice, text)
