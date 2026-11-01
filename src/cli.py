"""CLI Interface for PERSON 6 Modular Lexical RAG Support Agent."""
import os
import sys
from typing import Optional
from .conversation_agent import ConversationAgent


def _play_audio(audio_path: str, async_mode: bool = False):
    """Play audio file across macOS (afplay), Windows (winsound), and Linux (aplay)."""
    import platform
    plat = platform.system()
    if plat == 'Darwin':
        cmd = f"afplay '{audio_path}' 2>/dev/null"
        if async_mode:
            cmd += ' &'
        os.system(cmd)
    elif plat == 'Windows':
        try:
            import winsound
            flags = winsound.SND_FILENAME
            if async_mode:
                flags |= winsound.SND_ASYNC
            winsound.PlaySound(audio_path, flags)
        except Exception:
            ps_cmd = 'powershell -c "(New-Object Media.SoundPlayer \'%s\').PlaySync()"' % audio_path
            os.system(ps_cmd)
    else:
        cmd = f"aplay '{audio_path}' 2>/dev/null"
        if async_mode:
            cmd += ' &'
        os.system(cmd)


def run_interactive_cli(agent: ConversationAgent, voice_adapter=None):
    print("=" * 60)
    print("  ZentroMart Support Agent — PERSON 6 (Evaluation-Driven Adaptive RAG)")
    if voice_adapter and getattr(voice_adapter, 'is_live', False):
        print("  [Voice Output Enabled: Gnani.ai Kaveri TTS (speaking aloud via speakers)]")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 60)

    conv_id = "cli_session_1"
    # First greeting
    res = agent.process_turn(conv_id, "hello")
    print(f"\nAssistant: {res['response']}\n")
    if voice_adapter and getattr(voice_adapter, 'is_live', False):
        audio_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_outputs", "latest_chat_reply.wav")
        voice_adapter.synthesize(res['response'], audio_file)
        _play_audio(audio_file, async_mode=True)

    while True:
        try:
            user_input = input("Customer: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit'):
                print("\nAssistant: Thank you for contacting ZentroMart. Have a wonderful day!\n")
                break

            turn = agent.process_turn(conv_id, user_input)
            print(f"\nAssistant: {turn['response']}")
            if turn.get('evidence'):
                print(f"  [Evidence used: {', '.join(turn['evidence'])}]")
            if turn.get('tool_calls'):
                print(f"  [Tools executed: {', '.join(turn['tool_calls'])}]")
            print()

            if voice_adapter and getattr(voice_adapter, 'is_live', False):
                audio_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_outputs", "latest_chat_reply.wav")
                voice_adapter.synthesize(turn['response'], audio_file)
                _play_audio(audio_file, async_mode=True)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session...")
            break


def run_voice_cli(agent: ConversationAgent, voice_adapter):
    """Interactive Voice-to-Voice support assistant with microphone recording and audio playback."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "sample_outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("  ZentroMart Interactive Voice Assistant — PERSON 6 (Evaluation-Driven Adaptive RAG Speech-to-Speech)")
    print("  Powered by Gnani.ai STT (timbre-v2.5) & TTS (Kaveri voice)")
    print("  Speak via your microphone or provide audio files.")
    print("  Type 'exit' to end the session.")
    print("=" * 65)

    conv_id = "voice_cli_session"
    greeting = "Hello! I am ZentroMart's AI customer support assistant. You may speak in English or Hindi. How can I help you today?"
    print(f"\nAssistant: {greeting}\n")

    if voice_adapter and getattr(voice_adapter, 'is_live', False):
        greet_audio = os.path.join(output_dir, "voice_greeting.wav")
        voice_adapter.synthesize(greeting, greet_audio)
        _play_audio(greet_audio, async_mode=False)

    while True:
        try:
            print("\n[VOICE INPUT OPTIONS]:")
            print("  - Press [ENTER] to start speaking (press [ENTER] again to stop, or auto-stops after 25s)")
            print("  - Type 'reset' or 'new' to start a fresh scenario")
            print("  - Or type the path to a .wav audio file")
            print("  - Type 'exit' to quit")
            user_choice = input("\nCustomer Voice Input > ").strip()

            if user_choice.lower() in ('exit', 'quit'):
                farewell = "Thank you for contacting ZentroMart. Have a wonderful day!"
                print(f"\nAssistant: {farewell}\n")
                if voice_adapter and getattr(voice_adapter, 'is_live', False):
                    farewell_audio = os.path.join(output_dir, "voice_farewell.wav")
                    voice_adapter.synthesize(farewell, farewell_audio)
                    _play_audio(farewell_audio, async_mode=False)
                break

            if user_choice.lower() in ('reset', 'new', 'clear'):
                agent.reset_session(conv_id)
                print("\n" + "=" * 60)
                print("  [Session Reset]: Started a fresh new scenario.")
                print("=" * 60)
                greeting = "Hello! I am ZentroMart's AI customer support assistant. How can I help you today?"
                print(f"\nAssistant: {greeting}\n")
                if voice_adapter and getattr(voice_adapter, 'is_live', False):
                    greet_audio = os.path.join(output_dir, "voice_greeting.wav")
                    voice_adapter.synthesize(greeting, greet_audio)
                    _play_audio(greet_audio, async_mode=False)
                continue

            audio_file = ""
            if user_choice and os.path.isfile(user_choice):
                audio_file = user_choice
                print(f"Reading customer audio from: {audio_file}")
            else:
                # Record from microphone (Push-to-stop with enter, or max 25s timeout)
                mic_file = os.path.join(output_dir, "live_mic_input.wav")
                print("\n>>> RECORDING STARTED — PLEASE SPEAK NOW! <<<")
                print(">>> Press [ENTER] when done speaking (or auto-stops after 25 seconds)... <<<")
                try:
                    import sounddevice as sd
                    import wave
                    import numpy as np
                    import threading
                    import time

                    fs = 16000
                    max_duration = 25
                    audio_frames = []
                    stop_event = threading.Event()

                    def wait_for_enter():
                        try:
                            input()
                        except Exception:
                            pass
                        stop_event.set()

                    t_stop = threading.Thread(target=wait_for_enter, daemon=True)
                    t_stop.start()

                    def audio_callback(indata, frames, time_info, status):
                        audio_frames.append(indata.copy())

                    start_time = time.time()
                    with sd.InputStream(samplerate=fs, channels=1, dtype="int16", callback=audio_callback):
                        while not stop_event.is_set():
                            elapsed = time.time() - start_time
                            if elapsed >= max_duration:
                                print(f"\n>>> Auto-stopped: Maximum duration of 25 seconds reached. <<<")
                                break
                            time.sleep(0.05)

                    if audio_frames:
                        rec_data = np.concatenate(audio_frames, axis=0)
                    else:
                        rec_data = np.zeros((fs, 1), dtype="int16")

                    with wave.open(mic_file, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(fs)
                        wf.writeframes(rec_data.tobytes())

                    max_amp = int(np.max(np.abs(rec_data)))
                    if max_amp == 0:
                        print("\n[macOS Audio Warning]: The recorded audio is 100% silent (0 amplitude).")
                        print("macOS blocks microphone capture until Terminal is granted permission:")
                        print("  -> Open System Settings > Privacy & Security > Microphone")
                        print("  -> Toggle ON the switch next to 'Terminal' (or your terminal app).\n")
                        continue

                    audio_file = mic_file
                    print(">>> Recording complete. Transcribing via Gnani STT... <<<")
                except Exception as e:
                    print(f"Microphone recording error: {e}")
                    continue

            # Transcribe via Gnani STT
            transcript, stt_status = voice_adapter.transcribe(audio_file)
            if not transcript:
                print(f"\n[Gnani STT]: No speech detected ({stt_status}). Please try speaking closer to mic.")
                continue

            print(f"\nCustomer (Transcribed via Gnani STT): \"{transcript}\"")

            # Route to agent
            turn = agent.process_turn(conv_id, transcript)
            print(f"\nAssistant: {turn['response']}")
            if turn.get('evidence'):
                print(f"  [Evidence used: {', '.join(turn['evidence'])}]")
            if turn.get('tool_calls'):
                print(f"  [Tools executed: {', '.join(turn['tool_calls'])}]")

            # Synthesize reply via Gnani TTS and play
            reply_audio = os.path.join(output_dir, "voice_live_reply.wav")
            voice_adapter.synthesize(turn['response'], reply_audio)
            print("Speaking response aloud via speakers (Gnani Kaveri TTS)...")
            _play_audio(reply_audio, async_mode=False)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting voice session...")
            break
