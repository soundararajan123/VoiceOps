"""
Live Integration Test for Gnani.ai Speech Services (PERSON 6).

Verifies genuine live Gnani STT and TTS API connectivity:
- If GNANI_API_KEY is not configured: reports 'BLOCKED — credentials not configured'
  and does NOT report as passed.
- If GNANI_API_KEY is present: executes genuine STT and TTS API requests.
"""
import os
import sys
import unittest
import tempfile

def _load_env_file():
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(3):
        cand = os.path.join(cur, ".env")
        if os.path.isfile(cand):
            try:
                with open(cand, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k not in os.environ and v:
                                os.environ[k] = v
            except Exception:
                pass
            break
        cur = os.path.dirname(cur)

_load_env_file()


class TestGnaniLiveIntegration(unittest.TestCase):
    def test_live_gnani_stt_and_tts(self):
        """Live integration test for genuine Gnani.ai STT & TTS."""
        api_key = os.environ.get("GNANI_API_KEY")
        if not api_key:
            print("\n" + "=" * 60)
            print("  [LIVE GNANI TEST] STATUS: BLOCKED — credentials not configured")
            print("  Reason: GNANI_API_KEY environment variable is not set.")
            print("  Please configure GNANI_API_KEY in your environment or .env file.")
            print("=" * 60)
            self.skipTest("BLOCKED — credentials not configured")
            return

        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient

        stt_client = GnaniSTTClient(api_key=api_key)
        tts_client = GnaniTTSClient(api_key=api_key)

        with tempfile.TemporaryDirectory() as tmpdir:
            test_audio = os.path.join(tmpdir, "live_test_output.wav")
            test_phrase = "Hello from ZentroMart adaptive RAG support agent."

            print("\n[LIVE GNANI TEST] Testing live TTS synthesis...")
            tts_res = tts_client.synthesize(
                test_phrase,
                voice=os.environ.get("GNANI_TTS_VOICE", "Kaveri"),
                model=os.environ.get("GNANI_TTS_MODEL", "timbre-v2.5"),
                language=os.environ.get("GNANI_LANGUAGE_CODE", "en-IN"),
                output_file=test_audio
            )
            if isinstance(tts_res, bytes) and not os.path.exists(test_audio):
                with open(test_audio, "wb") as f:
                    f.write(tts_res)

            self.assertTrue(os.path.exists(test_audio), "TTS did not produce output audio file")
            self.assertGreater(os.path.getsize(test_audio), 100, "TTS audio output file is empty")
            print(f"[LIVE GNANI TEST] TTS Success: Generated {os.path.getsize(test_audio)} bytes.")

            print("[LIVE GNANI TEST] Testing live STT transcription...")
            stt_res = stt_client.transcribe(test_audio, language_code=os.environ.get("GNANI_LANGUAGE_CODE", "en-IN"))
            transcript = stt_res.get("transcript", "") if isinstance(stt_res, dict) else str(stt_res)
            print(f"[LIVE GNANI TEST] STT Success: Transcribed '{transcript}'")
            self.assertIsInstance(transcript, str)

if __name__ == "__main__":
    api_key = os.environ.get("GNANI_API_KEY")
    if not api_key:
        print("[LIVE GNANI TEST] STATUS: BLOCKED — credentials not configured")
        print("  Reason: GNANI_API_KEY environment variable is not set.")
        sys.exit(2)
    unittest.main()
