"""
Synthetic call audio generator using Windows System.Speech.Synthesis.
Generates realistic customer support and sales call WAV files for testing.
"""

import subprocess
from pathlib import Path

SAMPLES_DIR = Path("data/samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_CALLS = [
    {
        "filename": "sample_customer_call.wav",
        "script": """
Hello, thank you for calling Apex Cloud Support. My name is Alex. How can I help you today?
Hi Alex. Your product is good, but it is too expensive. I might cancel my subscription if the price increases again.
I completely understand your pricing concerns. Let me review your current usage and explain our discount plan options.
Okay, please do. If you have an annual discount, I would like to consider staying.
Wonderful. I can apply a fifteen percent loyalty discount to your account starting today.
That sounds reasonable. Please go ahead and apply it to my subscription. Thank you.
        """.strip()
    }
]


def generate_audio_samples():
    print("[*] Generating realistic speech WAV files using Windows Speech Synthesis...")
    for call in SAMPLE_CALLS:
        out_path = SAMPLES_DIR / call["filename"]
        clean_text = " ".join(call["script"].split())
        
        ps_cmd = f"""
        Add-Type -AssemblyName System.Speech
        $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
        $synth.Rate = 0
        $synth.SetOutputToWaveFile('{out_path.resolve()}')
        $synth.Speak('{clean_text}')
        $synth.Dispose()
        """
        
        res = subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, text=True)
        if res.returncode == 0 and out_path.exists():
            print(f"[OK] Generated: {out_path} ({out_path.stat().st_size:,} bytes)")
        else:
            print(f"[!] Error generating {out_path}: {res.stderr}")


if __name__ == "__main__":
    generate_audio_samples()
