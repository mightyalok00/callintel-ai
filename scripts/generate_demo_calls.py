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
        "call_id": "CALL_2026_APEX_001",
        "scenario": "Retention Crisis (Pricing & Cancellation Threat)",
        "script": """
Hello, thank you for calling Apex Cloud Support. My name is Alex. How can I help you today?
Hi Alex. Your product is good, but it is too expensive. I might cancel my subscription if the price increases again.
I completely understand your pricing concerns. Let me review your current usage and explain our discount plan options.
Okay, please do. If you have an annual discount, I would like to consider staying.
Wonderful. I can apply a fifteen percent loyalty discount to your account starting today.
That sounds reasonable. Please go ahead and apply it to my subscription. Thank you.
        """.strip()
    },
    {
        "filename": "sample_sales_expansion.wav",
        "call_id": "CALL_2026_SALES_002",
        "scenario": "Sales Expansion (Enterprise Tier & Buying Signals)",
        "script": """
Hello, welcome to Apex Cloud Sales. This is Sarah speaking. How can I assist your team today?
Hi Sarah, our engineering team loves the platform and we are expanding to fifty developers next month.
That is fantastic news. We have dedicated Enterprise tiers with higher throughput and custom SLAs.
We are very interested in upgrading our plan. Does the Enterprise tier include priority phone support and SOC Two compliance?
Yes, absolutely. Enterprise includes full compliance guarantees, designated technical account managers, and twenty-four seven priority support.
That fits our budget and security requirements perfectly. Please email me the contract proposal.
        """.strip()
    },
    {
        "filename": "sample_technical_escalation.wav",
        "call_id": "CALL_2026_TECH_003",
        "scenario": "Technical Outage (Critical Bug & Urgent Escalation)",
        "script": """
Apex Cloud Technical Support line, my name is David. What issue are you experiencing today?
David, our production database synchronization stopped working two hours ago and our transactions are failing.
I see the severity. Let me pull up your account telemetry and run a node health diagnosis immediately.
We are losing revenue every minute this persists. We need this escalated to tier three engineering right away.
Understood. I have escalated this as a Severity One emergency to our senior site reliability engineers.
Thank you, we will monitor the emergency Slack channel for their immediate response.
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
