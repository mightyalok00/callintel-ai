"""
Synthetic call audio generator using Windows System.Speech.Synthesis.
Generates realistic multi-speaker customer support and sales call WAV files
with distinct voices for Agent (David) and Customer (Zira).
"""

import subprocess
import tempfile
from pathlib import Path

SAMPLES_DIR = Path("data/samples")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_CALLS = [
    {
        "filename": "sample_customer_call.wav",
        "call_id": "CALL_2026_APEX_001",
        "scenario": "Retention Crisis (Pricing & Cancellation Threat)",
        "turns": [
            ("Agent", "Microsoft David Desktop", "Hello, thank you for calling Apex Cloud Support. My name is Alex. How can I help you today?"),
            ("Customer", "Microsoft Zira Desktop", "Hi Alex. Your product is good, but it is too expensive. I might cancel my subscription if the price increases again."),
            ("Agent", "Microsoft David Desktop", "I completely understand your pricing concerns. Let me review your current usage and explain our discount plan options."),
            ("Customer", "Microsoft Zira Desktop", "Okay, please do. If you have an annual discount, I would like to consider staying."),
            ("Agent", "Microsoft David Desktop", "Wonderful. I can apply a fifteen percent loyalty discount to your account starting today."),
            ("Customer", "Microsoft Zira Desktop", "That sounds reasonable. Please go ahead and apply it to my subscription. Thank you."),
        ],
    },
    {
        "filename": "sample_sales_expansion.wav",
        "call_id": "CALL_2026_SALES_002",
        "scenario": "Sales Expansion (Enterprise Tier & Buying Signals)",
        "turns": [
            ("Agent", "Microsoft Zira Desktop", "Hello, welcome to Apex Cloud Sales. This is Sarah speaking. How can I assist your team today?"),
            ("Customer", "Microsoft David Desktop", "Hi Sarah, our engineering team loves the platform and we are expanding to fifty developers next month."),
            ("Agent", "Microsoft Zira Desktop", "That is fantastic news. We have dedicated Enterprise tiers with higher throughput and custom SLAs."),
            ("Customer", "Microsoft David Desktop", "We are very interested in upgrading our plan. Does the Enterprise tier include priority phone support and SOC Two compliance?"),
            ("Agent", "Microsoft Zira Desktop", "Yes, absolutely. Enterprise includes full compliance guarantees, designated technical account managers, and twenty-four seven priority support."),
            ("Customer", "Microsoft David Desktop", "That fits our budget and security requirements perfectly. Please email me the contract proposal."),
        ],
    },
    {
        "filename": "sample_technical_escalation.wav",
        "call_id": "CALL_2026_TECH_003",
        "scenario": "Technical Outage (Critical Bug & Urgent Escalation)",
        "turns": [
            ("Agent", "Microsoft David Desktop", "Apex Cloud Technical Support line, my name is David. What issue are you experiencing today?"),
            ("Customer", "Microsoft Zira Desktop", "David, our production database synchronization stopped working two hours ago and our transactions are failing."),
            ("Agent", "Microsoft David Desktop", "I see the severity. Let me pull up your account telemetry and run a node health diagnosis immediately."),
            ("Customer", "Microsoft Zira Desktop", "We are losing revenue every minute this persists. We need this escalated to tier three engineering right away."),
            ("Agent", "Microsoft David Desktop", "Understood. I have escalated this as a Severity One emergency to our senior site reliability engineers."),
            ("Customer", "Microsoft Zira Desktop", "Thank you, we will monitor the emergency Slack channel for their immediate response."),
        ],
    },
]


def generate_audio_samples():
    print("[*] Generating realistic multi-speaker speech WAV files...")
    for call in SAMPLE_CALLS:
        filename: str = str(call["filename"])
        out_path = (SAMPLES_DIR / filename).resolve()
        
        # Build PowerShell script with alternating voices and natural pauses
        ps_lines = [
            "Add-Type -AssemblyName System.Speech",
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer",
            "$synth.Rate = 0",
            f"$synth.SetOutputToWaveFile('{out_path}')",
        ]
        
        turns_data = call.get("turns", [])
        for role, voice, text in turns_data:
            escaped_text = text.replace("'", "''")
            ps_lines.append(f"try {{ $synth.SelectVoice('{voice}') }} catch {{}}")
            ps_lines.append(f"$synth.Speak('{escaped_text}')")
            ps_lines.append("Start-Sleep -Milliseconds 400")
            
        ps_lines.append("$synth.Dispose()")
        
        ps_script = "\n".join(ps_lines)
        with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8") as tf:
            tf.write(ps_script)
            tmp_ps1 = tf.name
            
        try:
            res = subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", tmp_ps1],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0 and out_path.exists():
                print(f"[OK] Generated {call['call_id']}: {out_path} ({out_path.stat().st_size:,} bytes)")
            else:
                print(f"[!] Error generating {out_path}: {res.stderr}")
        finally:
            Path(tmp_ps1).unlink(missing_ok=True)


if __name__ == "__main__":
    generate_audio_samples()
