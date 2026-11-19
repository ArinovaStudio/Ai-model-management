"""Live Interactive CLI Chat with RealEstateAI Voice Agent (Aria).

Allows you to talk directly to Aria in real-time in your terminal:
- Natural conversation and niche discovery
- Grounded RAG search across verified properties
- Prompt injection & confidentiality deflection testing
- Live calendar slot hold and booking
- CalDAV synchronization
- (Optional) Real-time voice playback through your speakers using Piper TTS!
"""

import asyncio
import os
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.dialogue_manager import dialogue_manager
from src.audio.tts import tts_engine
from src.core.logging import logger

# Optional Windows audio playback
CAN_PLAY_AUDIO = False
if sys.platform == "win32":
    try:
        import winsound
        CAN_PLAY_AUDIO = True
    except Exception:
        pass


async def main():
    session_id = f"cli_user_{uuid.uuid4().hex[:8]}"
    print("=" * 72)
    print("       REALESTATE AI - LIVE INTERACTIVE CONVERSATION WITH ARIA       ")
    print("=" * 72)
    print(f"Session ID: {session_id}")
    print("You can chat freely about properties, site visits, or test confidentiality.")
    print("Commands:")
    print("  - Type your message and press [Enter]")
    print("  - Type 'audio on' or 'audio off' to toggle spoken voice playback")
    print("  - Type 'exit' or 'quit' to end the call")
    print("=" * 72)

    play_audio = False
    if CAN_PLAY_AUDIO:
        print("[TIP] You can hear Aria speak! Type 'audio on' to enable voice playback.\n")

    # Initial greeting from Aria
    greeting = "Hello! I am Aria, your AI Voice Representative. Are you looking for residential plots, commercial flats, or luxury villas today?"
    print(f"\nAria: {greeting}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCall ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("\nEnding session... Goodbye!")
            dialogue_manager.close_session(session_id)
            break

        if user_input.lower() == "audio on":
            play_audio = True
            print(">> Voice audio playback ENABLED. Aria will speak through your speakers.")
            continue
        elif user_input.lower() == "audio off":
            play_audio = False
            print(">> Voice audio playback DISABLED.")
            continue

        print("\n[Thinking...]")
        res = await dialogue_manager.process_turn(user_input, session_id=session_id)
        reply = res.get("voice_response", "")
        tool_used = res.get("tool_executed")

        print(f"\nAria: {reply}")
        if tool_used:
            print(f"  [Action Executed: {tool_used}]")

        session_state = res.get("session_state", {})
        if session_state.get("active_hold_slot_id"):
            print(f"  [Active Slot Hold: Slot #{session_state['active_hold_slot_id']} | 5-min timer running]")

        # Play spoken audio if enabled
        if play_audio and CAN_PLAY_AUDIO and reply:
            try:
                wav_path = os.path.join("scratch", "last_response.wav")
                os.makedirs("scratch", exist_ok=True)
                tts_engine.synthesize_to_wav(reply, output_path=wav_path)
                winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except Exception as e:
                print(f"  [Audio playback error: {e}]")

        print("-" * 72)


if __name__ == "__main__":
    asyncio.run(main())
