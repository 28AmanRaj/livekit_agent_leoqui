"""
LiveKit Voice Agent - Quick Start
==================================
The simplest possible LiveKit voice agent to get you started.
Requires only OpenAI and Deepgram API keys.
"""

import asyncio
import json
import os
from datetime import datetime

import httpx
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RunContext
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, openai, silero

# Load environment variables
load_dotenv(".env")

class Assistant(Agent):
    """Basic voice assistant."""

    def __init__(self, room=None, extra_instructions=""):
        self.room = room
        self.disconnect_task = None
        super().__init__(
            instructions="""You are an INFORMED medical interviewer conducting a pre-visit intake.
        Your goal is to collect accurate, useful clinical information in the shortest natural conversation possible.

        This is a REAL-TIME SPOKEN INTERVIEW, not a chat.

        Interview type:
        - Medical intake / symptom-gathering interview
        - Purpose: collect accurate information regarding patient's symptoms and medical condition before they visit the doctor.

        Strict behavior rules you MUST follow:

        1. Speak like a real human interviewer.
        - Short, clear sentences
        - Natural, calm, neutral tone
        - No formal or academic language

        2. Keep the interview efficient.
        - Prefer fewer turns over many small questions.
        - Group closely related questions into ONE short sentence.
        - Cover only ONE topic area per turn.
        - Do NOT mix unrelated topics.

        Examples:
        Good: “How long has this been happening, and how severe is it?”
        Bad: “How long, how severe, do you cough, and what medicines?”

        3. Do NOT overload the patient.
        - Maximum 3 related facts per question.
        - Keep questions simple and short when possible.
        - Never use long or complex sentences.

        4. Keep responses short.
        - 1-3 sentence is ideal.
        - Maximum 3 short sentences.

        5. Do NOT explain your reasoning.
        - Do not justify questions.
        - Do not describe the interview process.

        6. Do NOT sound like an AI.
        Avoid phrases like:
        “As an AI…”
        “I understand how you feel…”
        “Thank you for sharing…”
        “That must be difficult…”

        7. Maintain natural interview flow.
        - Finish one topic before moving to the next.
        - Ask relevant follow-ups only.
        - Do not jump between domains.

        8. Let the patient speak.
        - Start with an open-ended question.
        - Allow longer answers when useful.
        - Do not interrupt unnecessarily.

        9. If the patient gives a vague answer:
        - Ask a simple clarifying follow-up.
        - Example: “Can you tell me more about that?”

        10. If the patient pauses:
        - Wait briefly.
        - Then gently prompt.

        11. Never give medical advice, opinions, or conclusions.
        - You are ONLY gathering information.

        12. Interview Completion Logic:
        You must end the interview when ALL are reasonably covered:

        • Main complaint  
        • Duration and severity  
        • Associated symptoms  
        • Response to treatment  
        • Past medical history  
        • Current medications  
        • Allergies  
        • Other relevant context  
        • Final open question  

        Do NOT over-collect.
        “Good enough” is sufficient.

        When complete:
        - Thank briefly
        - Say the doctor will review
        - End conversation
        - Do NOT ask further questions unless user continues

        13. Do NOT insist on any information.
""" + extra_instructions
        )

    def extract_transcript(self, chat_ctx) -> str:
        """Helper to extract chat history to text.

        Uses the LiveKit ChatContext API:
        - chat_ctx.items returns list[ChatItem]
        - Each ChatMessage has .role and .text_content()
        """
        from livekit.agents.llm import ChatMessage

        transcript = []
        for item in chat_ctx.items:
            if isinstance(item, ChatMessage) and item.role in ("user", "assistant"):
                role_name = "Patient" if item.role == "user" else "Agent"
                text = item.text_content
                if text:
                    transcript.append(f"{role_name}: {text}")

        return "\n".join(transcript)


    @function_tool
    async def get_current_date_and_time(self, context: RunContext) -> str:
        """Get the current date and time."""
        current_datetime = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        return f"The current date and time is {current_datetime}"

    @function_tool
    async def end_interview(self, context: RunContext) -> str:
        """Call this function specifically when the interview is completely finished, you have gathered all necessary information, and you are ready to hang up."""
        print("Agent decided to end the interview. Disconnecting room in 8 seconds...")
        
        async def delayed_disconnect():
            await asyncio.sleep(8)  # Give time for the final message to be synthesized and played
            if hasattr(self, 'room') and self.room:
                try:
                    await self.room.disconnect()
                except Exception as e:
                    print(f"Error disconnecting: {e}")

        self.disconnect_task = asyncio.create_task(delayed_disconnect())
        return "SUCCESS. The system will disconnect the call in 8 seconds. Please say your final goodbye to the patient now, e.g. 'Thank you, the doctor will review your answers. Goodbye.'"

async def _submit_to_fastapi(appointment_id: str, transcription: str, summary: str, interview_status: str):
    """Submits the transcription, summary, and interview status to the FastAPI webhook"""
    base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    url = f"{base_url}/api/v1/appointment/{appointment_id}/interview-result"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "transcription": transcription,
        "summary": summary,
        "interview_status": interview_status
    }

    print(f"Submitting interview result to {url}...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            print("Successfully submitted interview result to FastAPI!")
            print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Failed to submit to FastAPI: {e}")

async def fetch_previous_interview(appointment_id: str) -> str | None:
    """Fetches the previous transcription from the backend."""
    base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    url = f"{base_url}/api/v1/appointment/{appointment_id}/interview-result"

    print(f"Checking for previous interview history at {url}...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                print(f"API response: {data}")
                
                # The API wraps the result in a "data" field:
                # {"status": true, "data": {"transcription": "...", "summary": "..."}}
                interview_data = data.get("data", data)  # unwrap, or fall back to root
                
                if interview_data and isinstance(interview_data, dict):
                    transcription = interview_data.get("transcription")
                    if transcription:
                        print("Found previous interview transcription!")
                        return transcription
                
                print("Response had no transcription data.")
            elif response.status_code == 404:
                print("No previous interview found (404).")
            else:
                print(f"Failed to fetch previous interview: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error fetching previous interview: {e}")
    
    return None

def parse_transcription_to_context(transcription: str, ctx: agents.llm.ChatContext):
    """Parses a saved transcription list back into ChatContext messages."""
    for line in transcription.split('\n'):
        if line.startswith("Patient: "):
            ctx.add_message(role="user", content=line[len("Patient: "):])
        elif line.startswith("Agent: "):
            ctx.add_message(role="assistant", content=line[len("Agent: "):])

async def entrypoint(ctx: agents.JobContext):
    """Entry point for the agent."""

    # Configure the voice pipeline with the essentials
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4.1-mini")),
        tts=deepgram.TTS(model="aura-asteria-en"),
        vad=silero.VAD.load(),
    )

    # Start the session
    # Agent is created below, after we check for previous interview history

    @session.on("user_state_changed")
    def on_user_state_changed(ev):
        if ev.new_state == "speaking":
            if hasattr(agent, 'disconnect_task') and agent.disconnect_task and not agent.disconnect_task.done():
                print("\n[INTERRUPTION] Patient started speaking again! Cancelling disconnect...")
                agent.disconnect_task.cancel()
                agent.disconnect_task = None

    @ctx.room.on("data_received")
    def on_data_received(data: bytes, participant, kind, topic: str):
        # Ignore cloud ping requests without warning
        if topic == "lk.agent.request":
            pass

    # Check for previous interview history
    appointment_id = ctx.room.name
    if appointment_id.startswith("appointment_"):
        appointment_id = appointment_id[len("appointment_"):]
    
    previous_transcription = await fetch_previous_interview(appointment_id)
    is_resuming = False
    resume_context = ""
    
    if previous_transcription:
        is_resuming = True
        print(f"\n--- Previous Transcription Found ---\n{previous_transcription[:500]}...\n---\n")
        # Build resume context to inject into the agent's instructions at construction
        resume_context = (
            "\n\n--- IMPORTANT: PREVIOUS INTERVIEW SESSION ---\n"
            "The patient disconnected during a previous interview session. "
            "Below is the transcript from that session. You MUST NOT repeat questions "
            "that were already answered. Review what information you already have and "
            "continue the interview from where it left off, only asking about topics "
            "not yet covered.\n\n"
            f"{previous_transcription}\n"
            "--- END OF PREVIOUS SESSION ---\n\n"
            "Resume the interview naturally. Acknowledge the patient is back and "
            "briefly summarize what you already know, then continue with remaining questions."
        )
        print("Resume context prepared for agent.")
    else:
        print("No previous interview found. Starting fresh.")

    # Connect to the room first so we can wait for participants (audio only)
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)

    # Wait for the patient participant to connect
    print("Waiting for participant to connect...")
    patient_participant = await ctx.wait_for_participant()
    print(f"Participant connected: {patient_participant.identity}, metadata: {patient_participant.metadata}")

    # Extract Patient Health Signals from Participant Metadata
    health_signals_instructions = ""
    if patient_participant and patient_participant.metadata:
        try:
            print(f"Raw participant metadata: {patient_participant.metadata}")
            metadata_dict = json.loads(patient_participant.metadata)
            health_signals = metadata_dict.get("health_signals", [])
            print(f"Extracted health_signals: {health_signals}")
            if health_signals:
                health_signals_instructions += """\n\n--- PATIENT HEALTH SIGNAL CONTEXT ---

The backend analyzed the patient's past lab reports and detected specific health trends.
These are background context to help guide your interview.

Rules:
• These signals are NOT diagnoses. Do NOT diagnose the patient.
• Use them only to guide relevant follow-up questions.
• Do NOT force questions if unrelated to the patient's complaint.
• Prefer asking about symptoms, lifestyle, or medical history.
• Explore each signal briefly if relevant.
• Do NOT interpret lab results or provide medical conclusions.

Detected Health Signals:\n"""
                for signal in health_signals:
                    cat = signal.get("category", "Unknown")
                    summary = signal.get("summary", "No summary")
                    trend = signal.get("trend", "Unknown")
                    priority = signal.get("priority", "Unknown")
                    health_signals_instructions += f"- [{priority.upper()}] {cat}: {summary} (Trend: {trend})\n"
                health_signals_instructions += "\n--- END HEALTH SIGNAL CONTEXT ---\n\n"
                print(f"Injected health signals into instructions: {health_signals}")
        except json.JSONDecodeError:
            print(f"Failed to parse participant metadata as JSON: {patient_participant.metadata}")
        except Exception as e:
            print(f"Error processing metadata: {e}")

    final_extra_instructions = resume_context + health_signals_instructions

    # Create agent with resume context and health signals baked into instructions
    agent = Assistant(room=ctx.room, extra_instructions=final_extra_instructions)

    await session.start(
        room=ctx.room,
        agent=agent
    )

    # Generate initial greeting based on whether we're resuming
    if is_resuming:
        await session.generate_reply(
            instructions="Welcome the patient back. Briefly mention what you already know from the previous session (e.g. their main complaint) and ask what remaining information you still need. Keep it short and natural."
        )
    else:
        await session.generate_reply(
            instructions="Greet user warmly by saying 'Welcome to Medjourney. I am here to gather some information regarding your symptoms and medical condition before you visit the doctor. Please tell me about your symptoms.'"
        )

    # Listen for job shutdown (triggered by room disconnect or server) to reliably save summary
    async def _on_shutdown():
        print("Job is shutting down. Starting data export process...")
        await run_post_interview_tasks(ctx.room.name, session, agent=agent, previous_transcription=previous_transcription)

    ctx.add_shutdown_callback(_on_shutdown)

async def run_post_interview_tasks(room_name: str, session: AgentSession, agent: Assistant, previous_transcription: str | None = None):
    """Extract transcript, summarize, and post to backend."""
    # 1. Extract Transcription using session.history (returns ChatContext)
    chat_ctx = session.history
    current_transcription = agent.extract_transcript(chat_ctx)
    print(f"\n--- Final Transcription ---\n{current_transcription}\n---------------------------\n")

    # 2. Combine with previous transcription if resuming
    if previous_transcription:
        transcription = previous_transcription + "\n" + current_transcription
        print("Combined previous + current transcription for submission.")
    else:
        transcription = current_transcription

    # 3. Generate Summary from full combined transcription
    print("Generating summary...")
    summary_prompt = """Generate a clinical, structured doctor pre-visit brief based exactly on the provided intake conversation transcript.

        Please organize the summary into the following sections using professional medical terminology:
        1. Chief Complaint: The primary reason for the patient's visit.
        2. History of Present Illness (HPI): Detailed symptom description including duration, severity, location, timing, and changing factors.
        3. Past Medical History: Any pre-existing conditions mentioned.
        4. Current Medications: Any medications the patient states they are taking.
        5. Allergies: Any allergies mentioned.
        6. Social/Lifestyle Factors: Relevant context (e.g., diet, activity, habits).
        7. Health Signal Context: Any findings or symptoms relevant to the provided health signals.

        Note: Do not include conversational filler. Keep the brief objective, concise, and structured.

        Transcript:\n\n""" + transcription

    # We create a new temporary ChatContext specifically for summarization
    summary_ctx = agents.llm.ChatContext()
    summary_ctx.add_message(role="system", content="You are a medical summarization assistant.")
    summary_ctx.add_message(role="user", content=summary_prompt)

    try:
        llm = openai.LLM(model=os.getenv("LLM_CHOICE", "gpt-4.1-mini"))
        res = llm.chat(chat_ctx=summary_ctx)

        # Accumulate the stream
        summary_text = ""
        async for chunk in res:
            if chunk.delta and chunk.delta.content:
                 summary_text += chunk.delta.content

        print(f"\n--- Summary ---\n{summary_text}\n---------------\n")

        # 4. Determine interview status
        print("Determining interview status...")
        status_prompt = "Based on the following intake conversation transcript, did the interviewer gather enough information about the patient's condition to consider the interview completed based on standard intake requirements? Reply with ONLY 'completed' or 'not completed'.\n\nTranscript:\n\n" + transcription
        status_ctx = agents.llm.ChatContext()
        status_ctx.add_message(role="system", content="You are an evaluator.")
        status_ctx.add_message(role="user", content=status_prompt)
        
        status_res = llm.chat(chat_ctx=status_ctx)
        status_text = ""
        async for chunk in status_res:
            if chunk.delta and chunk.delta.content:
                status_text += chunk.delta.content
                
        interview_status_raw = status_text.lower().strip()
        interview_status = "not completed" if "not completed" in interview_status_raw else "completed"
        print(f"Interview status evaluated as: {interview_status}")

        # 5. Send to FastAPI
        # The room_name typically has format "appointment_<uuid>"
        # We strip the prefix to get just the UUID for the API endpoint
        appointment_id = room_name
        if appointment_id.startswith("appointment_"):
            appointment_id = appointment_id[len("appointment_"):]

        # Fallback if room_name isn't a UUID
        if "-" not in appointment_id:
             print("Warning: room_name does not look like a UUID, but submitting anyway.")

        await _submit_to_fastapi(appointment_id, transcription, summary_text, interview_status)

    except Exception as e:
        print(f"Error during post-interview wrap-up: {e}")

if __name__ == "__main__":
    import asyncio
    # Run the agent
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
