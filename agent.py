import logging
import time

import httpx
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    AgentStateChangedEvent,
    AgentTask,
    JobContext,
    MetricsCollectedEvent,
    RunContext,
    ToolError,
    function_tool,
    inference,
    llm,
    mcp,
    metrics,
    room_io,
    stt,
    tts,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

logger = logging.getLogger(__name__)

CARTESIA_SUPPORT_VOICE = "cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
CARTESIA_MANAGER_VOICE = "cartesia/sonic-3:6f84f4b8-58a2-430c-8c79-688dad597532"

WMO_CONDITIONS = {
    0: "clear skies",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


class CollectConsent(AgentTask[bool]):
    def __init__(self, chat_ctx=None):
        super().__init__(
            instructions=(
                "Ask for recording consent and get a clear yes or no answer. "
                "Be polite and professional."
            ),
            chat_ctx=chat_ctx,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Briefly introduce yourself, then ask for permission to record "
                "the call for quality assurance and training purposes. "
                "Make it clear that they can decline."
            )
        )

    @function_tool()
    async def consent_given(self) -> None:
        """Use this when the user gives consent to record."""
        self.complete(True)

    @function_tool()
    async def consent_denied(self) -> None:
        """Use this when the user denies consent to record."""
        self.complete(False)


class ManagerAgent(Agent):
    def __init__(self, chat_ctx=None):
        super().__init__(
            instructions=(
                "You are a customer service manager. You handle escalated issues "
                "that frontline agents couldn't resolve. Be empathetic and "
                "solution-focused. You have authority to offer refunds, credits, "
                "or other accommodations. Keep replies under 3 sentences."
            ),
            chat_ctx=chat_ctx,
            tts=CARTESIA_MANAGER_VOICE,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Introduce yourself as a manager. Acknowledge that the customer "
                "asked to speak with someone senior. Ask how you can help resolve "
                "their concern."
            )
        )


class CustomerServiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are an upbeat, slightly sarcastic voice AI for tech support. "
                "Help the caller fix issues without rambling, and keep replies under 3 sentences. "
                "You can look up the weather if asked. You can also answer questions about "
                "LiveKit by searching the documentation. When users ask about LiveKit "
                "features, APIs, or how to build something, use the docs search tools "
                "to find accurate information. If they ask for a manager or you can't "
                "resolve their issue, use the escalate_to_manager tool."
            ),
        )

    async def on_enter(self) -> None:
        consent = await CollectConsent(chat_ctx=self.chat_ctx)
        if consent:
            await self.session.generate_reply(
                instructions="Thank them and offer your assistance."
            )
        else:
            await self.session.generate_reply(
                instructions="Let them know you understand and will proceed without recording."
            )

    @function_tool()
    async def lookup_weather(
        self,
        context: RunContext,
        location: str,
    ) -> dict:
        """Look up current weather for a location.

        Args:
            location: City name or location to get weather for.
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            geo_response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1},
            )
            geo_response.raise_for_status()
            geo_data = geo_response.json()

            if not geo_data.get("results"):
                raise ToolError(f"Could not find location: {location}")

            place = geo_data["results"][0]
            weather_response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,weather_code",
                    "temperature_unit": "fahrenheit",
                },
            )
            weather_response.raise_for_status()
            weather = weather_response.json()
            code = weather["current"]["weather_code"]

            return {
                "location": place["name"],
                "temperature_f": weather["current"]["temperature_2m"],
                "conditions": WMO_CONDITIONS.get(code, f"weather code {code}"),
            }

    @function_tool()
    async def escalate_to_manager(self, context: RunContext) -> tuple[ManagerAgent, str]:
        """Transfer the customer to a manager when requested or when you cannot resolve their issue."""
        return ManagerAgent(chat_ctx=self.chat_ctx), "Transferring you to a manager now."


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    vad = silero.VAD.load()

    session = AgentSession(
        llm=llm.FallbackAdapter(
            [
                inference.LLM(model="openai/gpt-4.1-mini"),
                inference.LLM(model="google/gemini-2.5-flash"),
            ]
        ),
        stt=stt.FallbackAdapter(
            [
                inference.STT.from_model_string("assemblyai/universal-streaming:en"),
                inference.STT.from_model_string("deepgram/nova-3"),
            ]
        ),
        tts=tts.FallbackAdapter(
            [
                inference.TTS.from_model_string(CARTESIA_SUPPORT_VOICE),
                inference.TTS.from_model_string("inworld/inworld-tts-1"),
            ]
        ),
        vad=vad,
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
        mcp_servers=[
            mcp.MCPServerHTTP(url="https://docs.livekit.io/mcp"),
        ],
    )

    usage_collector = metrics.UsageCollector()
    last_eou_metrics: metrics.EOUMetrics | None = None

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        nonlocal last_eou_metrics
        if ev.metrics.type == "eou_metrics":
            last_eou_metrics = ev.metrics
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        if ev.new_state == "speaking" and last_eou_metrics:
            elapsed = time.time() - last_eou_metrics.timestamp
            logger.info("Time to first audio: %.3fs", elapsed)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info("Usage summary: %s", summary)

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=CustomerServiceAgent(),
        room=ctx.room,
        record=True,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agents.cli.run_app(server)
