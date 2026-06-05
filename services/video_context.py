from livekit import rtc
import asyncio
import logging
from dataclasses import dataclass
from livekit.agents.utils.images import encode, EncodeOptions

logger = logging.getLogger("video_context")

@dataclass
class CapturedFrame:
    image_bytes: bytes
    width: int
    height: int

class VideoContextService:
    def __init__(self, room: rtc.Room):
        self.room = room

    async def capture_current_frame(self, participant_id: str) -> CapturedFrame:
        logger.info(f"Attempting to capture frame for participant: {participant_id}")
        
        # Find the participant
        participant = None
        if self.room.local_participant.identity == participant_id or self.room.local_participant.sid == participant_id:
            participant = self.room.local_participant
        else:
            for p in self.room.remote_participants.values():
                if p.identity == participant_id or p.sid == participant_id:
                    participant = p
                    break

        if not participant:
            raise ValueError(f"Participant '{participant_id}' not found in room.")

        # Find a video track
        video_track = None
        for publication in participant.track_publications.values():
            if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                video_track = publication.track
                break

        if not video_track:
            raise ValueError(f"No video track found for participant '{participant_id}'.")

        # Capture the next frame from the video stream
        video_stream = rtc.VideoStream(video_track)
        try:
            # Wait for up to 3 seconds for the first frame
            frame_event = await asyncio.wait_for(video_stream.__anext__(), timeout=3.0)
            frame = frame_event.frame
            
            # Encode frame to JPEG bytes
            options = EncodeOptions(format="JPEG", quality=85)
            jpeg_bytes = encode(frame, options=options)
            
            logger.info(f"Successfully captured frame: {frame.width}x{frame.height}")
            return CapturedFrame(
                image_bytes=jpeg_bytes,
                width=frame.width,
                height=frame.height
            )
        except StopAsyncIteration:
            raise ValueError("Video stream ended before a frame could be captured.")
        except asyncio.TimeoutError:
            raise TimeoutError("Timeout waiting for a video frame.")
        finally:
            await video_stream.aclose()
