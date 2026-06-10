import asyncio
import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np
from livekit import rtc
from livekit.agents.utils.images import EncodeOptions, encode

logger = logging.getLogger("video_context")

@dataclass
class CapturedFrame:
    image_bytes: bytes
    width: int
    height: int

class VideoContextService:
    def __init__(self, room: rtc.Room):
        self.room = room
        self.top_candidate_frames = []
        self._sampling_tasks = {}  # track_sid -> asyncio.Task
        self._is_sampling = False

        # Configurable normalization coefficients
        self.MAX_SHARPNESS = 1000.0
        self.MAX_EDGE_DENSITY = 0.5
        self.MAX_CORNER_DENSITY = 500.0

        # Quality weights (must sum to 1.0)
        self.WEIGHT_SHARPNESS = 0.4
        self.WEIGHT_EDGE_DENSITY = 0.3
        self.WEIGHT_CORNER_DENSITY = 0.3

    def start_sampling(self) -> None:
        """Register listeners and start sampling frames from video tracks."""
        if self._is_sampling:
            return
        self._is_sampling = True
        logger.info("Starting VideoContextService track sampling...")

        # Register event handlers on room for track events
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("track_unsubscribed", self._on_track_unsubscribed)

        # Start sampling for any tracks already subscribed
        for participant in self.room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_VIDEO:
                    self._start_track_sampling(publication.track)

    def stop_sampling(self) -> None:
        """Cancel all active sampling tasks and stop listening."""
        if not self._is_sampling:
            return
        self._is_sampling = False
        logger.info("Stopping VideoContextService track sampling...")

        # Cancel all active sampling tasks
        for task in list(self._sampling_tasks.values()):
            task.cancel()
        self._sampling_tasks.clear()

    def _on_track_subscribed(
        self, track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        if not self._is_sampling:
            return
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            self._start_track_sampling(track)

    def _on_track_unsubscribed(
        self, track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant
    ) -> None:
        self._stop_track_sampling(track)

    def _start_track_sampling(self, track: rtc.VideoTrack) -> None:
        track_sid = track.sid
        if track_sid in self._sampling_tasks:
            return
        logger.info(f"Starting frame sampling task for video track: {track_sid}")
        task = asyncio.create_task(self._sample_track_loop(track))
        self._sampling_tasks[track_sid] = task

    def _stop_track_sampling(self, track: rtc.VideoTrack) -> None:
        track_sid = track.sid
        task = self._sampling_tasks.pop(track_sid, None)
        if task:
            logger.info(f"Stopping frame sampling task for video track: {track_sid}")
            task.cancel()

    async def _sample_track_loop(self, track: rtc.VideoTrack) -> None:
        video_stream = rtc.VideoStream(track)
        last_time = 0.0
        try:
            async for frame_event in video_stream:
                now = asyncio.get_event_loop().time()
                # Sample at 1 fps
                if now - last_time >= 1.0:
                    last_time = now
                    frame = frame_event.frame
                    # Process frame in background thread to avoid blocking main thread
                    asyncio.create_task(self._process_frame(frame))
        except asyncio.CancelledError:
            logger.info(f"Sampling task for track {track.sid} cancelled.")
        except Exception as e:
            logger.error(f"Error sampling track {track.sid}: {e}")
        finally:
            await video_stream.aclose()

    async def _process_frame(self, frame: rtc.VideoFrame) -> None:
        try:
            # CPU-heavy CV analysis runs in thread pool
            candidate = await asyncio.to_thread(self._analyze_frame_quality, frame)
            if candidate is None:
                return
            await self._add_candidate_to_buffer(candidate)
        except Exception as e:
            logger.error(f"Error in background frame processing: {e}")

    def _analyze_frame_quality(self, frame: rtc.VideoFrame) -> dict | None:
        try:
            # Encode frame to JPEG bytes
            options = EncodeOptions(format="JPEG", quality=85)
            jpeg_bytes = encode(frame, options=options)

            # Decode using OpenCV
            np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                return None

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 1. Sharpness (Variance of Laplacian)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            # 2. Edge Density (Canny Edge Detection)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.count_nonzero(edges) / edges.size if edges.size > 0 else 0.0

            # 3. Corner Density (Shi-Tomasi Corners)
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
            corner_density = len(corners) if corners is not None else 0

            # 4. Normalize metrics
            normalized_sharpness = min(sharpness / self.MAX_SHARPNESS, 1.0)
            normalized_edge_density = min(edge_density / self.MAX_EDGE_DENSITY, 1.0)
            normalized_corner_density = min(corner_density / self.MAX_CORNER_DENSITY, 1.0)

            # 5. Compute overall quality score
            quality_score = (
                self.WEIGHT_SHARPNESS * normalized_sharpness +
                self.WEIGHT_EDGE_DENSITY * normalized_edge_density +
                self.WEIGHT_CORNER_DENSITY * normalized_corner_density
            )

            return {
                "frame_bytes": jpeg_bytes,
                "width": frame.width,
                "height": frame.height,
                "timestamp": time.time(),
                "quality_score": quality_score,
                "sharpness": sharpness,
                "edge_density": edge_density,
                "corner_density": corner_density
            }
        except Exception as e:
            logger.error(f"Failed to analyze frame quality: {e}")
            return None

    async def _add_candidate_to_buffer(self, candidate: dict) -> None:
        """Add candidate to buffer keeping only top 5 highest-scoring frames from the last 60 seconds."""
        now_time = time.time()
        # Keep only frames from the last 60 seconds
        self.top_candidate_frames = [f for f in self.top_candidate_frames if now_time - f["timestamp"] <= 60.0]

        if len(self.top_candidate_frames) < 5:
            self.top_candidate_frames.append(candidate)
            self.top_candidate_frames.sort(key=lambda x: x["quality_score"], reverse=True)
            logger.info(
                f"[VideoContext] Added frame. Buffer size: {len(self.top_candidate_frames)}, "
                f"Score: {candidate['quality_score']:.3f} (S:{candidate['sharpness']:.1f}, E:{candidate['edge_density']:.3f}, C:{candidate['corner_density']})"
            )
        else:
            # Replace lowest quality frame if new one is better
            if candidate["quality_score"] > self.top_candidate_frames[-1]["quality_score"]:
                old_score = self.top_candidate_frames[-1]["quality_score"]
                self.top_candidate_frames[-1] = candidate
                self.top_candidate_frames.sort(key=lambda x: x["quality_score"], reverse=True)
                logger.info(
                    f"[VideoContext] Replaced lowest frame (Score: {old_score:.3f}) "
                    f"with better frame (Score: {candidate['quality_score']:.3f})"
                )

    def get_top_candidates(self, limit: int = 3) -> list[dict]:
        """Returns the best frames from the last 60 seconds sorted by quality score."""
        now_time = time.time()
        self.top_candidate_frames = [f for f in self.top_candidate_frames if now_time - f["timestamp"] <= 60.0]
        return self.top_candidate_frames[:limit]

    def clear_buffer(self) -> None:
        """Clear all buffered candidate frames."""
        self.top_candidate_frames.clear()
        logger.info("[VideoContext] Candidate frame buffer cleared.")

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
