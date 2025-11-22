"""Video processing pipeline."""

import cv2
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from sources import VideoSource
from filters.base import BaseFilter


class VideoPipeline:
    """Video processing pipeline with filters."""

    def __init__(self, source: VideoSource, filter_obj: BaseFilter, source_path: Optional[Path] = None):
        self.source = source
        self.filter = filter_obj
        self.props = source.get_properties()
        self.source_path = source_path

    def process_to_file(self, output_path: Path, show_preview: bool = False, preserve_audio: bool = True):
        """
        Process video and save to file.

        Args:
            output_path: Path to save output video
            show_preview: Whether to show real-time preview window
            preserve_audio: Whether to preserve audio from source video (if available)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary file for video without audio
        temp_video = None
        if preserve_audio and self.source_path:
            temp_video = output_path.parent / f"temp_{output_path.name}"
            video_output = temp_video
        else:
            video_output = output_path

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(video_output),
            fourcc,
            self.props["fps"],
            (self.props["width"], self.props["height"])
        )

        frame_count = 0
        total_frames = self.props.get("frame_count", "unknown")

        print(f"Processing video...")
        print(f"Resolution: {self.props['width']}x{self.props['height']}")
        print(f"FPS: {self.props['fps']}")
        print(f"Total frames: {total_frames}")
        print(f"Output: {output_path}")

        try:
            while True:
                ret, frame = self.source.read()
                if not ret:
                    break

                # Apply filter
                processed_frame = self.filter.apply(frame)

                # Write to output
                out.write(processed_frame)

                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames", end="\r")

                # Show preview if requested
                if show_preview:
                    cv2.imshow('Processed Frame', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopped by user")
                        break

            print(f"\nProcessed {frame_count} frames total")

        finally:
            out.release()
            self.source.release()
            if show_preview:
                cv2.destroyAllWindows()

        # Merge with audio if requested and source has audio
        if preserve_audio and self.source_path and temp_video:
            try:
                print("Merging audio from source video...")
                # Use ffmpeg directly to copy audio stream from source
                result = subprocess.run([
                    'ffmpeg', '-y',
                    '-i', str(temp_video),
                    '-i', str(self.source_path),
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-map', '0:v:0',
                    '-map', '1:a:0?',
                    str(output_path)
                ], capture_output=True, text=True)

                if result.returncode == 0:
                    # Remove temporary file
                    temp_video.unlink()
                    print(f"Saved with audio to: {output_path}")
                else:
                    # Check if source has no audio stream
                    if "does not contain any stream" in result.stderr or "Stream map" in result.stderr:
                        shutil.move(str(temp_video), str(output_path))
                        print(f"No audio in source. Saved to: {output_path}")
                    else:
                        raise Exception(f"ffmpeg failed: {result.stderr}")

            except Exception as e:
                print(f"Warning: Could not merge audio ({e})")
                # Fallback: just rename temp file
                if temp_video.exists():
                    shutil.move(str(temp_video), str(output_path))
                print(f"Saved without audio to: {output_path}")
        else:
            print(f"Saved to: {output_path}")

    def process_realtime(self, window_name: str = "Real-time Filter"):
        """
        Process video in real-time with preview window.
        Press 'q' to quit.

        Args:
            window_name: Name of the preview window
        """
        print(f"Starting real-time processing...")
        print(f"Resolution: {self.props['width']}x{self.props['height']}")
        print(f"Press 'q' to quit")

        try:
            while True:
                ret, frame = self.source.read()
                if not ret:
                    break

                # Apply filter
                processed_frame = self.filter.apply(frame)

                # Display
                cv2.imshow(window_name, processed_frame)

                # Quit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stopped by user")
                    break

        finally:
            self.source.release()
            cv2.destroyAllWindows()
