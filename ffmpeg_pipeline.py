from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
import torch
from PIL import Image


DEFAULT_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (1080, 1080),
}


@dataclass
class MergeConfig:
    output_width: int
    output_height: int
    threads: int = 2
    segment_preset: str = "ultrafast"
    segment_crf: int = 28
    standardize_preset: str = "fast"
    standardize_crf: int = 18
    adjust_preset: str = "ultrafast"
    adjust_crf: int = 28
    final_preset: str = "fast"
    final_crf: int = 22
    tmp_root: str | None = None
    keep_temp: bool = False
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


@dataclass
class Form2Profile:
    name: str
    video_width: int
    video_height: int
    split_width: int
    split_half_height: int
    pip_width: int
    pip_height: int
    pip_overlay_position: str


class FFmpegVideoMergePipeline:
    def __init__(self, config: MergeConfig) -> None:
        self.config = config
        self.ffmpeg_bin = self._resolve_binary(
            config.ffmpeg_path,
            env_name="FFMPEG_PATH",
            executable_names=["ffmpeg", "ffmpeg.exe"],
        )
        self.ffprobe_bin = self._resolve_binary(
            config.ffprobe_path,
            env_name="FFPROBE_PATH",
            executable_names=["ffprobe", "ffprobe.exe"],
        )

    def run_merge(
        self,
        video_type: str,
        material_sources: list[str],
        storyboards: list[dict[str, Any]],
        output_dir: str,
        output_filename: str = "",
        task_id: str = "",
        lip_video_source: str = "",
        oral_audio_source: str = "",
        aspect_ratio: str = "16:9",
    ) -> dict[str, Any]:
        normalized_video_type = (video_type or "").strip()
        if normalized_video_type not in {"mix", "mix_oral"}:
            raise ValueError(f"unsupported video_type: {video_type}")
        if not material_sources:
            raise ValueError("material_sources is empty")
        if not storyboards:
            raise ValueError("storyboards is empty")
        if normalized_video_type == "mix_oral" and not lip_video_source.strip():
            raise ValueError("lip_video_source is empty")
        if normalized_video_type == "mix" and not oral_audio_source.strip():
            raise ValueError("oral_audio_source is empty")

        profile = self.get_form2_profile(aspect_ratio, self.config.output_width, self.config.output_height)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        task_prefix = self._safe_name(task_id or f"{normalized_video_type}_{uuid.uuid4().hex[:8]}")
        session_dir = self._create_session_dir(task_prefix)
        started_at = time.time()

        try:
            material_paths: list[str] = []
            total_material_duration = 0.0
            for index, source in enumerate(material_sources, start=1):
                material_path = self.resolve_source(
                    source,
                    session_dir,
                    f"material_{index}{self._guess_extension(source, '.mp4')}",
                )
                material_paths.append(material_path)
                total_material_duration += self.get_video_duration(material_path)

            total_storyboard_duration = sum(
                self.parse_time_period_duration(item.get("time_period")) for item in storyboards
            )
            if total_storyboard_duration <= 0:
                raise ValueError("total_storyboard_duration must be greater than 0")

            concat_material_path = str(Path(session_dir) / "concat_video.mp4")
            self.concat_videos_with_normalization(
                material_paths,
                concat_material_path,
                profile.video_width,
                profile.video_height,
                fps=30,
            )

            result_meta: dict[str, Any] = {
                "video_type": normalized_video_type,
                "task_id": task_id,
                "resolution": f"{profile.video_width}x{profile.video_height}",
                "segment_count": 0,
                "material_count": len(material_paths),
                "total_storyboard_duration": total_storyboard_duration,
                "total_material_duration": round(total_material_duration, 3),
                "session_dir": session_dir,
                "profile": profile.name,
            }

            if normalized_video_type == "mix_oral":
                result = self._run_mix_oral(
                    session_dir=session_dir,
                    storyboards=storyboards,
                    concat_material_path=concat_material_path,
                    lip_video_source=lip_video_source,
                    total_storyboard_duration=total_storyboard_duration,
                    profile=profile,
                )
                result_meta.update(result)
            else:
                result = self._run_mix(
                    session_dir=session_dir,
                    storyboards=storyboards,
                    concat_material_path=concat_material_path,
                    oral_audio_source=oral_audio_source,
                    total_storyboard_duration=total_storyboard_duration,
                    profile=profile,
                )
                result_meta.update(result)

            output_name = self._safe_name(output_filename) if output_filename else f"{task_prefix}.mp4"
            if not output_name.endswith(".mp4"):
                output_name = f"{output_name}.mp4"
            output_path = str(output_dir_path / output_name)
            shutil.copyfile(result_meta["final_video_path"], output_path)

            result_meta["video_path"] = output_path
            result_meta["elapsed_seconds"] = round(time.time() - started_at, 3)
            return result_meta
        finally:
            if not self.config.keep_temp:
                shutil.rmtree(session_dir, ignore_errors=True)

    def run_mix_oral(
        self,
        material_sources: list[str],
        lip_video_source: str,
        storyboards: list[dict[str, Any]],
        output_dir: str,
        output_filename: str = "",
        task_id: str = "",
        aspect_ratio: str = "16:9",
    ) -> dict[str, Any]:
        return self.run_merge(
            video_type="mix_oral",
            material_sources=material_sources,
            storyboards=storyboards,
            output_dir=output_dir,
            output_filename=output_filename,
            task_id=task_id,
            lip_video_source=lip_video_source,
            aspect_ratio=aspect_ratio,
        )

    def run_mix(
        self,
        material_sources: list[str],
        oral_audio_source: str,
        storyboards: list[dict[str, Any]],
        output_dir: str,
        output_filename: str = "",
        task_id: str = "",
        aspect_ratio: str = "16:9",
    ) -> dict[str, Any]:
        return self.run_merge(
            video_type="mix",
            material_sources=material_sources,
            storyboards=storyboards,
            output_dir=output_dir,
            output_filename=output_filename,
            task_id=task_id,
            oral_audio_source=oral_audio_source,
            aspect_ratio=aspect_ratio,
        )

    def resolve_source(self, source: str, session_dir: str, filename: str) -> str:
        source = source.strip()
        if not source:
            raise ValueError("source is empty")
        if self._is_url(source):
            destination = str(Path(session_dir) / filename)
            self.download_file(source, destination)
            return destination

        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"source file not found: {source}")
        return str(path.resolve())

    def download_file(self, url: str, output_path: str) -> str:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(output_path, "wb") as file_obj:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file_obj.write(chunk)
        return output_path

    def get_video_duration(self, video_path: str) -> float:
        result = self._run_command(
            [
                self.ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
        )
        return float((result.stdout or "0").strip() or "0")

    def get_audio_duration(self, audio_path: str) -> float:
        result = self._run_command(
            [
                self.ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            capture_output=True,
        )
        return float((result.stdout or "0").strip() or "0")

    def get_video_fps(self, video_path: str) -> float:
        result = self._run_command(
            [
                self.ffprobe_bin,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
        )
        raw_value = (result.stdout or "").strip()
        if not raw_value:
            return 25.0
        if "/" in raw_value:
            numerator, denominator = raw_value.split("/", 1)
            denominator_value = float(denominator or "1")
            if denominator_value == 0:
                return 25.0
            return float(numerator) / denominator_value
        return float(raw_value)

    def adjust_audio_speed(self, input_path: str, target_duration: float, output_path: str) -> str:
        current_duration = self.get_audio_duration(input_path)
        speed_ratio = current_duration / target_duration
        if speed_ratio <= 0:
            raise ValueError("audio speed_ratio must be greater than 0")
        audio_codec = "libmp3lame" if Path(output_path).suffix.lower() == ".mp3" else "aac"

        self._run_command(
            [
                self.ffmpeg_bin,
                "-y",
                "-i",
                input_path,
                "-filter:a",
                self._build_atempo_chain(speed_ratio),
                "-c:a",
                audio_codec,
                "-threads",
                self._threads_value(),
                "-movflags",
                "+faststart",
                output_path,
            ]
        )
        return output_path

    def adjust_video_speed_legacy(self, input_path: str, target_duration: float, output_path: str) -> str:
        current_duration = self.get_video_duration(input_path)
        speed_ratio = current_duration / target_duration
        if speed_ratio <= 0:
            raise ValueError("video speed_ratio must be greater than 0")

        self._run_command(
            [
                self.ffmpeg_bin,
                "-y",
                "-i",
                input_path,
                "-vf",
                f"setpts={(1 / speed_ratio):.4f}*PTS",
                "-af",
                self._build_atempo_chain(speed_ratio),
                output_path,
            ]
        )
        return output_path

    def standardize_video_for_concat(
        self,
        input_path: str,
        output_path: str,
        width: int,
        height: int,
        fps: int = 30,
    ) -> str:
        self._run_command(
            [
                self.ffmpeg_bin,
                "-y",
                "-i",
                input_path,
                "-vf",
                self._build_concat_standardize_filter(width, height, fps),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                self.config.standardize_preset,
                "-crf",
                str(self.config.standardize_crf),
                "-pix_fmt",
                "yuv420p",
                "-threads",
                self._threads_value(),
                "-movflags",
                "+faststart",
                output_path,
            ]
        )
        return output_path

    def concat_videos_with_normalization(
        self,
        video_paths: list[str],
        output_path: str,
        width: int,
        height: int,
        fps: int = 30,
    ) -> str:
        output_file = Path(output_path)
        prepared_paths: list[str] = []
        for index, video_path in enumerate(video_paths, start=1):
            prepared_path = output_file.with_name(
                f"{output_file.stem}.normalized_{index:03d}.mp4"
            )
            self.standardize_video_for_concat(video_path, str(prepared_path), width, height, fps)
            prepared_paths.append(str(prepared_path))
        return self.concat_prepared_segments_fast(prepared_paths, output_path, width, height)

    def create_split_screen_legacy(
        self,
        video_a_path: str,
        video_b_path: str,
        start_time: float,
        duration: float,
        output_path: str,
        width: int,
        half_height: int,
    ) -> str:
        filter_complex = (
            f"[0:v]scale={width}:{half_height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{half_height}[car];"
            f"[1:v]scale={width}:{half_height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{half_height}:(ow-iw)/2:(oh-ih)/2[per];"
            "[car][per]vstack[outv]"
        )
        self._run_command(
            [
                self.ffmpeg_bin,
                "-y",
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-i",
                video_a_path,
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-i",
                video_b_path,
                "-filter_complex",
                filter_complex,
                "-map",
                "[outv]",
                *self._segment_encode_options(),
                output_path,
            ]
        )
        return output_path

    def create_pip_legacy(
        self,
        video_a_path: str,
        video_b_path: str,
        start_time: float,
        duration: float,
        output_path: str,
        pip_width: int,
        pip_height: int,
        overlay_position: str,
    ) -> str:
        self._run_command(
            [
                self.ffmpeg_bin,
                "-y",
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-i",
                video_a_path,
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-i",
                video_b_path,
                "-filter_complex",
                ";".join(self._build_circular_pip_filters(pip_width, pip_height, overlay_position)),
                "-map",
                "[outv]",
                *self._segment_encode_options(),
                output_path,
            ]
        )
        return output_path

    def extract_full_video_legacy(
        self, input_path: str, start_time: float, duration: float, output_path: str
    ) -> str:
        self._run_command(
            [
                self.ffmpeg_bin,
                "-y",
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-i",
                input_path,
                output_path,
            ]
        )
        return output_path

    def concat_videos_reencode(
        self,
        video_paths: list[str],
        output_path: str,
        width: int,
        height: int,
    ) -> str:
        filter_parts = []
        for index in range(len(video_paths)):
            filter_parts.append(f"[{index}:v]format=yuv420p,scale={width}:{height},setsar=1[v{index}]")
        filter_parts.append(
            "".join(f"[v{index}]" for index in range(len(video_paths)))
            + f"concat=n={len(video_paths)}:v=1:a=0[outv]"
        )
        command = [self.ffmpeg_bin, "-y"]
        for video_path in video_paths:
            command.extend(["-i", video_path])
        command.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "[outv]",
                "-c:v",
                "libx264",
                "-preset",
                self.config.final_preset,
                "-crf",
                str(self.config.final_crf),
                "-pix_fmt",
                "yuv420p",
                output_path,
            ]
        )
        self._run_command(command)
        return output_path

    def concat_prepared_segments_fast(
        self,
        video_paths: list[str],
        output_path: str,
        width: int,
        height: int,
    ) -> str:
        list_file_path = f"{output_path}.list.txt"
        with open(list_file_path, "w", encoding="utf-8") as file_obj:
            for video_path in video_paths:
                normalized = video_path.replace("\\", "/").replace("'", "'\\''")
                file_obj.write(f"file '{normalized}'\n")

        try:
            result = self._run_command(
                [
                    self.ffmpeg_bin,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    list_file_path,
                    "-c:v",
                    "copy",
                    "-an",
                    "-movflags",
                    "+faststart",
                    output_path,
                ],
                capture_output=True,
                check=False,
            )
            stderr = result.stderr or ""
            if result.returncode != 0 or re.search(r"non-monotonic dts|out of order", stderr, re.IGNORECASE):
                self._delete_file(output_path)
                return self.concat_videos_reencode(video_paths, output_path, width, height)
            return output_path
        finally:
            self._delete_file(list_file_path)

    def concat_videos_with_audio_legacy(
        self,
        video_paths: list[str],
        audio_path: str,
        audio_duration: float,
        output_path: str,
        width: int,
        height: int,
    ) -> str:
        filter_parts = []
        for index in range(len(video_paths)):
            filter_parts.append(f"[{index}:v]format=yuv420p,scale={width}:{height},setsar=1[v{index}]")
        filter_parts.append(
            "".join(f"[v{index}]" for index in range(len(video_paths)))
            + f"concat=n={len(video_paths)}:v=1:a=0[outv]"
        )

        command = [self.ffmpeg_bin, "-y"]
        for video_path in video_paths:
            command.extend(["-i", video_path])
        command.extend(["-i", audio_path])
        command.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "[outv]",
                "-map",
                f"{len(video_paths)}:a",
                "-af",
                f"atrim=0:{audio_duration},asetpts=PTS-STARTPTS",
                "-c:v",
                "libx264",
                "-preset",
                self.config.final_preset,
                "-crf",
                str(self.config.final_crf),
                "-c:a",
                "aac",
                "-y",
                output_path,
            ]
        )
        self._run_command(command)
        return output_path

    def export_video_frames(self, video_path: str, output_dir: str) -> list[str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        pattern = str(output_path / "frame_%06d.png")
        self._run_command(
            [
                self.ffmpeg_bin,
                "-y",
                "-i",
                video_path,
                "-vsync",
                "0",
                pattern,
            ]
        )
        return [str(path) for path in sorted(output_path.glob("frame_*.png"))]

    def export_audio_wav(self, video_path: str, output_path: str) -> str:
        self._run_command(
            [
                self.ffmpeg_bin,
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "44100",
                "-ac",
                "2",
                output_path,
            ]
        )
        return output_path

    def video_to_comfy_payload(self, video_path: str, work_dir: str) -> tuple[torch.Tensor, dict[str, Any], float]:
        fps = self.get_video_fps(video_path)
        frames_dir = str(Path(work_dir) / "frames")
        audio_path = str(Path(work_dir) / "audio.wav")
        frame_paths = self.export_video_frames(video_path, frames_dir)
        if not frame_paths:
            raise RuntimeError(f"no frames extracted from {video_path}")
        images = self.load_images_to_tensor(frame_paths)
        audio = self.load_audio_to_comfy(self.export_audio_wav(video_path, audio_path))
        return images, audio, fps

    @staticmethod
    def load_images_to_tensor(frame_paths: list[str]) -> torch.Tensor:
        image_tensors: list[torch.Tensor] = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as image_obj:
                rgb_image = image_obj.convert("RGB")
                image_array = np.asarray(rgb_image, dtype=np.float32) / 255.0
            image_tensors.append(torch.from_numpy(image_array))
        return torch.stack(image_tensors, dim=0)

    @staticmethod
    def load_audio_to_comfy(audio_path: str) -> dict[str, Any]:
        with wave.open(audio_path, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channel_count = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            pcm_bytes = wav_file.readframes(frame_count)

        if sample_width != 2:
            raise RuntimeError(f"unsupported sample width: {sample_width}")

        waveform = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if channel_count > 1:
            waveform = waveform.reshape(-1, channel_count).transpose(1, 0)
        else:
            waveform = waveform.reshape(1, -1)

        audio_tensor = torch.from_numpy(np.ascontiguousarray(waveform)).unsqueeze(0)
        return {"waveform": audio_tensor, "sample_rate": sample_rate}

    @staticmethod
    def parse_time_period_duration(time_period: Any) -> int:
        if not time_period:
            return 5
        match = re.match(r"(\d+)-(\d+)s?", str(time_period))
        if not match:
            return 5
        return max(int(match.group(2)) - int(match.group(1)), 0)

    @staticmethod
    def normalize_edit_type(video_type: str, edit_type: Any) -> str:
        normalized = str(edit_type or "").strip()
        if video_type == "mix":
            return "a满屏"
        return normalized or "a满屏"

    @staticmethod
    def get_form2_profile(aspect_ratio: str, output_width: int, output_height: int) -> Form2Profile:
        if aspect_ratio == "9:16":
            return Form2Profile(
                name="form2-portrait-9:16",
                video_width=output_width,
                video_height=output_height,
                split_width=output_width,
                split_half_height=output_height // 2,
                pip_width=302 if output_width == 720 else int(output_width * 0.42),
                pip_height=170 if output_width == 720 else int(int(output_width * 0.42) * 9 / 16),
                pip_overlay_position="W-w-20:H-h-24",
            )

        return Form2Profile(
            name="form2-landscape-16:9",
            video_width=output_width,
            video_height=output_height,
            split_width=output_width,
            split_half_height=output_height // 2,
            pip_width=320 if output_width == 1280 else output_width // 4,
            pip_height=180 if output_height == 720 else output_height // 4,
            pip_overlay_position="W-w-20:H-h-20",
        )

    def _run_mix(
        self,
        session_dir: str,
        storyboards: list[dict[str, Any]],
        concat_material_path: str,
        oral_audio_source: str,
        total_storyboard_duration: int,
        profile: Form2Profile,
    ) -> dict[str, Any]:
        audio_path = self.resolve_source(
            oral_audio_source,
            session_dir,
            f"oral_audio{self._guess_extension(oral_audio_source, '.mp3')}",
        )
        oral_audio_duration = self.get_audio_duration(audio_path)
        adjusted_audio_path = str(Path(session_dir) / "oral_audio_adjusted.mp3")
        self.adjust_audio_speed(audio_path, total_storyboard_duration, adjusted_audio_path)

        segment_paths: list[str] = []
        current_time = 0
        for index, storyboard in enumerate(storyboards):
            segment_duration = self.parse_time_period_duration(storyboard.get("time_period"))
            edit_type = self.normalize_edit_type("mix", storyboard.get("edit_plan_convert"))
            segment_path = str(Path(session_dir) / f"segment_{index}.mp4")
            self.extract_full_video_legacy(
                concat_material_path,
                current_time,
                segment_duration,
                segment_path,
            )
            segment_paths.append(segment_path)
            current_time += segment_duration

        final_video_path = str(Path(session_dir) / "final_video.mp4")
        self.concat_videos_with_audio_legacy(
            segment_paths,
            adjusted_audio_path,
            total_storyboard_duration,
            final_video_path,
            profile.video_width,
            profile.video_height,
        )
        return {
            "segment_count": len(segment_paths),
            "oral_audio_duration": round(oral_audio_duration, 3),
            "final_video_path": final_video_path,
        }

    def _run_mix_oral(
        self,
        session_dir: str,
        storyboards: list[dict[str, Any]],
        concat_material_path: str,
        lip_video_source: str,
        total_storyboard_duration: int,
        profile: Form2Profile,
    ) -> dict[str, Any]:
        lip_video_path = self.resolve_source(
            lip_video_source,
            session_dir,
            f"lip_video_source{self._guess_extension(lip_video_source, '.mp4')}",
        )
        lip_video_duration = self.get_video_duration(lip_video_path)
        adjusted_lip_path = str(Path(session_dir) / "lip_video_adjusted.mp4")
        self.adjust_video_speed_legacy(lip_video_path, total_storyboard_duration, adjusted_lip_path)

        segment_paths: list[str] = []
        current_time = 0
        for index, storyboard in enumerate(storyboards):
            segment_duration = self.parse_time_period_duration(storyboard.get("time_period"))
            edit_type = self.normalize_edit_type("mix_oral", storyboard.get("edit_plan_convert"))
            segment_path = str(Path(session_dir) / f"segment_{index}.mp4")

            if edit_type == "ab上下分屏":
                self.create_split_screen_legacy(
                    concat_material_path,
                    adjusted_lip_path,
                    current_time,
                    segment_duration,
                    segment_path,
                    profile.split_width,
                    profile.split_half_height,
                )
            elif edit_type == "b右下画中画":
                self.create_pip_legacy(
                    concat_material_path,
                    adjusted_lip_path,
                    current_time,
                    segment_duration,
                    segment_path,
                    profile.pip_width,
                    profile.pip_height,
                    profile.pip_overlay_position,
                )
            elif edit_type == "b满屏":
                self.extract_full_video_legacy(
                    adjusted_lip_path,
                    current_time,
                    segment_duration,
                    segment_path,
                )
            else:
                self.extract_full_video_legacy(
                    concat_material_path,
                    current_time,
                    segment_duration,
                    segment_path,
                )

            segment_paths.append(segment_path)
            current_time += segment_duration

        final_video_path = str(Path(session_dir) / "final_video.mp4")
        self.concat_videos_with_audio_legacy(
            segment_paths,
            adjusted_lip_path,
            total_storyboard_duration,
            final_video_path,
            profile.video_width,
            profile.video_height,
        )
        return {
            "segment_count": len(segment_paths),
            "lip_video_duration": round(lip_video_duration, 3),
            "final_video_path": final_video_path,
        }

    def _segment_encode_options(self) -> list[str]:
        return [
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            self.config.segment_preset,
            "-crf",
            str(self.config.segment_crf),
            "-pix_fmt",
            "yuv420p",
            "-threads",
            self._threads_value(),
            "-movflags",
            "+faststart",
        ]

    @staticmethod
    def _build_atempo_chain(speed_ratio: float) -> str:
        filters: list[str] = []
        remaining = speed_ratio
        while remaining > 2:
            filters.append("atempo=2")
            remaining /= 2
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.6f}")
        return ",".join(filters)

    @staticmethod
    def _build_concat_standardize_filter(width: int, height: int, fps: int) -> str:
        return ",".join(
            [
                f"scale={width}:{height}:force_original_aspect_ratio=decrease",
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
                "setsar=1",
                f"fps={fps}",
                "format=yuv420p",
            ]
        )

    @staticmethod
    def _build_circular_pip_filters(pip_width: int, pip_height: int, overlay_position: str) -> list[str]:
        inner_diameter = max(48, min(int(pip_width), int(pip_height)))
        border_width = max(3, round(inner_diameter * 0.027))
        outer_diameter = inner_diameter + border_width * 2
        inner_radius_expr = f"({inner_diameter}/2-1)"
        outer_radius_expr = f"({outer_diameter}/2-1)"
        return [
            "[0:v]setpts=PTS-STARTPTS[main_video]",
            f"[1:v]setpts=PTS-STARTPTS,scale={inner_diameter}:{inner_diameter}:force_original_aspect_ratio=increase,"
            f"crop={inner_diameter}:{inner_diameter},format=rgba[small_square]",
            f"[small_square]geq=r='r(X,Y)':g='g(X,Y)':b='b(X,Y)':"
            f"a='if(lte((X-W/2)*(X-W/2)+(Y-H/2)*(Y-H/2),{inner_radius_expr}*{inner_radius_expr}),255,0)'[small_circle]",
            f"color=c=white:s={outer_diameter}x{outer_diameter},format=rgba[border_base]",
            f"[border_base]geq=r='255':g='255':b='255':"
            f"a='if(lte((X-W/2)*(X-W/2)+(Y-H/2)*(Y-H/2),{outer_radius_expr}*{outer_radius_expr}),255,0)'[border_circle]",
            "[border_circle][small_circle]overlay=(W-w)/2:(H-h)/2:shortest=1[circular_pip]",
            f"[main_video][circular_pip]overlay={overlay_position}:shortest=1:eof_action=pass[outv]",
        ]

    def _threads_value(self) -> str:
        return str(self.config.threads if self.config.threads > 0 else 2)

    def _create_session_dir(self, task_prefix: str) -> str:
        root = Path(self.config.tmp_root) if self.config.tmp_root else Path(tempfile.gettempdir())
        root.mkdir(parents=True, exist_ok=True)
        session_dir = root / f"comfyui_video_merge_{task_prefix}_{int(time.time() * 1000)}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return str(session_dir)

    @staticmethod
    def _safe_name(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
        return cleaned.strip("._") or uuid.uuid4().hex[:8]

    @staticmethod
    def _guess_extension(source: str, default: str) -> str:
        parsed = urlparse(source)
        suffix = Path(parsed.path).suffix if parsed.scheme else Path(source).suffix
        return suffix if suffix else default

    @staticmethod
    def _is_url(value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"}

    @staticmethod
    def _delete_file(file_path: str) -> None:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass

    @staticmethod
    def _resolve_binary(
        configured_path: str | None,
        env_name: str,
        executable_names: list[str],
    ) -> str:
        candidates: list[str] = []

        if configured_path and configured_path.strip():
            candidates.append(configured_path.strip())

        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            candidates.append(env_value)

        for name in executable_names:
            resolved = shutil.which(name)
            if resolved:
                candidates.append(resolved)

        if os.name == "nt":
            common_dirs = [
                r"C:\ffmpeg\bin",
                r"C:\Program Files\ffmpeg\bin",
                r"C:\Program Files (x86)\ffmpeg\bin",
                r"D:\ffmpeg\bin",
                r"D:\tools\ffmpeg\bin",
            ]
            for directory in common_dirs:
                for name in executable_names:
                    candidates.append(str(Path(directory) / name))

        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return str(Path(candidate))

        readable_names = ", ".join(executable_names)
        raise FileNotFoundError(
            f"Cannot find {readable_names}. "
            f"Set {env_name} or provide an explicit path in the node input."
        )

    def _run_command(
        self,
        command: list[str],
        capture_output: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command,
            check=False,
            capture_output=capture_output,
            text=True,
        )
        if check and result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            raise RuntimeError(f"command failed: {' '.join(command)}\nstdout: {stdout}\nstderr: {stderr}")
        return result


def parse_storyboards(value: str) -> list[dict[str, Any]]:
    data = json.loads(value)
    if isinstance(data, dict) and isinstance(data.get("storyboard_list"), list):
        data = data["storyboard_list"]
    if not isinstance(data, list):
        raise ValueError("storyboard_json must be a JSON array or an object with storyboard_list")
    normalized: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("each storyboard item must be a JSON object")
        normalized.append(item)
    return normalized


def parse_material_sources(value: str) -> list[str]:
    stripped = value.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise ValueError("material_sources JSON must be an array")
        return [str(item).strip() for item in data if str(item).strip()]
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    if "," in stripped:
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return [stripped]
