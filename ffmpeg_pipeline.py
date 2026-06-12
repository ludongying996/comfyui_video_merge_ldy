from __future__ import annotations

import json
import math
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
    threads: int = 0
    segment_preset: str = "ultrafast"
    segment_crf: int = 28
    concat_preset: str = "ultrafast"
    concat_crf: int = 28
    adjust_preset: str = "ultrafast"
    adjust_crf: int = 28
    tmp_root: str | None = None
    keep_temp: bool = False
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None


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

    def run_mix_oral(
        self,
        material_sources: list[str],
        lip_video_source: str,
        storyboards: list[dict[str, Any]],
        output_dir: str,
        output_filename: str = "",
        task_id: str = "",
    ) -> dict[str, Any]:
        if not material_sources:
            raise ValueError("material_sources is empty")
        if not lip_video_source:
            raise ValueError("lip_video_source is empty")
        if not storyboards:
            raise ValueError("storyboards is empty")

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        task_prefix = self._safe_name(task_id or f"mix_oral_{uuid.uuid4().hex[:8]}")
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

            lip_video_path = self.resolve_source(
                lip_video_source,
                session_dir,
                f"lip_video_source{self._guess_extension(lip_video_source, '.mp4')}",
            )
            lip_video_duration = self.get_video_duration(lip_video_path)

            total_storyboard_duration = sum(
                self.parse_time_period_duration(item.get("time_period")) for item in storyboards
            )
            if total_storyboard_duration <= 0:
                raise ValueError("total_storyboard_duration must be greater than 0")

            adjusted_lip_path = str(Path(session_dir) / "lip_video_adjusted.mp4")
            self.adjust_video_speed(lip_video_path, total_storyboard_duration, adjusted_lip_path)

            concat_material_path = str(Path(session_dir) / "concat_material.mp4")
            self.concat_videos_lightweight(material_paths, concat_material_path)

            segment_paths: list[str] = []
            current_time = 0
            for index, storyboard in enumerate(storyboards):
                duration = self.parse_time_period_duration(storyboard.get("time_period"))
                edit_type = str(storyboard.get("edit_plan_convert") or "")
                segment_path = str(Path(session_dir) / f"segment_{index}.mp4")

                if edit_type == "ab上下分屏":
                    self.create_split_screen(
                        concat_material_path,
                        adjusted_lip_path,
                        current_time,
                        duration,
                        segment_path,
                    )
                elif edit_type == "b右下画中画":
                    self.create_pip(
                        concat_material_path,
                        adjusted_lip_path,
                        current_time,
                        duration,
                        segment_path,
                    )
                elif edit_type == "a满屏":
                    self.extract_full_video_fast(
                        concat_material_path,
                        current_time,
                        duration,
                        segment_path,
                    )
                elif edit_type == "b满屏":
                    self.extract_full_video_fast(
                        adjusted_lip_path,
                        current_time,
                        duration,
                        segment_path,
                    )
                else:
                    self.extract_full_video(
                        concat_material_path,
                        current_time,
                        duration,
                        segment_path,
                    )

                segment_paths.append(segment_path)
                current_time += duration

            concat_video_path = str(Path(session_dir) / "concat_video.mp4")
            self.concat_prepared_segments_fast(
                segment_paths,
                concat_video_path,
                self.config.output_width,
                self.config.output_height,
            )

            final_video_path = str(Path(session_dir) / "final_video.mp4")
            self.add_audio_to_video(
                concat_video_path,
                adjusted_lip_path,
                total_storyboard_duration,
                final_video_path,
            )

            output_name = self._safe_name(output_filename) if output_filename else f"{task_prefix}.mp4"
            if not output_name.endswith(".mp4"):
                output_name = f"{output_name}.mp4"
            output_path = str(output_dir_path / output_name)
            shutil.copyfile(final_video_path, output_path)

            return {
                "video_path": output_path,
                "task_id": task_id,
                "resolution": f"{self.config.output_width}x{self.config.output_height}",
                "segment_count": len(segment_paths),
                "material_count": len(material_paths),
                "total_storyboard_duration": total_storyboard_duration,
                "total_material_duration": round(total_material_duration, 3),
                "lip_video_duration": round(lip_video_duration, 3),
                "elapsed_seconds": round(time.time() - started_at, 3),
                "session_dir": session_dir,
            }
        finally:
            if not self.config.keep_temp:
                shutil.rmtree(session_dir, ignore_errors=True)

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

    def adjust_video_speed(self, input_path: str, target_duration: float, output_path: str) -> str:
        current_duration = self.get_video_duration(input_path)
        speed_ratio = current_duration / target_duration
        if speed_ratio <= 0:
            raise ValueError("speed_ratio must be greater than 0")

        audio_filter = self._build_atempo_chain(speed_ratio)
        video_filter = f"setpts={(1 / speed_ratio):.4f}*PTS"
        self._run_command(
            [
                self.ffmpeg_bin,
                "-y",
                "-i",
                input_path,
                "-vf",
                video_filter,
                "-af",
                audio_filter,
                "-c:v",
                "libx264",
                "-preset",
                self.config.adjust_preset,
                "-crf",
                str(self.config.adjust_crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-threads",
                str(self.config.threads),
                "-movflags",
                "+faststart",
                output_path,
            ]
        )
        return output_path

    def concat_videos_lightweight(self, video_paths: list[str], output_path: str) -> str:
        filter_complex = "".join(f"[{index}:v]" for index in range(len(video_paths)))
        filter_complex += f"concat=n={len(video_paths)}:v=1:a=0[outv]"

        command = [self.ffmpeg_bin, "-y"]
        for video_path in video_paths:
            command.extend(["-i", video_path])
        command.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[outv]",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                self.config.concat_preset,
                "-crf",
                str(self.config.concat_crf),
                "-pix_fmt",
                "yuv420p",
                "-threads",
                str(self.config.threads),
                "-movflags",
                "+faststart",
                output_path,
            ]
        )
        self._run_command(command)
        return output_path

    def create_split_screen(
        self,
        video_a_path: str,
        video_b_path: str,
        start_time: float,
        duration: float,
        output_path: str,
    ) -> str:
        width = self.config.output_width
        half_height = self.config.output_height // 2
        filter_complex = (
            f"[0:v]scale={width}:{half_height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{half_height}:(ow-iw)/2:(oh-ih)/2[car];"
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

    def create_pip(
        self,
        video_a_path: str,
        video_b_path: str,
        start_time: float,
        duration: float,
        output_path: str,
    ) -> str:
        width = self.config.output_width
        height = self.config.output_height
        is_portrait = height > width

        if is_portrait:
            pip_width = int(width * 0.42)
            pip_height = int(pip_width * 9 / 16)
            overlay_position = "W-w-20:H-h-24"
        else:
            pip_width = width // 4
            pip_height = height // 4
            overlay_position = "W-w-20:H-h-20"

        filter_complex = (
            f"[1:v]scale={pip_width}:{pip_height}[small];"
            f"[0:v][small]overlay={overlay_position}[outv]"
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

    def extract_full_video(self, input_path: str, start_time: float, duration: float, output_path: str) -> str:
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
                *self._segment_encode_options(),
                output_path,
            ]
        )
        return output_path

    def extract_full_video_fast(self, input_path: str, start_time: float, duration: float, output_path: str) -> str:
        try:
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
                    "-map",
                    "0:v:0",
                    "-c:v",
                    "copy",
                    "-an",
                    "-avoid_negative_ts",
                    "make_zero",
                    "-movflags",
                    "+faststart",
                    output_path,
                ]
            )
            actual_duration = self.get_video_duration(output_path)
            if abs(actual_duration - duration) > 0.8:
                raise RuntimeError(
                    f"extract_full_video_fast duration mismatch: expected={duration}, actual={actual_duration}"
                )
            return output_path
        except Exception:
            self._delete_file(output_path)
            return self.extract_full_video(input_path, start_time, duration, output_path)

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
                "fast",
                "-crf",
                "22",
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

    def add_audio_to_video(
        self,
        video_path: str,
        audio_path: str,
        audio_duration: float,
        output_path: str,
    ) -> str:
        self._run_command(
            [
                self.ffmpeg_bin,
                "-y",
                "-i",
                video_path,
                "-i",
                audio_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-af",
                f"atrim=0:{audio_duration},asetpts=PTS-STARTPTS",
                output_path,
            ]
        )
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
            str(self.config.threads),
            "-movflags",
            "+faststart",
        ]

    @staticmethod
    def _build_atempo_chain(speed_ratio: float) -> str:
        if 0.5 <= speed_ratio <= 2:
            return f"atempo={speed_ratio}"
        if speed_ratio < 0.5:
            return f"atempo=0.5,atempo={speed_ratio / 0.5}"
        return f"atempo=2,atempo={speed_ratio / 2}"

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
    if not isinstance(data, list):
        raise ValueError("storyboard_json must be a JSON array")
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
