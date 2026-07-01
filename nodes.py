from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .ffmpeg_pipeline import (
    DEFAULT_RESOLUTIONS,
    FFmpegVideoMergePipeline,
    MergeConfig,
    parse_material_sources,
    parse_storyboards,
)

try:
    import folder_paths  # type: ignore
except ImportError:  # pragma: no cover
    folder_paths = None


class _PQTVideoMergeNodeBase:
    CATEGORY = "PingQutong/Video"

    @staticmethod
    def _common_required_inputs():
        return {
            "video_type": (["mix_oral", "mix"], {"default": "mix_oral"}),
            "material_sources": ("STRING", {"multiline": True, "default": ""}),
            "lip_video_source": ("STRING", {"multiline": False, "default": ""}),
            "oral_audio_source": ("STRING", {"multiline": False, "default": ""}),
            "storyboard_json": ("STRING", {"multiline": True, "default": "[]"}),
            "aspect_ratio": (["16:9", "9:16", "1:1", "custom"], {"default": "16:9"}),
            "output_width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 2}),
            "output_height": ("INT", {"default": 720, "min": 64, "max": 8192, "step": 2}),
            "ffmpeg_threads": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1}),
            "segment_preset": ("STRING", {"default": "ultrafast"}),
            "segment_crf": ("INT", {"default": 28, "min": 0, "max": 51, "step": 1}),
            "standardize_preset": ("STRING", {"default": "fast"}),
            "standardize_crf": ("INT", {"default": 18, "min": 0, "max": 51, "step": 1}),
            "adjust_preset": ("STRING", {"default": "ultrafast"}),
            "adjust_crf": ("INT", {"default": 28, "min": 0, "max": 51, "step": 1}),
            "final_preset": ("STRING", {"default": "fast"}),
            "final_crf": ("INT", {"default": 22, "min": 0, "max": 51, "step": 1}),
            "task_id": ("STRING", {"multiline": False, "default": "video_merge_task"}),
            "output_filename": ("STRING", {"multiline": False, "default": ""}),
            "keep_temp": ("BOOLEAN", {"default": False}),
        }

    @staticmethod
    def _common_optional_inputs():
        return {
            "output_dir": ("STRING", {"multiline": False, "default": ""}),
            "tmp_dir": ("STRING", {"multiline": False, "default": ""}),
            "ffmpeg_path": ("STRING", {"multiline": False, "default": ""}),
            "ffprobe_path": ("STRING", {"multiline": False, "default": ""}),
        }

    @staticmethod
    def _resolve_resolution(aspect_ratio: str, output_width: int, output_height: int) -> tuple[int, int]:
        if aspect_ratio in DEFAULT_RESOLUTIONS:
            return DEFAULT_RESOLUTIONS[aspect_ratio]
        return output_width, output_height

    @staticmethod
    def _resolve_output_dir(output_dir: str) -> str:
        if output_dir and output_dir.strip():
            target = Path(output_dir.strip())
        elif folder_paths is not None:
            target = Path(folder_paths.get_output_directory())
        else:
            target = Path.cwd() / "output"
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    @staticmethod
    def _build_video_ui_result(video_path: str) -> dict | None:
        if folder_paths is None:
            return None

        output_root = Path(folder_paths.get_output_directory()).resolve()
        file_path = Path(video_path).resolve()
        try:
            relative_path = file_path.relative_to(output_root)
        except ValueError:
            return None

        subfolder = str(relative_path.parent).replace("\\", "/")
        if subfolder == ".":
            subfolder = ""

        return {
            "filename": file_path.name,
            "subfolder": subfolder,
            "type": "output",
            "format": file_path.suffix.lstrip(".").lower() or "mp4",
            "fullpath": str(file_path),
        }

    @classmethod
    def _build_pipeline(cls, width, height, ffmpeg_threads, segment_preset, segment_crf, standardize_preset,
                        standardize_crf, adjust_preset, adjust_crf, final_preset, final_crf, keep_temp,
                        tmp_dir="", ffmpeg_path="", ffprobe_path=""):
        return FFmpegVideoMergePipeline(
            MergeConfig(
                output_width=width,
                output_height=height,
                threads=ffmpeg_threads,
                segment_preset=segment_preset,
                segment_crf=segment_crf,
                standardize_preset=standardize_preset,
                standardize_crf=standardize_crf,
                adjust_preset=adjust_preset,
                adjust_crf=adjust_crf,
                final_preset=final_preset,
                final_crf=final_crf,
                tmp_root=tmp_dir.strip() or None,
                keep_temp=keep_temp,
                ffmpeg_path=ffmpeg_path.strip() or None,
                ffprobe_path=ffprobe_path.strip() or None,
            )
        )

    @classmethod
    def _run_pipeline(
        cls,
        video_type,
        material_sources,
        lip_video_source,
        oral_audio_source,
        storyboard_json,
        aspect_ratio,
        output_width,
        output_height,
        ffmpeg_threads,
        segment_preset,
        segment_crf,
        standardize_preset,
        standardize_crf,
        adjust_preset,
        adjust_crf,
        final_preset,
        final_crf,
        task_id,
        output_filename,
        keep_temp,
        output_dir="",
        tmp_dir="",
        ffmpeg_path="",
        ffprobe_path="",
    ):
        material_list = parse_material_sources(material_sources)
        storyboards = parse_storyboards(storyboard_json)
        width, height = cls._resolve_resolution(aspect_ratio, output_width, output_height)
        resolved_output_dir = cls._resolve_output_dir(output_dir)

        pipeline = cls._build_pipeline(
            width,
            height,
            ffmpeg_threads,
            segment_preset,
            segment_crf,
            standardize_preset,
            standardize_crf,
            adjust_preset,
            adjust_crf,
            final_preset,
            final_crf,
            keep_temp,
            tmp_dir=tmp_dir,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
        )

        result = pipeline.run_merge(
            video_type=video_type,
            material_sources=material_list,
            lip_video_source=lip_video_source,
            oral_audio_source=oral_audio_source,
            storyboards=storyboards,
            output_dir=resolved_output_dir,
            output_filename=output_filename,
            task_id=task_id,
            aspect_ratio=aspect_ratio,
        )
        return pipeline, result


class PQTVideoMergeNode(_PQTVideoMergeNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": cls._common_required_inputs(),
            "optional": cls._common_optional_inputs(),
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "video_path", "summary_json")
    FUNCTION = "run"
    OUTPUT_NODE = True

    def run(
        self,
        video_type,
        material_sources,
        lip_video_source,
        oral_audio_source,
        storyboard_json,
        aspect_ratio,
        output_width,
        output_height,
        ffmpeg_threads,
        segment_preset,
        segment_crf,
        standardize_preset,
        standardize_crf,
        adjust_preset,
        adjust_crf,
        final_preset,
        final_crf,
        task_id,
        output_filename,
        keep_temp,
        output_dir="",
        tmp_dir="",
        ffmpeg_path="",
        ffprobe_path="",
    ):
        pipeline, result = self._run_pipeline(
            video_type,
            material_sources,
            lip_video_source,
            oral_audio_source,
            storyboard_json,
            aspect_ratio,
            output_width,
            output_height,
            ffmpeg_threads,
            segment_preset,
            segment_crf,
            standardize_preset,
            standardize_crf,
            adjust_preset,
            adjust_crf,
            final_preset,
            final_crf,
            task_id,
            output_filename,
            keep_temp,
            output_dir=output_dir,
            tmp_dir=tmp_dir,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
        )

        payload_dir = tempfile.mkdtemp(prefix="pqt_comfyui_payload_")
        try:
            images, audio, fps = pipeline.video_to_comfy_payload(result["video_path"], payload_dir)
        finally:
            shutil.rmtree(payload_dir, ignore_errors=True)

        return (
            images,
            audio,
            float(fps),
            result["video_path"],
            json.dumps(result, ensure_ascii=False, indent=2),
        )


class PQTVideoMergeSaveNode(_PQTVideoMergeNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": cls._common_required_inputs(),
            "optional": cls._common_optional_inputs(),
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "summary_json")
    FUNCTION = "run"
    OUTPUT_NODE = True

    def run(
        self,
        video_type,
        material_sources,
        lip_video_source,
        oral_audio_source,
        storyboard_json,
        aspect_ratio,
        output_width,
        output_height,
        ffmpeg_threads,
        segment_preset,
        segment_crf,
        standardize_preset,
        standardize_crf,
        adjust_preset,
        adjust_crf,
        final_preset,
        final_crf,
        task_id,
        output_filename,
        keep_temp,
        output_dir="",
        tmp_dir="",
        ffmpeg_path="",
        ffprobe_path="",
    ):
        _, result = self._run_pipeline(
            video_type,
            material_sources,
            lip_video_source,
            oral_audio_source,
            storyboard_json,
            aspect_ratio,
            output_width,
            output_height,
            ffmpeg_threads,
            segment_preset,
            segment_crf,
            standardize_preset,
            standardize_crf,
            adjust_preset,
            adjust_crf,
            final_preset,
            final_crf,
            task_id,
            output_filename,
            keep_temp,
            output_dir=output_dir,
            tmp_dir=tmp_dir,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
        )

        summary_json = json.dumps(result, ensure_ascii=False, indent=2)
        preview = self._build_video_ui_result(result["video_path"])
        if preview:
            return {
                "ui": {"gifs": [preview]},
                "result": (
                    result["video_path"],
                    summary_json,
                ),
            }

        return (
            result["video_path"],
            summary_json,
        )


class PQTMixOralVideoMergeNode(_PQTVideoMergeNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        required = dict(cls._common_required_inputs())
        required.pop("video_type")
        required.pop("oral_audio_source")
        required["task_id"] = ("STRING", {"multiline": False, "default": "mix_oral_task"})
        return {
            "required": required,
            "optional": cls._common_optional_inputs(),
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "video_path", "summary_json")
    FUNCTION = "run"
    CATEGORY = "PingQutong/Video"
    OUTPUT_NODE = True

    def run(
        self,
        material_sources,
        lip_video_source,
        storyboard_json,
        aspect_ratio,
        output_width,
        output_height,
        ffmpeg_threads,
        segment_preset,
        segment_crf,
        standardize_preset,
        standardize_crf,
        adjust_preset,
        adjust_crf,
        final_preset,
        final_crf,
        task_id,
        output_filename,
        keep_temp,
        output_dir="",
        tmp_dir="",
        ffmpeg_path="",
        ffprobe_path="",
    ):
        return PQTVideoMergeNode().run(
            "mix_oral",
            material_sources,
            lip_video_source,
            "",
            storyboard_json,
            aspect_ratio,
            output_width,
            output_height,
            ffmpeg_threads,
            segment_preset,
            segment_crf,
            standardize_preset,
            standardize_crf,
            adjust_preset,
            adjust_crf,
            final_preset,
            final_crf,
            task_id,
            output_filename,
            keep_temp,
            output_dir=output_dir,
            tmp_dir=tmp_dir,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
        )


class PQTMixOralVideoMergeSaveNode(_PQTVideoMergeNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return PQTMixOralVideoMergeNode.INPUT_TYPES()

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "summary_json")
    FUNCTION = "run"
    CATEGORY = "PingQutong/Video"
    OUTPUT_NODE = True

    def run(
        self,
        material_sources,
        lip_video_source,
        storyboard_json,
        aspect_ratio,
        output_width,
        output_height,
        ffmpeg_threads,
        segment_preset,
        segment_crf,
        standardize_preset,
        standardize_crf,
        adjust_preset,
        adjust_crf,
        final_preset,
        final_crf,
        task_id,
        output_filename,
        keep_temp,
        output_dir="",
        tmp_dir="",
        ffmpeg_path="",
        ffprobe_path="",
    ):
        return PQTVideoMergeSaveNode().run(
            "mix_oral",
            material_sources,
            lip_video_source,
            "",
            storyboard_json,
            aspect_ratio,
            output_width,
            output_height,
            ffmpeg_threads,
            segment_preset,
            segment_crf,
            standardize_preset,
            standardize_crf,
            adjust_preset,
            adjust_crf,
            final_preset,
            final_crf,
            task_id,
            output_filename,
            keep_temp,
            output_dir=output_dir,
            tmp_dir=tmp_dir,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
        )


NODE_CLASS_MAPPINGS = {
    "PQTVideoMergeNode": PQTVideoMergeNode,
    "PQTVideoMergeSaveNode": PQTVideoMergeSaveNode,
    "PQTMixOralVideoMergeNode": PQTMixOralVideoMergeNode,
    "PQTMixOralVideoMergeSaveNode": PQTMixOralVideoMergeSaveNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PQTVideoMergeNode": "PQT Video Merge",
    "PQTVideoMergeSaveNode": "PQT Video Merge Save",
    "PQTMixOralVideoMergeNode": "PQT Mix Oral Video Merge",
    "PQTMixOralVideoMergeSaveNode": "PQT Mix Oral Video Merge Save",
}
