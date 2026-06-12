from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .ffmpeg_pipeline import DEFAULT_RESOLUTIONS, FFmpegVideoMergePipeline, MergeConfig, parse_material_sources, parse_storyboards

try:
    import folder_paths  # type: ignore
except ImportError:  # pragma: no cover
    folder_paths = None


class PQTMixOralVideoMergeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "material_sources": ("STRING", {"multiline": True, "default": ""}),
                "lip_video_source": ("STRING", {"multiline": False, "default": ""}),
                "storyboard_json": ("STRING", {"multiline": True, "default": "[]"}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "custom"], {"default": "16:9"}),
                "output_width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 2}),
                "output_height": ("INT", {"default": 720, "min": 64, "max": 8192, "step": 2}),
                "ffmpeg_threads": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "segment_preset": ("STRING", {"default": "ultrafast"}),
                "segment_crf": ("INT", {"default": 28, "min": 0, "max": 51, "step": 1}),
                "concat_preset": ("STRING", {"default": "ultrafast"}),
                "concat_crf": ("INT", {"default": 28, "min": 0, "max": 51, "step": 1}),
                "adjust_preset": ("STRING", {"default": "ultrafast"}),
                "adjust_crf": ("INT", {"default": 28, "min": 0, "max": 51, "step": 1}),
                "task_id": ("STRING", {"multiline": False, "default": "mix_oral_task"}),
                "output_filename": ("STRING", {"multiline": False, "default": ""}),
                "keep_temp": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "output_dir": ("STRING", {"multiline": False, "default": ""}),
                "tmp_dir": ("STRING", {"multiline": False, "default": ""}),
                "ffmpeg_path": ("STRING", {"multiline": False, "default": ""}),
                "ffprobe_path": ("STRING", {"multiline": False, "default": ""}),
            },
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
        concat_preset,
        concat_crf,
        adjust_preset,
        adjust_crf,
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

        width, height = self._resolve_resolution(aspect_ratio, output_width, output_height)
        resolved_output_dir = self._resolve_output_dir(output_dir)

        pipeline = FFmpegVideoMergePipeline(
            MergeConfig(
                output_width=width,
                output_height=height,
                threads=ffmpeg_threads,
                segment_preset=segment_preset,
                segment_crf=segment_crf,
                concat_preset=concat_preset,
                concat_crf=concat_crf,
                adjust_preset=adjust_preset,
                adjust_crf=adjust_crf,
                tmp_root=tmp_dir.strip() or None,
                keep_temp=keep_temp,
                ffmpeg_path=ffmpeg_path.strip() or None,
                ffprobe_path=ffprobe_path.strip() or None,
            )
        )

        result = pipeline.run_mix_oral(
            material_sources=material_list,
            lip_video_source=lip_video_source,
            storyboards=storyboards,
            output_dir=resolved_output_dir,
            output_filename=output_filename,
            task_id=task_id,
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


NODE_CLASS_MAPPINGS = {
    "PQTMixOralVideoMergeNode": PQTMixOralVideoMergeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PQTMixOralVideoMergeNode": "PQT Mix Oral Video Merge",
}
