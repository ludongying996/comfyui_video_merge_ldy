# ComfyUI Video Merge

This directory is a standalone ComfyUI custom node project for the PingQutong
video merge flow. It now mirrors the current backend logic in
`backend/src/services/cron/handlers/videoMergeHandler.ts` and
`backend/src/services/cron/api/ffmpegService.ts`, so it can run on a RunningHub
ComfyUI server with behavior close to the production backend.

## What It Does

- Supports both `mix` and `mix_oral`
- Downloads or reads local material videos
- Downloads or reads a lip-sync video for `mix_oral`
- Downloads or reads an oral audio file for `mix`
- Calculates total storyboard duration from `time_period`
- Normalizes all material videos to a unified output resolution, fps, SAR, and pixel format before concat
- Builds a material timeline first, then renders storyboard segments one by one
- Applies the current backend segment rules:
  - `mix`: all edit types are normalized to `a满屏`
  - `mix_oral`: supports `a满屏`, `b满屏`, `ab上下分屏`, `b右下画中画`
- Uses the current split-screen behavior:
  - `a` top half fills and center-crops
  - `b` bottom half keeps full content with proportional scaling and padding
- Uses the current circular picture-in-picture behavior:
  - circular lip video
  - white outer border
  - right-bottom overlay
- Concatenates prepared segments with audio using the same overall backend strategy
- Writes the final MP4 into the ComfyUI output directory
- Exports the final result as ComfyUI `IMAGE` frames and `AUDIO`

## Directory Layout

- `__init__.py`: ComfyUI node export
- `nodes.py`: custom node definition
- `ffmpeg_pipeline.py`: FFmpeg pipeline implementation
- `requirements.txt`: Python dependencies

## Install In ComfyUI

Copy this whole directory into:

```text
ComfyUI/custom_nodes/comfyui_video_merge
```

Then install dependencies inside the ComfyUI Python environment:

```bash
pip install -r custom_nodes/comfyui_video_merge/requirements.txt
```

The server must already have these binaries in `PATH`:

```text
ffmpeg
ffprobe
```

## Node Names

The custom nodes will appear as:

```text
PingQutong/Video -> PQT Video Merge
PingQutong/Video -> PQT Video Merge Save
PingQutong/Video -> PQT Mix Oral Video Merge
PingQutong/Video -> PQT Mix Oral Video Merge Save
```

## Main Inputs

- `PQT Video Merge` / `PQT Video Merge Save`
  - `video_type`: `mix_oral` or `mix`
  - `material_sources`: one material path or URL per line, or a JSON array
  - `lip_video_source`: local path or URL, required for `mix_oral`
  - `oral_audio_source`: local path or URL, required for `mix`
  - `storyboard_json`: storyboard JSON array, or a JSON object containing `storyboard_list`
  - `aspect_ratio`: `16:9`, `9:16`, `1:1`, or `custom`
  - `output_width` / `output_height`: used only when `aspect_ratio=custom`
  - `ffmpeg_threads`: defaults to `2`
  - `segment_preset` / `segment_crf`: used for segment rendering
  - `standardize_preset` / `standardize_crf`: used for material normalization before concat
  - `adjust_preset` / `adjust_crf`: used for duration adjustment outputs
  - `final_preset` / `final_crf`: used for final concat with audio
  - `task_id`: output naming helper
  - `output_filename`: final MP4 name
  - `keep_temp`: keep session files for debugging
  - `output_dir` (optional): override ComfyUI output directory
  - `tmp_dir` (optional): override temp working directory
- `PQT Mix Oral Video Merge` / `PQT Mix Oral Video Merge Save`
  - Backward-compatible wrapper nodes for `mix_oral`
  - Keep the original simplified input style, but internally use the new backend-aligned pipeline

## Storyboard JSON Example

```json
[
  {
    "time_period": "0-5s",
    "edit_plan_convert": "ab上下分屏"
  },
  {
    "time_period": "5-10s",
    "edit_plan_convert": "b右下画中画"
  },
  {
    "time_period": "10-15s",
    "edit_plan_convert": "a满屏"
  },
  {
    "time_period": "25-30s",
    "edit_plan_convert": "b满屏"
  }
]
```

Or backend-style object:

```json
{
  "storyboard_list": [
    {
      "time_period": "0-5s",
      "edit_plan_convert": "ab上下分屏"
    },
    {
      "time_period": "5-10s",
      "edit_plan_convert": "b右下画中画"
    }
  ]
}
```

## Material Sources Example

Multiline string:

```text
https://example.com/material-1.mp4
https://example.com/material-2.mp4
/data/videos/material-3.mp4
```

Or JSON array:

```json
[
  "https://example.com/material-1.mp4",
  "https://example.com/material-2.mp4",
  "/data/videos/material-3.mp4"
]
```

## Output

### `PQT Video Merge`

The node returns:

- `images`: ComfyUI `IMAGE` batch, ready for `Video Combine`
- `audio`: ComfyUI `AUDIO`, ready for `Video Combine`
- `fps`: detected video fps from the rendered MP4
- `video_path`: final MP4 path for debugging or direct file use
- `summary_json`: JSON summary with resolution, segment count, elapsed time, and task metadata

### `PQT Video Merge Save`

This node does not export frames or audio tensors. It directly saves the rendered MP4
into the output directory and returns:

- `video_path`: final MP4 file path
- `summary_json`: JSON summary with resolution, segment count, elapsed time, and task metadata

## Connect To Video Combine

`PQT Video Merge` is designed to feed a downstream `Video Combine` node:

- connect `images` -> `Video Combine.images`
- connect `audio` -> `Video Combine.audio`
- use `fps` as the frame rate reference when configuring `Video Combine`

The node still saves the rendered MP4 so you can inspect the intermediate result if needed.

If you do not want to go through `Video Combine`, use `PQT Video Merge Save`
instead.

## Notes

- This version mirrors the current PingQutong backend merge logic for both `mix` and `mix_oral`.
- It does not include COS upload. RunningHub can pick up the output MP4 directly.
- The ComfyUI-facing output is produced by decoding the rendered MP4 into frame tensors plus WAV audio.
- If you want this project to be driven directly by your backend, the next step is to
  build a RunningHub workflow JSON that passes the same URLs and storyboard JSON into this node.
