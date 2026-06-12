# ComfyUI Video Merge

This directory is a standalone ComfyUI custom node project for the PingQutong
`mix_oral` video merge flow. It ports the core FFmpeg logic from the current
backend into Python so it can run on a RunningHub ComfyUI server.

## What It Does

- Downloads or reads local material videos
- Downloads or reads a lip-sync video
- Adjusts lip-sync duration to the total storyboard duration
- Concatenates source materials into one material timeline
- Renders storyboard segments with these edit types:
  - `a满屏`
  - `b满屏`
  - `ab上下分屏`
  - `b右下画中画`
- Concatenates prepared segments
- Adds back the lip-sync audio
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
PingQutong/Video -> PQT Mix Oral Video Merge
PingQutong/Video -> PQT Mix Oral Video Merge Save
```

## Main Inputs

- `material_sources`: one material path or URL per line, or a JSON array
- `lip_video_source`: local path or URL
- `storyboard_json`: storyboard JSON array
- `aspect_ratio`: `16:9`, `9:16`, `1:1`, or `custom`
- `output_width` / `output_height`: used only when `aspect_ratio=custom`
- `ffmpeg_threads`: `0` means auto
- `segment_preset`, `concat_preset`, `adjust_preset`: FFmpeg presets
- `task_id`: output naming helper
- `output_filename`: final MP4 name
- `keep_temp`: keep session files for debugging
- `output_dir` (optional): override ComfyUI output directory
- `tmp_dir` (optional): override temp working directory

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

### `PQT Mix Oral Video Merge`

The node returns:

- `images`: ComfyUI `IMAGE` batch, ready for `Video Combine`
- `audio`: ComfyUI `AUDIO`, ready for `Video Combine`
- `fps`: detected video fps from the rendered MP4
- `video_path`: final MP4 path for debugging or direct file use
- `summary_json`: JSON summary with resolution, segment count, elapsed time, and task metadata

### `PQT Mix Oral Video Merge Save`

This node does not export frames or audio tensors. It directly saves the rendered MP4
into the output directory and returns:

- `video_path`: final MP4 file path
- `summary_json`: JSON summary with resolution, segment count, elapsed time, and task metadata

## Connect To Video Combine

`PQT Mix Oral Video Merge` is designed to feed a downstream `Video Combine` node:

- connect `images` -> `Video Combine.images`
- connect `audio` -> `Video Combine.audio`
- use `fps` as the frame rate reference when configuring `Video Combine`

The node still saves the rendered MP4 so you can inspect the intermediate result if needed.

If you do not want to go through `Video Combine`, use `PQT Mix Oral Video Merge Save`
instead.

## Notes

- This version focuses on the current PingQutong `mix_oral` merge path.
- It does not include COS upload. RunningHub can pick up the output MP4 directly.
- Fast full-screen extraction and fast segment concat both include safe fallback logic.
- The ComfyUI-facing output is produced by decoding the rendered MP4 into frame tensors plus WAV audio.
- If you want this project to be driven directly by your backend, the next step is to
  build a RunningHub workflow JSON that passes the same URLs and storyboard JSON into this node.
