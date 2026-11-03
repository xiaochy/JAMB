# 3D track playback viewer

Standalone, no-build static page: predicted-track + dense-point-cloud
playback for RoboTwin eval episodes, built from the per-chunk trackvis logs
`scripts/eval_policy.py` writes when `TRACK_VIS_LOG_DIR` is set.

## Files

- `index.html` — the viewer (three.js from CDN, no bundler needed)
- `manifest.json` — list of available episodes: `{key, label, setting, outcome, file}`
- `episodes/<key>.json` — one file per episode, `{"b64": "<base64 of the packed binary trackvis export>"}`

## Adding an episode

Run `scripts/export_trackvis_binary.py <trackvis_pkl> <out_dir>/episodes/<key>.json`
(packs one episode's `episodeN_trackvis.pkl` into the binary layout the
viewer parses), then add an entry to `manifest.json` pointing at it.

## Using it standalone

It's a static site — any web server works, e.g.:

```
python3 -m http.server 8000 --directory webviz/track3d
```

then open `http://localhost:8000`. (Opening `index.html` directly via
`file://` will NOT work — `fetch()` of local files is blocked by browser
CORS rules for the `file://` origin.)

## Embedding in another site

Copy this whole directory (or just `index.html` + `manifest.json` + `episodes/`)
under your site's static assets and link/iframe it — it has no dependency on
anything outside itself except the three.js CDN scripts pinned at the top of
`index.html`.

## Binary format (`episodes/<key>.json`'s decoded `b64`)

Little-endian. The dense point cloud is only captured every few chunks
during eval (`DENSE_PC_EVERY`, default 3) to keep long episodes under the
16MB per-file budget, so chunks reference a shared snapshot table by index
rather than each carrying their own point cloud:

```
u32 n_chunks, n_snapshots, n_dense, n_patch, horizon
snapshots (n_snapshots of):
  f32[n_dense*3] xyz   (world-frame, from compute_dense_pointcloud_world)
  u8[n_dense*3]  rgb
chunks (n_chunks of):
  u32 n_frames
  u32 snapshot_idx       (index into the snapshots table above)
  f32[n_patch*3]         patch_centers  (world-frame, from compute_patch_centers_world)
  f32[n_patch*horizon*3] track_pred     (displacement from patch_centers)
```
