# eu4-clone

Hacking on a deterministic, browser-based grand-strategy renderer in Rust + wgpu.
Currently a multi-pass terrain renderer ported from the
[Advanced Terrain Erosion Filter](https://www.shadertoy.com/view/sf23W1) Shadertoy,
with real heightmap data of central Europe loaded from public DEM tiles.

## Running

You need:

- Rust (any reasonably recent stable; see `Cargo.toml` for the resolved versions)
- `wasm-bindgen-cli` matching the `wasm-bindgen` version in `Cargo.lock`
- Python 3 (only for the dev server)

```bash
# One-time setup
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli  # install whatever version Cargo.lock says

# Build + serve
./script/build-wasm
./script/serve   # http://127.0.0.1:8091
```

Open the page and you should see a top-down rendered chunk of the Alps.

## Layout

```
crates/
  math/    Deterministic fixed-point arithmetic (Fixed)
  app/     The wasm crate: wgpu render pipeline + Shadertoy port
    src/
      lib.rs           Setup, render loop, input handling
      shaders/
        common.wgsl    Shared uniforms + fullscreen vs
        noise.wgsl     hash() + gradient noise
        base_heightmap.wgsl  Source heightmap (procedural bump or PNG sample)
        terrain.wgsl   Erosion filter on top of the base heightmap
        detail_noise.wgsl  Surface detail noise
        image.wgsl     Raymarched terrain rendering
    assets/
      heightmap.png    8192² 16-bit elevation, central Europe (Mapzen Terrarium tiles)
      water_mask.png   8192² 8-bit water mask (Natural Earth)
script/
  build-wasm  Compile + bundle into ./dist
  serve       python3 -m http.server bound to 127.0.0.1
```

## Third-party content

- **Heightmap data**: Stitched from Mapzen Terrain Tiles on
  [AWS Open Data](https://registry.opendata.aws/terrain-tiles/) (public domain
  with attribution to data providers — primarily SRTM, NED, ETOPO1, GMTED, etc.).
- **Water mask**: [Natural Earth](https://www.naturalearthdata.com/) 1:10m
  physical features (public domain).
- **Erosion shader**: `crates/app/src/shaders/terrain.wgsl` ports
  Phacelle Noise + the Advanced Terrain Erosion Filter from the original
  Shadertoy by Rune Skovbo Johansen — those functions are © 2025 Rune
  Skovbo Johansen and licensed under the Mozilla Public License 2.0.
