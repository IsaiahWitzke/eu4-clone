# eu4-clone

Grand-strategy renderer in Rust + wgpu, running in the browser via wasm.

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
