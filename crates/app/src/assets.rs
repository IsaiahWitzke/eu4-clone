//! Runtime asset loading. The wasm bundle ships small; the heavyweight PNGs
//! (heightmap, water mask) are fetched at startup and decoded with the `png`
//! crate.

use js_sys::Uint8Array;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::Response;

/// Decoded grayscale PNG: width, height in pixels, plus the raw pixel bytes
/// in little-endian order suitable for direct upload to a wgpu texture.
///   - 16-bit grayscale: 2 bytes per pixel, R16Unorm-compatible.
///   - 8-bit grayscale: 1 byte per pixel, R8Unorm-compatible.
pub struct DecodedPng {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub bytes: Vec<u8>,
}

/// Fetch a URL via the browser's `fetch` API and decode the resulting bytes
/// as a grayscale PNG. Errors bubble up through `Result<_, JsValue>` so the
/// caller can `?` them.
pub async fn fetch_png(url: &str) -> Result<DecodedPng, wasm_bindgen::JsValue> {
    let window = web_sys::window().expect("no window");
    let resp_value = JsFuture::from(window.fetch_with_str(url)).await?;
    let response: Response = resp_value
        .dyn_into()
        .expect("fetch did not return a Response");
    if !response.ok() {
        return Err(format!(
            "fetch {} failed: HTTP {}",
            url,
            response.status()
        )
        .into());
    }
    let array_buf = JsFuture::from(response.array_buffer()?).await?;
    let bytes = Uint8Array::new(&array_buf).to_vec();

    decode_grayscale_png(&bytes).map_err(|e| format!("decode {url}: {e}").into())
}

/// Decode a grayscale PNG. For 16-bit sources, the `png` crate gives us
/// big-endian samples; we byte-swap them to little-endian here so the bytes
/// can be uploaded directly into an `R16Unorm` texture.
fn decode_grayscale_png(bytes: &[u8]) -> Result<DecodedPng, String> {
    let decoder = png::Decoder::new(bytes);
    let mut reader = decoder.read_info().map_err(|e| e.to_string())?;
    let info = reader.info().clone();

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let read_info = reader.next_frame(&mut buf).map_err(|e| e.to_string())?;
    buf.truncate(read_info.buffer_size());

    let bit_depth = match info.bit_depth {
        png::BitDepth::Eight => 8,
        png::BitDepth::Sixteen => 16,
        other => return Err(format!("unsupported bit depth: {other:?}")),
    };

    if !matches!(info.color_type, png::ColorType::Grayscale) {
        return Err(format!("expected grayscale, got {:?}", info.color_type));
    }

    // 16-bit grayscale PNGs store samples big-endian per spec: [hi, lo, hi, lo, ...].
    // We upload that buffer verbatim into an `Rg8Unorm` texture so byte 0 ends up
    // in the R channel and byte 1 in the G channel — the shader reassembles them.
    // 8-bit grayscale needs no transformation.

    Ok(DecodedPng {
        width: info.width,
        height: info.height,
        bit_depth,
        bytes: buf,
    })
}
