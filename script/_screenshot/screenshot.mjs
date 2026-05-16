// Puppeteer driver for the eu4-clone screenshot harness.
//
// Invoked by `script/screenshot` (the bash wrapper). Builds a URL with
// camera-control query params, drives a headless Chrome at it, waits for
// `window.__eu4_ready` (set by `lib.rs::mark_screenshot_ready` once all
// asset PNGs have loaded + been baked into the LoD atlases), and saves
// the canvas pixels to a PNG.
//
// Why puppeteer-core, not puppeteer? puppeteer downloads its own
// ~170 MB Chromium on install. On macOS we already have Google Chrome
// at the standard path, so puppeteer-core (which has no bundled
// browser, just the protocol client) is fine and ~50× smaller.
//
// Why a real Chrome, not headless Chromium? WebGPU is the renderer's
// primary backend; running it in a real GPU-backed Chrome is the only
// way to keep parity with what you see when you visit the page in your
// dev browser. The wgpu code falls back to WebGL2 if WebGPU isn't
// available, but the visual output isn't bit-identical between the two.

import { parseArgs } from "node:util";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import puppeteer from "puppeteer-core";

const { values } = parseArgs({
  options: {
    cx:        { type: "string" },
    cz:        { type: "string" },
    dist:      { type: "string" },
    tilt:      { type: "string" },
    width:     { type: "string", default: "1280" },
    height:    { type: "string", default: "800" },
    output:    { type: "string", default: "screenshot.png" },
    url:       { type: "string", default: "http://127.0.0.1:8091/" },
    timeout:   { type: "string", default: "30" },
    "keep-open": { type: "boolean", default: false },
  },
});

const width   = parseInt(values.width, 10);
const height  = parseInt(values.height, 10);
const timeoutMs = Math.max(1, parseInt(values.timeout, 10)) * 1000;

// Build the query string. Only set keys the user explicitly passed;
// omitted keys let the Rust side use its defaults so we don't have to
// duplicate them here.
const qs = new URLSearchParams();
for (const k of ["cx", "cz", "dist", "tilt"]) {
  if (values[k] !== undefined) qs.set(k, values[k]);
}
const base = values.url.endsWith("/") ? values.url : `${values.url}/`;
const fullUrl = qs.toString() ? `${base}?${qs.toString()}` : base;

// The puppeteer-core docs recommend Chrome over Chromium for end-user
// rendering. macOS path is hard-coded — if/when this harness needs to
// run anywhere else, factor this out into an env var or autodetect.
const CHROME_BIN =
  process.env.CHROME_BIN ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

if (!existsSync(CHROME_BIN)) {
  console.error(`screenshot: Chrome not found at ${CHROME_BIN}`);
  console.error("  Set CHROME_BIN=... or install Google Chrome.");
  process.exit(1);
}

const outputPath = resolve(values.output);
mkdirSync(dirname(outputPath), { recursive: true });

console.error(`screenshot: launching Chrome → ${fullUrl}`);

const browser = await puppeteer.launch({
  executablePath: CHROME_BIN,
  // `--headless=new` is the post-2022 mode that actually runs the full
  // browser pipeline (including GPU) instead of the old code path.
  // `--enable-unsafe-webgpu` is required for WebGPU in headless on most
  // platforms; harmless when WebGPU isn't supported (wgpu falls back to
  // WebGL2 either way thanks to the `webgl` feature in Cargo.toml).
  headless: "new",
  args: [
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--use-angle=metal",
    "--ignore-gpu-blocklist",
  ],
  defaultViewport: { width, height, deviceScaleFactor: 1 },
});

let exitCode = 0;
try {
  const page = await browser.newPage();
  // Surface wasm-side console.log / console.error in our terminal so
  // failures don't disappear into a black box.
  page.on("console", (msg) => {
    const t = msg.type();
    const prefix = t === "error" ? "page!" : "page";
    console.error(`${prefix}: ${msg.text()}`);
  });
  page.on("pageerror", (err) => {
    console.error(`pageerror: ${err.message}`);
  });

  await page.goto(fullUrl, { waitUntil: "domcontentloaded" });

  // Wait for `window.__eu4_ready` (set by Rust after all PNG assets
  // have loaded + been baked). The Rust side fires this *exactly once*
  // per page load, so polling with waitForFunction is the simplest
  // path.
  await page.waitForFunction("window.__eu4_ready === true", {
    timeout: timeoutMs,
    polling: 100,
  });

  // Give the renderer one extra rAF tick — `mark_screenshot_ready`
  // fires *during* a frame so the swapchain present for that frame is
  // already in flight, but the next requestAnimationFrame is the
  // earliest point we can be confident it's reached the compositor.
  await page.evaluate(
    () =>
      new Promise((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      ),
  );

  // Screenshot just the canvas, not the full viewport — that way the
  // city-panel overlay (when re-enabled) and any future HTML chrome
  // doesn't end up in the PNG.
  const canvas = await page.$("#game");
  if (!canvas) {
    throw new Error("#game canvas not found in the page");
  }
  await canvas.screenshot({ path: outputPath, type: "png" });

  console.error(`screenshot: wrote ${outputPath} (${width}×${height})`);
} catch (err) {
  console.error(`screenshot: ${err.message}`);
  exitCode = 1;
} finally {
  if (values["keep-open"]) {
    console.error("screenshot: --keep-open, leaving Chrome running");
  } else {
    await browser.close();
  }
}

process.exit(exitCode);
