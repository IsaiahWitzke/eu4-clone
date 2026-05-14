//! HTML-overlay UI. The map renders into `<canvas id="game">`; everything
//! the player reads or clicks (panels, tooltips, dialogs) lives in a
//! sibling `<div id="ui-root">` with `pointer-events: none`. Each
//! interactive child opts back into pointer events, which keeps the
//! canvas's own drag/zoom behaviour clean.
//!
//! For now the only piece of UI is the "selected city" panel — populated
//! when the user clicks on a city, hidden otherwise. The markup itself
//! lives in `web/index.html`; this module just looks elements up by id
//! and pokes their text / class state.
//!
//! Recommendation moving forward: until this gets to ~5 panels, keep
//! using plain `web_sys` here. When the per-frame state-sync starts to
//! hurt, swap in a Rust-side framework (Dioxus / Leptos) — only the
//! markup-construction side needs to change, the Rust→DOM contract
//! exposed by this module stays the same.

use wasm_bindgen::JsCast;
use web_sys::{Document, Element, HtmlElement};

use crate::settlements::{Settlement, realm_color_hex};

/// The selected-city panel. Holds cached references to the DOM nodes so
/// we don't re-query the document on every `show()` call.
pub struct CityPanel {
    /// Root `<div id="city-panel">`. Toggled visible/hidden by adding /
    /// removing the `visible` class.
    root: Element,
    /// Text nodes / containers inside the panel. Filled by `show()`.
    swatch: HtmlElement,
    name: Element,
    realm: Element,
    pop: Element,
    pos: Element,
}

impl CityPanel {
    /// Look up the city-panel elements once at boot. Panics if the
    /// expected ids aren't in the page (which means `web/index.html`
    /// is out of sync with this code — fail loud, not silently).
    pub fn new(document: &Document) -> Self {
        let root = require_element(document, "city-panel");
        let swatch = require_element(document, "cp-swatch")
            .dyn_into::<HtmlElement>()
            .expect("#cp-swatch is not an HtmlElement");
        let name = require_element(document, "cp-name");
        let realm = require_element(document, "cp-realm");
        let pop = require_element(document, "cp-pop");
        let pos = require_element(document, "cp-pos");
        Self {
            root,
            swatch,
            name,
            realm,
            pop,
            pos,
        }
    }

    /// Populate the panel with a settlement's details and reveal it.
    pub fn show(&self, s: &Settlement) {
        self.name.set_text_content(Some(&s.name));
        self.realm
            .set_text_content(Some(&format!("Realm {}", s.realm_id)));
        // Strength is "kilo-population" in the seed data. Show it as
        // an integer count of thousands; small towns (strength < 10)
        // render with a `<10k` suffix so they don't look like "0k".
        let pop_str = if s.strength < 1.0 {
            "<1k".to_string()
        } else if s.strength < 10.0 {
            format!("~{:.0}k", s.strength)
        } else {
            format!("{:.0}k", s.strength)
        };
        self.pop.set_text_content(Some(&pop_str));
        self.pos.set_text_content(Some(&format!(
            "{:.0}, {:.0} km",
            s.world_xz[0], s.world_xz[1]
        )));

        // Realm colour swatch — set background-color directly so we
        // don't have to maintain N CSS classes for every realm.
        let _ = self
            .swatch
            .style()
            .set_property("background-color", realm_color_hex(s.realm_id));

        self.set_visible(true);
    }

    /// Hide the panel. The DOM stays mounted so the next `show()` can
    /// reuse the cached element handles; CSS `visibility: hidden` +
    /// transition gives a small fade-out.
    pub fn hide(&self) {
        self.set_visible(false);
    }

    /// Add or remove the `visible` class. Driven by `display`/`opacity`
    /// transitions in CSS; see `web/index.html` for the rule.
    fn set_visible(&self, visible: bool) {
        let cls = self.root.class_list();
        if visible {
            let _ = cls.add_1("visible");
        } else {
            let _ = cls.remove_1("visible");
        }
    }
}

/// Look up a required element by id, panicking with a helpful message if
/// it's missing — fast-fail when `web/index.html` is out of sync with
/// the Rust side.
fn require_element(document: &Document, id: &str) -> Element {
    document
        .get_element_by_id(id)
        .unwrap_or_else(|| panic!("missing #{} in index.html", id))
}

// ---- Realm name labels ----------------------------------------------------
//
// Country-name labels are now rendered in-engine via the SDF glyph atlas
// pass (see `crate::passes::realm_labels`); the previous HTML overlay
// lived here and was removed when the SDF pipeline landed.
