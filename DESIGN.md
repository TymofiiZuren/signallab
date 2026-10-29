---
name: SignalLab
description: An inspectable charcoal and orange sentiment research workspace.
colors:
  canvas: "#111111"
  surface: "#1b1b1b"
  ink: "#f5f2ed"
  muted: "#b9b5af"
  accent: "#f6821f"
  on-accent: "#17120c"
  line: "#414141"
  interactive: "#ffa257"
  table-header: "#252525"
  positive: "#8cb5f6"
  chart-grid: "#333333"
  dot-ink: "rgb(48 24 7 / 24%)"
  dot-light: "rgb(255 238 215 / 80%)"
typography:
  display:
    fontFamily: "Signal Sans, sans-serif"
    fontSize: "clamp(2.25rem, 4.5vw, 4rem)"
    fontWeight: 500
    lineHeight: 1.05
    letterSpacing: "-0.03em"
  headline:
    fontFamily: "Signal Sans, sans-serif"
    fontSize: "2.25rem"
    fontWeight: 500
    letterSpacing: "-0.03em"
  title:
    fontFamily: "Signal Sans, sans-serif"
    fontSize: "1.5rem"
    fontWeight: 500
    letterSpacing: "-0.03em"
  body:
    fontFamily: "Signal Sans, sans-serif"
    fontSize: "16px"
  intro-body:
    fontFamily: "Signal Sans, sans-serif"
    fontSize: "1.0625rem"
    lineHeight: 1.6
rounded:
  control: "0.5rem"
  intro: "1rem"
  button: "full"
components:
  page-intro:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.on-accent}"
    rounded: "{rounded.intro}"
    padding: "2.5rem 3rem"
  page-intro-mobile:
    padding: "2rem 1.5rem"
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.on-accent}"
    rounded: "{rounded.button}"
  button-primary-hover:
    backgroundColor: "{colors.interactive}"
    textColor: "{colors.on-accent}"
---

# Design System: SignalLab

## Overview

**Creative North Star: "The inspectable research workspace"**

The user-approved Cloudflare.com reference supplies the charcoal canvas, saturated orange introduction, warm-white text, rounded actions and spacious sans typography. This direction supersedes the previous light-green visual system. SignalLab retains its own identity and operational purpose; the reference does not authorize Cloudflare branding or marketing claims.

Native Streamlit controls, real charts and evidence tables remain the working surface. The orange introduction establishes hierarchy while visible provenance and limitations keep experiments honest.

**Key Characteristics:**
- Charcoal workspace with an orange introduction per workflow.
- Locally hosted sans typography and restrained outlines.
- Native semantic controls with explicit states and sentiment labels.

## Colors

### Primary

Orange accent anchors introductions, primary actions, selection and the negative chart series. The lighter interactive orange identifies links, selected navigation and primary-button hover. Dark on-accent ink supplies text on orange surfaces.

### Secondary

Blue identifies the positive chart series. Keep explicit Positive and Negative labels alongside color; orange here identifies a class, not an error state.

### Neutral

Canvas and surface provide the dark workspace. Warm ink carries primary text, muted ink carries captions, line separates the sidebar and forms, and table-header distinguishes dataframe headers. Chart-grid is specific to Plotly axes.

**The Evidence Rule.** Color supports labels and measured data; it never substitutes for sentiment names or provenance.

## Typography

Signal Sans is the theme alias for locally hosted DM Sans (`static/fonts/DMSans-variable.ttf`, normal variable weight range 100–1000), with sans-serif fallback. Body and heading roles are defined above. Native top-level headings are 3.75rem; the orange introduction overrides that with the fluid display role. Sidebar headings are 1.75rem, 1.25rem and 1.125rem at weights 600, 500 and 500.

Headings use balanced wrapping. Introduction copy has a 65ch maximum measure; captions use a 1.6 line-height. Dataframes use tabular numbers. Plotly labels explicitly use Signal Sans at 14px.

## Layout

The wide Streamlit shell has a persistent sidebar for navigation and dataset selection. Main content is capped at 1440px with padding of 4.5rem 3rem 4rem. The introduction uses the frontmatter padding and a 1rem gap. Other widget spacing remains native Streamlit behavior; there is no custom spacing scale. The declared `--signal-space` property is unused and is not a normative spacing token.

At widths up to 1100px, horizontal blocks within main content wrap and their columns take a 100% minimum width. At widths up to 768px, main padding becomes 4.5rem 1.25rem 3rem, introduction padding becomes the mobile variant, all horizontal blocks wrap with full-width columns, and subheading top padding reduces from 1rem to 0.5rem. Native Streamlit owns sidebar collapse behavior.

## Elevation & Depth

The custom presentation adds no shadows. Tonal surfaces, outlines and the orange introduction establish hierarchy; native widget state treatments remain Streamlit-owned. The introduction reveals through a 500ms clip-path animation using cubic-bezier(0.16, 1, 0.3, 1). Reduced-motion preference disables that animation. The orange introduction also has a masked radial-dot texture (6px spacing; 5px at 768px and below), with a light-dot sweep lasting 4 seconds for one iteration. The sweep settles to invisible while the static dots remain. Reduced-motion preference hides the sweep. Both decorative layers ignore pointer events and sit behind the text; no continuous animation or JavaScript loop runs.

## Shapes

The introduction has softly rounded corners; theme controls use the control radius and native buttons use the theme's `full` radius. Borders remain visible on native widgets and forms. The SignalLab mark comprises three orange bars with heights 12px, 24px and 18px.

## Components

### Buttons

Primary actions use accent fill and dark text at weight 600; hover uses interactive orange and active returns to accent. Native action, download and link controls have a 44px minimum height and 1.5rem inline padding. Radio and checkbox labels share the 44px minimum hit height, including forms. Disabled behavior and secondary treatments remain native. The shared focus-visible rule adds a 2px interactive-orange outline with a 4px offset.

### Inputs / Fields

Selectboxes, text inputs, text areas, sliders, uploaders and forms remain native Streamlit widgets. The dark theme supplies surface, text, border and control radius; text entry uses an orange caret. Validation, empty, loading, success and stale-result messages stay beside their owning task.

### Navigation

Native sidebar radio navigation exposes Explore, Experiments, Synthetic Review Lab, Analysis, Saved runs, Case Study, Predict and Methodology. Saved runs uses an explicit Refresh action, a status table and a selected numeric result; browsing summaries does not require an attached dataset. Analysis labels its Current session or Saved experiment source explicitly. Predict labels Session baseline and Saved transformer sources; saved-model controls render independently of dataset attachment. Model selection clears prior output, loading stays beside submission, and token truncation receives a visible notice. Labels have a 44px minimum height; selected text becomes interactive orange at weight 600. Dataset selection and provenance remain in the same sidebar.

### Case Study

A linear reading flow reuses native controls, the orange introduction and ordinary unboxed sections: source/evidence, measured results, paired mistakes, optional exact-data examples, limitations and one primary Markdown download. Numeric reading works without a CSV. Source and exploratory warnings sit beside the evidence, with neutral copy for positive, negative and tied results. Review attachment is a closed opt-in section; literal examples are bounded and excluded from Markdown. The Analysis handoff retains run/partition but may require data reattachment. Read failures remove prior story/download state; attachment failures preserve numeric results. No new visual tokens or animation.

### Synthetic Review Lab

An Operate workspace inherits the orange introduction, native controls and unboxed sections. Protocol/pilot identity and status precede mandatory exact-development attachment, then original/candidate comparison in two columns (stacked original first at the shared breakpoint), a native decision form and closed evidence/CLI details. Text is literal and wraps. Human decision and automatic training eligibility remain separate labels. Pending includes excluded outputs; no verdict is preselected for a new candidate. Explicit saves show revisions; conflicts retain a separate draft and require reload. Numeric summaries do not require a CSV; source mismatch removes the pair/form. Feasibility and current/historical seal status come from validated evidence. A native sealing section follows review: it lists unmet gates or permits an explicit seal with exact attachment, pins the displayed revision, and requires reload on concurrent edits. Review edits warn that current seals become historical; a closed section shows full identities and text-free CLI inspection. No training runs from this page. No new tokens or animation; the shared 44px label target rule also covers review radios and checkboxes.

T6 verification (2026-10-29): the authored review workflow was inspected in native Chrome at actual 200% browser zoom, including keyboard focus through the decision form. Per-tab reduced-motion emulation reported `introAnimation: none`, `sweepDisplay: none` and `sweepAnimation: none`; emulation and zoom were restored afterwards. These bounded checks complement the responsive evidence; they are not a complete WCAG or assistive-technology audit. Evidence lives in `.scratch/synthetic-integration-implementation/`.

### Introduction / Containers

Each workflow has the same named orange introduction, display heading and concise supporting copy. Ordinary analysis sections remain unboxed. Dataframes keep native interaction, visible borders and dark headers; charts use transparent plot backgrounds and explicit axes.

## Do's and Don'ts

### Do:
- **Do** use native Streamlit theme settings and semantic controls before adding presentation CSS.
- **Do** retain explicit sentiment labels, provenance and measured-result caveats.
- **Do** preserve the 1100px and 768px wrapping behavior and reduced-motion override.

### Don't:
- **Don't** restore the superseded light-green palette.
- **Don't** copy Cloudflare branding or imply a relationship with Cloudflare.
- **Don't** replace working controls, charts or tables with decorative HTML imitations.
