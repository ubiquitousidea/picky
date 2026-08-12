# Picky: Capabilities & Technical Architecture Summary

This document provides a comprehensive technical overview of **Picky**, a non-destructive, browser-based image effects and library management application. It details the system architecture, core application capabilities, mathematical and algorithmic implementations of the visual filter pipeline, and the design and execution of the integrated agentic interface.

---

## 1. High-Level Overview & Core Capabilities

Picky is designed around non-destructive, exploratory image editing and semantic library navigation. Rather than employing a destructive linear undo/redo stack, Picky models all edits as a **branching work tree (Directed Acyclic Graph)** for every imported image.

```
                          [ Original JPG (Root) ]
                                /         \
                 [ Node 1: Tone Curve ]   [ Node 2: Hex Pixelate ]
                         |                         |
                 [ Node 3: Bokeh ]                 |
                         \                         /
                          \                       /
                       [ Node 4: Blend (DAG Merge) ]
```

### Key System Capabilities:
1. **Branching Work Tree (DAG) Engine**:
   - Every effect applied creates a child node referencing its parent node (`parent_id`).
   - Blend operations carry a secondary parent link (`parent2_id`), forming a multi-parent DAG.
   - Any historical node can be selected, previewed, branched from, re-tuned in place, or exported.
   - Node IDs are strictly topological ($parent\_id < node\_id$), enabling linear-time forward layout passes for commit-graph rendering and cycle-free validation.
   - Node deletion cascades recursively through both parent links using SQLite recursive Common Table Expressions (CTEs) under deferred foreign keys.

2. **Real-Time In-Memory Live Preview**:
   - Adjusting effect parameters immediately renders a live preview in memory via `POST /api/nodes/{id}/preview` without writing files to disk or creating database rows.
   - Preview rendering shares the exact same core pipeline (`rendering._apply`) as persisted node rendering, guaranteeing pixel-perfect fidelity between preview and committed nodes.

3. **Persistent SQLite & Multi-Tiered Cache Architecture**:
   - State and metadata are persisted in SQLite (`data/picky.db`), while binary assets are organized across dedicated directories:
     - `data/originals/`: Read-only source images (`<image_id>.jpg`).
     - `data/renders/`: Regenerable cache containing rendered node JPEGs (`<node_id>.jpg`), framed output JPEGs (`<node_id>.out.jpg`), thumbnails (`<node_id>.thumb.jpg`), posterize k-means cluster data (`<node_id>.clusters.json`), and SAM encoder embeddings (`<node_id>.embedding.npy`).
     - `data/masks/`: User data containing frozen 1-bit boolean masks (`<mask_id>.png`).
     - `data/models/`: Pinned ONNX model weight files downloaded lazily on demand.
   - File existence serves as the primary cache key. When a node's parameters are edited in place via `PATCH /api/nodes/{id}`, the node row is updated first, and then all descendant cache files are swept from disk (`db.descendant_ids` + `rendering.delete_render_files`), re-rendering the target node eagerly.

4. **Non-Destructive Output Stage (Crop & Rotate)**:
   - Crop and straightening rotations (`images.crop`) deliberately reside **outside** the work tree DAG as a final output stage applied in `rendering.render_output()`.
   - Every node in the work tree maintains the exact pixel dimensions of the original image. This guarantees that saved masks, click coordinates, and blend overlays remain valid across all nodes without coordinate warping.
   - The server publishes the closed-form inverse affine transformation matrix with each image (`crop_geometry`), enabling the frontend to map clicks on rotated/cropped previews back into native node coordinates with zero client-side trigonometry.

5. **Saved Masks & Interactive Segmentation**:
   - Subjects are isolated using a **MobileSAM** model by clicking a single point.
   - Picked selections are automatically "banked" into durable 1-bit PNG masks (`data/masks/<mask_id>.png`) when an effect is applied, detaching the mask from volatile click coordinates and preventing model drift.
   - Multiple saved masks can be combined via boolean unions to apply an effect across several distinct objects simultaneously.

6. **Portable Relative Presets**:
   - Effect chains are captured as reusable recipes (`presets`) by traversing ancestor closures.
   - Parent dependencies are converted from absolute node IDs into relative indices ($0$ representing the target base node, $1 \dots i-1$ representing earlier recipe steps).
   - Replaying a preset executes sequentially with transactional rollback: if any intermediate step fails, all newly generated nodes and orphaned render files are cleaned up.

7. **Interactive 3D Semantic Image Map**:
   - Visualizes the entire photo library in an interactive 3D point cloud rendered via an HTML5 canvas with custom orbit and inertial spin physics.
   - Uses OpenAI **CLIP ViT-B/32 (Vision Tower)** embeddings to group photos by semantic content rather than color palette.
   - Embeddings are projected to 3D space using **PCA** (for deterministic, stable incremental updates) or **t-SNE** (for tight semantic clustering).
   - Automatic library clustering with natural language labels generated via cosine similarity against a pre-encoded vocabulary of 857 terms.
   - Multi-dimensional filtering across text search queries, effect type chips, and Euclidean proximity.

---

## 2. Technical Details: Image Filter Features

Every image filter is implemented as a pure transformation function mapping an RGB `uint8` NumPy array $(\text{Height} \times \text{Width} \times 3)$ to an array of identical dimensions. Filter definitions and parameter specifications are centralized in the `EFFECTS` registry in `server/effects.py`.

```
               [ Input RGB uint8 Array ]
                           |
    +----------------------+----------------------+
    |                      |                      |
[ Standard Filters ]  [ Spatial Blur/Defocus ]  [ Deep-Learning Bokeh ]
- Posterize (K-Means)  - Gaussian Blur          - Depth Anything V2 Disparity
- Tone Curve / Gamma   - Fast 1D Disk Defocus   - Layered Gather Compositing
- Hexagonal Pixelate     (Prefix Sum Runs)      - Shift-Variant Optical Swirl
- Sobel Edges / Dither                          - Catadioptric Annular Aperture
    |                      |                    - Specular Highlight Bloom
    +----------------------+----------------------+
                           |
               [ Mask Stencil Composite ]  <--- (Optional Boolean Mask Union)
                           |
              [ Output RGB uint8 Array ]
```

### 2.1. Posterize (PCA-Whitened K-Means)
- **Algorithm**: Color quantization using `sklearn.cluster.MiniBatchKMeans`.
- **PCA Whitening**: In standard RGB space, pixel variance is overwhelmingly dominated by the luminance diagonal $\langle 1, 1, 1 \rangle$, causing standard k-means centroids to bunch along brightness levels rather than chromaticity. Picky fits a 3-component PCA over a 50,000-pixel sample and whitens the coordinates:
  $$\text{whiten}(x) = \frac{\text{PCA}.\text{transform}(x)}{\sqrt{\text{explained\_variance} + 10^{-4}}}$$
  K-means clusters in this whitened space equalize variance across principal axes, distributing clusters across distinct hues.
- **Diagnostics**: Endpoints return sampled 3D scatter points and cluster centroids (`/api/nodes/{id}/clusters`) for interactive WebGL/Canvas 3D visualization.

### 2.2. Blur (Gaussian & Fast Exact Disk Defocus)
- **Gaussian Blur**: Standard separable Gaussian filter executed via Pillow.
- **Disk Defocus Blur (`_disk_blur`)**: Simulates optical out-of-focus blur (flat circular aperture).
  - *Complexity Optimization*: Naive spatial convolution with a flat disk of radius $r$ requires $O(W \cdot H \cdot r^2)$ operations ($\sim 31{,}000$ taps per pixel at $r=100$, taking over $100\text{ s}$ on a $40\text{ MP}$ image).
  - *1D Prefix Sum Decomposition*: Picky decomposes the circular disk into horizontal scanline runs $\Delta y \in [-r, r]$ with half-widths $h_w(\Delta y) = \lfloor\sqrt{r^2 - \Delta y^2}\rfloor$. A cumulative sum array is computed along the horizontal axis:
    $$\text{Sum}(y, x) = \sum_{i=0}^{x-1} \text{Image}(y, i)$$
    Any horizontal run from $x - h_w$ to $x + h_w$ is evaluated with exactly two lookups: $\text{Sum}(y, x + h_w + 1) - \text{Sum}(y, x - h_w)$.
  - *Performance & Precision*: Reduces total complexity to $O(W \cdot H \cdot r)$ ($8.5\text{ s}$ at $40\text{ MP}$ for $r=100$). Processing is executed in vertical bands of $1{,}024$ rows with exact `int32` accumulators, eliminating float rounding and bounding peak memory.

### 2.3. Bokeh (Physics-Based Depth-of-Field Defocus)
The Bokeh filter simulates real photographic lens defocus driven by monocular depth estimation rather than artificial silhouette cutouts.

```
                           [ Input Image (40 MP) ]
                                      |
                 +--------------------+--------------------+
                 |                                         |
     [ Depth Anything V2 ONNX ]                   [ Downsample Canvas ]
     (Inverse Depth / Disparity)                 (<= 2048px, r <= 96px)
                 |                                         |
                 +--------------------+--------------------+
                                      |
                         [ Linear Depth Slicing ]
                         (4 to 12 Depth Bands)
                                      |
                     [ Per-Band Aperture Filtering ]
            - Shift-Variant Cat's Eye Lens Convolution
            - Annular Mirror Lens Defocus
            - Highlight Bloom Soft-Maximum
                                      |
                    [ Far-to-Near Alpha Composite ]
                     (Coverage Un-premultiplication)
                                      |
                     [ Full-Resolution Recombine ]
                  (Smoothstep High-Frequency Preserve)
                                      |
                    [ Physical Corner Vignetting ]
```

1. **Monocular Depth Estimation (`server/depth.py`)**:
   - Backed by **Depth Anything V2 Small** exported to ONNX ($\sim 99\text{ MB}$).
   - Preprocessing resizes the shortest image dimension to $518\text{ px}$ (snapped to patch multiples of $14$), normalized using ImageNet constants ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
   - The model natively outputs **relative inverse depth (disparity)**. Because the optical circle of confusion (CoC) of a physical lens is proportional to $|1/z_{\text{focus}} - 1/z|$, disparity is linear with respect to blur radius.
   - Outputs are normalized between the 1st and 99th percentiles to prevent specular or noise outliers from compressing scene depth. Cached in-process using BLAKE2b content hashes.

2. **Layered Gather Defocus Compositing (`_bokeh_layers`)**:
   - *The Scatter vs. Gather Problem*: Defocus in nature is a *scatter* operation (points spread light into disks). Image convolution is a *gather* operation (pixels average their neighbors). A naive global blur lets sharp foreground subjects bleed into the background, creating unnatural smearing and halos.
   - *Band Partitioning*: Depth is divided into $L \in [4, 12]$ discrete depth bands. Each pixel is partitioned between adjacent bands using linear interpolation weights ($w_k + w_{k+1} = 1$).
   - *Compositing*: Each band carries RGB premultiplied by weight and alpha coverage. Bands are individually filtered at their depth-specific radius $r_k = r_{\text{full}} \cdot \left(\frac{|d_k - \text{focus}|}{\text{span}}\right)^{\text{falloff}}$ and composited from back to front using premultiplied alpha:
     $$\text{Out} \leftarrow \text{RGB}_k + (1 - \alpha_k) \cdot \text{Out}, \quad \text{Coverage} \leftarrow \alpha_k + (1 - \alpha_k) \cdot \text{Coverage}$$
   - *Coverage Correction*: Accumulated color is un-premultiplied by final coverage to prevent boundary darkening where blur spreads faster than background fill.

3. **Aperture Modeling & Spatial Variation**:
   - **Optical Swirl / Cat's Eye (`_swirl_mean`)**:
     - Models mechanical barrel vignetting (e.g., the Helios 44-2 lens look) where the circular entrance pupil is clipped by the lens barrel into two intersecting circular arcs off-axis.
     - Parametrized by aspect ratio $t(\rho) = 1 - \text{swirl} \cdot \rho^2$, where $\rho$ is normalized radial distance from the optical center.
     - Because the kernel changes across the frame, this is a **linear shift-variant filter**. Picky evaluates it blockwise over $64 \times 64$ tiles (`_SWIRL_TILE`), reusing a single full-frame 1D prefix sum array so per-tile evaluation incurs no halo boundaries or extra allocations.
   - **True Physical Corner Vignetting (`_vignette`)**:
     - Real mechanical vignetting darkens corners because a clipped pupil admits less light. Because `_swirl_mean` normalizes each kernel by its own area, the light loss is computed analytically via circular segment integration:
       $$\frac{\text{Area}(t)}{\pi r^2} = \frac{2 r_c^2 \arccos\left(\frac{d}{2 r_c}\right) - \frac{d}{2}\sqrt{4 r_c^2 - d^2}}{\pi}$$
       where $d = r(1/t - t)$ and $r_c = \frac{r}{2}(1/t + t)$. Applied as a smooth, continuous floating-point gain ramp in 1024-row bands.
   - **Catadioptric / Mirror Lens (`_disk_mean` with Hole)**:
     - Simulates reflecting telephoto lenses with central secondary mirror obstructions, creating characteristic doughnut-shaped bokeh rings.
     - Evaluated by subtracting a secondary inner disk run of radius $r_{\text{hole}} = \text{round}(r \cdot \text{obstruction})$ directly from the primary prefix sum accumulator.

4. **Specular Highlight Bloom (`_bloom_luminance`)**:
   - Standard averaging diminishes pinpoint highlights into dim smudges. Picky applies an exponential soft-maximum over luminance:
     $$L' = 1 + \frac{\ln\left(\text{mean}\left(e^{k(L - 1)}\right)\right)}{k}$$
   - Modulates luminance only while preserving linear chromaticity ratios, avoiding color shifts on overexposed highlights.

5. **Performance Optimization (Working Canvas & Recombination)**:
   - Slices computation to a downscaled working canvas ($\le 2048\text{ px}$ on the long side, maximum working radius $\le 96\text{ px}$).
   - Recombines the blurred canvas with the full-resolution original using a smoothstep crossover mask:
     $$\text{mix} = \text{smoothstep}\left(\frac{r_{\text{full}} \cdot \text{diff}^{\text{falloff}} - \text{lo}}{\text{hi} - \text{lo}}\right)$$
   - In-focus areas retain $100\%$ original sensor resolution with zero upsampling artifacts.

6. **Diagnostics**:
   - `show: "depth"`: Direct bilinear visualization of the estimated disparity map.
   - `show: "kernels"`: Renders true-scale rasterized and eroded aperture stamps (`_aperture_stamp`) across a configurable grid, visualizing exact blur footprints, swirl deformations, and doughnut holes across the frame.

### 2.4. Tone Curve & Gamma
- **Monotone Cubic Spline Tone Curve (`_curve_lut`)**:
  - Employs **Fritsch–Carlson monotone cubic Hermite interpolation** across up to 16 user-defined control points over a 256-bin lookup table (LUT).
  - Enforces monotonicity by clamping tangents ($s = a^2 + b^2 > 9 \implies \tau = 3/\sqrt{s}$), strictly preventing spline overshooting and local contrast inversions.
  - Mirrored line-for-line in JavaScript (`web/app.js:curveLut`) for instantaneous client-side curve rendering.
- **Gamma Correction**: Fast single-parameter power-law LUT:
  $$\text{LUT}[i] = \text{round}\left(255 \cdot \left(\frac{i}{255}\right)^{1/\gamma}\right)$$

### 2.5. Pixelate (Square & Hexagonal Honeycomb)
- **Square Pixelate**: Fast area downsampling with box filtering followed by nearest-neighbor upscaling.
- **Hexagonal Pixelate (`_hex_pixelate`)**:
  - Quantizes pixels into regular pointy-top hexagonal bins (Voronoi cells of a triangular lattice).
  - Avoids spatial searching by splitting the triangular lattice into two rectangular sublattices (even rows and half-cell offset odd rows). Nearest centers are identified via 1D broadcast rounding.
  - Bin labels are assigned in 1024-row bands and colors are accumulated via exact `np.bincount` weighted sums.

### 2.6. Edge Detection & Dithering
- **Sobel Edges**: Converts image to Rec. 601 luma ($Y = 0.299R + 0.587G + 0.114B$), applies $3 \times 3$ shifted kernel convolutions for horizontal ($G_x$) and vertical ($G_y$) gradients, computes gradient magnitudes $\sqrt{G_x^2 + G_y^2}$, and applies optional thresholding.
- **Floyd–Steinberg Dither**: Color quantization with 2D error diffusion dithering via Pillow.

### 2.7. Blend (Multi-Parent Node Merging)
Combines two nodes of the same image DAG (`parent_id` and `parent2_id`) across four modes:
- **Average**: $\text{round}(A \cdot (1 - w) + B \cdot w)$
- **Additive**: $\min(255, A + B)$
- **Multiplicative**: $\lfloor(A \cdot B) / 255\rfloor$
- **Subtractive**: $\max(0, A - B)$

---

## 3. Technical Details: The Agentic Interface

The agentic interface allows users to operate Picky via natural language commands entered into a dedicated **Command Row** in the web interface.

```
+-----------------------------------------------------------------------------------+
| [ Web Frontend ]                                                                  |
| - User enters prompt in Command Row                                               |
| - Manages conversation history in `state.agent.history`                           |
| - Dispatches to `POST /api/agent`                                                 |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| [ server/agent.py (Orchestrator) ]                                                |
| - Model: Claude 3.5 Haiku (`claude-haiku-4-5`) via Anthropic Messages API        |
| - System prompt injected with dynamic `_effects_doc()` and `_context_doc()`       |
| - Conversation loop (bounded to MAX_ROUNDS = 10, trimmed to MAX_HISTORY = 20)     |
+-----------------------------------------------------------------------------------+
                                         |
                       [ Tool Execution Dispatch Loop ]
                                         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
[ Navigation / Inspection ]     [ Edits & Applications ]      [ Destructive Safety Gate ]
- `find_photos` (CLIP/Filename)  - `apply_effect`             - `delete_node`
- `show_photo`                   - `edit_node`                - `delete_photo`
- `list_photo_nodes`             - `apply_preset`             (Returns `pending` payload;
- `list_presets`                 (Executes via `main.py`)      requires GUI confirmation)
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| [ Re-entry into server/main.py Handlers ]                                         |
| Direct function calls: `create_node()`, `update_node()`, `search_embedding_map()` |
| Reuses all validation, clamping, cache invalidation, and eager render pipelines.  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| [ Response Payload to Client ]                                                    |
| - `reply`: Natural language summary                                               |
| - `steps`: Structured execution transcript with status markers                    |
| - `focus`: Target navigation object `{image_id, node_id}`                         |
| - `pending`: Interactive confirmation card payload (if deletion queued)           |
+-----------------------------------------------------------------------------------+
```

### 3.1. Core Architectural Principle: Second Front End, Not Second Implementation
The agent is designed strictly as a **second front end** over the existing API.
- The tool execution handlers in `server/agent.py` do not manipulate the database or filesystem directly; instead, they import and call the exact FastAPI endpoint handler functions in `server/main.py` (`create_node`, `update_node`, `apply_preset`, `search_embedding_map`).
- **Benefits**:
  - Parameter clamping (`validate_params`) applies identically.
  - Image ownership checks and security boundaries are shared.
  - Eager rendering, cache invalidation closures (`delete_render_files`), and model readiness checks (`_check_effect_ready`) behave identically whether triggered by a human GUI click or an AI tool call.

### 3.2. Dynamic System Prompt & Vocabulary Generation
The system prompt in `server/agent.py` is compiled dynamically on every turn:
1. **Dynamic Effect Schema (`_effects_doc()`)**: Generated directly from `effect_specs()`. Any new effect added to the Python registry is immediately exposed to the agent with its exact parameter types, numerical ranges, defaults, and choice options without manual prompt editing.
2. **Dynamic Context (`_context_doc()`)**: Injects the active state of the user's viewport, including current image ID, image filename, active node ID, and the human-readable ancestor effect chain (e.g., `original → curves → bokeh`). This resolves ambiguous references such as "make it brighter" or "blur this".

### 3.3. Complete Tool Definitions & Schema

| Tool Name | Purpose | Key Inputs | Behavior / Safety Invariants |
| :--- | :--- | :--- | :--- |
| `find_photos` | Semantic & keyword library search | `description` (str), `limit` (int) | Combines substring filename search with 512-d CLIP text-vision cosine search (`search_embedding_map`). Scores against `STRONG_MATCH = 0.275`. |
| `show_photo` | Switch UI focus | `image_id` (int), `node_id` (int, opt) | Sets `turn.focus` to update client viewport. |
| `list_photo_nodes` | Inspect DAG history | `image_id` (int) | Returns full tree structure of all versions. |
| `apply_effect` | Create and render a new child node | `image_id`, `node_id`, `effect`, `params`, `second_node_id` | Validates, calls `main.create_node`, renders eagerly, updates `turn.focus`. |
| `edit_node` | Re-tune existing node in place | `node_id` (int), `params` (dict) | Merges new params over existing params, updates DB, invalidates descendant render caches. |
| `list_presets` | List saved recipes | None | Returns saved preset names and step summaries. |
| `apply_preset` | Replay multi-step recipe | `node_id` (int), `preset_name` (str) | Replays relative DAG steps; rolls back on failure. |
| `delete_node` | Queue node deletion | `node_id` (int) | **Non-destructive**: Queues a `pending` confirmation action. |
| `delete_photo` | Queue image deletion | `image_id` (int) | **Non-destructive**: Queues a `pending` confirmation action. |

### 3.4. Safety Guardrails & Robustness Mechanisms

1. **Two-Phase Commit for Destructive Operations**:
   - `delete_node` and `delete_photo` never execute deletions directly.
   - They compute the exact blast radius (e.g., "Remove version #4 and the 2 versions built on it") and return a structured `pending` dictionary.
   - The web client intercepts `pending` and renders an explicit interactive confirmation card with a red **Delete** button and a **Cancel** button (`renderAgentPending`), ensuring users retain absolute control over data deletion.

2. **Confidence-Gated Hallucination Prevention**:
   - CLIP similarity search computes cosine scores against the library. When query matches fall below `STRONG_MATCH = 0.275`, the tool output injects an explicit system directive:
     > *"NOTE: none of these is a confident match... Do NOT edit any of them. Say what the closest photos are and ask which one they meant."*
   - Placing the refusal directive directly inside the tool result ensures the model adheres to it consistently.

3. **Explicit Handling of Selections**:
   - LLMs cannot inspect pixel coordinates or interactively click objects.
   - The system prompt explicitly instructs the agent that selections are unavailable to tools. When asked to "blur the background", the agent applies the effect to the full frame and clearly informs the user that they can restrict the effect using the UI selection picker.

4. **Asynchronous Model Download Gates**:
   - If the agent invokes an effect requiring unloaded weights (e.g., Bokeh or semantic text search), the server returns a 409 status and triggers the background download job (`depth_job.start()` or `text_job.start()`).
   - The agent catches this, informs the user of the active download percentage, and prompts them to try again shortly, avoiding hung requests.

5. **Stateless Backend with Client-Side Conversation History**:
   - The server maintains no conversational session state.
   - The browser stores `state.agent.history` and posts it with each request.
   - Before dispatching to Claude, `agent._trim()` truncates history to `MAX_HISTORY = 20` messages while preserving valid message boundaries (ensuring `tool_result` blocks are never orphaned from their preceding assistant calls).

6. **Structured Step Transcripts**:
   - Each tool execution appends a human-readable summary step (`turn.steps`).
   - The frontend renders these sequentially with bullet indicators (`·` for success, `×` for failure) alongside the agent's textual response, providing total transparency into all automated actions.

---

## 4. Summary Table of Endpoints & Component Mapping

| Subsystem | Primary Server Files | Primary Frontend Functions | Core Endpoints |
| :--- | :--- | :--- | :--- |
| **DAG Work Tree** | `server/main.py`, `server/db.py`, `server/rendering.py` | `selectImage()`, `layoutGraph()`, `renderTree()` | `GET/POST /api/images/{id}/nodes`<br>`PATCH/DELETE /api/nodes/{id}`<br>`GET /api/nodes/{id}/render` |
| **Effects Engine** | `server/effects.py`, `server/rendering.py` | `buildParamControls()`, `readParams()`, `setEffect()` | `GET /api/effects`<br>`POST /api/nodes/{id}/preview`<br>`GET /api/nodes/{id}/histogram` |
| **Bokeh & Depth** | `server/effects.py`, `server/depth.py`, `server/depth_job.py` | `buildDepthPick()`, `syncParamVisibility()` | `GET /api/nodes/{id}/depth-at`<br>`POST/GET /api/depth-model/prepare` |
| **Segmentation** | `server/sam.py`, `server/rendering.py` | `bankSelection()`, `renderSelectControls()` | `POST /api/nodes/{id}/mask`<br>`GET/POST/DELETE /api/images/{id}/masks` |
| **Presets** | `server/main.py`, `server/db.py` | `openPresetModal()`, `applyPreset()` | `GET/POST/DELETE /api/presets`<br>`POST /api/nodes/{id}/apply-preset` |
| **Output Framing** | `server/effects.py`, `server/rendering.py` | `initCropOverlay()`, `saveCrop()` | `PUT /api/images/{id}/crop`<br>`POST /api/images/{id}/crop-preview` |
| **3D Image Map** | `server/embed.py`, `server/text_embed.py`, `server/labels.py` | `openEmbedMap()`, `drawEmbedMap()` | `GET /api/embedding-map`<br>`GET /api/embedding-map/search`<br>`POST/GET /api/embedding-map/prepare` |
| **Agent Interface** | `server/agent.py`, `server/main.py` | `runAgent()`, `renderAgentPending()`, `agentLine()` | `GET/POST /api/agent` |
