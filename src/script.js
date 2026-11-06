// CanopyGIS Client Controller

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide icons on page load
    lucide.createIcons();

    // ----------------------------------------------------
    // 1. Initialize Map & Basemaps with Pixel Zoom
    // ----------------------------------------------------
    // Ramintra Sports Park (สวนกีฬารามอินทรา), Bang Khen, Bangkok — primary thesis site
    const defaultCenter = [13.878551, 100.619507];
    const map = L.map('map', {
        center: defaultCenter,
        zoom: 18,
        maxZoom: 23,
        zoomControl: true,
        attributionControl: true
    });

    // maxNativeZoom = last zoom that still fetches real tiles.
    // Google Satellite serves native VHR imagery over Bangkok up to z22 (3.6 cm/px at scale=2);
    // Leaflet overzooms cleanly to z23 when inspecting sub-crown polygons.
    const PIXEL_ZOOM_CAP = 23;
    const ESRI_NATIVE_MAX = 19;
    const GOOGLE_NATIVE_MAX = 22;
    const OSM_NATIVE_MAX = 19;

    const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: PIXEL_ZOOM_CAP,
        maxNativeZoom: OSM_NATIVE_MAX,
        attribution: '© OpenStreetMap contributors'
    });

    const esriSatellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        maxZoom: PIXEL_ZOOM_CAP,
        maxNativeZoom: ESRI_NATIVE_MAX,
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    });

    // lyrs=s = pure satellite (better overzoom continuity than hybrid labels)
    const googleSatellite = L.tileLayer('https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
        maxZoom: PIXEL_ZOOM_CAP,
        maxNativeZoom: GOOGLE_NATIVE_MAX,
        attribution: 'Map data &copy; Google'
    });

    // Default: Google satellite — denser VHR coverage over Bangkok than Esri blanks
    googleSatellite.addTo(map);

    const baseMaps = {
        google: googleSatellite,
        esri: esriSatellite,
        osm: osm
    };

    map.createPane('denseCandidatesPane');
    map.getPane('denseCandidatesPane').style.zIndex = 460;
    map.getPane('denseCandidatesPane').style.pointerEvents = 'auto';
    const denseCandidateRenderer = L.canvas({
        pane: 'denseCandidatesPane',
        padding: 0.5,
        tolerance: 8
    });

    // Layers for overlays
    let boundsLayer = null;
    let imageOverlayLayer = null;
    let canopyGeoJsonLayer = null;
    let canopyCoverageLayer = null;
    let canopyCoverageCentroidsLayer = null;
    let crownGroupsLayer = null;
    let denseCandidatesLayer = null;
    let denseCandidateCentroidsLayer = null;
    let centroidsLayer = L.layerGroup().addTo(map);
    let lastFilteredFeatures = [];

    // ----------------------------------------------------
    // 2. Global State Variables
    // ----------------------------------------------------
    let uploadedFilePath = null;
    let uploadedMetadata = null;
    let extractedGeoJson = null;
    let activeHighlightId = null;
    let selectedTreeId = null;
    let isDrawingActive = false;
    let currentRoiPolygon = null;
    let previousRoiPolygon = null;
    let drawnVertices = [];
    let drawVertexGroup = L.layerGroup().addTo(map);
    let drawPreviewLayer = null;

    const STORAGE_KEY_BOUNDARY = 'canopygis_saved_boundary_v1';
    const STORAGE_KEY_PRESET_MAP = 'canopygis_preset_boundaries_v1';

    const OFFICIAL_ONNUT_POLYGON = [[13.70829,100.684747],[13.70825,100.684611],[13.708182,100.684449],[13.708356,100.684344],[13.70838,100.684316],[13.708762,100.684089],[13.709221,100.683839],[13.709357,100.683807],[13.709615,100.683733],[13.709682,100.683721],[13.710076,100.683603],[13.710581,100.683328],[13.710681,100.68324],[13.711068,100.682819],[13.711354,100.682533],[13.71159,100.682345],[13.711865,100.68213],[13.712174,100.681938],[13.71253,100.681674],[13.712636,100.681565],[13.712762,100.681438],[13.712852,100.681305],[13.713115,100.680937],[13.713198,100.680812],[13.712927,100.680422],[13.71377,100.678154],[13.714078,100.677545],[13.714359,100.677062],[13.714468,100.677194],[13.714829,100.677587],[13.715253,100.677983],[13.715659,100.678286],[13.715837,100.678397],[13.716277,100.678575],[13.716687,100.678733],[13.716954,100.678808],[13.717008,100.678992],[13.716786,100.679037],[13.716431,100.67915],[13.716167,100.679258],[13.71604,100.679389],[13.71669,100.683076],[13.71673,100.683249],[13.716868,100.683712],[13.716926,100.683883],[13.71708,100.684293],[13.717115,100.684313],[13.717266,100.684273],[13.717893,100.684143],[13.718036,100.68464],[13.718065,100.684735],[13.718186,100.685161],[13.718291,100.685499],[13.718366,100.685481],[13.718482,100.685461],[13.718644,100.685431],[13.718634,100.685317],[13.71857,100.684663],[13.718771,100.684653],[13.719065,100.684654],[13.719312,100.684652],[13.719405,100.685329],[13.719492,100.686087],[13.719519,100.686636],[13.719517,100.686739],[13.719521,100.686847],[13.719524,100.687033],[13.719533,100.687502],[13.719047,100.687516],[13.719054,100.687616],[13.719075,100.687795],[13.719147,100.688243],[13.719192,100.688569],[13.719361,100.689978],[13.719524,100.689996],[13.719779,100.690004],[13.720703,100.689957],[13.720712,100.690161],[13.719789,100.690207],[13.719503,100.690198],[13.719316,100.690177],[13.719258,100.690238],[13.717501,100.690362],[13.71589,100.690527],[13.715096,100.690553],[13.71448,100.690654],[13.713314,100.690917],[13.713467,100.690363],[13.713336,100.690251],[13.713254,100.689929],[13.713133,100.689515],[13.713062,100.689271],[13.712879,100.688611],[13.712338,100.688641],[13.711967,100.687544],[13.711819,100.687814],[13.711499,100.687657],[13.711019,100.687429],[13.710739,100.687289],[13.710057,100.686952],[13.710491,100.686034],[13.708961,100.685152],[13.708756,100.684988]];

    const OFFICIAL_NONGKHAEM_POLYGON = [[13.717297,100.357709],[13.717962,100.357633],[13.717959,100.357206],[13.71792,100.357178],[13.717337,100.357223],[13.71735,100.356508],[13.717338,100.356374],[13.717298,100.355931],[13.718266,100.355605],[13.719362,100.355234],[13.719,100.353917],[13.71882,100.353199],[13.719429,100.352983],[13.720792,100.352665],[13.721949,100.352493],[13.722028,100.352507],[13.722117,100.352602],[13.722174,100.352817],[13.722506,100.354059],[13.722759,100.355001],[13.72288,100.355112],[13.722978,100.355354],[13.723088,100.35563],[13.723178,100.355918],[13.723269,100.356027],[13.72326,100.35637],[13.723509,100.356525],[13.723546,100.356626],[13.723652,100.356913],[13.72254,100.357081],[13.721866,100.357263],[13.721587,100.357336],[13.72131,100.357414],[13.721084,100.357472],[13.720775,100.357559],[13.721185,100.35897],[13.721317,100.359528],[13.721511,100.360285],[13.721801,100.361371],[13.721228,100.361335],[13.72054,100.361298],[13.720889,100.362215],[13.721467,100.36376],[13.721298,100.363708],[13.72061,100.363704],[13.720233,100.363718],[13.720101,100.363338],[13.719811,100.3625],[13.719752,100.362329],[13.71963,100.361977],[13.71939,100.361283],[13.718859,100.361281],[13.718559,100.36128],[13.71855,100.361423],[13.718489,100.36143],[13.717742,100.36128],[13.717555,100.361243],[13.717627,100.361637],[13.717661,100.361659],[13.717725,100.36193],[13.71816,100.363759],[13.717942,100.363747],[13.71766,100.362583],[13.717479,100.362629],[13.717418,100.36235],[13.717178,100.361293],[13.717131,100.360971],[13.717211,100.3606],[13.716604,100.360856],[13.716439,100.360898],[13.71648,100.361941],[13.715406,100.362055],[13.715079,100.362089],[13.714156,100.362256],[13.714151,100.36168],[13.713116,100.361691],[13.712882,100.361436],[13.712857,100.360247],[13.713632,100.360099],[13.714303,100.36004],[13.714225,100.35945],[13.714209,100.359274],[13.714182,100.358633],[13.714203,100.35818],[13.714634,100.358151],[13.715832,100.358031],[13.716756,100.357953],[13.716823,100.359069],[13.715351,100.359296],[13.715459,100.360078],[13.717243,100.359758],[13.717149,100.358627],[13.717217,100.357956]];

    const OFFICIAL_RAMINTRA_POLYGON = [
        [13.8811468, 100.616371],
        [13.8790533, 100.6165051],
        [13.8775847, 100.6164408],
        [13.877642, 100.6180555],
        [13.8778503, 100.621596],
        [13.8790429, 100.621891],
        [13.8798918, 100.6225187],
        [13.8804282, 100.6228298],
        [13.8810479, 100.6229478],
        [13.8808812, 100.6192893]
    ];

    const DEFAULT_SAVED_BOUNDARY = {
        sitePreset: 'onnut',
        centerLat: 13.714447,
        centerLon: 100.683989,
        width: 1540,
        length: 1440,
        roiPolygon: OFFICIAL_ONNUT_POLYGON
    };

    // Distinct Crown Palette (10 high-contrast colors for adjacent-different graph coloring)
    const DISTINCT_CROWN_PALETTE = [
        '#10b981', '#3b82f6', '#f59e0b', '#f43f5e', '#8b5cf6',
        '#06b6d4', '#f97316', '#84cc16', '#ec4899', '#14b8a6'
    ];
    let polygonColorMode = 'distinct'; // 'distinct' (default) or 'confidence'

    // Sort & Filter state
    let currentSortCol = null;
    let currentSortDir = 'none'; // 'asc', 'desc', 'none'
    const filters = {
        tree_id: '',
        centroid: '',
        area_m2: '',
        crown_diameter_m: '',
        estimated_height_m: '',
        estimated_dbh_cm: '',
        estimated_volume_m3: '',
        confidence_pct: ''
    };

    // ----------------------------------------------------
    // 2B. Custom Map Layers & Display Panel Controls
    // ----------------------------------------------------
    const mapLayerPanel = document.getElementById('map-layer-panel');
    const mapLayerHeaderToggle = document.getElementById('map-layer-header-toggle');
    const basemapPills = document.querySelectorAll('.basemap-pill');
    const colormodePills = document.querySelectorAll('.colormode-pill');
    const chkLayerPolygons = document.getElementById('chk-layer-polygons');
    const chkLayerPoints = document.getElementById('chk-layer-points');
    const chkLayerFootprint = document.getElementById('chk-layer-footprint');
    const chkLayerCandidates = document.getElementById('chk-layer-candidates');
    const chkLayerGroups = document.getElementById('chk-layer-groups');
    const chkLayerBounds = document.getElementById('chk-layer-bounds');
    const chkLayerRaster = document.getElementById('chk-layer-raster');
    const sliderPolyOpacity = document.getElementById('slider-poly-opacity');
    const valPolyOpacity = document.getElementById('val-poly-opacity');
    const sliderMinConf = document.getElementById('slider-min-conf');
    const valMinConf = document.getElementById('val-min-conf');
    const sliderMinRelief = document.getElementById('slider-min-relief');
    const valMinRelief = document.getElementById('val-min-relief');
    const sliderMaxCyan = document.getElementById('slider-max-cyan');
    const valMaxCyan = document.getElementById('val-max-cyan');
    const sliderLiveMinArea = document.getElementById('slider-live-min-area');
    const valLiveMinArea = document.getElementById('val-live-min-area');
    const sliderLiveMaxArea = document.getElementById('slider-live-max-area');
    const valLiveMaxArea = document.getElementById('val-live-max-area');
    const engineFilterPills = document.querySelectorAll('.engine-pill');
    const btnResetLiveFilters = document.getElementById('btn-reset-live-filters');
    const liveFilterCountBadge = document.getElementById('live-filter-count-badge');
    const activeSourceEngines = new Set(['sam3', 'deepforest_2.1', 'compact_watershed']);

    if (mapLayerPanel && typeof L !== 'undefined' && L.DomEvent) {
        L.DomEvent.disableClickPropagation(mapLayerPanel);
        L.DomEvent.disableScrollPropagation(mapLayerPanel);
    }

    if (mapLayerHeaderToggle && mapLayerPanel) {
        mapLayerHeaderToggle.addEventListener('click', () => {
            mapLayerPanel.classList.toggle('collapsed');
        });
    }

    basemapPills.forEach(pill => {
        pill.addEventListener('click', () => {
            basemapPills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            const key = pill.getAttribute('data-basemap');
            Object.values(baseMaps).forEach(layer => {
                if (map.hasLayer(layer)) map.removeLayer(layer);
            });
            const selectedBase = baseMaps[key] || googleSatellite;
            selectedBase.addTo(map);
        });
    });

    colormodePills.forEach(pill => {
        pill.addEventListener('click', () => {
            colormodePills.forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            const mode = pill.getAttribute('data-colormode');
            polygonColorMode = mode === 'confidence' ? 'confidence' : 'distinct';
            updateMapLayers(lastFilteredFeatures);
        });
    });

    function syncLayerToggle(layer, checkbox) {
        if (!layer || !checkbox) return;
        if (checkbox.checked) {
            if (!map.hasLayer(layer)) map.addLayer(layer);
        } else {
            if (map.hasLayer(layer)) map.removeLayer(layer);
        }
    }

    function syncAllOverlayToggles() {
        syncLayerToggle(canopyGeoJsonLayer, chkLayerPolygons);
        syncLayerToggle(centroidsLayer, chkLayerPoints);
        syncLayerToggle(canopyCoverageLayer, chkLayerFootprint);
        syncLayerToggle(denseCandidatesLayer, chkLayerCandidates);
        syncLayerToggle(crownGroupsLayer, chkLayerGroups);
        syncLayerToggle(boundsLayer, chkLayerBounds);
        syncLayerToggle(imageOverlayLayer, chkLayerRaster);
    }

    if (chkLayerPolygons) {
        chkLayerPolygons.addEventListener('change', () => {
            syncLayerToggle(canopyGeoJsonLayer, chkLayerPolygons);
            updateMapLayers(lastFilteredFeatures);
        });
    }
    if (chkLayerPoints) {
        chkLayerPoints.addEventListener('change', () => {
            syncLayerToggle(centroidsLayer, chkLayerPoints);
            updateMapLayers(lastFilteredFeatures);
        });
    }
    if (chkLayerFootprint) {
        chkLayerFootprint.addEventListener('change', () => {
            syncLayerToggle(canopyCoverageLayer, chkLayerFootprint);
        });
    }
    if (chkLayerCandidates) {
        chkLayerCandidates.addEventListener('change', () => {
            syncLayerToggle(denseCandidatesLayer, chkLayerCandidates);
        });
    }
    if (chkLayerGroups) {
        chkLayerGroups.addEventListener('change', () => {
            syncLayerToggle(crownGroupsLayer, chkLayerGroups);
        });
    }
    if (chkLayerBounds) {
        chkLayerBounds.addEventListener('change', () => {
            syncLayerToggle(boundsLayer, chkLayerBounds);
        });
    }
    if (chkLayerRaster) {
        chkLayerRaster.addEventListener('change', () => {
            syncLayerToggle(imageOverlayLayer, chkLayerRaster);
        });
    }
    if (sliderPolyOpacity) {
        sliderPolyOpacity.addEventListener('input', (e) => {
            if (valPolyOpacity) valPolyOpacity.innerText = `${e.target.value}%`;
            updateMapLayers(lastFilteredFeatures);
        });
    }
    if (sliderMinConf) {
        sliderMinConf.addEventListener('input', (e) => {
            if (valMinConf) valMinConf.innerText = `${e.target.value}%`;
            sortAndRenderTable();
        });
    }
    if (sliderMinRelief) {
        sliderMinRelief.addEventListener('input', (e) => {
            if (valMinRelief) valMinRelief.innerText = `${e.target.value}%`;
            sortAndRenderTable();
        });
    }
    if (sliderMaxCyan) {
        sliderMaxCyan.addEventListener('input', (e) => {
            if (valMaxCyan) valMaxCyan.innerText = `${e.target.value}%`;
            sortAndRenderTable();
        });
    }
    if (sliderLiveMinArea) {
        sliderLiveMinArea.addEventListener('input', (e) => {
            if (valLiveMinArea) valLiveMinArea.innerText = `${e.target.value} m²`;
            sortAndRenderTable();
        });
    }
    if (sliderLiveMaxArea) {
        sliderLiveMaxArea.addEventListener('input', (e) => {
            if (valLiveMaxArea) valLiveMaxArea.innerText = `${e.target.value} m²`;
            sortAndRenderTable();
        });
    }
    engineFilterPills.forEach(pill => {
        pill.addEventListener('click', () => {
            const eng = pill.getAttribute('data-engine');
            if (!eng) return;
            if (activeSourceEngines.has(eng)) {
                if (activeSourceEngines.size > 1) {
                    activeSourceEngines.delete(eng);
                    pill.classList.remove('active');
                }
            } else {
                activeSourceEngines.add(eng);
                pill.classList.add('active');
            }
            sortAndRenderTable();
        });
    });

    function resetLivePostFilters() {
        if (sliderMinConf) {
            sliderMinConf.value = 0;
            if (valMinConf) valMinConf.innerText = '0%';
        }
        if (sliderMinRelief) {
            sliderMinRelief.value = 0;
            if (valMinRelief) valMinRelief.innerText = '0%';
        }
        if (sliderMaxCyan) {
            sliderMaxCyan.value = 0;
            if (valMaxCyan) valMaxCyan.innerText = '0%';
        }
        if (sliderLiveMinArea) {
            sliderLiveMinArea.value = 0;
            if (valLiveMinArea) valLiveMinArea.innerText = '0 m²';
        }
        if (sliderLiveMaxArea) {
            sliderLiveMaxArea.value = 1000;
            if (valLiveMaxArea) valLiveMaxArea.innerText = '1000 m²';
        }
        activeSourceEngines.clear();
        ['sam3', 'deepforest_2.1', 'compact_watershed'].forEach(e => activeSourceEngines.add(e));
        engineFilterPills.forEach(p => p.classList.add('active'));
    }

    if (btnResetLiveFilters) {
        btnResetLiveFilters.addEventListener('click', () => {
            resetLivePostFilters();
            sortAndRenderTable();
        });
    }

    // ----------------------------------------------------
    // 3. Selectors & DOM elements
    // ----------------------------------------------------
    // Tab switching buttons
    const tabBtns = document.querySelectorAll('.sidebar-tabs .sidebar-tab-btn');
    const tabPanes = document.querySelectorAll('.sidebar-content .tab-pane');

    // Source Imagery Selector & Panels
    const tabBtnUpload = document.getElementById('tab-btn-upload');
    const tabBtnSatellite = document.getElementById('tab-btn-satellite');
    const uploadPanel = document.getElementById('upload-panel');
    const satellitePanel = document.getElementById('satellite-panel');

    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const fileNameEl = document.getElementById('file-name');
    const fileDimsEl = document.getElementById('file-dims');
    const btnRemoveFile = document.getElementById('btn-remove-file');

    // BBox Coordinates Inputs
    const inputLat = document.getElementById('input-lat');
    const inputLon = document.getElementById('input-lon');
    const inputWidth = document.getElementById('input-width');
    const inputLength = document.getElementById('input-length');
    const btnDrawBBox = document.getElementById('btn-draw-bbox');
    const btnFetchSatellite = document.getElementById('btn-fetch-satellite');

    // UTM (N/E) Inputs
    const coordFormatLatlngBtn = document.getElementById('coord-format-latlng');
    const coordFormatUtmBtn = document.getElementById('coord-format-utm');
    const latlngInputsPanel = document.getElementById('latlng-inputs-panel');
    const utmInputsPanel = document.getElementById('utm-inputs-panel');
    const inputUtmZone = document.getElementById('input-utm-zone');
    const inputUtmHemi = document.getElementById('input-utm-hemi');
    const inputUtmNorthing = document.getElementById('input-utm-northing');
    const inputUtmEasting = document.getElementById('input-utm-easting');

    // Model Selection & Sliders (Canopy V2)
    const modelSelect = document.getElementById('model-select');
    const confGroup = document.getElementById('conf-group');
    const confSlider = document.getElementById('conf-slider');
    const confVal = document.getElementById('conf-val');

    const variSlider = document.getElementById('vari-slider');
    const variVal = document.getElementById('vari-val');
    const tileSlider = document.getElementById('v2-tile-slider');
    const tileVal = document.getElementById('v2-tile-val');
    const minAreaSlider = document.getElementById('min-area-slider');
    const minAreaVal = document.getElementById('min-area-val');
    const maxAreaSlider = document.getElementById('max-area-slider');
    const maxAreaVal = document.getElementById('max-area-val');
    const heightSlider = document.getElementById('height-slider');
    const heightVal = document.getElementById('height-val');

    const btnExtract = document.getElementById('btn-extract');
    const btnExportCsv = document.getElementById('btn-export-csv');
    const btnExportGeoJson = document.getElementById('btn-export-geojson');

    // Presets, Help, and Log elements
    const btnToggleHelp = document.getElementById('btn-toggle-help');
    const settingsHelpCard = document.getElementById('settings-help-card');
    const logPanel = document.getElementById('log-panel');
    const logContent = document.getElementById('log-content');

    function addLog(msg) {
        if (!logPanel || !logContent) return;
        logPanel.classList.remove('hidden');
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        logContent.innerHTML += `[${timestamp}] ${msg}\n`;
        logPanel.scrollTop = logPanel.scrollHeight;
    }

    if (btnToggleHelp && settingsHelpCard) {
        btnToggleHelp.addEventListener('click', () => {
            settingsHelpCard.classList.toggle('hidden');
        });
    }

    const V2_AUTO_SUMMARY = 'SAM 3 (Single Crowns) + DeepForest 2.1 (Closed-Canopy Nadir Detector) + Global Compact Geodesic Watershed · conf 0.18 · RGB 0.43 · Tile 1280px';

    let isApplyingAutoTune = false;
    let currentAutoTuneProfile = null;

    function setSlider(slider, valEl, value) {
        if (!slider) return;
        slider.value = value;
        if (valEl) valEl.innerText = value;
        slider.dispatchEvent(new Event('input'));
    }

    function renderAutoTuneCardState(isAuto, summaryText) {
        const card = document.getElementById('auto-tune-card');
        const titleEl = document.getElementById('auto-tune-title');
        const pillEl = document.getElementById('auto-tune-mode-pill');
        const labelEl = document.getElementById('chk-auto-tune-label');
        const summary = document.getElementById('auto-tune-summary');
        const btnAuto = document.getElementById('btn-mode-auto-tune');
        const btnManual = document.getElementById('btn-mode-manual-override');

        if (isAuto) {
            if (card) {
                card.style.background = 'rgba(92, 131, 83, 0.08)';
                card.style.borderColor = 'rgba(92, 131, 83, 0.3)';
            }
            if (titleEl) {
                titleEl.innerText = 'Auto tuning';
                titleEl.style.color = 'var(--primary)';
            }
            if (pillEl) {
                pillEl.innerText = 'SCENE AUTO';
                pillEl.style.background = 'rgba(92, 131, 83, 0.18)';
                pillEl.style.color = 'var(--primary)';
            }
            if (btnAuto) {
                btnAuto.style.background = 'var(--primary)';
                btnAuto.style.color = '#ffffff';
                btnAuto.style.borderColor = 'var(--primary)';
            }
            if (btnManual) {
                btnManual.style.background = '#ffffff';
                btnManual.style.color = 'var(--text-secondary)';
                btnManual.style.borderColor = 'var(--border-color)';
            }
            if (labelEl) labelEl.innerText = 'Auto On';
            if (summary && summaryText) summary.innerText = summaryText;
        } else {
            if (card) {
                card.style.background = 'rgba(217, 119, 6, 0.09)';
                card.style.borderColor = 'rgba(217, 119, 6, 0.45)';
            }
            if (titleEl) {
                titleEl.innerText = 'Manual Override Active';
                titleEl.style.color = '#b45309';
            }
            if (pillEl) {
                pillEl.innerText = 'MANUAL OVERRIDE';
                pillEl.style.background = 'rgba(217, 119, 6, 0.18)';
                pillEl.style.color = '#b45309';
            }
            if (btnAuto) {
                btnAuto.style.background = '#ffffff';
                btnAuto.style.color = 'var(--text-secondary)';
                btnAuto.style.borderColor = 'var(--border-color)';
            }
            if (btnManual) {
                btnManual.style.background = '#d97706';
                btnManual.style.color = '#ffffff';
                btnManual.style.borderColor = '#d97706';
            }
            if (labelEl) labelEl.innerText = 'Auto Off';
            if (summary) {
                summary.innerText = summaryText || `Custom parameters locked: conf ${confSlider.value} · RGB ${variSlider.value} · Tile ${tileSlider.value}px · Area ${minAreaSlider.value}–${maxAreaSlider.value}m² (click 'Scene Auto-Tune' to restore)`;
            }
        }
    }

    function applyAutoTune(profileOverride = null) {
        const prof = (profileOverride && typeof profileOverride === 'object')
            ? profileOverride
            : currentAutoTuneProfile;
        const confStr = prof && prof.v2_instance_threshold != null
            ? Number(prof.v2_instance_threshold).toFixed(2)
            : '0.18';
        const rgbStr = prof && prof.v2_coverage_threshold != null
            ? Number(prof.v2_coverage_threshold).toFixed(2)
            : '0.43';
        const tileStr = prof && prof.v2_tile_size != null
            ? String(Math.round(Number(prof.v2_tile_size)))
            : '1280';
        const minAreaStr = prof && prof.min_area_m2 != null
            ? Number(prof.min_area_m2).toFixed(1)
            : '0.5';
        const maxAreaStr = prof && prof.max_area_m2 != null
            ? String(Math.round(Number(prof.max_area_m2)))
            : '380';
        const summaryText = (prof && prof.summary_text)
            ? prof.summary_text
            : V2_AUTO_SUMMARY;

        isApplyingAutoTune = true;
        try {
            setSlider(confSlider, confVal, confStr);
            setSlider(variSlider, variVal, rgbStr);
            setSlider(tileSlider, tileVal, tileStr);
            setSlider(minAreaSlider, minAreaVal, minAreaStr);
            setSlider(maxAreaSlider, maxAreaVal, maxAreaStr);
        } finally {
            isApplyingAutoTune = false;
        }
        renderAutoTuneCardState(true, summaryText);
        addLog(`Auto tuning: ${summaryText}`);
    }

    const chkAutoTune = document.getElementById('chk-auto-tune');
    const btnModeAutoTune = document.getElementById('btn-mode-auto-tune');
    const btnModeManualOverride = document.getElementById('btn-mode-manual-override');

    function activateManualOverrideMode() {
        if (chkAutoTune) chkAutoTune.checked = false;
        renderAutoTuneCardState(false);
        const adv = document.getElementById('advanced-settings');
        const btnAdv = document.getElementById('btn-toggle-advanced');
        if (adv && adv.classList.contains('hidden')) {
            adv.classList.remove('hidden');
            if (btnAdv) btnAdv.innerText = 'Hide advanced settings';
        }
        addLog('Switched to Manual Override mode — custom slider values below will be used for extraction.');
    }

    if (chkAutoTune) {
        chkAutoTune.addEventListener('change', () => {
            if (chkAutoTune.checked) {
                applyAutoTune(currentAutoTuneProfile);
            } else {
                activateManualOverrideMode();
            }
        });
    }
    if (btnModeAutoTune) {
        btnModeAutoTune.addEventListener('click', () => {
            if (chkAutoTune) chkAutoTune.checked = true;
            applyAutoTune(currentAutoTuneProfile);
        });
    }
    if (btnModeManualOverride) {
        btnModeManualOverride.addEventListener('click', () => {
            activateManualOverrideMode();
        });
    }

    // Touching advanced sliders turns Auto off so user overrides stick
    [confSlider, variSlider, tileSlider, minAreaSlider, maxAreaSlider, heightSlider].forEach(el => {
        if (!el) return;
        const disableAutoOnManualTweak = () => {
            if (isApplyingAutoTune) return;
            const wasAuto = Boolean(chkAutoTune && chkAutoTune.checked);
            if (chkAutoTune) chkAutoTune.checked = false;
            renderAutoTuneCardState(false);
            if (wasAuto) {
                addLog('Manual Canopy parameter tweak detected — switched to Manual Override mode.');
            }
        };
        el.addEventListener('input', disableAutoOnManualTweak);
        el.addEventListener('change', disableAutoOnManualTweak);
    });

    const btnToggleAdvanced = document.getElementById('btn-toggle-advanced');
    const advancedSettings = document.getElementById('advanced-settings');
    if (btnToggleAdvanced && advancedSettings) {
        btnToggleAdvanced.addEventListener('click', () => {
            const open = advancedSettings.classList.toggle('hidden') === false;
            btnToggleAdvanced.innerText = open
                ? 'Hide advanced settings'
                : 'Advanced settings (optional)';
        });
    }

    // Bottom Table & Metrics Drawer
    const metricsDrawer = document.getElementById('metrics-drawer');
    const drawerToggle = document.getElementById('drawer-toggle');
    const canopyCountBadge = document.getElementById('canopy-count-badge') || document.getElementById('canopy-count');
    const canopyTableBody = document.querySelector('#canopy-table tbody');
    const systemStatusEl = document.getElementById('system-status');
    const statusIndicator = document.querySelector('.status-indicator');

    // Details Drawer & Polaroid Elements
    const detailDrawer = document.getElementById('detail-drawer');
    const btnCloseDrawer = document.getElementById('btn-close-drawer');
    const polaroidCanvas = document.getElementById('polaroid-canvas');
    const polaroidIdText = document.getElementById('polaroid-id-text');
    const drawerTreeIdBadge = document.getElementById('drawer-tree-id-badge');
    const drawerNdviVal = document.getElementById('drawer-ndvi-val');
    const drawerNdviBar = document.getElementById('drawer-ndvi-bar');
    const drawerHealthLabel = document.getElementById('drawer-health-label');
    const drawerVolumeVal = document.getElementById('drawer-volume-val');
    const drawerVolumeBar = document.getElementById('drawer-volume-bar');
    const drawerSourceModel = document.getElementById('drawer-source-model');
    const drawerLat = document.getElementById('drawer-lat');
    const drawerLng = document.getElementById('drawer-lng');
    const drawerArea = document.getElementById('drawer-area');
    const drawerDiameter = document.getElementById('drawer-diameter');
    const drawerHeight = document.getElementById('drawer-height');
    const drawerDbh = document.getElementById('drawer-dbh');
    const drawerRelief = document.getElementById('drawer-relief');
    const drawerFoliage = document.getElementById('drawer-foliage');
    const drawerCyan = document.getElementById('drawer-cyan');

    // Sorting & Filtering DOM Elements
    const btnToggleFilters = document.getElementById('btn-toggle-filters');
    const filterRow = document.getElementById('filter-row');
    const btnClearFilters = document.getElementById('btn-clear-filters');
    const filterInputs = document.querySelectorAll('.filter-input');
    const tableHeaderCols = document.querySelectorAll('#canopy-table thead tr.header-row th[data-col]');

    const BACKEND_URL = (
        typeof window !== 'undefined' &&
        window.location &&
        window.location.protocol.startsWith('http') &&
        window.location.port !== '5500' &&
        window.location.port !== '5501'
    )
        ? window.location.origin
        : 'http://127.0.0.1:8000';

    // ----------------------------------------------------
    // 4. Connect to Backend & Check Device Status
    // ----------------------------------------------------
    async function checkBackendStatus() {
        try {
            const response = await fetch(`${BACKEND_URL}/health/status`, {
                headers: { 'Accept': 'application/json' }
            });
            const data = await response.json();
            if (data.status === 'running') {
                statusIndicator.className = 'status-indicator online';
                systemStatusEl.innerText = `Backend Connected (${data.device.toUpperCase()})`;
                checkSamModelStatus();
            }
        } catch (error) {
            statusIndicator.className = 'status-indicator offline';
            systemStatusEl.innerText = 'Backend Offline';
            console.error('Failed to connect to FastAPI backend:', error);

            const samEl = document.getElementById('model-sam');
            if (samEl) { samEl.className = 'token-status error'; samEl.innerText = 'Offline'; }
        }
    }

    async function checkSamModelStatus() {
        const samEl = document.getElementById('model-sam');
        if (!samEl) return;
        samEl.className = 'token-status checking';
        samEl.innerText = 'Checking...';
        try {
            const response = await fetch(`${BACKEND_URL}/health/models`);
            const h = await response.json();
            if (h.samgeo && h.deepforest) {
                samEl.className = 'token-status active';
                samEl.innerText = 'SAM3 + DF Ready';
                samEl.title = 'SAM 3 + Weecology DeepForest 2.1 + Compact Geodesic Watershed';
            } else if (h.samgeo) {
                samEl.className = 'token-status active';
                samEl.innerText = 'SAM3 Ready';
                samEl.title = h.message || 'SAM 3 + Compact Geodesic Watershed ready';
            } else if (h.deepforest) {
                samEl.className = 'token-status active';
                samEl.innerText = 'DF + Watershed';
                samEl.title = h.message || 'Weecology DeepForest 2.1 + Compact Geodesic Watershed ready';
            } else {
                samEl.className = 'token-status checking';
                samEl.innerText = 'RGB Fallback';
                samEl.title = h.message || 'SAM 3 unavailable; Canopy V2.2 running with RGB + Compact Geodesic Watershed support';
            }
        } catch (e) {
            samEl.className = 'token-status error';
            samEl.innerText = 'Error';
        }
    }
    const checkModelsHealth = checkSamModelStatus;

    checkBackendStatus();

    // ----------------------------------------------------
    // 5. Sidebar Tabs Navigation Handlers
    // ----------------------------------------------------
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const tabName = btn.getAttribute('data-tab');
            document.getElementById(`tab-pane-${tabName}`).classList.add('active');
        });
    });

    // Sub-Imagery upload vs fetch satellite panels switcher
    tabBtnUpload.addEventListener('click', () => {
        tabBtnUpload.classList.add('active');
        tabBtnSatellite.classList.remove('active');
        uploadPanel.classList.remove('hidden');
        satellitePanel.classList.add('hidden');
    });

    tabBtnSatellite.addEventListener('click', () => {
        tabBtnSatellite.classList.add('active');
        tabBtnUpload.classList.remove('active');
        satellitePanel.classList.remove('hidden');
        uploadPanel.classList.add('hidden');
    });

    // ----------------------------------------------------
    // Coordinate Format Switcher & UTM Conversion Math
    // ----------------------------------------------------
    coordFormatLatlngBtn.addEventListener('click', () => {
        coordFormatLatlngBtn.classList.add('active');
        coordFormatUtmBtn.classList.remove('active');
        latlngInputsPanel.classList.remove('hidden');
        utmInputsPanel.classList.add('hidden');
        syncUtmToLatLng();
    });

    coordFormatUtmBtn.addEventListener('click', () => {
        coordFormatUtmBtn.classList.add('active');
        coordFormatLatlngBtn.classList.remove('active');
        utmInputsPanel.classList.remove('hidden');
        latlngInputsPanel.classList.add('hidden');
        syncLatLngToUtm();
    });

    // UTM to WGS84 Lat/Lng conversion
    function utmToLatLng(zone, easting, northing, northernHemisphere) {
        if (!northernHemisphere) {
            northing = 10000000 - northing;
        }
        
        const a = 6378137.0;
        const e = 0.0818191908426;
        const e1sq = 0.0067394967423;
        const k0 = 0.9996;
        
        const arc = northing / k0;
        const mu = arc / (a * (1 - e*e / 4.0 - 3 * Math.pow(e, 4) / 64.0 - 5 * Math.pow(e, 6) / 256.0));
        
        const ei = (1 - Math.sqrt(1 - e*e)) / (1 + Math.sqrt(1 - e*e));
        
        const ca = 3 * ei / 2 - 27 * Math.pow(ei, 3) / 32.0;
        const cb = 21 * Math.pow(ei, 2) / 16 - 55 * Math.pow(ei, 4) / 32.0;
        const cc = 151 * Math.pow(ei, 3) / 96.0;
        const cd = 1097 * Math.pow(ei, 4) / 512.0;
        
        const phi1 = mu + ca * Math.sin(2 * mu) + cb * Math.sin(4 * mu) + cc * Math.sin(6 * mu) + cd * Math.sin(8 * mu);
        
        const q0 = 1 - e*e * Math.sin(phi1) * Math.sin(phi1);
        
        const n0 = a / Math.sqrt(q0);
        const r0 = a * (1 - e*e) / Math.pow(q0, 1.5);
        const dd0 = (easting - 500000) / (n0 * k0);
        
        const t0 = Math.tan(phi1) * Math.tan(phi1);
        const eta0 = e1sq * Math.cos(phi1) * Math.cos(phi1);
        
        const temp1 = n0 * Math.tan(phi1) / r0;
        const temp2 = dd0 * dd0 / 2.0;
        const temp3 = (5 + 3 * t0 + 10 * eta0 - 4 * eta0 * eta0 - 9 * e1sq) * Math.pow(dd0, 4) / 24.0;
        const temp4 = (61 + 90 * t0 + 298 * eta0 + 45 * t0 * t0 - 252 * e1sq - 3 * eta0 * eta0) * Math.pow(dd0, 6) / 720.0;
        
        const lat = phi1 - temp1 * (temp2 - temp3 + temp4);
        
        const temp5 = dd0;
        const temp6 = (1 + 2 * t0 + eta0) * Math.pow(dd0, 3) / 6.0;
        const temp7 = (5 - 2 * eta0 + 28 * t0 - 3 * eta0 * eta0 + 8 * e1sq + 24 * t0 * t0) * Math.pow(dd0, 5) / 120.0;
        
        const lon = (temp5 - temp6 + temp7) / Math.cos(phi1);
        
        const lat0 = lat * 180 / Math.PI;
        const lon0 = (lon * 180 / Math.PI) + (zone * 6 - 183);
        
        return { lat: lat0, lng: lon0 };
    }

    // WGS84 Lat/Lng to UTM conversion
    function latLngToUtm(lat, lon) {
        const zone = Math.floor((lon + 180) / 6) + 1;
        const southernHemisphere = lat < 0;
        
        const latRad = lat * Math.PI / 180;
        const lonRad = lon * Math.PI / 180;
        
        const k0 = 0.9996;
        const a = 6378137.0;
        const e = 0.0818191908426;
        const e1sq = (e * e) / (1 - e * e);
        
        const lonOrigin = (zone - 1) * 6 - 180 + 3;
        const lonOriginRad = lonOrigin * Math.PI / 180;
        
        const N = a / Math.sqrt(1 - e * e * Math.sin(latRad) * Math.sin(latRad));
        const T = Math.tan(latRad) * Math.tan(latRad);
        const C = e1sq * Math.cos(latRad) * Math.cos(latRad);
        const A = (lonRad - lonOriginRad) * Math.cos(latRad);
        
        const M = a * (
            (1 - e * e / 4 - 3 * e * e * e * e / 64 - 5 * e * e * e * e * e * e / 256) * latRad
            - (3 * e * e / 8 + 3 * e * e * e * e / 32 + 45 * e * e * e * e * e * e / 1024) * Math.sin(2 * latRad)
            + (15 * e * e * e * e / 256 + 45 * e * e * e * e * e * e / 1024) * Math.sin(4 * latRad)
            - (35 * e * e * e * e * e * e / 3072) * Math.sin(6 * latRad)
        );
        
        const easting = k0 * N * (
            A 
            + (1 - T + C) * A * A * A / 6 
            + (5 - 18 * T + T * T + 72 * C - 58 * e1sq) * Math.pow(A, 5) / 120
        ) + 500000.0;
        
        let northing = k0 * (
            M 
            + N * Math.tan(latRad) * (
                A * A / 2 
                + (5 - T + 9 * C + 4 * C * C) * Math.pow(A, 4) / 24 
                + (61 - 58 * T + T * T + 600 * C - 330 * e1sq) * Math.pow(A, 6) / 720
            )
        );
        
        if (southernHemisphere) {
            northing += 10000000.0;
        }
        
        return {
            easting: easting,
            northing: northing,
            zone: zone,
            hemi: southernHemisphere ? 'S' : 'N'
        };
    }

    // Helper: Sync UTM fields to Lat/Lng fields
    function syncUtmToLatLng() {
        const zone = parseInt(inputUtmZone.value);
        const hemi = inputUtmHemi.value;
        const northing = parseFloat(inputUtmNorthing.value);
        const easting = parseFloat(inputUtmEasting.value);
        
        if (isNaN(zone) || isNaN(northing) || isNaN(easting)) return;
        
        const result = utmToLatLng(zone, easting, northing, hemi === 'N');
        if (!isNaN(result.lat) && !isNaN(result.lng)) {
            inputLat.value = result.lat.toFixed(6);
            inputLon.value = result.lng.toFixed(6);
        }
    }

    // Helper: Sync Lat/Lng fields to UTM fields
    function syncLatLngToUtm() {
        const lat = parseFloat(inputLat.value);
        const lon = parseFloat(inputLon.value);
        
        if (isNaN(lat) || isNaN(lon)) return;
        
        const result = latLngToUtm(lat, lon);
        inputUtmZone.value = result.zone;
        inputUtmHemi.value = result.hemi;
        inputUtmNorthing.value = result.northing.toFixed(3);
        inputUtmEasting.value = result.easting.toFixed(3);
    }

    // Connect event listeners to UTM inputs
    [inputUtmZone, inputUtmHemi, inputUtmNorthing, inputUtmEasting].forEach(input => {
        input.addEventListener('change', () => {
            syncUtmToLatLng();
            updateBBoxOnMap();
        });
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                input.blur();
                syncUtmToLatLng();
                updateBBoxOnMap();
            }
        });
    });

    function syncModelUi() {
        if (confGroup) confGroup.classList.remove('hidden');
        const sam3PromptGroup = document.getElementById('sam3-prompt-group');
        if (sam3PromptGroup) sam3PromptGroup.classList.remove('hidden');
        const autoSummary = document.getElementById('auto-tune-summary');
        if (autoSummary) {
            autoSummary.innerText = V2_AUTO_SUMMARY;
        }
    }
    if (modelSelect) {
        modelSelect.addEventListener('change', syncModelUi);
    }
    syncModelUi();

    // Sync slider values with text indicators
    ['conf', 'vari', 'v2-tile', 'min-area', 'max-area', 'height'].forEach(name => {
        const slider = document.getElementById(`${name}-slider`);
        const valEl = document.getElementById(`${name}-val`);
        if (slider && valEl) {
            slider.addEventListener('input', (e) => { valEl.innerText = e.target.value; });
        }
    });

    // Expand bottom table drawer
    drawerToggle.addEventListener('click', () => {
        metricsDrawer.classList.toggle('collapsed');
        const isExpanded = metricsDrawer.classList.toggle('expanded');
        document.querySelector('.main-map-area').classList.toggle('bottom-expanded', isExpanded);
    });

    // ----------------------------------------------------
    // 6. Interactive Search Area Polygon Drawing (`Draw Polygon`)
    // ----------------------------------------------------
    function cleanupPolygonDrawingUI() {
        isDrawingActive = false;
        drawVertexGroup.clearLayers();
        if (drawPreviewLayer) {
            if (map.hasLayer(drawPreviewLayer)) map.removeLayer(drawPreviewLayer);
            drawPreviewLayer = null;
        }
        btnDrawBBox.classList.remove('btn-primary');
        btnDrawBBox.classList.add('btn-secondary');
        btnDrawBBox.innerText = 'Draw Polygon';
        map.getContainer().style.cursor = '';
        setTimeout(() => {
            if (map.doubleClickZoom) map.doubleClickZoom.enable();
        }, 120);
    }

    function startPolygonDrawing() {
        isDrawingActive = true;
        previousRoiPolygon = currentRoiPolygon ? currentRoiPolygon.map(pt => [...pt]) : null;
        drawnVertices = [];
        drawVertexGroup.clearLayers();
        if (drawPreviewLayer) {
            if (map.hasLayer(drawPreviewLayer)) map.removeLayer(drawPreviewLayer);
            drawPreviewLayer = null;
        }
        if (boundsLayer) {
            if (map.hasLayer(boundsLayer)) map.removeLayer(boundsLayer);
            boundsLayer = null;
        }
        btnDrawBBox.classList.add('btn-primary');
        btnDrawBBox.classList.remove('btn-secondary');
        btnDrawBBox.innerText = 'Finish Polygon (0 pts)';
        map.getContainer().style.cursor = 'crosshair';
        if (map.doubleClickZoom) map.doubleClickZoom.disable();
        addLog('Polygon draw mode active: click map to add vertices, click first vertex / double-click / press Enter or click Finish Polygon to complete (Esc to cancel).');
    }

    function finishPolygonDrawing() {
        if (drawnVertices.length < 3) {
            cancelPolygonDrawing();
            return;
        }
        currentRoiPolygon = drawnVertices.map(v => [
            Number(v[0].toFixed(7)),
            Number(v[1].toFixed(7))
        ]);
        cleanupPolygonDrawingUI();

        if (boundsLayer) {
            if (map.hasLayer(boundsLayer)) map.removeLayer(boundsLayer);
            boundsLayer = null;
        }
        boundsLayer = L.polygon(currentRoiPolygon, {
            color: '#3b82f6',
            weight: 2,
            fillOpacity: 0.06,
            dashArray: '5, 5'
        });
        if (!chkLayerBounds || chkLayerBounds.checked) {
            boundsLayer.addTo(map);
        }

        const lats = currentRoiPolygon.map(p => p[0]);
        const lons = currentRoiPolygon.map(p => p[1]);
        const south = Math.min(...lats);
        const north = Math.max(...lats);
        const west = Math.min(...lons);
        const east = Math.max(...lons);
        const centerLat = (south + north) / 2.0;
        const centerLon = (west + east) / 2.0;
        const width = Math.max(20, Math.round(map.distance(L.latLng(centerLat, west), L.latLng(centerLat, east))));
        const length = Math.max(20, Math.round(map.distance(L.latLng(south, centerLon), L.latLng(north, centerLon))));

        inputLat.value = centerLat.toFixed(6);
        inputLon.value = centerLon.toFixed(6);
        inputWidth.value = width;
        inputLength.value = length;
        syncLatLngToUtm();
        saveBoundaryToStorage();
        addLog(`Drew custom search polygon (${currentRoiPolygon.length} vertices · BBox ${width}×${length} m)`);
    }

    function cancelPolygonDrawing() {
        cleanupPolygonDrawingUI();
        currentRoiPolygon = previousRoiPolygon;
        if (currentRoiPolygon && currentRoiPolygon.length >= 3) {
            if (boundsLayer && map.hasLayer(boundsLayer)) map.removeLayer(boundsLayer);
            boundsLayer = L.polygon(currentRoiPolygon, {
                color: '#3b82f6',
                weight: 2,
                fillOpacity: 0.06,
                dashArray: '5, 5'
            });
            if (!chkLayerBounds || chkLayerBounds.checked) {
                boundsLayer.addTo(map);
            }
        } else {
            updateBBoxOnMap();
        }
    }

    btnDrawBBox.addEventListener('click', () => {
        if (!isDrawingActive) {
            startPolygonDrawing();
        } else if (drawnVertices.length >= 3) {
            finishPolygonDrawing();
        } else {
            cancelPolygonDrawing();
        }
    });

    map.on('click', (e) => {
        if (!isDrawingActive) return;

        // If >= 3 vertices and user clicked near first vertex, close polygon
        if (drawnVertices.length >= 3) {
            const firstPt = map.latLngToContainerPoint(L.latLng(drawnVertices[0][0], drawnVertices[0][1]));
            const clickPt = map.latLngToContainerPoint(e.latlng);
            if (clickPt.distanceTo(firstPt) <= 12) {
                finishPolygonDrawing();
                return;
            }
        }

        // Ignore duplicate rapid clicks at the exact same pixel
        if (drawnVertices.length > 0) {
            const last = drawnVertices[drawnVertices.length - 1];
            const lastPt = map.latLngToContainerPoint(L.latLng(last[0], last[1]));
            const clickPt = map.latLngToContainerPoint(e.latlng);
            if (clickPt.distanceTo(lastPt) <= 4) {
                return;
            }
        }

        const vertex = [e.latlng.lat, e.latlng.lng];
        drawnVertices.push(vertex);
        btnDrawBBox.innerText = `Finish Polygon (${drawnVertices.length} pts)`;

        const isFirst = drawnVertices.length === 1;
        const vertexMarker = L.circleMarker(vertex, {
            radius: isFirst ? 6 : 4.5,
            color: isFirst ? '#1d4ed8' : '#3b82f6',
            fillColor: '#ffffff',
            fillOpacity: 1,
            weight: 2
        });
        if (isFirst) {
            vertexMarker.bindTooltip('Click first vertex to finish polygon', { direction: 'top' });
            vertexMarker.on('click', (ev) => {
                if (isDrawingActive && drawnVertices.length >= 3) {
                    L.DomEvent.stopPropagation(ev);
                    finishPolygonDrawing();
                }
            });
        }
        drawVertexGroup.addLayer(vertexMarker);

        if (boundsLayer && map.hasLayer(boundsLayer)) {
            map.removeLayer(boundsLayer);
        }
        boundsLayer = L.polygon(drawnVertices, {
            color: '#3b82f6',
            weight: 2,
            fillOpacity: 0.06,
            dashArray: '5, 5'
        });
        if (!chkLayerBounds || chkLayerBounds.checked) {
            boundsLayer.addTo(map);
        }
    });

    map.on('mousemove', (e) => {
        if (!isDrawingActive || drawnVertices.length === 0) return;
        const previewPts = [...drawnVertices, [e.latlng.lat, e.latlng.lng]];
        if (drawPreviewLayer) {
            drawPreviewLayer.setLatLngs(previewPts);
        } else {
            drawPreviewLayer = L.polygon(previewPts, {
                color: '#2563eb',
                weight: 1.5,
                fillOpacity: 0.08,
                dashArray: '4, 4',
                interactive: false
            }).addTo(map);
        }
    });

    map.on('dblclick', (e) => {
        if (!isDrawingActive) return;
        L.DomEvent.stopPropagation(e);
        if (drawnVertices.length >= 3) {
            finishPolygonDrawing();
        }
    });

    document.addEventListener('keydown', (e) => {
        if (!isDrawingActive) return;
        if (e.key === 'Enter' && drawnVertices.length >= 3) {
            e.preventDefault();
            finishPolygonDrawing();
        } else if (e.key === 'Escape') {
            e.preventDefault();
            cancelPolygonDrawing();
        }
    });

    function buildRectCorners(south, west, north, east) {
        return [
            [south, west],
            [south, east],
            [north, east],
            [north, west]
        ];
    }

    function updateBBoxOnMap() {
        // Sync Lat/Lng to UTM if Lat/Lng tab is active (to keep UTM inputs in sync)
        if (coordFormatLatlngBtn.classList.contains('active')) {
            syncLatLngToUtm();
        }
        
        const lat = parseFloat(inputLat.value);
        const lon = parseFloat(inputLon.value);
        const width = parseFloat(inputWidth.value);
        const length = parseFloat(inputLength.value);
        
        if (isNaN(lat) || isNaN(lon) || isNaN(width) || isNaN(length)) return;
        
        if (boundsLayer) {
            map.removeLayer(boundsLayer);
        }
        
        const EarthRadius = 6378137.0;
        const c_lat_rad = lat * Math.PI / 180.0;
        const d_lat = (length / 2.0 / EarthRadius) * (180.0 / Math.PI);
        const d_lon = (width / 2.0 / (EarthRadius * Math.cos(c_lat_rad))) * (180.0 / Math.PI);
        
        const south = lat - d_lat;
        const north = lat + d_lat;
        const west = lon - d_lon;
        const east = lon + d_lon;
        const corners = buildRectCorners(south, west, north, east);
        currentRoiPolygon = null;
        
        boundsLayer = L.polygon(corners, {
            color: '#3b82f6',
            weight: 2,
            fillOpacity: 0.06,
            dashArray: '5, 5'
        });
        if (!chkLayerBounds || chkLayerBounds.checked) {
            boundsLayer.addTo(map);
        }
        map.fitBounds(boundsLayer.getBounds());
        saveBoundaryToStorage();
    }

    const sitePresetSelect = document.getElementById('site-preset-select');

    const SITE_PRESETS = {
        onnut: {
            name: 'ศูนย์กำจัดมูลฝอยอ่อนนุช (ON · 580.11-Rai Official Boundary)',
            lat: 13.714447,
            lon: 100.683989,
            width: 1540,
            length: 1440,
            roiPolygon: OFFICIAL_ONNUT_POLYGON
        },
        nongkhaem: {
            name: 'ศูนย์กำจัดมูลฝอยหนองแขม (NK · Official Boundary)',
            lat: 13.718255,
            lon: 100.358126,
            width: 1260,
            length: 1250,
            roiPolygon: OFFICIAL_NONGKHAEM_POLYGON
        },
        ramintra: {
            name: 'สวนกีฬารามอินทรา (Ramintra Sports Park)',
            lat: 13.879366,
            lon: 100.619659,
            width: 710,
            length: 396,
            roiPolygon: OFFICIAL_RAMINTRA_POLYGON
        },
        sapansung: {
            name: 'สวนสาธารณะสะพานสูง (Saphan Sung Park)',
            lat: 13.768900,
            lon: 100.684500,
            width: 300,
            length: 300,
            roiPolygon: null
        },
        piyaphirom: {
            name: 'สวนปิยะภิรมย์ (Piyaphirom Park)',
            lat: 13.743600,
            lon: 100.640900,
            width: 250,
            length: 250,
            roiPolygon: null
        }
    };

    function detectPresetFromCoords(lat, lon) {
        for (const [key, preset] of Object.entries(SITE_PRESETS)) {
            if (Math.abs(Number(lat) - preset.lat) <= 0.008 && Math.abs(Number(lon) - preset.lon) <= 0.008) {
                return key;
            }
        }
        return sitePresetSelect ? sitePresetSelect.value : 'onnut';
    }

    function saveBoundaryToStorage(syncBackend = true) {
        try {
            const activePreset = (sitePresetSelect && sitePresetSelect.value)
                ? sitePresetSelect.value
                : detectPresetFromCoords(parseFloat(inputLat.value), parseFloat(inputLon.value));
            const payload = {
                sitePreset: activePreset,
                centerLat: parseFloat(inputLat.value),
                centerLon: parseFloat(inputLon.value),
                width: parseFloat(inputWidth.value),
                length: parseFloat(inputLength.value),
                roiPolygon: (currentRoiPolygon && currentRoiPolygon.length >= 3) ? currentRoiPolygon : null,
                updatedAt: Date.now()
            };
            if (!isNaN(payload.centerLat) && !isNaN(payload.centerLon) && !isNaN(payload.width) && !isNaN(payload.length)) {
                localStorage.setItem(STORAGE_KEY_BOUNDARY, JSON.stringify(payload));
                let presetMap = {};
                try {
                    const rawMap = localStorage.getItem(STORAGE_KEY_PRESET_MAP);
                    if (rawMap) presetMap = JSON.parse(rawMap) || {};
                } catch (_) {}
                if (activePreset) {
                    presetMap[activePreset] = payload;
                    localStorage.setItem(STORAGE_KEY_PRESET_MAP, JSON.stringify(presetMap));
                }
                if (syncBackend) {
                    const fd = new FormData();
                    if (activePreset) fd.append('site_preset', activePreset);
                    fd.append('center_lat', payload.centerLat);
                    fd.append('center_lon', payload.centerLon);
                    fd.append('width_m', payload.width);
                    fd.append('length_m', payload.length);
                    if (payload.roiPolygon) {
                        fd.append('roi_polygon', JSON.stringify(payload.roiPolygon));
                    }
                    fetch(`${BACKEND_URL}/save_boundary_session`, {
                        method: 'POST',
                        body: fd
                    }).catch(() => {});
                }
            }
        } catch (e) {
            console.warn('Failed to save boundary to localStorage:', e);
        }
    }

    function loadBoundaryFromStorage() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY_BOUNDARY);
            if (raw) {
                const parsed = JSON.parse(raw);
                if (
                    parsed &&
                    Number.isFinite(Number(parsed.centerLat)) &&
                    Number.isFinite(Number(parsed.centerLon)) &&
                    Number.isFinite(Number(parsed.width)) &&
                    Number.isFinite(Number(parsed.length))
                ) {
                    return parsed;
                }
            }
        } catch (e) {
            console.warn('Failed to load boundary from localStorage:', e);
        }
        return DEFAULT_SAVED_BOUNDARY;
    }

    function applySavedBoundary(boundaryObj, fitMap = true, syncBackend = false) {
        if (!boundaryObj) return;
        const resolvedPreset = boundaryObj.sitePreset || detectPresetFromCoords(boundaryObj.centerLat, boundaryObj.centerLon);
        if (sitePresetSelect && resolvedPreset && SITE_PRESETS[resolvedPreset]) {
            sitePresetSelect.value = resolvedPreset;
        }
        inputLat.value = Number(boundaryObj.centerLat).toFixed(6);
        inputLon.value = Number(boundaryObj.centerLon).toFixed(6);
        inputWidth.value = Math.round(Number(boundaryObj.width));
        inputLength.value = Math.round(Number(boundaryObj.length));
        syncLatLngToUtm();

        if (Array.isArray(boundaryObj.roiPolygon) && boundaryObj.roiPolygon.length >= 3) {
            currentRoiPolygon = boundaryObj.roiPolygon.map(pt => [Number(pt[0]), Number(pt[1])]);
            if (boundsLayer && map.hasLayer(boundsLayer)) {
                map.removeLayer(boundsLayer);
            }
            boundsLayer = L.polygon(currentRoiPolygon, {
                color: '#3b82f6',
                weight: 2,
                fillOpacity: 0.06,
                dashArray: '5, 5'
            });
            if (!chkLayerBounds || chkLayerBounds.checked) {
                boundsLayer.addTo(map);
            }
            if (fitMap) {
                map.fitBounds(boundsLayer.getBounds());
            }
            saveBoundaryToStorage(syncBackend);
        } else {
            updateBBoxOnMap();
        }
    }

    async function applySessionMetadataToMap(metadata, fitMap = true) {
        if (!metadata) return;
        const resolvedPreset = metadata.site_preset || detectPresetFromCoords(metadata.center_lat, metadata.center_lon);
        if (sitePresetSelect && resolvedPreset && SITE_PRESETS[resolvedPreset]) {
            sitePresetSelect.value = resolvedPreset;
        }

        const roiPoly = (Array.isArray(metadata.roi_polygon) && metadata.roi_polygon.length >= 3)
            ? metadata.roi_polygon.map(pt => [Number(pt[0]), Number(pt[1])])
            : (SITE_PRESETS[resolvedPreset]?.roiPolygon || currentRoiPolygon);

        applySavedBoundary({
            sitePreset: resolvedPreset,
            centerLat: metadata.center_lat ?? parseFloat(inputLat.value),
            centerLon: metadata.center_lon ?? parseFloat(inputLon.value),
            width: metadata.width_m ?? parseFloat(inputWidth.value),
            length: metadata.length_m ?? parseFloat(inputLength.value),
            roiPolygon: roiPoly
        }, fitMap, false);

        if (metadata.file_path) {
            uploadedFilePath = metadata.file_path;
            uploadedMetadata = metadata;
            fileNameEl.innerText = metadata.filename || 'vhr_latest.tif';
            const srcLabel = metadata.imagery_source || 'VHR';
            const gsdLabel = (typeof metadata.gsd_m === 'number')
                ? ` · ~${metadata.gsd_m.toFixed(3)} m/px`
                : '';
            const zLabel = metadata.zoom != null ? ` · z${metadata.zoom}` : '';
            fileDimsEl.innerText = `${metadata.width}×${metadata.height} px · GEO · ${srcLabel}${zLabel}${gsdLabel}`;
            fileInfo.classList.remove('hidden');
            btnExtract.disabled = false;
            setPipelineModeUI(metadata);
        } else {
            uploadedFilePath = null;
            uploadedMetadata = null;
            fileInfo.classList.add('hidden');
            btnExtract.disabled = true;
            if (imageOverlayLayer && map.hasLayer(imageOverlayLayer)) {
                map.removeLayer(imageOverlayLayer);
            }
            clearExtractedLayers();
            resetTable();
        }

        if (metadata.bounds) {
            const southWest = L.latLng(metadata.bounds.south, metadata.bounds.west);
            const northEast = L.latLng(metadata.bounds.north, metadata.bounds.east);
            const bounds = L.latLngBounds(southWest, northEast);
            const polyRing = (currentRoiPolygon && currentRoiPolygon.length >= 3)
                ? currentRoiPolygon
                : buildRectCorners(metadata.bounds.south, metadata.bounds.west, metadata.bounds.north, metadata.bounds.east);

            if (boundsLayer && map.hasLayer(boundsLayer)) {
                map.removeLayer(boundsLayer);
            }
            boundsLayer = L.polygon(polyRing, {
                color: '#3b82f6',
                weight: 2,
                fillOpacity: 0.06,
                dashArray: '5, 5'
            });
            if (!chkLayerBounds || chkLayerBounds.checked) {
                boundsLayer.addTo(map);
            }

            if (metadata.preview_url) {
                if (imageOverlayLayer && map.hasLayer(imageOverlayLayer)) {
                    map.removeLayer(imageOverlayLayer);
                }
                const imgUrl = `${BACKEND_URL}${metadata.preview_url}`;
                imageOverlayLayer = L.imageOverlay(imgUrl, bounds, {
                    opacity: 0.9,
                    interactive: false
                });
                if (chkLayerRaster && chkLayerRaster.checked) {
                    imageOverlayLayer.addTo(map);
                }
            }

            if (fitMap) {
                map.fitBounds(boundsLayer.getBounds());
            }
        }

        if (metadata.result_geojson_url) {
            try {
                const gjRes = await fetch(`${BACKEND_URL}${metadata.result_geojson_url}`);
                if (gjRes.ok) {
                    extractedGeoJson = await gjRes.json();
                    renderCanopies(extractedGeoJson);
                    btnExportCsv.disabled = false;
                    btnExportGeoJson.disabled = false;
                    addLog(`Restored ${extractedGeoJson.features ? extractedGeoJson.features.length : 0} extracted crowns from cached session (${metadata.filename}).`);
                }
            } catch (e) {
                console.warn('Could not restore cached extraction GeoJSON:', e);
            }
        }
    }

    async function restoreLatestBackendSession() {
        try {
            const res = await fetch(`${BACKEND_URL}/latest_session`, {
                headers: { 'Accept': 'application/json' }
            });
            if (!res.ok) return;
            const data = await res.json();
            if (!data || !data.has_session || !data.metadata) return;

            const metadata = data.metadata;
            await applySessionMetadataToMap(metadata, true);
            addLog('Restored saved site (' + (metadata.site_preset || 'custom') + '), search polygon (' + (currentRoiPolygon ? currentRoiPolygon.length + ' vertices' : 'BBox') + ')' + (metadata.filename ? ' & satellite raster (' + metadata.filename + ')' : '') + '.');
        } catch (err) {
            console.warn('Could not restore latest backend session:', err);
        }
    }

    if (sitePresetSelect) {
        sitePresetSelect.addEventListener('change', async (e) => {
            const key = e.target.value;
            const preset = SITE_PRESETS[key];
            if (!preset) return;

            let savedForPreset = null;
            try {
                const rawMap = localStorage.getItem(STORAGE_KEY_PRESET_MAP);
                if (rawMap) {
                    const presetMap = JSON.parse(rawMap) || {};
                    if (presetMap[key] && Array.isArray(presetMap[key].roiPolygon) && presetMap[key].roiPolygon.length >= 3) {
                        savedForPreset = presetMap[key];
                    }
                }
            } catch (_) {}

            const boundaryToApply = {
                sitePreset: key,
                centerLat: savedForPreset?.centerLat ?? preset.lat,
                centerLon: savedForPreset?.centerLon ?? preset.lon,
                width: savedForPreset?.width ?? preset.width,
                length: savedForPreset?.length ?? preset.length,
                roiPolygon: (savedForPreset?.roiPolygon && savedForPreset.roiPolygon.length >= 3)
                    ? savedForPreset.roiPolygon
                    : preset.roiPolygon
            };

            applySavedBoundary(boundaryToApply, true, true);
            addLog(`Selected site preset: ${preset.name} (${boundaryToApply.centerLat.toFixed(6)}, ${boundaryToApply.centerLon.toFixed(6)} · ${boundaryToApply.width}×${boundaryToApply.length} m · ${boundaryToApply.roiPolygon ? boundaryToApply.roiPolygon.length + ' polygon vertices' : 'BBox'})`);

            try {
                const res = await fetch(`${BACKEND_URL}/latest_session?site_preset=${encodeURIComponent(key)}`, {
                    headers: { 'Accept': 'application/json' }
                });
                if (res.ok) {
                    const data = await res.json();
                    if (data && data.has_session && data.metadata) {
                        await applySessionMetadataToMap(data.metadata, true);
                        addLog(`Loaded cached VHR raster (${data.metadata.filename}) for preset ${key.toUpperCase()}.`);
                    }
                }
            } catch (_) {}
        });
    }

    [inputWidth, inputLength, inputLat, inputLon].forEach(input => {
        input.addEventListener('change', updateBBoxOnMap);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                input.blur();
                updateBBoxOnMap();
            }
        });
    });

    // Restore saved boundary & latest backend session
    applySavedBoundary(loadBoundaryFromStorage(), true, false);
    restoreLatestBackendSession();

    // ----------------------------------------------------
    // 7. Fetch Satellite Imagery
    // ----------------------------------------------------
    btnFetchSatellite.addEventListener('click', async () => {
        if (isDrawingActive && drawnVertices.length >= 3) {
            finishPolygonDrawing();
        }
        const lat = parseFloat(inputLat.value);
        const lon = parseFloat(inputLon.value);
        const width = parseFloat(inputWidth.value);
        const length = parseFloat(inputLength.value);
        
        if (isNaN(lat) || isNaN(lon) || isNaN(width) || isNaN(length)) {
            alert('Please specify valid coordinates and dimension parameters.');
            return;
        }
        
        btnFetchSatellite.disabled = true;
        btnFetchSatellite.innerText = 'Fetching satellite...';
        
        const activePolygon = (currentRoiPolygon && currentRoiPolygon.length >= 3)
            ? currentRoiPolygon.map(pt => [...pt])
            : null;
        const activePreset = (sitePresetSelect && sitePresetSelect.value)
            ? sitePresetSelect.value
            : detectPresetFromCoords(lat, lon);

        const formData = new FormData();
        formData.append('center_lat', lat);
        formData.append('center_lon', lon);
        formData.append('width_m', width);
        formData.append('length_m', length);
        if (activePreset) {
            formData.append('site_preset', activePreset);
        }
        const zoomSelect = document.getElementById('satellite-zoom-select');
        const targetZoom = zoomSelect ? zoomSelect.value : '0';
        formData.append('target_zoom', targetZoom);
        if (activePolygon) {
            formData.append('roi_polygon', JSON.stringify(activePolygon));
        }
        
        resetUploadState();
        currentRoiPolygon = activePolygon;
        
        try {
            const response = await fetch(`${BACKEND_URL}/fetch_imagery`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Fetch failed');
            }
            
            const metadata = await response.json();
            uploadedFilePath = metadata.file_path;
            uploadedMetadata = metadata;
            if (!currentRoiPolygon && Array.isArray(metadata.roi_polygon) && metadata.roi_polygon.length >= 3) {
                currentRoiPolygon = metadata.roi_polygon;
            }
            
            fileNameEl.innerText = metadata.filename;
            const srcLabel = metadata.imagery_source || 'VHR';
            const gsdLabel = (typeof metadata.gsd_m === 'number')
                ? ` · ~${metadata.gsd_m.toFixed(3)} m/px`
                : '';
            const zLabel = metadata.zoom != null ? ` · z${metadata.zoom}` : '';
            fileDimsEl.innerText = `${metadata.width}×${metadata.height} px · GEO · ${srcLabel}${zLabel}${gsdLabel}`;
            fileInfo.classList.remove('hidden');
            btnExtract.disabled = false;
            setPipelineModeUI(metadata);
            if (typeof addLog === 'function') {
                addLog(`Fetched GEO ${srcLabel} z${metadata.zoom ?? '?'} GSD~${typeof metadata.gsd_m === 'number' ? metadata.gsd_m.toFixed(3) : (metadata.gsd_m ?? '?')} m/px`);
            }
            
            if (metadata.bounds) {
                const southWest = L.latLng(metadata.bounds.south, metadata.bounds.west);
                const northEast = L.latLng(metadata.bounds.north, metadata.bounds.east);
                const bounds = L.latLngBounds(southWest, northEast);
                const polyRing = (currentRoiPolygon && currentRoiPolygon.length >= 3)
                    ? currentRoiPolygon
                    : buildRectCorners(metadata.bounds.south, metadata.bounds.west, metadata.bounds.north, metadata.bounds.east);
                
                if (boundsLayer && map.hasLayer(boundsLayer)) {
                    map.removeLayer(boundsLayer);
                }
                boundsLayer = L.polygon(polyRing, {
                    color: '#3b82f6',
                    weight: 2,
                    fillOpacity: 0.06,
                    dashArray: '5, 5'
                });
                if (!chkLayerBounds || chkLayerBounds.checked) {
                    boundsLayer.addTo(map);
                }
                
                if (metadata.preview_url) {
                    const imgUrl = `${BACKEND_URL}${metadata.preview_url}`;
                    imageOverlayLayer = L.imageOverlay(imgUrl, bounds, {
                        opacity: 0.9,
                        interactive: false
                    });
                    if (chkLayerRaster && chkLayerRaster.checked) {
                        imageOverlayLayer.addTo(map);
                    }
                }
                
                map.fitBounds(bounds);
            }
            saveBoundaryToStorage();
        } catch (error) {
            alert(`Error fetching satellite data: ${error.message}`);
            resetUploadState();
        } finally {
            btnFetchSatellite.disabled = false;
            btnFetchSatellite.innerText = 'Fetch Data';
            lucide.createIcons();
        }
    });

    // ----------------------------------------------------
    // 8. File Upload Handling
    // ----------------------------------------------------
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    btnRemoveFile.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUploadState();
    });

    function setPipelineModeUI(metadata) {
        const banner = document.getElementById('pipeline-mode-banner');
        const badge = document.getElementById('pipeline-mode-badge');
        const title = document.getElementById('pipeline-mode-title');
        const msg = document.getElementById('pipeline-mode-msg');
        const exportNote = document.getElementById('export-mode-note');
        const exportLabel = document.getElementById('btn-export-geojson-label');
        if (!banner || !metadata) return;

        const mode = metadata.pipeline_mode || 'geo';
        banner.classList.remove('hidden');
        if (mode === 'image_only') {
            badge.innerText = 'IMAGE-ONLY';
            badge.style.background = 'rgba(217, 107, 82, 0.15)';
            badge.style.color = 'var(--danger)';
            title.innerText = 'Detection QA (not map-survey)';
            banner.style.background = 'rgba(217, 107, 82, 0.06)';
            banner.style.borderColor = 'rgba(217, 107, 82, 0.35)';
            if (exportNote) {
                exportNote.classList.remove('hidden');
                exportNote.innerText =
                    'IMAGE-ONLY export: coordinates/areas use preview placement and assumed GSD. Do not use as primary Rhino/MCDA layer. Prefer GeoTIFF or Map extent (GEO).';
            }
            if (exportLabel) exportLabel.innerText = 'Export GeoJSON (preview coords)';
        } else {
            badge.innerText = 'GEO';
            badge.style.background = 'rgba(92, 131, 83, 0.15)';
            badge.style.color = 'var(--primary)';
            title.innerText = mode === 'geo' ? 'Map-ready georeferenced' : 'GEO';
            banner.style.background = 'rgba(92, 131, 83, 0.08)';
            banner.style.borderColor = 'rgba(92, 131, 83, 0.35)';
            if (exportNote) {
                exportNote.classList.remove('hidden');
                exportNote.style.background = 'rgba(92, 131, 83, 0.08)';
                exportNote.style.borderColor = 'rgba(92, 131, 83, 0.35)';
                exportNote.innerText =
                    'GEO export: WGS84 boundaries suitable for QGIS / Rhino-MCDA (verify GSD for satellite mosaics).';
            }
            if (exportLabel) exportLabel.innerText = 'Export GeoJSON Boundaries (WGS84)';
        }
        if (msg) {
            msg.innerText = metadata.mode_message ||
                (mode === 'image_only'
                    ? 'No real georeference on this file.'
                    : 'Georeferenced source.');
        }
        addLog(`Pipeline mode: ${mode.toUpperCase()}`);
        if (metadata && metadata.auto_tune && typeof metadata.auto_tune === 'object') {
            currentAutoTuneProfile = metadata.auto_tune;
        }
        // Auto-tune detection params for this mode (unless user disabled Auto)
        const autoOn = document.getElementById('chk-auto-tune');
        if (!autoOn || autoOn.checked) {
            applyAutoTune(currentAutoTuneProfile);
        }
    }

    function clearPipelineModeUI() {
        const banner = document.getElementById('pipeline-mode-banner');
        const exportNote = document.getElementById('export-mode-note');
        if (banner) banner.classList.add('hidden');
        if (exportNote) {
            exportNote.classList.add('hidden');
            exportNote.style.background = 'rgba(217, 107, 82, 0.08)';
            exportNote.style.borderColor = 'rgba(217, 107, 82, 0.35)';
        }
    }

    function resetUploadState() {
        uploadedFilePath = null;
        uploadedMetadata = null;
        extractedGeoJson = null;
        fileInput.value = '';
        fileInfo.classList.add('hidden');
        dropZone.classList.remove('hidden');
        btnExtract.disabled = true;
        btnExportCsv.disabled = true;
        btnExportGeoJson.disabled = true;
        clearPipelineModeUI();
        
        if (boundsLayer) {
            map.removeLayer(boundsLayer);
            boundsLayer = null;
        }
        if (currentRoiPolygon && currentRoiPolygon.length >= 3) {
            boundsLayer = L.polygon(currentRoiPolygon, {
                color: '#3b82f6',
                weight: 2,
                fillOpacity: 0.06,
                dashArray: '5, 5'
            });
            if (!chkLayerBounds || chkLayerBounds.checked) {
                boundsLayer.addTo(map);
            }
        }
        if (imageOverlayLayer) {
            map.removeLayer(imageOverlayLayer);
            imageOverlayLayer = null;
        }
        closeDetailDrawer();
        clearExtractedLayers();
        resetTable();
    }

    async function handleFileUpload(file) {
        resetUploadState();

        const formData = new FormData();
        formData.append('file', file);

        dropZone.classList.add('dragover');
        const uploadText = dropZone.querySelector('.upload-text');
        const originalText = uploadText.innerText;
        uploadText.innerText = 'Uploading and processing metadata...';

        try {
            const response = await fetch(`${BACKEND_URL}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Upload failed');
            }

            const metadata = await response.json();
            uploadedFilePath = metadata.file_path;
            uploadedMetadata = metadata;

            fileNameEl.innerText = file.name;
            const modeTag = metadata.pipeline_mode === 'image_only' ? 'IMAGE-ONLY' : 'GEO';
            const gsdBit = metadata.gsd_m != null
                ? ` · ~${Number(metadata.gsd_m).toFixed(3)} m/px${metadata.pipeline_mode === 'image_only' ? ' (assumed)' : ''}`
                : '';
            fileDimsEl.innerText = `${metadata.width}×${metadata.height} px · ${modeTag}${gsdBit}`;
            
            dropZone.classList.add('hidden');
            fileInfo.classList.remove('hidden');
            btnExtract.disabled = false;
            setPipelineModeUI(metadata);

            if (metadata.bounds) {
                const southWest = L.latLng(metadata.bounds.south, metadata.bounds.west);
                const northEast = L.latLng(metadata.bounds.north, metadata.bounds.east);
                const bounds = L.latLngBounds(southWest, northEast);
                const corners = buildRectCorners(metadata.bounds.south, metadata.bounds.west, metadata.bounds.north, metadata.bounds.east);
                currentRoiPolygon = null;
                
                if (boundsLayer && map.hasLayer(boundsLayer)) {
                    map.removeLayer(boundsLayer);
                }
                boundsLayer = L.polygon(corners, {
                    color: '#3b82f6',
                    weight: 2,
                    fillOpacity: 0.06,
                    dashArray: '5, 5'
                });
                if (!chkLayerBounds || chkLayerBounds.checked) {
                    boundsLayer.addTo(map);
                }

                if (metadata.preview_url) {
                    const imgUrl = `${BACKEND_URL}${metadata.preview_url}`;
                    imageOverlayLayer = L.imageOverlay(imgUrl, bounds, {
                        opacity: 0.9,
                        interactive: false
                    });
                    if (metadata.pipeline_mode === 'image_only' && chkLayerRaster) {
                        chkLayerRaster.checked = true;
                    }
                    if (chkLayerRaster && chkLayerRaster.checked) {
                        imageOverlayLayer.addTo(map);
                    }
                }

                map.fitBounds(bounds);
            }
        } catch (error) {
            alert(`Error uploading file: ${error.message}`);
            resetUploadState();
        } finally {
            uploadText.innerText = originalText;
            dropZone.classList.remove('dragover');
            lucide.createIcons();
        }
    }

    // ----------------------------------------------------
    // 9. Run Tree Canopy Segmentation
    // ----------------------------------------------------
    btnExtract.addEventListener('click', async () => {
        if (!uploadedFilePath) return;

        btnExtract.disabled = true;
        const btnText = btnExtract.querySelector('.btn-text');
        const spinner = btnExtract.querySelector('.spinner');
        const originalBtnText = btnText.innerText;
        btnText.innerText = 'Processing model...';
        spinner.classList.remove('hidden');

        closeDetailDrawer();
        clearExtractedLayers();
        resetTable();
        
        if (logContent) logContent.innerHTML = '';
        logPanel.classList.remove('hidden');

        const autoEl = document.getElementById('chk-auto-tune');
        if (autoEl && autoEl.checked) {
            applyAutoTune();
        }

        const clampedCoverage = Math.max(0.10, Math.min(0.90, parseFloat(variSlider?.value) || 0.43)).toFixed(2);
        const tileSizeVal = tileSlider ? tileSlider.value : '1280';
        const overlapVal = String(Math.min(384, Math.floor((Number(tileSizeVal) || 1280) * 0.30)));
        const promptInput = document.getElementById('sam3-prompt-input');
        const sam3PromptVal = promptInput && promptInput.value.trim() ? promptInput.value.trim() : 'tree';

        addLog('Initializing canopy extraction request...');
        addLog('Engine: Canopy V3.0 Hybrid Tri-Engine (SAM 3 + DeepForest 2.1 + Global Compact Geodesic Watershed)');
        addLog(`SAM 3 Prompt: "${sam3PromptVal}"`);
        addLog(`SAM 3 Instance Threshold: ${confSlider.value}`);
        addLog(`RGB Coverage Threshold: ${clampedCoverage}`);
        addLog(`Tile Size: ${tileSizeVal} px (overlap ${overlapVal} px)`);
        addLog('Sending request to FastAPI backend...');

        const formData = new FormData();
        formData.append('file_path', uploadedFilePath);
        formData.append('model_type', 'canopy_v2');
        formData.append('sam3_prompt', sam3PromptVal);
        formData.append('v2_instance_threshold', confSlider.value);
        formData.append('v2_coverage_threshold', clampedCoverage);
        formData.append('v2_tile_size', tileSizeVal);
        formData.append('v2_overlap', overlapVal);
        formData.append('min_area_m2', minAreaSlider.value);
        formData.append('max_area_m2', maxAreaSlider.value);
        formData.append('default_height_m', heightSlider.value);
        if (currentRoiPolygon && currentRoiPolygon.length >= 3) {
            formData.append('roi_polygon', JSON.stringify(currentRoiPolygon));
            addLog(`ROI Polygon Clip: ${currentRoiPolygon.length} vertices active`);
        }

        // Start polling the backend log file
        let pollInterval = setInterval(async () => {
            try {
                const logRes = await fetch(`${BACKEND_URL}/static_uploads/extraction.log?t=${Date.now()}`);
                if (logRes.ok) {
                    const text = await logRes.text();
                    logContent.innerText = text;
                    logPanel.scrollTop = logPanel.scrollHeight;
                }
            } catch (err) {
                console.error('Failed to poll logs:', err);
            }
        }, 1000);

        try {
            const response = await fetch(`${BACKEND_URL}/segment`, {
                method: 'POST',
                body: formData
            });

            if (pollInterval) {
                clearInterval(pollInterval);
                pollInterval = null;
            }

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Segmentation failed');
            }

            extractedGeoJson = await response.json();
            const count = extractedGeoJson.features ? extractedGeoJson.features.length : 0;
            const summary = extractedGeoJson.properties?.v2_summary;
            
            // Fetch logs one final time to show final lines
            try {
                const finalLogRes = await fetch(`${BACKEND_URL}/static_uploads/extraction.log?t=${Date.now()}`);
                if (finalLogRes.ok) {
                    logContent.innerText = await finalLogRes.text();
                }
            } catch (_) {}
            
            if (summary) {
                addLog(
                    `SUCCESS: ${summary.individual_count} provisional individuals, ` +
                    `${summary.dense_candidate_count || 0} dense crown candidates, ` +
                    `${summary.group_count} unresolved groups, ` +
                    `${summary.coverage_percent}% semantic canopy coverage.`
                );
                addLog('Rendering coverage, groups, instances, and centroids...');
            } else {
                addLog(`SUCCESS: Delineation complete. Extracted ${count} tree crowns.`);
                addLog('Rendering centroids and boundaries on Leaflet map...');
            }
            
            renderCanopies(extractedGeoJson);
            
            btnExportCsv.disabled = false;
            btnExportGeoJson.disabled = false;
            
            metricsDrawer.classList.remove('collapsed');
            metricsDrawer.classList.add('expanded');
            document.querySelector('.main-map-area').classList.add('bottom-expanded');

        } catch (error) {
            if (pollInterval) {
                clearInterval(pollInterval);
                pollInterval = null;
            }
            addLog(`FAILED: ${error.message}`);
            alert(`Error extracting canopies: ${error.message}`);
        } finally {
            if (pollInterval) {
                clearInterval(pollInterval);
            }
            btnText.innerText = originalBtnText;
            spinner.classList.add('hidden');
            btnExtract.disabled = false;
            lucide.createIcons();
        }
    });

    // ----------------------------------------------------
    // 10. Map Visualization Rendering & Separation Confidence (0–100%)
    // ----------------------------------------------------
    function getConfidencePct(props) {
        if (!props) return 70.0;
        const raw = Number(props.confidence_pct ?? (Number(props.confidence || 0.7) * 100));
        if (!Number.isFinite(raw)) return 70.0;
        return Math.max(0, Math.min(100, raw));
    }

    function getCanopyConfidenceColor(confPct) {
        if (confPct >= 80) return '#059669'; // Deep Emerald Green — High Confidence (80–98%)
        if (confPct >= 60) return '#2563eb'; // Royal Blue — Good Confidence (60–79%)
        if (confPct >= 45) return '#eab308'; // Amber Yellow — Moderate Confidence (45–59%)
        return '#dc2626';                    // Crimson Red — Lower Confidence (<45%)
    }

    function getConfidenceTierLabel(confPct, props) {
        if (props && props.confidence_label) return props.confidence_label;
        if (confPct >= 80) return 'High — Verified Crown Separation';
        if (confPct >= 60) return 'Good — Clear Treetop & Relief';
        if (confPct >= 45) return 'Moderate — Closed-Canopy Split';
        return 'Low — Ambiguous / Scrub Partition';
    }

    function extractSampledRingPoints(feature) {
        const geom = feature && feature.geometry;
        if (!geom || !geom.coordinates) return [];
        let ring = [];
        if (geom.type === 'Polygon') {
            ring = geom.coordinates[0] || [];
        } else if (geom.type === 'MultiPolygon') {
            ring = (geom.coordinates[0] && geom.coordinates[0][0]) || [];
        }
        if (!Array.isArray(ring) || ring.length === 0) return [];
        const dense = [];
        const step = Math.max(1, Math.floor(ring.length / 28));
        for (let i = 0; i < ring.length; i += step) {
            const pt = ring[i];
            const next = ring[(i + 1) % ring.length];
            if (Array.isArray(pt) && pt.length >= 2) {
                dense.push([Number(pt[0]), Number(pt[1])]);
                if (Array.isArray(next) && next.length >= 2) {
                    dense.push([(Number(pt[0]) + Number(next[0])) * 0.5, (Number(pt[1]) + Number(next[1])) * 0.5]);
                }
            }
        }
        return dense;
    }

    function ensureAdjacentDistinctColors(features) {
        if (!Array.isArray(features) || features.length === 0) return;
        if (features.every(f => f && f.properties && f.properties.distinct_color)) return;

        // Shuffle palette once per extraction run
        const palette = [...DISTINCT_CROWN_PALETTE];
        for (let i = palette.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            const tmp = palette[i];
            palette[i] = palette[j];
            palette[j] = tmp;
        }

        const n = features.length;
        const rings = new Array(n);
        const geoBBoxes = new Array(n);
        const pxBBoxes = new Array(n);

        for (let i = 0; i < n; i++) {
            const feat = features[i] || {};
            const props = feat.properties || (feat.properties = {});
            const pts = extractSampledRingPoints(feat);
            rings[i] = pts;
            if (pts.length > 0) {
                let minLon = Infinity, minLat = Infinity, maxLon = -Infinity, maxLat = -Infinity;
                for (let k = 0; k < pts.length; k++) {
                    const [lon, lat] = pts[k];
                    if (lon < minLon) minLon = lon;
                    if (lon > maxLon) maxLon = lon;
                    if (lat < minLat) minLat = lat;
                    if (lat > maxLat) maxLat = lat;
                }
                geoBBoxes[i] = [minLon, minLat, maxLon, maxLat];
            } else {
                geoBBoxes[i] = null;
            }
            const pb = props.pixel_bounds;
            if (Array.isArray(pb) && pb.length === 4) {
                pxBBoxes[i] = [Number(pb[0]), Number(pb[1]), Number(pb[0]) + Number(pb[2]), Number(pb[1]) + Number(pb[3])];
            } else {
                pxBBoxes[i] = null;
            }
        }

        const adj = Array.from({ length: n }, () => []);
        const degPad = 0.00002; // ~2.2 m
        const degThreshSq = 0.000022 * 0.000022;
        const pxPad = 10;

        for (let i = 0; i < n; i++) {
            const gbA = geoBBoxes[i];
            const pbA = pxBBoxes[i];
            const ringA = rings[i];
            for (let j = i + 1; j < n; j++) {
                let isNeighbor = false;
                const gbB = geoBBoxes[j];
                const ringB = rings[j];
                if (gbA && gbB &&
                    gbA[0] - degPad <= gbB[2] + degPad &&
                    gbA[2] + degPad >= gbB[0] - degPad &&
                    gbA[1] - degPad <= gbB[3] + degPad &&
                    gbA[3] + degPad >= gbB[1] - degPad) {
                    outerLoop:
                    for (let a = 0; a < ringA.length; a++) {
                        const lonA = ringA[a][0];
                        const latA = ringA[a][1];
                        for (let b = 0; b < ringB.length; b++) {
                            const dLon = lonA - ringB[b][0];
                            const dLat = latA - ringB[b][1];
                            if (dLon * dLon + dLat * dLat <= degThreshSq) {
                                isNeighbor = true;
                                break outerLoop;
                            }
                        }
                    }
                }
                if (!isNeighbor && (!ringA.length || !ringB.length)) {
                    const pbB = pxBBoxes[j];
                    if (pbA && pbB &&
                        pbA[0] - pxPad <= pbB[2] + pxPad &&
                        pbA[2] + pxPad >= pbB[0] - pxPad &&
                        pbA[1] - pxPad <= pbB[3] + pxPad &&
                        pbA[3] + pxPad >= pbB[1] - pxPad) {
                        isNeighbor = true;
                    }
                }
                if (isNeighbor) {
                    adj[i].push(j);
                    adj[j].push(i);
                }
            }
        }

        // Greedy DSatur graph coloring with balanced color usage
        const colorAssigned = new Array(n).fill(-1);
        const neighborColorSets = Array.from({ length: n }, () => new Set());
        const colorUsageCount = new Array(palette.length).fill(0);

        for (let step = 0; step < n; step++) {
            let bestU = -1;
            let bestSat = -1;
            let bestDeg = -1;
            for (let u = 0; u < n; u++) {
                if (colorAssigned[u] !== -1) continue;
                const sat = neighborColorSets[u].size;
                const deg = adj[u].length;
                if (sat > bestSat || (sat === bestSat && deg > bestDeg)) {
                    bestSat = sat;
                    bestDeg = deg;
                    bestU = u;
                }
            }
            if (bestU === -1) break;

            let chosenColorIdx = -1;
            let minUsage = Infinity;
            for (let c = 0; c < palette.length; c++) {
                const idx = (c + step) % palette.length;
                if (!neighborColorSets[bestU].has(idx)) {
                    if (colorUsageCount[idx] < minUsage) {
                        minUsage = colorUsageCount[idx];
                        chosenColorIdx = idx;
                    }
                }
            }
            if (chosenColorIdx === -1) {
                chosenColorIdx = step % palette.length;
            }

            colorAssigned[bestU] = chosenColorIdx;
            colorUsageCount[chosenColorIdx]++;
            features[bestU].properties.distinct_color = palette[chosenColorIdx];

            for (const v of adj[bestU]) {
                if (colorAssigned[v] === -1) {
                    neighborColorSets[v].add(chosenColorIdx);
                }
            }
        }
    }

    function getFeatureDisplayColor(props) {
        if (!props) return DISTINCT_CROWN_PALETTE[0];
        if (polygonColorMode === 'confidence') {
            return getCanopyConfidenceColor(getConfidencePct(props));
        }
        return props.distinct_color || DISTINCT_CROWN_PALETTE[(Number(props.tree_id) || 0) % DISTINCT_CROWN_PALETTE.length];
    }

    function getSourceModelLabel(sourceModel, geometryQuality) {
        const raw = String(sourceModel || geometryQuality || '').toLowerCase();
        if (raw === 'sam3_native' || raw.includes('sam3')) {
            return 'SAM 3 Single Crown (sam3_native)';
        }
        if (raw === 'deepforest_watershed' || raw.includes('deepforest')) {
            return 'DeepForest 2.1 + Geodesic Watershed (deepforest_watershed)';
        }
        if (raw === 'watershed_forest_crown' || raw.includes('watershed')) {
            return 'Compact Geodesic Watershed (watershed_forest_crown)';
        }
        return sourceModel || geometryQuality || 'Canopy V2.3 Hybrid';
    }

    function clearExtractedLayers() {
        if (canopyGeoJsonLayer) {
            map.removeLayer(canopyGeoJsonLayer);
            canopyGeoJsonLayer = null;
        }
        if (canopyCoverageLayer) {
            map.removeLayer(canopyCoverageLayer);
            canopyCoverageLayer = null;
        }
        if (canopyCoverageCentroidsLayer) {
            map.removeLayer(canopyCoverageCentroidsLayer);
            canopyCoverageCentroidsLayer = null;
        }
        if (crownGroupsLayer) {
            map.removeLayer(crownGroupsLayer);
            crownGroupsLayer = null;
        }
        if (denseCandidatesLayer) {
            map.removeLayer(denseCandidatesLayer);
            denseCandidatesLayer = null;
        }
        if (denseCandidateCentroidsLayer) {
            map.removeLayer(denseCandidateCentroidsLayer);
            denseCandidateCentroidsLayer = null;
        }
        centroidsLayer.clearLayers();
    }

    function renderCanopies(geoJsonData) {
        clearExtractedLayers();
        if (geoJsonData && Array.isArray(geoJsonData.features)) {
            ensureAdjacentDistinctColors(geoJsonData.features);
        }

        const rootStyle = getComputedStyle(document.documentElement);
        const teal = rootStyle.getPropertyValue('--teal').trim() || '#0d9488';
        const warning = rootStyle.getPropertyValue('--warn').trim() || '#d97706';
        const candidateColor = '#2563eb';
        const defaultFillOpacity = Number(sliderPolyOpacity ? sliderPolyOpacity.value : 40) / 100.0;

        function reviewPopupHtml(feature, label) {
            const props = feature.properties || {};
            const area = Number(props.area_m2 || 0);
            const confPct = getConfidencePct(props);
            const scoreLabel = props.feature_class === 'dense_crown_candidate'
                ? 'Crown-top prominence'
                : 'Separation confidence';
            const confidenceLine = confPct > 0
                ? `<br>${scoreLabel}: ${confPct.toFixed(1)}%`
                : '';
            const featureId = props.feature_class === 'dense_crown_candidate'
                ? `D-${props.tree_id}`
                : props.tree_id;
            return (
                `<strong>${label} ${featureId ? `#${featureId}` : ''}</strong><br>` +
                `Area: ${area.toFixed(1)} m²${confidenceLine}<br>` +
                `<span style="color:#64748b">` +
                `${props.qa_note || props.measurement_note || 'Review before counting.'}` +
                `</span>`
            );
        }

        function bindReviewFeature(feature, layer, label) {
            layer.bindPopup(reviewPopupHtml(feature, label));
            layer.on('click', (event) => {
                L.DomEvent.stopPropagation(event);
                layer.openPopup(event.latlng);
            });
        }

        if (geoJsonData.coverage_features?.length) {
            canopyCoverageLayer = L.geoJSON({
                type: 'FeatureCollection',
                features: geoJsonData.coverage_features
            }, {
                style: {
                    color: teal,
                    weight: 1.5,
                    fillColor: teal,
                    fillOpacity: 0.14
                },
                onEachFeature: (feature, layer) => {
                    bindReviewFeature(feature, layer, 'Merged forest footprint');
                }
            });
            if (chkLayerFootprint && chkLayerFootprint.checked) {
                canopyCoverageLayer.addTo(map);
            }
        }

        if (geoJsonData.group_features?.length) {
            crownGroupsLayer = L.geoJSON({
                type: 'FeatureCollection',
                features: geoJsonData.group_features
            }, {
                style: {
                    color: warning,
                    weight: 1.5,
                    fillColor: warning,
                    fillOpacity: 0.20,
                    dashArray: '5 4'
                },
                onEachFeature: (feature, layer) => {
                    bindReviewFeature(feature, layer, 'Unresolved crown group');
                }
            });
            if (chkLayerGroups && chkLayerGroups.checked) {
                crownGroupsLayer.addTo(map);
            }
        }

        if (geoJsonData.candidate_features?.length) {
            denseCandidatesLayer = L.geoJSON({
                type: 'FeatureCollection',
                features: geoJsonData.candidate_features
            }, {
                pane: 'denseCandidatesPane',
                renderer: denseCandidateRenderer,
                interactive: true,
                bubblingMouseEvents: false,
                style: {
                    color: candidateColor,
                    weight: 1.5,
                    fillColor: candidateColor,
                    fillOpacity: 0.18,
                    dashArray: '3 3'
                },
                onEachFeature: (feature, layer) => {
                    bindReviewFeature(feature, layer, 'Dense crown candidate');
                    layer.bindTooltip(
                        `Dense crown D-${feature.properties.tree_id}`,
                        { sticky: true }
                    );
                }
            });
            if (chkLayerCandidates && chkLayerCandidates.checked) {
                denseCandidatesLayer.addTo(map);
            }
        }

        canopyGeoJsonLayer = L.geoJSON(geoJsonData, {
            style: (feature) => {
                const props = feature.properties || {};
                const color = getFeatureDisplayColor(props);
                return {
                    color: color,
                    weight: 1.5,
                    fillColor: color,
                    fillOpacity: defaultFillOpacity,
                    className: `canopy-poly-${props.tree_id}`
                };
            },
            onEachFeature: (feature, layer) => {
                const props = feature.properties;
                const srcLabel = getSourceModelLabel(props.source_model, props.geometry_quality || props.boundary_quality);
                const confPct = getConfidencePct(props);
                const tierLabel = getConfidenceTierLabel(confPct, props);
                const phys = getFeaturePhysicalScores(props);
                const color = getFeatureDisplayColor(props);
                layer.bindTooltip(
                    `<strong>Tree #${props.tree_id}</strong> (${props.area_m2} m²)<br>` +
                    `${srcLabel}<br>` +
                    `Confidence: <strong>${confPct.toFixed(1)}%</strong> (${tierLabel})<br>` +
                    `<span style="font-size:11px;opacity:0.88;">3D Relief: ${(phys.relief3d * 100).toFixed(0)}% · Foliage: ${(phys.foliageProb * 100).toFixed(0)}% · Cyan: ${(phys.cyanScore * 100).toFixed(0)}%</span>`,
                    { sticky: true, direction: 'top' }
                );
                
                layer.on({
                    mouseover: (e) => {
                        if (selectedTreeId !== props.tree_id) {
                            const target = e.target;
                            const baseOp = Number(sliderPolyOpacity ? sliderPolyOpacity.value : 40) / 100.0;
                            target.setStyle({
                                fillOpacity: Math.min(0.85, baseOp + 0.25),
                                weight: 2.5,
                                color: '#047857'
                            });
                            highlightTableRow(props.tree_id);
                        }
                    },
                    mouseout: (e) => {
                        if (selectedTreeId !== props.tree_id) {
                            const curColor = getFeatureDisplayColor(props);
                            const baseOp = Number(sliderPolyOpacity ? sliderPolyOpacity.value : 40) / 100.0;
                            e.target.setStyle({
                                color: curColor,
                                fillColor: curColor,
                                weight: 1.5,
                                fillOpacity: baseOp
                            });
                            unhighlightTableRow(props.tree_id);
                        }
                    },
                    click: (e) => {
                        L.DomEvent.stopPropagation(e);
                        selectTree(props.tree_id, true);
                    }
                });

                const centroidMarker = L.circleMarker([props.centroid_lat, props.centroid_lon], {
                    radius: 3,
                    fillColor: "#ffffff",
                    color: color,
                    weight: 1.5,
                    fillOpacity: 0.9,
                    opacity: 0.85
                });
                
                centroidMarker.tree_id = props.tree_id;
                centroidMarker.featureProps = props;

                centroidMarker.on('click', (e) => {
                    L.DomEvent.stopPropagation(e);
                    selectTree(props.tree_id, true);
                });

                centroidsLayer.addLayer(centroidMarker);
            }
        });

        if (!chkLayerPolygons || chkLayerPolygons.checked) {
            canopyGeoJsonLayer.addTo(map);
        }
        if (chkLayerPoints && !chkLayerPoints.checked) {
            if (map.hasLayer(centroidsLayer)) map.removeLayer(centroidsLayer);
        } else if (!map.hasLayer(centroidsLayer)) {
            centroidsLayer.addTo(map);
        }

        // Initial table load
        sortAndRenderTable();
    }

    // ----------------------------------------------------
    // 11. Results Table Management (Sorting, Filtering, Highlight)
    // ----------------------------------------------------
    function formatMetric(value, unit, digits = null) {
        if (value === null || value === undefined || !Number.isFinite(Number(value))) {
            return '—';
        }
        const numeric = Number(value);
        const rendered = digits === null ? numeric : numeric.toFixed(digits);
        return `${rendered} ${unit}`;
    }

    function resetTable() {
        canopyTableBody.innerHTML = `
            <tr class="empty-state">
                <td colspan="9">Draw a search polygon or upload a raster and run extraction to populate results.</td>
            </tr>
        `;
        if (canopyCountBadge) canopyCountBadge.innerText = '0 detected';
        if (liveFilterCountBadge) liveFilterCountBadge.innerText = '0 / 0 Crowns';
        lastFilteredFeatures = [];
    }

    function renderFilteredTable(features) {
        const totalCount = extractedGeoJson && Array.isArray(extractedGeoJson.features)
            ? extractedGeoJson.features.length
            : features.length;
        if (liveFilterCountBadge) {
            liveFilterCountBadge.innerText =
                features.length === totalCount
                    ? `Showing All ${totalCount.toLocaleString()} Crowns`
                    : `Showing ${features.length.toLocaleString()} / ${totalCount.toLocaleString()} Crowns`;
        }
        if (features.length === 0) {
            canopyTableBody.innerHTML = `
                <tr class="empty-state">
                    <td colspan="9">No plant canopies match active search criteria.</td>
                </tr>
            `;
            if (canopyCountBadge) canopyCountBadge.innerText = '0 crowns · 0.0% avg confidence';
            updateMapLayers([]);
            return;
        }

        const meanConf = features.reduce((sum, f) => sum + getConfidencePct(f.properties), 0) / features.length;
        if (canopyCountBadge) {
            canopyCountBadge.innerText = `${features.length.toLocaleString()} crowns · ${meanConf.toFixed(1)}% avg confidence`;
        }
        canopyTableBody.innerHTML = '';

        features.forEach(feat => {
            const props = feat.properties;
            const row = document.createElement('tr');
            row.id = `table-row-${props.tree_id}`;
            
            if (selectedTreeId === props.tree_id) {
                row.classList.add('active-row');
            }

            const confPct = getConfidencePct(props);
            const confColor = getCanopyConfidenceColor(confPct);
            const tierLabel = getConfidenceTierLabel(confPct, props);
            const srcLabel = getSourceModelLabel(props.source_model, props.geometry_quality || props.boundary_quality);
            
            row.innerHTML = `
                <td title="${srcLabel}">#${props.tree_id}</td>
                <td class="coords-cell">${props.centroid_lat.toFixed(6)}, ${props.centroid_lon.toFixed(6)}</td>
                <td class="area-cell">${props.area_m2} m²</td>
                <td>${props.crown_diameter_m} m</td>
                <td>${formatMetric(props.estimated_height_m, 'm')}</td>
                <td>${formatMetric(props.estimated_dbh_cm, 'cm')}</td>
                <td>${formatMetric(props.estimated_volume_m3, 'm³')}</td>
                <td style="font-weight:700; color: ${confColor}" title="${tierLabel}">${confPct.toFixed(1)}%</td>
                <td><button class="btn-zoom" data-id="${props.tree_id}">Focus</button></td>
            `;

            row.addEventListener('mouseenter', () => {
                if (selectedTreeId !== props.tree_id) {
                    row.classList.add('active-row');
                    highlightMapFeature(props.tree_id);
                }
            });
            row.addEventListener('mouseleave', () => {
                if (selectedTreeId !== props.tree_id) {
                    row.classList.remove('active-row');
                    unhighlightMapFeature(props.tree_id);
                }
            });

            row.addEventListener('click', () => {
                selectTree(props.tree_id, false);
            });

            row.querySelector('.btn-zoom').addEventListener('click', (e) => {
                e.stopPropagation();
                selectTree(props.tree_id, true);
            });

            canopyTableBody.appendChild(row);
        });

        // Sync map layer visibilities with currently filtered elements
        updateMapLayers(features);
    }

    function selectTree(treeId, fitMap = false) {
        selectedTreeId = treeId;
        
        // Remove previous row active styles
        document.querySelectorAll('#canopy-table tbody tr').forEach(r => r.classList.remove('active-row'));
        
        // Add active style to selected row
        const activeRow = document.getElementById(`table-row-${treeId}`);
        if (activeRow) {
            activeRow.classList.add('active-row');
            activeRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
        
        const baseOp = Number(sliderPolyOpacity ? sliderPolyOpacity.value : 40) / 100.0;
        // Reset and highlight map features
        if (canopyGeoJsonLayer) {
            canopyGeoJsonLayer.eachLayer(layer => {
                if (layer.feature.properties.tree_id === treeId) {
                    layer.setStyle({
                        fillOpacity: Math.min(0.85, baseOp + 0.25),
                        weight: 3,
                        color: "#047857"
                    });
                    
                    if (fitMap) {
                        map.setView([layer.feature.properties.centroid_lat, layer.feature.properties.centroid_lon], Math.max(map.getZoom(), 21));
                    }
                    
                    openDetailDrawer(layer.feature.properties);
                } else {
                    canopyGeoJsonLayer.resetStyle(layer);
                }
            });
            updateMapLayers(lastFilteredFeatures);
        }
    }

    function highlightTableRow(id) {
        const row = document.getElementById(`table-row-${id}`);
        if (row) {
            row.classList.add('active-row');
        }
    }

    function unhighlightTableRow(id) {
        if (selectedTreeId !== id) {
            const row = document.getElementById(`table-row-${id}`);
            if (row) {
                row.classList.remove('active-row');
            }
        }
    }

    function highlightMapFeature(id) {
        activeHighlightId = id;
        const baseOp = Number(sliderPolyOpacity ? sliderPolyOpacity.value : 40) / 100.0;
        if (canopyGeoJsonLayer) {
            canopyGeoJsonLayer.eachLayer(layer => {
                if (layer.feature.properties.tree_id === id) {
                    layer.setStyle({
                        fillOpacity: Math.min(0.85, baseOp + 0.25),
                        weight: 3,
                        color: "#047857"
                    });
                }
            });
        }
    }

    function unhighlightMapFeature(id) {
        activeHighlightId = null;
        if (canopyGeoJsonLayer && selectedTreeId !== id) {
            updateMapLayers(lastFilteredFeatures);
        }
    }

    // ----------------------------------------------------
    // 12. Details Drawer Panel Handlers
    // ----------------------------------------------------
    btnCloseDrawer.addEventListener('click', () => {
        closeDetailDrawer();
    });

    function closeDetailDrawer() {
        detailDrawer.classList.add('collapsed');
        selectedTreeId = null;
        
        // Reset active selections
        document.querySelectorAll('#canopy-table tbody tr').forEach(r => r.classList.remove('active-row'));
        if (canopyGeoJsonLayer) {
            canopyGeoJsonLayer.eachLayer(layer => {
                canopyGeoJsonLayer.resetStyle(layer);
            });
            updateMapLayers(lastFilteredFeatures);
        }
    }

    function openDetailDrawer(props) {
        detailDrawer.classList.remove('collapsed');
        
        // Fill IDs & Badges
        drawerTreeIdBadge.innerText = `#${props.tree_id}`;
        polaroidIdText.innerText = props.tree_id;
        
        // Coordinates and stats
        if (drawerSourceModel) {
            drawerSourceModel.innerText = getSourceModelLabel(
                props.source_model,
                props.geometry_quality || props.boundary_quality
            );
        }
        drawerLat.innerText = props.centroid_lat.toFixed(6);
        drawerLng.innerText = props.centroid_lon.toFixed(6);
        drawerArea.innerText = `${props.area_m2} m²`;
        drawerDiameter.innerText = `${props.crown_diameter_m} m`;
        drawerHeight.innerText = formatMetric(props.estimated_height_m, 'm');
        drawerDbh.innerText = formatMetric(props.estimated_dbh_cm, 'cm');
        const physScores = getFeaturePhysicalScores(props);
        if (drawerRelief) drawerRelief.innerText = `${(physScores.relief3d * 100).toFixed(1)}%`;
        if (drawerFoliage) drawerFoliage.innerText = `${(physScores.foliageProb * 100).toFixed(1)}%`;
        if (drawerCyan) drawerCyan.innerText = `${(physScores.cyanScore * 100).toFixed(1)}%`;
        
        // Separation Confidence (0-100%) Indicators
        const confPct = getConfidencePct(props);
        const confColor = getCanopyConfidenceColor(confPct);
        const tierLabel = getConfidenceTierLabel(confPct, props);
        drawerNdviVal.innerText = `${confPct.toFixed(1)}%`;
        drawerNdviBar.style.width = `${confPct}%`;
        drawerNdviBar.style.backgroundColor = confColor;
        drawerHealthLabel.innerText = tierLabel;
        drawerHealthLabel.style.color = confColor;
        
        // Volume Indicators
        const vol = props.estimated_volume_m3;
        drawerVolumeVal.innerText = formatMetric(vol, 'm³', 3);
        const volPct = vol === null || vol === undefined
            ? 0
            : Math.max(0, Math.min(100, (vol / 4.0) * 100));
        drawerVolumeBar.style.width = `${volPct}%`;

        // Render cropped canvas image
        drawPolaroidCrop(props);
        lucide.createIcons();
    }

    function drawPolaroidCrop(props) {
        const ctx = polaroidCanvas.getContext('2d');
        polaroidCanvas.width = 250;
        polaroidCanvas.height = 250;

        // Clear with background light color
        ctx.fillStyle = '#faf8f5';
        ctx.fillRect(0, 0, polaroidCanvas.width, polaroidCanvas.height);

        if (!uploadedMetadata || !uploadedMetadata.preview_url) {
            ctx.fillStyle = '#808c79';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No image preview loaded', 125, 125);
            return;
        }

        const imgUrl = `${BACKEND_URL}${uploadedMetadata.preview_url}`;
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.src = imgUrl;

        img.onload = () => {
            const bounds = props.pixel_bounds; // [x, y, w, h]
            if (!bounds || bounds.length !== 4) return;

            const [px, py, pw, ph] = bounds;

            const scaleX = img.naturalWidth / uploadedMetadata.width;
            const scaleY = img.naturalHeight / uploadedMetadata.height;

            const sx = px * scaleX;
            const sy = py * scaleY;
            const sw = pw * scaleX;
            const sh = ph * scaleY;

            const padFactor = 0.20;
            const padX = sw * padFactor;
            const padY = sh * padFactor;

            const cropX = Math.max(0, sx - padX);
            const cropY = Math.max(0, sy - padY);
            const cropW = Math.min(img.naturalWidth - cropX, sw + 2 * padX);
            const cropH = Math.min(img.naturalHeight - cropY, sh + 2 * padY);

            const aspect = cropW / cropH;
            let dx = 0, dy = 0, dWidth = polaroidCanvas.width, dHeight = polaroidCanvas.height;

            if (aspect > 1) {
                dHeight = polaroidCanvas.width / aspect;
                dy = (polaroidCanvas.height - dHeight) / 2;
            } else {
                dWidth = polaroidCanvas.height * aspect;
                dx = (polaroidCanvas.width - dWidth) / 2;
            }

            ctx.drawImage(img, cropX, cropY, cropW, cropH, dx, dy, dWidth, dHeight);
        };

        img.onerror = () => {
            ctx.fillStyle = '#ef4444';
            ctx.font = '11px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Failed to crop imagery', 125, 125);
        };
    }

    // ----------------------------------------------------
    // 13. Sorting & Filtering Handlers
    // ----------------------------------------------------
    btnToggleFilters.addEventListener('click', (e) => {
        e.stopPropagation();
        const isHidden = filterRow.classList.toggle('hidden');
        btnToggleFilters.classList.toggle('active', !isHidden);
        lucide.createIcons();
    });

    btnClearFilters.addEventListener('click', (e) => {
        e.stopPropagation();
        filterInputs.forEach(input => {
            input.value = '';
        });
        Object.keys(filters).forEach(k => filters[k] = '');
        resetLivePostFilters();
        sortAndRenderTable();
    });

    filterInputs.forEach(input => {
        input.addEventListener('input', () => {
            const colName = input.getAttribute('data-col');
            filters[colName] = input.value.toLowerCase().trim();
            sortAndRenderTable();
        });
    });

    tableHeaderCols.forEach(th => {
        th.addEventListener('click', () => {
            const col = th.getAttribute('data-col');
            if (currentSortCol === col) {
                if (currentSortDir === 'asc') {
                    currentSortDir = 'desc';
                } else if (currentSortDir === 'desc') {
                    currentSortDir = 'none';
                } else {
                    currentSortDir = 'asc';
                }
            } else {
                currentSortCol = col;
                currentSortDir = 'asc';
            }
            
            tableHeaderCols.forEach(el => {
                el.classList.remove('asc', 'desc');
            });
            
            if (currentSortDir !== 'none') {
                th.classList.add(currentSortDir);
            }
            
            sortAndRenderTable();
        });
    });

    function sortAndRenderTable() {
        if (!extractedGeoJson || !extractedGeoJson.features) return;

        let items = [...extractedGeoJson.features];

        // 1. Apply active text inputs + min-confidence slider filters
        items = applyActiveFilters(items);

        // 2. Apply active column sorting
        if (currentSortCol && currentSortDir !== 'none') {
            items.sort((a, b) => {
                let valA = currentSortCol === 'confidence_pct'
                    ? getConfidencePct(a.properties)
                    : a.properties[currentSortCol];
                let valB = currentSortCol === 'confidence_pct'
                    ? getConfidencePct(b.properties)
                    : b.properties[currentSortCol];

                if (currentSortCol === 'centroid') {
                    valA = a.properties.centroid_lat;
                    valB = b.properties.centroid_lat;
                }

                const aNull = valA === null || valA === undefined;
                const bNull = valB === null || valB === undefined;
                if (aNull && bNull) return 0;
                if (aNull) return 1;
                if (bNull) return -1;

                if (typeof valA === 'string') {
                    return currentSortDir === 'asc' 
                        ? valA.localeCompare(String(valB)) 
                        : String(valB).localeCompare(valA);
                } else {
                    return currentSortDir === 'asc' 
                        ? valA - valB 
                        : valB - valA;
                }
            });
        }

        // 3. Render
        renderFilteredTable(items);
    }

    function getFeaturePhysicalScores(props) {
        if (!props) {
            return { foliageProb: 0.5, relief3d: 0.5, cyanScore: 0.0 };
        }
        if (
            typeof props.relief_3d_score === 'number' &&
            typeof props.artificial_cyan_score === 'number' &&
            typeof props.foliage_prob === 'number'
        ) {
            return {
                foliageProb: props.foliage_prob,
                relief3d: props.relief_3d_score,
                cyanScore: props.artificial_cyan_score
            };
        }
        const rgb = Array.isArray(props.mean_rgb) && props.mean_rgb.length >= 3
            ? props.mean_rgb
            : [52, 82, 62];
        const rM = Number(rgb[0]) || 52;
        const gM = Number(rgb[1]) || 82;
        const bM = Number(rgb[2]) || 62;
        const gliM = typeof props.foliar_gli === 'number'
            ? props.foliar_gli
            : (2 * gM - rM - bM) / (2 * gM + rM + bM + 1e-5);
        const variM = typeof props.raw_vari === 'number'
            ? props.raw_vari
            : (gM - rM) / (gM + rM - bM + 1e-5);
        const confPct = getConfidencePct(props);
        const gliNorm = Math.max(0, Math.min(1, gliM / 0.14));
        const variNorm = Math.max(0, Math.min(1, (variM + 0.05) / 0.25));
        const foliageProb = Math.max(0, Math.min(1, 0.60 * gliNorm + 0.40 * variNorm));
        const confNorm = Math.max(0, Math.min(1, (confPct - 25.0) / 65.0));
        const chlDepth = Math.max(0, Math.min(1, ((gM - rM) - 8.0) / 24.0));
        const darkWoody = Math.max(0, Math.min(1, (92.0 - rM) / 55.0));
        const relief3d = Math.max(0, Math.min(1, 0.50 * confNorm + 0.30 * chlDepth + 0.20 * darkWoody));
        const brExcess = (bM - rM) - 8.0;
        const bgRatio = (bM - (gM - 12.0)) / 18.0;
        const cyanChroma = Math.max(0, Math.min(1, 0.60 * Math.max(0, brExcess / 18.0) + 0.40 * Math.max(0, bgRatio)));
        const brightFactor = Math.max(0.15, Math.min(1, 0.55 * (1.0 - darkWoody) + 0.45 * (1.0 - 0.5 * confNorm)));
        const cyanScore = Math.max(0, Math.min(1, cyanChroma * brightFactor));
        return { foliageProb, relief3d, cyanScore };
    }

    function normalizeFeatureEngineKey(props) {
        const raw = String(props?.source_model || props?.geometry_quality || '').toLowerCase();
        if (raw.includes('sam3')) return 'sam3';
        if (raw.includes('deepforest')) return 'deepforest_2.1';
        return 'compact_watershed';
    }

    function applyActiveFilters(features) {
        const minConf = Number(sliderMinConf ? sliderMinConf.value : 0);
        const minReliefPct = Number(sliderMinRelief ? sliderMinRelief.value : 0);
        const cyanStrictnessPct = Number(sliderMaxCyan ? sliderMaxCyan.value : 0);
        const liveMinArea = Number(sliderLiveMinArea ? sliderLiveMinArea.value : 0);
        const liveMaxArea = Number(sliderLiveMaxArea ? sliderLiveMaxArea.value : 1000);

        return features.filter(feat => {
            const props = feat.properties;
            const confPct = getConfidencePct(props);

            if (confPct < minConf) {
                return false;
            }

            const engKey = normalizeFeatureEngineKey(props);
            if (!activeSourceEngines.has(engKey)) {
                return false;
            }

            const areaM2 = Number(props.area_m2) || 0;
            if (liveMinArea > 0 && areaM2 < liveMinArea) {
                return false;
            }
            if (liveMaxArea < 1000 && areaM2 > liveMaxArea) {
                return false;
            }

            if (minReliefPct > 0 || cyanStrictnessPct > 0) {
                const phys = getFeaturePhysicalScores(props);
                if (minReliefPct > 0 && (phys.relief3d * 100.0) < minReliefPct) {
                    return false;
                }
                if (cyanStrictnessPct > 0 && (phys.cyanScore * 100.0) > (95.0 - cyanStrictnessPct)) {
                    return false;
                }
            }

            if (filters.tree_id && !props.tree_id.toString().includes(filters.tree_id)) {
                return false;
            }

            if (filters.centroid) {
                const coordStr = `${props.centroid_lat.toFixed(6)}, ${props.centroid_lon.toFixed(6)}`;
                if (!coordStr.includes(filters.centroid)) return false;
            }

            const numericColumns = ['area_m2', 'crown_diameter_m', 'estimated_height_m', 'estimated_dbh_cm', 'estimated_volume_m3', 'confidence_pct'];
            for (const col of numericColumns) {
                const query = filters[col];
                if (query) {
                    const recordVal = col === 'confidence_pct' ? Number(confPct.toFixed(1)) : props[col];
                    if (recordVal === null || recordVal === undefined) return false;

                    if (query.startsWith('>')) {
                        const num = parseFloat(query.substring(1));
                        if (isNaN(num) || recordVal <= num) return false;
                    } else if (query.startsWith('<')) {
                        const num = parseFloat(query.substring(1));
                        if (isNaN(num) || recordVal >= num) return false;
                    } else {
                        if (!recordVal.toString().includes(query)) return false;
                    }
                }
            }
            return true;
        });
    }

    function updateMapLayers(filteredFeatures) {
        lastFilteredFeatures = Array.isArray(filteredFeatures) ? filteredFeatures : [];

        const showPolygons = !chkLayerPolygons || chkLayerPolygons.checked;
        const showPoints = !chkLayerPoints || chkLayerPoints.checked;
        const polyOpacity = Number(sliderPolyOpacity ? sliderPolyOpacity.value : 40) / 100.0;
        const activeIds = new Set(lastFilteredFeatures.map(f => f.properties.tree_id));

        if (canopyGeoJsonLayer) {
            if (!showPolygons) {
                if (map.hasLayer(canopyGeoJsonLayer)) map.removeLayer(canopyGeoJsonLayer);
            } else {
                if (!map.hasLayer(canopyGeoJsonLayer)) canopyGeoJsonLayer.addTo(map);
                canopyGeoJsonLayer.eachLayer(layer => {
                    const props = layer.feature.properties;
                    const id = props.tree_id;
                    const displayColor = getFeatureDisplayColor(props);
                    if (activeIds.has(id)) {
                        const isSelected = selectedTreeId === id;
                        layer.setStyle({
                            color: isSelected ? '#047857' : displayColor,
                            fillColor: displayColor,
                            weight: isSelected ? 3 : 1.5,
                            opacity: 0.95,
                            fillOpacity: isSelected ? Math.min(0.85, polyOpacity + 0.25) : polyOpacity,
                            stroke: true
                        });
                    } else {
                        layer.setStyle({
                            opacity: 0,
                            fillOpacity: 0,
                            stroke: false
                        });
                        if (layer.isPopupOpen()) layer.closePopup();
                    }
                });
            }
        }

        if (centroidsLayer) {
            if (!showPoints) {
                if (map.hasLayer(centroidsLayer)) map.removeLayer(centroidsLayer);
            } else {
                if (!map.hasLayer(centroidsLayer)) centroidsLayer.addTo(map);
                centroidsLayer.eachLayer(marker => {
                    if (marker.tree_id) {
                        if (activeIds.has(marker.tree_id)) {
                            const mColor = marker.featureProps
                                ? getFeatureDisplayColor(marker.featureProps)
                                : '#059669';
                            marker.setStyle({
                                color: mColor,
                                opacity: 0.85,
                                fillOpacity: 0.9
                            });
                        } else {
                            marker.setStyle({ opacity: 0, fillOpacity: 0 });
                            if (marker.isPopupOpen()) marker.closePopup();
                        }
                    }
                });
            }
        }
    }

    // ----------------------------------------------------
    // 14. Data Exporters
    // ----------------------------------------------------
    function downloadFile(content, filename, contentType) {
        const blob = new Blob([content], { type: contentType });
        const url = URL.createObjectURL(blob);
        const link = Object.assign(document.createElement("a"), { href: url, download: filename });
        link.click();
        URL.revokeObjectURL(url);
    }

    btnExportCsv.addEventListener('click', () => {
        if (!extractedGeoJson || extractedGeoJson.features.length === 0) return;
        
        let exportedFeatures = applyActiveFilters(extractedGeoJson.features);
        if (exportedFeatures.length === 0) {
            alert('No matching records to export with current filters.');
            return;
        }

        const csvContent = ["Tree_ID,Latitude,Longitude,Area_m2,Crown_Diameter_m,Estimated_Height_m,Estimated_DBH_cm,Estimated_Volume_m3,Confidence_Pct,Relief_3D_Score,Foliage_Prob,Artificial_Cyan_Score,Source_Model,Circularity,Pixel_X,Pixel_Y"]
            .concat(exportedFeatures.map(feat => {
                const props = feat.properties;
                const confPct = getConfidencePct(props).toFixed(1);
                const phys = getFeaturePhysicalScores(props);
                return `${props.tree_id},${props.centroid_lat},${props.centroid_lon},${props.area_m2},${props.crown_diameter_m},${props.estimated_height_m},${props.estimated_dbh_cm},${props.estimated_volume_m3},${confPct},${phys.relief3d.toFixed(3)},${phys.foliageProb.toFixed(3)},${phys.cyanScore.toFixed(3)},${props.source_model || props.geometry_quality || 'canopy_v2'},${props.circularity},${props.pixel_cx},${props.pixel_cy}`;
            })).join("\n");
        const mode = uploadedMetadata && uploadedMetadata.pipeline_mode === 'image_only' ? 'image_only' : 'geo';
        if (mode === 'image_only') {
            const ok = confirm(
                'This is IMAGE-ONLY mode (no real georeference).\n\nExport will use preview coordinates and assumed GSD — not survey-grade for Rhino/MCDA.\n\nContinue export?'
            );
            if (!ok) return;
        }
        const fileBase = uploadedMetadata ? uploadedMetadata.filename.split('.')[0] : 'canopy';
        downloadFile(csvContent, `${fileBase}_${mode}_centroids.csv`, 'text/csv;charset=utf-8;');
    });

    btnExportGeoJson.addEventListener('click', () => {
        if (!extractedGeoJson) return;
        
        let exportedGeoJson = { ...extractedGeoJson };
        exportedGeoJson.features = applyActiveFilters(extractedGeoJson.features);
        
        if (exportedGeoJson.features.length === 0) {
            alert('No matching records to export with current filters.');
            return;
        }

        const mode = uploadedMetadata && uploadedMetadata.pipeline_mode === 'image_only' ? 'image_only' : 'geo';
        if (mode === 'image_only') {
            const ok = confirm(
                'IMAGE-ONLY GeoJSON uses preview map placement and assumed GSD.\n\nNot for primary Rhino/MCDA coordinates. Prefer GeoTIFF or Map extent (GEO).\n\nExport anyway?'
            );
            if (!ok) return;
        }
        if (!exportedGeoJson.properties) exportedGeoJson.properties = {};
        exportedGeoJson.properties.pipeline_mode = mode;
        exportedGeoJson.properties.exported_at = new Date().toISOString();
        if (uploadedMetadata) {
            exportedGeoJson.properties.source_mode_message = uploadedMetadata.mode_message || '';
            exportedGeoJson.properties.georef_quality = uploadedMetadata.georef_quality || '';
        }

        const fileBase = uploadedMetadata ? uploadedMetadata.filename.split('.')[0] : 'canopy';
        downloadFile(
            JSON.stringify(exportedGeoJson, null, 2),
            `${fileBase}_${mode}_boundaries.geojson`,
            'application/json;charset=utf-8;'
        );
    });
});
