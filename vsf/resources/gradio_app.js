() => {
    const POPOVER_SCRIPT_VERSION = "fast-advanced-popover-v2";
    if (window.__vsfAdvancedPopoverInstalled === POPOVER_SCRIPT_VERSION) {
        return;
    }
    window.__vsfAdvancedPopoverInstalled = POPOVER_SCRIPT_VERSION;

    const POPOVER_OPEN_CLASS = "vsf-advanced-popover-open";
    const CLIENT_OPTIONS_VISIBLE_CLASS = "vsf-show-client-options";
    const IMAGE_ANALYSIS_VISIBLE_CLASS = "vsf-show-image-analysis";
    const OCR_LANGUAGE_VISIBLE_CLASS = "vsf-show-ocr-language";
    const FORCE_OCR_HIDDEN_CLASS = "vsf-hide-force-ocr";
    const HYBRID_EFFORT_HIDDEN_CLASS = "vsf-hide-hybrid-effort";
    const OFFICE_PREVIEW_NOTICE_STORAGE_KEY = "vsf.officePreviewNoticeIgnored";
    const ANIMATION_DELAY_MS = 90;
    const CLIPBOARD_MIME_EXTENSIONS = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/bmp": "bmp",
        "image/tiff": "tiff",
        "application/pdf": "pdf",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    };
    // Implementation detail.
    const normalizeVSFLocale = (locale) => {
        const normalized = String(locale || "").toLowerCase();
        if (normalized.startsWith("zh")) {
            return "zh";
        }
        return "en";
    };

    // Implementation detail.
    const resolveVSFLocale = () => {
        if (typeof navigator !== "undefined") {
            const languages = Array.from(navigator.languages || []);
            const primaryLocale = languages[0] || navigator.language;
            if (primaryLocale) {
                return normalizeVSFLocale(primaryLocale);
            }
        }
        return normalizeVSFLocale(document.documentElement.getAttribute("lang"));
    };

    // Implementation detail.
    const localizeVSFCustomText = () => {
        const locale = resolveVSFLocale();
        document.querySelectorAll("[data-vsf-i18n-key]").forEach((item) => {
            const localizedText = item.getAttribute(`data-vsf-i18n-${locale}`)
                || item.getAttribute("data-vsf-i18n-en");
            if (localizedText !== null && item.textContent !== localizedText) {
                item.textContent = localizedText;
            }
        });
    };

    // Extract the required value.
    const getOfficePreviewNoticeIgnored = () => {
        try {
            return localStorage.getItem(OFFICE_PREVIEW_NOTICE_STORAGE_KEY) === "1";
        } catch (error) {
            return false;
        }
    };

    // Add the value to the result.
    const setOfficePreviewNoticeIgnored = () => {
        try {
            localStorage.setItem(OFFICE_PREVIEW_NOTICE_STORAGE_KEY, "1");
        } catch (error) {
            return false;
        }
        return true;
    };

    const findOfficePreviewNotices = () =>
        document.querySelectorAll(".office-preview-notice");

    // Implementation detail.
    const applyOfficePreviewNoticePreference = () => {
        if (!getOfficePreviewNoticeIgnored()) {
            return;
        }
        findOfficePreviewNotices().forEach((notice) => {
            notice.classList.add("is-dismissed");
        });
    };

    // Implementation detail.
    const refreshVSFCustomHtml = () => {
        localizeVSFCustomText();
        applyOfficePreviewNoticePreference();
        refreshVSFOptionVisibility();
    };

    // Implementation detail.
    const findButton = () => document.querySelector(
        "button.vsf-advanced-open, .vsf-advanced-open button, .vsf-advanced-open"
    );
    const findPopover = () => document.querySelector(".vsf-advanced-popover");
    const findBackendRoot = () => document.querySelector(".vsf-backend-select");
    const findEffortRoot = () => document.querySelector(".vsf-hybrid-effort");
    let visibilityTimer = null;
    let optionRefreshFrame = null;
    let customHtmlRefreshFrame = null;

    // Validate the current value.
    const getBackendValue = () => {
        const backendRoot = findBackendRoot();
        const backendControl = backendRoot?.querySelector('[role="listbox"]');
        return (backendControl?.value || backendControl?.textContent || "").trim();
    };

    // Extract the required value.
    const getEffortValue = () => {
        const effortRoot = findEffortRoot();
        const checkedRadio = effortRoot?.querySelector(
            'input[type="radio"]:checked, input[type="radio"][aria-checked="true"]'
        );
        return (checkedRadio?.value || "").trim();
    };

    // Implementation detail.
    const refreshVSFOptionVisibility = () => {
        const backend = getBackendValue();
        const effort = getEffortValue();
        const showClientOptions = backend.endsWith("http-client");
        const showImageAnalysis = backend.startsWith("vlm")
            || (backend.startsWith("hybrid") && effort === "high");
        const showOcrLanguage = backend === "pipeline";
        const hideForceOcr = backend !== "pipeline" && !backend.startsWith("hybrid");
        const hideHybridEffort = !backend.startsWith("hybrid");

        document.body.classList.toggle(CLIENT_OPTIONS_VISIBLE_CLASS, showClientOptions);
        document.body.classList.toggle(IMAGE_ANALYSIS_VISIBLE_CLASS, showImageAnalysis);
        document.body.classList.toggle(OCR_LANGUAGE_VISIBLE_CLASS, showOcrLanguage);
        document.body.classList.toggle(FORCE_OCR_HIDDEN_CLASS, hideForceOcr);
        document.body.classList.toggle(HYBRID_EFFORT_HIDDEN_CLASS, hideHybridEffort);
        if (document.body.classList.contains(POPOVER_OPEN_CLASS)) {
            positionPopover();
        }
    };

    // Implementation detail.
    const queueVSFOptionVisibilityRefresh = () => {
        if (optionRefreshFrame !== null) {
            return;
        }
        optionRefreshFrame = requestAnimationFrame(() => {
            optionRefreshFrame = null;
            refreshVSFOptionVisibility();
        });
    };
    const findUploadFileInput = () => {
        const uploadRoot = document.querySelector(".vsf-upload-file");
        if (!uploadRoot) {
            return null;
        }
        return uploadRoot.querySelector('input[type="file"]');
    };

    // Extract the required value.
    const getUploadAcceptedTypes = (uploadInput) => {
        const accept = uploadInput?.getAttribute("accept") || "";
        return accept.split(",").map((item) => item.trim().toLowerCase()).filter(Boolean);
    };

    // Validate the current value.
    const fileMatchesAcceptedType = (file, acceptedTypes) => {
        if (!acceptedTypes.length) {
            return true;
        }
        const name = (file.name || "").toLowerCase();
        const type = (file.type || "").toLowerCase();
        return acceptedTypes.some((accepted) => {
            if (accepted.startsWith(".")) {
                return name.endsWith(accepted);
            }
            if (accepted.endsWith("/*")) {
                return type.startsWith(accepted.slice(0, -1));
            }
            return type === accepted;
        });
    };

    // Parse the input data.
    const buildClipboardFileName = (file) => {
        const type = (file.type || "").toLowerCase();
        const extension = CLIPBOARD_MIME_EXTENSIONS[type];
        if (!extension) {
            return "";
        }
        const timestamp = new Date().toISOString()
            .replace(/[-:]/g, "")
            .replace(/[.].+/, "")
            .replace("T", "-");
        const prefix = type.startsWith("image/") ? "clipboard-image" : "clipboard-file";
        return `${prefix}-${timestamp}.${extension}`;
    };

    // Process the file path.
    const normalizeClipboardFile = (file) => {
        if (/[.][^.]+$/.test(file.name || "")) {
            return file;
        }
        const fileName = buildClipboardFileName(file);
        if (!fileName || typeof File === "undefined") {
            return file;
        }
        return new File([file], fileName, {
            type: file.type,
            lastModified: file.lastModified || Date.now(),
        });
    };

    // Implementation detail.
    const collectClipboardFiles = (clipboardData) => {
        const files = Array.from(clipboardData.files || []);
        if (files.length) {
            return files;
        }
        return Array.from(clipboardData.items || [])
            .filter((item) => item.kind === "file")
            .map((item) => item.getAsFile())
            .filter(Boolean);
    };

    // Process the file path.
    const createUploadFileList = (file) => {
        try {
            const transfer = new DataTransfer();
            transfer.items.add(file);
            return transfer.files;
        } catch (error) {
            return null;
        }
    };

    // Process the file path.
    const assignClipboardFileToUpload = (uploadInput, uploadFiles) => {
        if (!uploadFiles) {
            return false;
        }
        try {
            uploadInput.files = uploadFiles;
        } catch (error) {
            return false;
        }
        uploadInput.dispatchEvent(new Event("input", { bubbles: true }));
        uploadInput.dispatchEvent(new Event("change", { bubbles: true }));
        return true;
    };

    // Process image content.
    const uploadClipboardFile = (event) => {
        const clipboardData = event.clipboardData;
        const uploadInput = findUploadFileInput();
        if (!clipboardData || !uploadInput) {
            return false;
        }

        const acceptedTypes = getUploadAcceptedTypes(uploadInput);
        const rawClipboardFiles = clipboardData.files || null;
        const clipboardFiles = collectClipboardFiles(clipboardData)
            .map((rawFile) => ({ rawFile, uploadFile: normalizeClipboardFile(rawFile) }))
            .filter(({ uploadFile }) => fileMatchesAcceptedType(uploadFile, acceptedTypes));
        if (!clipboardFiles.length) {
            return false;
        }

        const { rawFile, uploadFile } = clipboardFiles[0];
        const uploadFiles = createUploadFileList(uploadFile)
            || (
                rawClipboardFiles?.length === 1
                && rawClipboardFiles[0] === rawFile
                && rawFile === uploadFile
                    ? rawClipboardFiles
                    : null
            );
        return assignClipboardFileToUpload(uploadInput, uploadFiles);
    };

    // Implementation detail.
    const positionAdvancedDropdowns = () => {
        const popover = findPopover();
        if (!popover || !document.body.classList.contains(POPOVER_OPEN_CLASS)) {
            return;
        }

        popover.querySelectorAll("ul.options").forEach((options) => {
            const wrap = options.closest(".wrap");
            if (!wrap) {
                return;
            }

            popover.querySelectorAll(".wrap").forEach((item) => {
                item.style.removeProperty("z-index");
            });

            const wrapRect = wrap.getBoundingClientRect();
            const popoverRect = popover.getBoundingClientRect();
            const viewportPadding = 12;
            const gap = 6;
            const belowSpace = Math.max(0, popoverRect.bottom - wrapRect.bottom - viewportPadding);
            const aboveSpace = Math.max(0, wrapRect.top - popoverRect.top - viewportPadding);
            const naturalHeight = Math.max(36, Math.min(options.scrollHeight || 220, 240));
            const openBelow = belowSpace >= Math.min(180, naturalHeight) || belowSpace >= aboveSpace;
            const availableHeight = Math.max(84, openBelow ? belowSpace : aboveSpace);
            const height = Math.min(naturalHeight, availableHeight);
            const top = openBelow ? wrap.offsetHeight + gap : -height - gap;

            wrap.style.setProperty("z-index", "1003", "important");
            options.style.setProperty("position", "absolute", "important");
            options.style.setProperty("left", "0", "important");
            options.style.setProperty("top", `${top}px`, "important");
            options.style.setProperty("bottom", "auto", "important");
            options.style.setProperty("width", `${wrapRect.width}px`, "important");
            options.style.setProperty("max-height", `${height}px`, "important");
            options.style.setProperty("z-index", "1004", "important");
        });
    };

    // Implementation detail.
    const cancelPopoverTimers = () => {
        if (visibilityTimer !== null) {
            clearTimeout(visibilityTimer);
            visibilityTimer = null;
        }
    };

    // Remove invalid or unnecessary data.
    const clearLegacyPopoverDisplay = (popover) => {
        if (popover) {
            popover.style.removeProperty("display");
        }
    };

    // Implementation detail.
    const applyOpenPopoverStyle = (popover) => {
        if (!popover) {
            return;
        }
        popover.style.setProperty("visibility", "visible", "important");
        popover.style.setProperty("opacity", "1", "important");
        popover.style.setProperty("pointer-events", "auto", "important");
        popover.style.setProperty("transform", "translateY(0)", "important");
    };

    // Implementation detail.
    const applyClosedPopoverStyle = (popover) => {
        if (!popover) {
            return;
        }
        popover.style.setProperty("opacity", "0", "important");
        popover.style.setProperty("pointer-events", "none", "important");
        popover.style.setProperty("transform", "translateY(-2px)", "important");
        visibilityTimer = window.setTimeout(() => {
            if (!document.body.classList.contains(POPOVER_OPEN_CLASS)) {
                popover.style.setProperty("visibility", "hidden", "important");
            }
            visibilityTimer = null;
        }, ANIMATION_DELAY_MS);
    };

    // Implementation detail.
    const queueDropdownPosition = () => {
        requestAnimationFrame(() => {
            positionAdvancedDropdowns();
        });
    };

    // Implementation detail.
    const positionPopover = () => {
        const button = findButton();
        const popover = findPopover();
        if (!button || !popover) {
            return;
        }

        const buttonRect = button.getBoundingClientRect();
        const preferredWidth = Math.min(420, window.innerWidth - 36);
        const left = Math.min(
            Math.max(18, buttonRect.right + 12),
            Math.max(18, window.innerWidth - preferredWidth - 18)
        );
        const availableHeight = Math.max(260, window.innerHeight - 36);
        const measuredHeight = Math.min(
            popover.scrollHeight || 520,
            availableHeight,
            Math.round(window.innerHeight * 0.7)
        );
        const centeredTop = buttonRect.top + buttonRect.height / 2 - measuredHeight / 2;
        const top = Math.min(
            Math.max(18, centeredTop),
            Math.max(18, window.innerHeight - measuredHeight - 18)
        );

        popover.style.setProperty("--vsf-popover-left", `${left}px`);
        popover.style.setProperty("--vsf-popover-top", `${top}px`);
    };

    // Calculate the result.
    const openPopover = () => {
        const popover = findPopover();
        cancelPopoverTimers();
        clearLegacyPopoverDisplay(popover);
        positionPopover();
        document.body.classList.add(POPOVER_OPEN_CLASS);
        applyOpenPopoverStyle(popover);
        queueDropdownPosition();
    };

    // Implementation detail.
    const closePopover = () => {
        const popover = findPopover();
        cancelPopoverTimers();
        clearLegacyPopoverDisplay(popover);
        document.body.classList.remove(POPOVER_OPEN_CLASS);
        applyClosedPopoverStyle(popover);
    };

    refreshVSFCustomHtml();
    const queueCustomHtmlRefresh = () => {
        if (customHtmlRefreshFrame !== null) {
            return;
        }
        customHtmlRefreshFrame = requestAnimationFrame(() => {
            customHtmlRefreshFrame = null;
            refreshVSFCustomHtml();
        });
    };
    queueCustomHtmlRefresh();
    if (typeof MutationObserver !== "undefined") {
        const uiObserver = new MutationObserver(() => {
            queueCustomHtmlRefresh();
        });
        uiObserver.observe(document.body, { childList: true, subtree: true });
    }

    document.addEventListener("click", (event) => {
        const target = event.target;
        if (!(target instanceof Element)) {
            return;
        }
        if (target.closest(".office-preview-ignore-forever")) {
            const notice = target.closest(".office-preview-notice");
            if (setOfficePreviewNoticeIgnored()) {
                applyOfficePreviewNoticePreference();
            } else {
                notice?.classList.add("is-dismissed");
            }
            return;
        }
        if (target.closest(".office-preview-ignore-once")) {
            target.closest(".office-preview-notice")?.classList.add("is-dismissed");
            return;
        }
        if (target.closest(".vsf-advanced-open")) {
            if (document.body.classList.contains(POPOVER_OPEN_CLASS)) {
                closePopover();
            } else {
                openPopover();
            }
            return;
        }
        if (target.closest(".vsf-advanced-popover")) {
            queueDropdownPosition();
        }
        if (!target.closest(".vsf-advanced-popover")) {
            closePopover();
        }
    });

    document.addEventListener("focusin", (event) => {
        const target = event.target;
        if (target instanceof Element && target.closest(".vsf-advanced-popover")) {
            queueDropdownPosition();
        }
    });

    document.addEventListener("input", (event) => {
        const target = event.target;
        queueVSFOptionVisibilityRefresh();
        if (target instanceof Element && target.closest(".vsf-advanced-popover")) {
            queueDropdownPosition();
        }
    });

    document.addEventListener("change", () => {
        queueVSFOptionVisibilityRefresh();
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            closePopover();
            return;
        }
        const target = event.target;
        if (target instanceof Element && target.closest(".vsf-advanced-popover")) {
            queueDropdownPosition();
        }
    });

    document.addEventListener("paste", (event) => {
        if (uploadClipboardFile(event)) {
            event.preventDefault();
        }
    });

    window.addEventListener("resize", () => {
        if (document.body.classList.contains(POPOVER_OPEN_CLASS)) {
            positionPopover();
            positionAdvancedDropdowns();
        }
    });
}
