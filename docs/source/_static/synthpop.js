(function () {
  "use strict";

  const MIN_ZOOM = 1;
  const MAX_ZOOM = 8;
  const ZOOM_STEP = 1.2;

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function createButton(label, title, onClick) {
    const button = document.createElement("button");

    button.type = "button";
    button.className = "synthpop-zoombox__button";
    button.textContent = label;
    button.title = title;
    button.setAttribute("aria-label", title);

    button.addEventListener("click", function (event) {
      event.stopPropagation();
      onClick();
    });

    return button;
  }

  function setupZoombox(box) {
    if (box.dataset.synthpopZoomboxInitialized === "true") {
      return;
    }

    const image = box.querySelector("img");

    if (!image) {
      return;
    }

    box.dataset.synthpopZoomboxInitialized = "true";

    let zoom = 1;
    let x = 0;
    let y = 0;

    let baseWidth = 0;
    let baseHeight = 0;

    let dragging = false;
    let lastPointerX = 0;
    let lastPointerY = 0;

    /*
     * The image is initially allowed to render normally.
     * We measure that rendered size before taking it out of normal flow.
     */
    function measureImage() {
      const rect = image.getBoundingClientRect();

      baseWidth = rect.width;
      baseHeight = rect.height;

      return {
        width: baseWidth,
        height: baseHeight
      };
    }

    function applyTransform() {
      const scale = zoom;

      image.style.width = baseWidth + "px";
      image.style.height = baseHeight + "px";

      image.style.transform =
        "translate(" +
        x +
        "px, " +
        y +
        "px) scale(" +
        scale +
        ")";
    }

    function reset() {
      /*
       * Temporarily remove the transform so that getBoundingClientRect()
       * tells us the image's normal Sphinx-rendered dimensions.
       */
      image.style.transform = "none";

      /*
       * Restore normal sizing while measuring.
       */
      image.style.position = "static";
      image.style.width = "";
      image.style.height = "";
      image.style.maxWidth = "";
      image.style.maxHeight = "";

      const measured = measureImage();

      baseWidth = measured.width;
      baseHeight = measured.height;

      /*
       * Now put the image into the zoom viewport.
       */
      image.style.position = "absolute";
      image.style.width = baseWidth + "px";
      image.style.height = baseHeight + "px";

      zoom = 1;

      /*
       * Match normal document rendering horizontally.
       *
       * Vertically center the image in the zoombox.
       */
      x = Math.max(
        (box.clientWidth - baseWidth) / 2,
        0
      );

      y = Math.max(
        (box.clientHeight - baseHeight) / 2,
        0
      );

      applyTransform();
    }

    function zoomAt(newZoom, centerX, centerY) {
      newZoom = clamp(
        newZoom,
        MIN_ZOOM,
        MAX_ZOOM
      );

      const oldScale = zoom;
      const newScale = newZoom;

      /*
       * Keep the point underneath the pointer stationary.
       */
      const imageX =
        (centerX - x) /
        oldScale;

      const imageY =
        (centerY - y) /
        oldScale;

      x =
        centerX -
        imageX * newScale;

      y =
        centerY -
        imageY * newScale;

      zoom = newZoom;

      applyTransform();
    }

    function zoomIn() {
      zoomAt(
        zoom * ZOOM_STEP,
        box.clientWidth / 2,
        box.clientHeight / 2
      );
    }

    function zoomOut() {
      zoomAt(
        zoom / ZOOM_STEP,
        box.clientWidth / 2,
        box.clientHeight / 2
      );
    }

    /*
     * Controls
     */
    const controls =
      document.createElement("div");

    controls.className =
      "synthpop-zoombox__controls";

    controls.appendChild(
      createButton(
        "+",
        "Zoom in",
        zoomIn
      )
    );

    controls.appendChild(
      createButton(
        "−",
        "Zoom out",
        zoomOut
      )
    );

    controls.appendChild(
      createButton(
        "Reset",
        "Reset zoom and position",
        reset
      )
    );

    box.appendChild(controls);

    /*
     * Help text
     */
    const help =
      document.createElement("div");

    help.className =
      "synthpop-zoombox__help";

    help.textContent =
      "Scroll to zoom · Drag to pan";

    box.appendChild(help);

    /*
     * Mouse-wheel zoom.
     */
    box.addEventListener(
      "wheel",
      function (event) {
        event.preventDefault();

        const rect =
          box.getBoundingClientRect();

        const mouseX =
          event.clientX -
          rect.left;

        const mouseY =
          event.clientY -
          rect.top;

        const factor =
          event.deltaY < 0
            ? ZOOM_STEP
            : 1 / ZOOM_STEP;

        zoomAt(
          zoom * factor,
          mouseX,
          mouseY
        );
      },
      { passive: false }
    );

    /*
     * Dragging / panning.
     */
    box.addEventListener(
      "pointerdown",
      function (event) {
        if (
          event.target.closest(
            ".synthpop-zoombox__controls"
          )
        ) {
          return;
        }

        dragging = true;

        lastPointerX =
          event.clientX;

        lastPointerY =
          event.clientY;

        box.classList.add(
          "is-dragging"
        );

        box.setPointerCapture(
          event.pointerId
        );
      }
    );

    box.addEventListener(
      "pointermove",
      function (event) {
        if (!dragging) {
          return;
        }

        const dx =
          event.clientX -
          lastPointerX;

        const dy =
          event.clientY -
          lastPointerY;

        x += dx;
        y += dy;

        lastPointerX =
          event.clientX;

        lastPointerY =
          event.clientY;

        applyTransform();
      }
    );

    function stopDragging(event) {
      if (!dragging) {
        return;
      }

      dragging = false;

      box.classList.remove(
        "is-dragging"
      );

      if (
        event.pointerId !== undefined
      ) {
        try {
          box.releasePointerCapture(
            event.pointerId
          );
        } catch (_) {
          /* Already released. */
        }
      }
    }

    box.addEventListener(
      "pointerup",
      stopDragging
    );

    box.addEventListener(
      "pointercancel",
      stopDragging
    );

    box.addEventListener(
      "pointerleave",
      function (event) {
        if (event.buttons === 0) {
          stopDragging(event);
        }
      }
    );

    /*
     * Prevent the browser's default image dragging.
     */
    image.addEventListener(
      "dragstart",
      function (event) {
        event.preventDefault();
      }
    );

    /*
     * Initialise after the image has loaded.
     */
    if (image.complete) {
      requestAnimationFrame(reset);
    } else {
      image.addEventListener(
        "load",
        function () {
          requestAnimationFrame(reset);
        },
        { once: true }
      );
    }

    /*
     * Recalculate the baseline when the page width changes.
     */
    let resizeTimer;

    window.addEventListener(
      "resize",
      function () {
        clearTimeout(resizeTimer);

        resizeTimer = setTimeout(
          function () {
            reset();
          },
          100
        );
      }
    );
  }


  /*
   * Automatically turn sufficiently large documentation
   * images into zoomboxes.
   */
  function wrapImage(image) {
    if (
      image.closest(
        ".synthpop-zoombox"
      )
    ) {
      return;
    }

    /*
     * Don't touch theme/navigation images.
     */
    if (
      image.closest(
        ".navbar, " +
        ".bd-header, " +
        ".sd-card, " +
        ".sidebar-primary, " +
        ".sidebar-secondary"
      )
    ) {
      return;
    }

    const width =
      image.naturalWidth ||
      image.width ||
      0;

    const height =
      image.naturalHeight ||
      image.height ||
      0;

    /*
     * Only wrap reasonably large images.
     */
    if (
      width < 300 ||
      height < 200
    ) {
      return;
    }

    const box =
      document.createElement("div");

    box.className =
      "synthpop-zoombox";

    const figure =
      image.closest("figure");

    if (figure) {
      figure.insertBefore(
        box,
        image
      );
    } else {
      image.parentNode.insertBefore(
        box,
        image
      );
    }

    box.appendChild(image);

    setupZoombox(box);
  }


  function initialise() {
    /*
     * Explicit zoomboxes.
     */
    document
      .querySelectorAll(
        ".synthpop-zoombox"
      )
      .forEach(setupZoombox);
  }


  if (
    document.readyState ===
    "loading"
  ) {
    document.addEventListener(
      "DOMContentLoaded",
      initialise
    );
  } else {
    initialise();
  }
})();