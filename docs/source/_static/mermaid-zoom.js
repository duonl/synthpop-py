window.addEventListener("load", function () {
  const svgs = document.querySelectorAll(".mermaid svg");

  svgs.forEach((svg) => {
    svgPanZoom(svg, {
      zoomEnabled: true,
      controlIconsEnabled: true,
      fit: true,
      center: true,
    });
  });
});