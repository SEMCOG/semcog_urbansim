document.addEventListener("DOMContentLoaded", function () {
  // Wait for Material theme to finish initializing.
  setTimeout(function () {
    function isOpen(toggle, nav) {
      return toggle.checked ||
             nav.getAttribute("aria-expanded") === "true" ||
             window.getComputedStyle(nav).display !== "none";
    }

    function setOpen(item, toggle, nav, open, collapsedClass) {
      item.classList.toggle(collapsedClass, !open);
      toggle.checked = open;
      nav.setAttribute("aria-expanded", open ? "true" : "false");
    }

    function makeToggle(item, toggle, controls, nav, collapsedClass) {
      setOpen(item, toggle, nav, isOpen(toggle, nav), collapsedClass);

      controls.forEach(function (control) {
        control.addEventListener("click", function (event) {
          var shouldOpen = !isOpen(toggle, nav);

          if (control.tagName === "A") {
            event.preventDefault();
          }

          // Let the Material handler finish first, then apply the requested state.
          setTimeout(function () {
            setOpen(item, toggle, nav, shouldOpen, collapsedClass);
          }, 30);
        });
      });
    }

    document.querySelectorAll(".md-nav__item--nested").forEach(function (item) {
      var toggle = item.querySelector(":scope > input.md-nav__toggle");
      var label  = item.querySelector(":scope > label.md-nav__link");
      var nav    = item.querySelector(":scope > .md-nav");
      if (!toggle || !label || !nav) return;

      makeToggle(item, toggle, [label], nav, "nav-collapsed");
    });

    document.querySelectorAll(".md-nav__item--active").forEach(function (item) {
      var toggle = item.querySelector(":scope > input#__toc");
      var label  = item.querySelector(":scope > label.md-nav__link");
      var link   = item.querySelector(":scope > a.md-nav__link--active");
      var nav    = item.querySelector(":scope > .md-nav--secondary");
      var controls = [label, link].filter(Boolean);
      if (!toggle || !controls.length || !nav) return;

      makeToggle(item, toggle, controls, nav, "toc-collapsed");
    });
  }, 150);
});
