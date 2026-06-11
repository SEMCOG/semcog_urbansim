document.addEventListener("DOMContentLoaded", function () {
  var tabsList = document.querySelector(".md-tabs__list");
  if (!tabsList) return;

  var item = document.createElement("li");
  item.className = "md-tabs__item";
  item.style.marginLeft = "auto";

  var link = document.createElement("a");
  link.href = "mailto:li@semcog.org";
  link.className = "md-tabs__link";
  link.textContent = "Contact: li@semcog.org";
  link.style.fontSize = "0.72rem";
  link.style.fontWeight = "400";
  link.style.opacity = "0.7";
  link.style.letterSpacing = "0";

  item.appendChild(link);
  tabsList.appendChild(item);
});
