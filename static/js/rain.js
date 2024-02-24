// static/js/rain.js

document.addEventListener("DOMContentLoaded", function () {
    createRainDrops();
});

function createRainDrops() {
    const rainContainer = document.createElement("div");
    rainContainer.className = "rain-container";
    document.body.appendChild(rainContainer);

    for (let i = 0; i < 50; i++) {
        const rainDrop = document.createElement("div");
        rainDrop.className = "rain-drop"; // Updated class name
        rainDrop.style.left = `${Math.random() * 100}vw`;
        rainDrop.style.animationDuration = `${Math.random() * 2 + 1}s`;
        rainContainer.appendChild(rainDrop);
    }
}
