document.getElementById("crop-form").addEventListener("submit", function(event) {
    event.preventDefault();

    let formData = new FormData(this);
    
    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById("result").innerText = "Error: " + data.error;
        } else {
            let explanationText = "Feature Impact:\n";
            
            for (let key in data.explanation) {
                explanationText += `${key}: ${data.explanation[key].toFixed(4)}\n`;
            }

            document.getElementById("result").innerText = "Recommended Crop: " + data.crop;
            document.getElementById("explanation").innerText = explanationText;
        }
    })
    .catch(error => console.error("Error:", error));
});
