document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const uploadLabel = document.getElementById('uploadLabel');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.getElementById('preview');
    const resultsContainer = document.getElementById('results');
    const loader = document.getElementById('loader');
    const resultContent = document.getElementById('resultContent');
    const descriptionEl = document.getElementById('description');
    const labelsEl = document.getElementById('labels');

    const API_ENDPOINT = 'https://onwsf3b6ka.execute-api.us-east-1.amazonaws.com/v1/analyze'; // <-- IMPORTANT: REPLACE THIS

    let base64Image = null;

    imageUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            // Display image preview
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                uploadLabel.querySelector('span').textContent = file.name;
                analyzeBtn.disabled = false;
            };
            reader.readAsDataURL(file);

            // Convert image to base64 for sending to API
            const readerForBase64 = new FileReader();
            readerForBase64.onload = (e) => {
                // Remove the data URL prefix (e.g., "data:image/jpeg;base64,")
                base64Image = e.target.result.split(',')[1];
            };
            readerForBase64.readAsDataURL(file);
        }
    });

    analyzeBtn.addEventListener('click', async () => {
        if (!base64Image || API_ENDPOINT === 'YOUR_API_GATEWAY_INVOKE_URL') {
            alert('Please select an image first or configure the API endpoint in script.js.');
            return;
        }

        // Show loader and results section
        resultsContainer.classList.remove('hidden');
        loader.style.display = 'block';
        resultContent.style.display = 'none';
        descriptionEl.textContent = '';
        labelsEl.innerHTML = '';

        try {
            const response = await fetch(API_ENDPOINT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: base64Image }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Display results
            descriptionEl.textContent = data.description;
            data.labels.forEach(label => {
                const labelTag = document.createElement('div');
                labelTag.className = 'label-tag';
                labelTag.textContent = label;
                labelsEl.appendChild(labelTag);
            });

        } catch (error) {
            console.error('Error:', error);
            descriptionEl.textContent = `An error occurred: ${error.message}`;
        } finally {
            // Hide loader and show content
            loader.style.display = 'none';
            resultContent.style.display = 'block';
        }
    });
});