document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileUpload = document.getElementById('file-upload');
    const fileNameDisplay = document.getElementById('file-name');
    const analyzeBtn = document.getElementById('analyze-btn');
    const spinner = document.getElementById('spinner');
    const resultArea = document.getElementById('result-area');
    const emotionText = document.getElementById('emotion-text');
    const errorArea = document.getElementById('error-area');
    const emotionResultIcon = document.querySelector('.emotion-result i');

    let selectedFile = null;

    // Emotion Icons mapping
    const emotionIcons = {
        'neutral': 'fa-face-meh',
        'calm': 'fa-face-relieved',
        'happy': 'fa-face-smile',
        'sad': 'fa-face-sad-tear',
        'angry': 'fa-face-angry',
        'fearful': 'fa-face-grimace',
        'disgust': 'fa-face-dizzy',
        'surprised': 'fa-face-surprise'
    };

    // Handle Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('dragover');
        }, false);
    });

    dropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }, false);

    fileUpload.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type === 'audio/wav' || file.name.endsWith('.wav')) {
                selectedFile = file;
                fileNameDisplay.textContent = file.name;
                analyzeBtn.disabled = false;
                resultArea.style.display = 'none';
                errorArea.textContent = '';
            } else {
                errorArea.textContent = 'Please upload a valid .wav audio file.';
                analyzeBtn.disabled = true;
                selectedFile = null;
                fileNameDisplay.textContent = '';
            }
        }
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI Loading State
        analyzeBtn.disabled = true;
        analyzeBtn.querySelector('span').style.display = 'none';
        spinner.style.display = 'block';
        resultArea.style.display = 'none';
        errorArea.textContent = '';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                // Show Result
                const emotion = data.emotion;
                emotionText.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
                
                // Update Icon
                emotionResultIcon.className = 'fa-solid';
                if(emotionIcons[emotion]) {
                    emotionResultIcon.classList.add(emotionIcons[emotion]);
                } else {
                    emotionResultIcon.classList.add('fa-face-smile');
                }

                resultArea.style.display = 'block';
            } else {
                throw new Error(data.error || 'Something went wrong during analysis.');
            }
        } catch (error) {
            errorArea.textContent = error.message;
        } finally {
            // Restore UI
            analyzeBtn.disabled = false;
            analyzeBtn.querySelector('span').style.display = 'block';
            spinner.style.display = 'none';
        }
    });
});
