document.addEventListener('DOMContentLoaded', () => {
    const toggleButton = document.getElementById('theme-toggle');
    const darkModeText = "Dark Mode";
    const lightModeText = "Light Mode";

    const updateButtonText = () => {
        if (document.body.classList.contains('dark-mode')) {
            toggleButton.textContent = lightModeText;
        } else {
            toggleButton.textContent = darkModeText;
        }
    };

    toggleButton.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        updateButtonText();
    });

    const diagnosisButton = document.getElementById('diagnosisButton');
    const resultDiv = document.getElementById('result');
    const modal = document.getElementById('result-modal');
    const closeButton = document.querySelector('.modal .close');

    diagnosisButton.addEventListener('click', () => {
        // Lấy dữ liệu từ form chẩn đoán
        const formData = new FormData(document.getElementById('diagnosis-form'));

        // Lấy thuật toán được chọn
        const algorithm = document.getElementById('algorithm').value;
        formData.append('algorithm', algorithm);

        // Gửi dữ liệu POST đến server (fetch API)
        fetch('/submit', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json()) 
        .then(data => {
            // Hiển thị kết quả trong div#result
            let icon;
            if (data.diagnosis === 'Negative') {
                icon = '<img src="D:/HUIT/Học Máy/NhomEEE_DoAnCuoiKy/Demo/Templates/images/check.png" alt="Check Icon" class="result-icon">';
            } else {
                icon = '<img src="D:/HUIT/Học Máy/NhomEEE_DoAnCuoiKy/Demo/Templates/images/cross.png" alt="Cross Icon" class="result-icon">';
            }
            resultDiv.innerHTML = `${icon}<h3>Kết quả chẩn đoán:</h3> <p>${data.diagnosis}</p>`;
            // Hiển thị modal
            modal.style.display = 'flex';
        })
    });
    closeButton.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    // Close the modal when clicking outside the modal content
    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
    updateButtonText();
});
