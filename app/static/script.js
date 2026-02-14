document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predict-btn');
    const premiseInput = document.getElementById('premise');
    const hypothesisInput = document.getElementById('hypothesis');
    const resultText = document.getElementById('result-text');

    predictBtn.addEventListener('click', async () => {
        const premise = premiseInput.value;
        const hypothesis = hypothesisInput.value;

        if (!premise || !hypothesis) {
            alert('Please enter both a premise and a hypothesis.');
            return;
        }

        // Show loading state
        resultText.textContent = 'Predicting...';
        resultText.className = '';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ premise, hypothesis }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                resultText.textContent = `Error: ${data.error}`;
                resultText.className = 'contradiction'; // Use red for errors
            } else {
                const prediction = data.prediction;
                resultText.textContent = prediction;
                // Add a class to the result text for specific styling
                resultText.className = prediction.toLowerCase();
            }

        } catch (error) {
            console.error('Prediction error:', error);
            resultText.textContent = 'Failed to get a prediction.';
            resultText.className = 'contradiction'; // Use red for errors
        }
    });
});
