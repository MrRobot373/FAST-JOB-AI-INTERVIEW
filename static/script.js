document.addEventListener('DOMContentLoaded', (event) => {
    // Initial display for chat history if empty
    const chatHistoryDiv = document.getElementById('chatHistory');
    if (!chatHistoryDiv.innerHTML.trim()) {
        chatHistoryDiv.innerHTML = '<p style="text-align: center; color: #888;">Start the interview by sending your first message.</p>';
    }
});


document.getElementById('chatForm').addEventListener('submit', async function (e) {
    e.preventDefault(); // Prevent default form submission

    const formData = new FormData(e.target);
    const userId = formData.get("user_id"); // Get user_id from form
    const responseDiv = document.getElementById('response');
    const audioPlayer = document.getElementById('audioPlayer');
    const chatHistoryDiv = document.getElementById('chatHistory');
    const messageInput = document.querySelector('textarea[name="message"]'); // Get message input field

    // Reset UI for new interaction
    responseDiv.innerText = "Processing AI response...";
    audioPlayer.style.display = "none";
    audioPlayer.src = ''; // Clear any previous audio source
    audioPlayer.pause(); // Stop any currently playing audio

    try {
        // Send message to /talk endpoint
        const talkRes = await fetch('/talk', {
            method: 'POST',
            body: formData // FormData automatically sets content-type
        });

        if (!talkRes.ok) {
            const errorData = await talkRes.json();
            throw new Error(errorData.detail || `HTTP error! status: ${talkRes.status}`);
        }

        const data = await talkRes.json();
        responseDiv.innerText = data.response; // Display AI's text response

        // Clear the message input field after sending
        messageInput.value = '';

        // Update and display full chat history
        updateChatHistoryDisplay(data.full_history, chatHistoryDiv);

        const sessionId = data.session_id; // Get the session_id from the /talk response

        // --- Polling for Audio ---
        let pollAttempts = 0;
        const maxPollAttempts = 20; // Poll for about 60 seconds (20 * 3 seconds)
        responseDiv.innerText += "\n(Generating audio...)"; // Add a message while waiting

            const pollInterval = setInterval(async () => {
                pollAttempts++;
                if (pollAttempts > maxPollAttempts) {
                    clearInterval(pollInterval);
                    responseDiv.innerText = data.response + "\n(Audio generation timed out or failed.)";
                    return;
                }

                try {
                    // 1) HEAD to see if the file exists (no body, just headers)
                    const headRes = await fetch(`/get_audio/${sessionId}`, { method: 'HEAD' });

                    if (headRes.ok) {
                    // 2) Now GET the full MP3 blob
                    clearInterval(pollInterval);
                    const getRes = await fetch(`/get_audio/${sessionId}`);
                    const blob = await getRes.blob();

                    audioPlayer.src = URL.createObjectURL(blob);
                    audioPlayer.load();              // ensure it actually loads the whole file
                    audioPlayer.style.display = "block";
                    audioPlayer.play().catch(e => console.error(e));

                    responseDiv.innerText = data.response; // back to just text
                    } else if (headRes.status === 404) {
                    console.log("Audio still processing…");
                    // keep polling
                    } else {
                    clearInterval(pollInterval);
                    const txt = await headRes.text();
                    responseDiv.innerText = data.response + `\n(Error checking audio: ${txt})`;
                    console.error("Head error:", headRes.status, txt);
                    }
                } catch (err) {
                    clearInterval(pollInterval);
                    responseDiv.innerText = data.response + `\n(Network error: ${err.message})`;
                    console.error(err);
                }
                }, 3000);

    } catch (error) {
        console.error('Error in chatForm submission:', error);
        responseDiv.innerText = `Error: ${error.message}`;
    }
});

// --- Speech-to-Text (STT) Microphone Logic ---
document.getElementById('micBtn').addEventListener('click', function () {
    const messageInput = document.querySelector('textarea[name="message"]');
    if (!('webkitSpeechRecognition' in window)) {
        alert("Your browser does not support Speech-to-Text. Please use Chrome or Edge.");
        return;
    }

    const recognition = new webkitSpeechRecognition();
    recognition.lang = 'en-US'; // Set recognition language
    recognition.interimResults = false; // Only return final results
    recognition.continuous = false; // Stop after a single utterance

    // Update input field with recognized speech
    recognition.onresult = event => {
        const transcript = event.results[0][0].transcript;
        messageInput.value = transcript;
        console.log("Speech recognized:", transcript);
    };

    // Handle errors (e.g., no microphone, permission denied)
    recognition.onerror = event => {
        console.error("Speech recognition error:", event.error);
        let errorMessage = "Speech recognition error. Please check your microphone.";
        if (event.error === 'not-allowed') {
            errorMessage = "Microphone access denied. Please allow microphone permissions in your browser settings.";
        } else if (event.error === 'no-speech') {
            errorMessage = "No speech detected. Please speak clearly.";
        }
        alert(errorMessage);
    };

    // Start listening
    recognition.start();
    alert("Listening... Please speak now.");
});

// --- Helper Function to Update Chat History Display ---
function updateChatHistoryDisplay(history, chatHistoryDiv) {
    chatHistoryDiv.innerHTML = ''; // Clear previous history display

    if (!history || history.length === 0) {
        chatHistoryDiv.innerHTML = '<p style="text-align: center; color: #888;">Start the interview by sending your first message.</p>';
        return;
    }

    history.forEach(entry => {
        const messageElement = document.createElement('div');
        messageElement.classList.add(entry.role === 'user' ? 'user-message' : 'model-message');
        // Sanitize text to prevent XSS if it were user-controlled and directly inserted as HTML without innerText
        messageElement.innerHTML = `<strong>${entry.role.charAt(0).toUpperCase() + entry.role.slice(1)}:</strong> ${escapeHTML(entry.text)}`;
        chatHistoryDiv.appendChild(messageElement);
    });

    // Scroll to the bottom of the chat history
    chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
}

// Basic HTML escaping for display
function escapeHTML(str) {
    var div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}


// --- Clear History Button Logic ---
document.getElementById('clearHistoryBtn').addEventListener('click', async function() {
    const userId = document.querySelector('input[name="user_id"]').value;
    if (!userId) {
        alert("Please enter your User ID to clear history.");
        return;
    }

    if (!confirm("Are you sure you want to clear your chat history? This cannot be undone.")) {
        return; // User cancelled
    }

    try {
        // Corrected: Use POST method and path parameter
        const res = await fetch(`/clear_chat/${userId}`, {
            method: 'POST'
        });

        if (!res.ok) {
            const errorData = await res.json();
            throw new Error(errorData.detail || `HTTP error! status: ${res.status}`);
        }

        const data = await res.json();
        alert(data.message); // Show success/error message

        // Clear UI elements
        document.getElementById('response').innerText = '';
        document.getElementById('audioPlayer').style.display = "none";
        document.getElementById('audioPlayer').src = '';
        document.getElementById('chatHistory').innerHTML = '<p style="text-align: center; color: #888;">History cleared. Start a new interview.</p>';

    } catch (error) {
        console.error('Error clearing history:', error);
        alert(`Failed to clear history: ${error.message}. Please try again.`);
    }
});

// --- Generate Final Report Button Logic ---
document.getElementById('finalReportBtn').addEventListener('click', async function() {
    const userId = document.querySelector('input[name="user_id"]').value;
    if (!userId) {
        alert("Please enter your User ID to generate a final report.");
        return;
    }

    const responseDiv = document.getElementById('response');
    responseDiv.innerText = "Generating final report... This may take a moment.";
    document.getElementById('audioPlayer').style.display = "none";
    document.getElementById('audioPlayer').src = ''; // Clear audio

    try {
        // Corrected: Use POST method and send FormData
        const reportFormData = new FormData();
        reportFormData.append('user_id', userId);

        const res = await fetch(`/final_report`, {
            method: 'POST',
            body: reportFormData
        });

        if (!res.ok) {
            const errorData = await res.json();
            throw new Error(errorData.detail || `HTTP error! status: ${res.status}`);
        }

        const data = await res.json();
        if (data.report) { // Changed from data.final_report to data.report based on your main.py
            responseDiv.innerText = "--- Final Interview Report ---\n\n" + data.report;
        } else {
            responseDiv.innerText = "No report generated. The interview history might be empty.";
        }
    } catch (error) {
        console.error('Error generating final report:', error);
        responseDiv.innerText = `Error generating report: ${error.message}.`;
    }
});
