const express = require('express');
const path = require('path');
const app = express();
const PORT = 8080;

// Serve static files from current directory
app.use(express.static(__dirname));

// Route for studio
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'studio.html'));
});

app.listen(PORT, () => {
    console.log(`🌍 World Engine Studio running at http://localhost:${PORT}`);
    console.log(`🎭 Multimodal interface ready for testing`);
});