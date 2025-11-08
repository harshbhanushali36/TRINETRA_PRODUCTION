// ============================================
// New Dashboard.js - Loads from GitHub
// Dashboard.js - Loads visualization with AI Chatbot
// ============================================

import React from 'react';
import Chatbot from './chatbot';

function Dashboard() {
  // Note: Since the iframe is from a different origin (GitHub Pages),
  // we cannot directly access its internal state. However, users can
  // mention object names in their questions, and the chatbot will
  // use that context to provide relevant answers.

  return (
    <div style={{ 
      width: '100%', 
      height: '100vh',
      overflow: 'hidden',
      margin: 0,
      padding: 0,
      position: 'relative'
    }}>
      <iframe
        title="TRINETRA Visualization"
        src="https://harshbhanushali36.github.io/TRINETRA_PRODUCTION/"
        width="100%"
        height="100%"
        frameBorder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        sandbox="allow-same-origin allow-scripts allow-forms"
      />
      {/* AI Chatbot - floats over the visualization */}
      <Chatbot selectedObject={null} />
    </div>
  );
}

export default Dashboard;