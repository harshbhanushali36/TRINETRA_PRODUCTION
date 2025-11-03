// ============================================
// New Dashboard.js - Loads from GitHub
// ============================================

import React from 'react';

function Dashboard() {
  return (
    <div style={{ 
      width: '100%', 
      height: '100vh',
      overflow: 'hidden',
      margin: 0,
      padding: 0
    }}>
      <iframe
        src="https://harshbhanushali36.github.io/TRINETRA_PRODUCTION/"
        title="Dashboard"
        style={{
          width: '100%',
          height: '100%',
          border: 'none',
          margin: 0,
          padding: 0,
          display: 'block'
        }}
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        sandbox="allow-same-origin allow-scripts allow-forms"
      />
    </div>
  );
}

export default Dashboard;