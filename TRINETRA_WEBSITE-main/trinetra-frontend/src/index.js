import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Import Iceland font from Google Fonts
import '@fontsource/iceland'; // npm install @fontsource/iceland

// OR use this CSS import if you don't want to install the package
// Create a style element to import from Google Fonts directly
const style = document.createElement('style');
style.textContent = `
  @import url('https://fonts.googleapis.com/css2?family=Iceland&display=swap');
`;
document.head.appendChild(style);

// Create MUI theme with Iceland font
const theme = createTheme({
  typography: {
    fontFamily: 'Iceland, sans-serif',
    h1: {
      fontFamily: 'Iceland, sans-serif',
    },
    h2: {
      fontFamily: 'Iceland, sans-serif',
    },
    h3: {
      fontFamily: 'Iceland, sans-serif',
    },
    h4: {
      fontFamily: 'Iceland, sans-serif',
    },
    h5: {
      fontFamily: 'Iceland, sans-serif',
    },
    h6: {
      fontFamily: 'Iceland, sans-serif',
    },
    body1: {
      fontFamily: 'Iceland, sans-serif',
    },
    body2: {
      fontFamily: 'Iceland, sans-serif',
    },
    button: {
      fontFamily: 'Iceland, sans-serif',
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: `
        * {
          font-family: 'Iceland', sans-serif;
        }
        body {
          font-family: 'Iceland', sans-serif;
        }
      `,
    },
  },
});

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </React.StrictMode>
);

reportWebVitals();