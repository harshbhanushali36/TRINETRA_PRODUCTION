import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Home from './Home';
import SignIn from './SignIn';
import SignUp from './SignUp';
import EmergencyEnroll from './EmergencyEnroll';
import Dashboard from './dashboard';

// Define theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#78dbff',
      light: '#a6e8ff',
      dark: '#0081cb',
    },
    secondary: {
      main: '#ff77c6',
      light: '#ff99d0',
      dark: '#c60055',
    },
    background: {
      default: '#0a0a2a',
      paper: '#1a1a4a',
    },
    text: {
      primary: '#ffffff',
      secondary: '#8ecae6',
    }
  },
  typography: {
    h1: {
      fontWeight: 700,
      fontFamily: "'Orbitron', sans-serif",
    },
    h2: {
      fontWeight: 600,
      fontFamily: "'Orbitron', sans-serif",
    },
    h3: {
      fontWeight: 600,
      fontFamily: "'Orbitron', sans-serif",
    },
    h4: {
      fontWeight: 600,
      fontFamily: "'Orbitron', sans-serif",
    },
    h5: {
      fontWeight: 600,
      fontFamily: "'Orbitron', sans-serif",
    },
    h6: {
      fontWeight: 600,
      fontFamily: "'Orbitron', sans-serif",
    },
    button: {
      textTransform: 'none',
      fontWeight: 600,
    }
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: '16px',
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: '12px',
        }
      }
    }
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/signin" element={<SignIn />} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/enroll" element={<EmergencyEnroll />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;