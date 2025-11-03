import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  TextField,
  Button,
  Typography,
  Card,
  CardContent,
  Divider,
  MenuItem,
  FormControlLabel,
  Checkbox
} from '@mui/material';
import { SatelliteAlt } from '@mui/icons-material';
import Galaxy from './Galaxy';

function EmergencyEnroll() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    phone: '',
    alertChannel: 'sms',
    location: '',
    agree: false
  });

  const handleChange = (e) => {
    const { name, value, checked, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Emergency enroll:', formData);
    // TODO: submit to backend API
    navigate('/');
  };

  return (
    <Box sx={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Galaxy Background */}
      <Box sx={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 0
      }}>
        <Galaxy
          density={1.2}
          starSpeed={0.3}
          hueShift={220}
          speed={0.8}
          glowIntensity={0.4}
          saturation={0.3}
          mouseRepulsion={true}
          mouseInteraction={true}
          twinkleIntensity={0.5}
          rotationSpeed={0.05}
          repulsionStrength={2}
          transparent={false}
          style={{ width: '100%', height: '100%' }}
        />
      </Box>

      {/* Logo in top left */}
      <Box sx={{
        position: 'absolute',
        top: 32,
        left: 32,
        zIndex: 10,
        display: 'flex',
        alignItems: 'center',
        gap: 1.5
      }}>
        <Box sx={{
          width: 40,
          height: 40,
          borderRadius: '10px',
          background: 'linear-gradient(135deg, #78dbff 0%, #7877c6 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 4px 20px rgba(120, 219, 255, 0.3)'
        }}>
          <SatelliteAlt sx={{ fontSize: 24, color: 'white' }} />
        </Box>
        <Typography variant="h5" sx={{ fontWeight: 'bold', color: 'white', letterSpacing: '1px' }}>
          TRINETRA
        </Typography>
      </Box>

      {/* Enrollment Card */}
      <Container maxWidth="sm" sx={{ position: 'relative', zIndex: 1, py: 4 }}>
        <Card sx={{
          background: 'rgba(26, 26, 74, 0.5)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(120, 219, 255, 0.2)',
          borderRadius: '24px',
          boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.5)',
          overflow: 'visible'
        }}>
          <CardContent sx={{ p: 5 }}>
            <Box sx={{ textAlign: 'center', mb: 4 }}>
              <Typography variant="h4" sx={{ fontWeight: 'bold', color: 'white', mb: 1, textShadow: '0 0 20px rgba(120, 219, 255, 0.3)' }}>
                Emergency Alerts Enrollment
              </Typography>
              <Typography sx={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: '1rem' }}>
                Receive critical notifications via SMS or Email
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center', my: 3 }}>
              <Divider sx={{ flex: 1, borderColor: 'rgba(120, 219, 255, 0.2)' }} />
              <Typography sx={{ px: 2, color: 'rgba(255, 255, 255, 0.5)', fontSize: '0.9rem' }}>
                SECURE FORM
              </Typography>
              <Divider sx={{ flex: 1, borderColor: 'rgba(120, 219, 255, 0.2)' }} />
            </Box>

            <form onSubmit={handleSubmit}>
              <TextField
                fullWidth
                label="Full Name"
                name="fullName"
                value={formData.fullName}
                onChange={handleChange}
                margin="normal"
                required
                sx={{
                  mb: 2,
                  '& .MuiOutlinedInput-root': {
                    color: 'white',
                    '& fieldset': { borderColor: 'rgba(120, 219, 255, 0.3)' },
                    '&:hover fieldset': { borderColor: 'rgba(120, 219, 255, 0.5)' },
                    '&.Mui-focused fieldset': { borderColor: '#78dbff' },
                  },
                  '& .MuiInputLabel-root': { color: 'rgba(255, 255, 255, 0.7)' },
                  '& .MuiInputLabel-root.Mui-focused': { color: '#78dbff' }
                }}
              />

              <TextField
                fullWidth
                label="Email Address"
                name="email"
                type="email"
                value={formData.email}
                onChange={handleChange}
                margin="normal"
                required
                sx={{
                  mb: 2,
                  '& .MuiOutlinedInput-root': {
                    color: 'white',
                    '& fieldset': { borderColor: 'rgba(120, 219, 255, 0.3)' },
                    '&:hover fieldset': { borderColor: 'rgba(120, 219, 255, 0.5)' },
                    '&.Mui-focused fieldset': { borderColor: '#78dbff' },
                  },
                  '& .MuiInputLabel-root': { color: 'rgba(255, 255, 255, 0.7)' },
                  '& .MuiInputLabel-root.Mui-focused': { color: '#78dbff' }
                }}
              />

              <TextField
                fullWidth
                label="Phone Number"
                name="phone"
                type="tel"
                value={formData.phone}
                onChange={handleChange}
                margin="normal"
                required
                sx={{
                  mb: 2,
                  '& .MuiOutlinedInput-root': {
                    color: 'white',
                    '& fieldset': { borderColor: 'rgba(120, 219, 255, 0.3)' },
                    '&:hover fieldset': { borderColor: 'rgba(120, 219, 255, 0.5)' },
                    '&.Mui-focused fieldset': { borderColor: '#78dbff' },
                  },
                  '& .MuiInputLabel-root': { color: 'rgba(255, 255, 255, 0.7)' },
                  '& .MuiInputLabel-root.Mui-focused': { color: '#78dbff' }
                }}
              />

              <TextField
                select
                fullWidth
                label="Preferred Channel"
                name="alertChannel"
                value={formData.alertChannel}
                onChange={handleChange}
                margin="normal"
                sx={{
                  mb: 2,
                  '& .MuiOutlinedInput-root': {
                    color: 'white',
                    '& fieldset': { borderColor: 'rgba(120, 219, 255, 0.3)' },
                    '&:hover fieldset': { borderColor: 'rgba(120, 219, 255, 0.5)' },
                    '&.Mui-focused fieldset': { borderColor: '#78dbff' },
                  },
                  '& .MuiInputLabel-root': { color: 'rgba(255, 255, 255, 0.7)' },
                  '& .MuiInputLabel-root.Mui-focused': { color: '#78dbff' }
                }}
              >
                <MenuItem value="sms">SMS</MenuItem>
                <MenuItem value="email">Email</MenuItem>
                <MenuItem value="both">SMS + Email</MenuItem>
              </TextField>

              <TextField
                fullWidth
                label="Location (City, Country)"
                name="location"
                value={formData.location}
                onChange={handleChange}
                margin="normal"
                placeholder="e.g., Mumbai, India"
                sx={{
                  mb: 2,
                  '& .MuiOutlinedInput-root': {
                    color: 'white',
                    '& fieldset': { borderColor: 'rgba(120, 219, 255, 0.3)' },
                    '&:hover fieldset': { borderColor: 'rgba(120, 219, 255, 0.5)' },
                    '&.Mui-focused fieldset': { borderColor: '#78dbff' },
                  },
                  '& .MuiInputLabel-root': { color: 'rgba(255, 255, 255, 0.7)' },
                  '& .MuiInputLabel-root.Mui-focused': { color: '#78dbff' }
                }}
              />

              <FormControlLabel
                control={<Checkbox name="agree" checked={formData.agree} onChange={handleChange} sx={{ color: 'rgba(120, 219, 255, 0.5)', '&.Mui-checked': { color: '#78dbff' } }} />}
                label={<Typography sx={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: '0.9rem' }}>I agree to receive emergency alerts and communications</Typography>}
                sx={{ mb: 3, mt: 1 }}
              />

              <Button
                type="submit"
                fullWidth
                variant="contained"
                size="large"
                disabled={!formData.agree}
                sx={{
                  py: 1.5,
                  mb: 2,
                  backgroundColor: '#78dbff',
                  color: '#0a0a2a',
                  fontWeight: 'bold',
                  fontSize: '1rem',
                  '&:hover': { backgroundColor: '#a6e8ff', boxShadow: '0 0 20px rgba(120, 219, 255, 0.5)' }
                }}
              >
                Enroll Now
              </Button>

              <Box sx={{ textAlign: 'center', mt: 3 }}>
                <Typography sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  Prefer to sign up first?{' '}
                  <Link to="/signup" style={{ color: '#78dbff', textDecoration: 'none', fontWeight: 600 }}>Create Account</Link>
                </Typography>
              </Box>
            </form>
          </CardContent>
        </Card>

        {/* Back to Home Link */}
        <Box sx={{ textAlign: 'center', mt: 3 }}>
          <Link to="/" style={{ color: 'rgba(255, 255, 255, 0.6)', textDecoration: 'none', fontSize: '0.9rem' }}>
            ← Back to Home
          </Link>
        </Box>
      </Container>
    </Box>
  );
}

export default EmergencyEnroll;


