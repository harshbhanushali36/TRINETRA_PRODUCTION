import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Box,
  Container,
  TextField,
  Button,
  Typography,
  Card,
  CardContent,
  IconButton,
  InputAdornment,
  Divider,
  Checkbox,
  FormControlLabel
} from '@mui/material';
import { Visibility, VisibilityOff, SatelliteAlt, Google, GitHub } from '@mui/icons-material';
import Galaxy from './Galaxy';

function SignUp() {
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [formData, setFormData] = useState({
    fullName: '',
    email: '',
    password: '',
    confirmPassword: '',
    agreeToTerms: false
  });

  const handleClickShowPassword = () => setShowPassword(!showPassword);
  const handleClickShowConfirmPassword = () => setShowConfirmPassword(!showConfirmPassword);

  const handleChange = (e) => {
    const { name, value, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'agreeToTerms' ? checked : value
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Sign up data:', formData);
    // Add your sign up logic here
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
          density={0.8}
          starSpeed={0.2}
          hueShift={220}
          speed={0.6}
          glowIntensity={0.3}
          saturation={0.3}
          mouseRepulsion={false}
          mouseInteraction={false}
          twinkleIntensity={0.3}
          rotationSpeed={0.03}
          repulsionStrength={2}
          transparent={false}
          style={{
            width: '100%',
            height: '100%'
          }}
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

      {/* Sign Up Card */}
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
            {/* Header */}
            <Box sx={{ textAlign: 'center', mb: 4 }}>
              <Typography variant="h4" sx={{
                fontWeight: 'bold',
                color: 'white',
                mb: 1,
                textShadow: '0 0 20px rgba(120, 219, 255, 0.3)'
              }}>
                Create Account
              </Typography>
              <Typography sx={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: '1rem' }}>
                Join the future of space monitoring
              </Typography>
            </Box>

            {/* Social Sign Up Buttons */}
            <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
              <Button
                fullWidth
                variant="outlined"
                startIcon={<Google />}
                sx={{
                  borderColor: 'rgba(120, 219, 255, 0.3)',
                  color: 'white',
                  py: 1.2,
                  '&:hover': {
                    borderColor: 'rgba(120, 219, 255, 0.5)',
                    backgroundColor: 'rgba(120, 219, 255, 0.05)'
                  }
                }}
              >
                Google
              </Button>
              <Button
                fullWidth
                variant="outlined"
                startIcon={<GitHub />}
                sx={{
                  borderColor: 'rgba(120, 219, 255, 0.3)',
                  color: 'white',
                  py: 1.2,
                  '&:hover': {
                    borderColor: 'rgba(120, 219, 255, 0.5)',
                    backgroundColor: 'rgba(120, 219, 255, 0.05)'
                  }
                }}
              >
                GitHub
              </Button>
            </Box>

            {/* Divider */}
            <Box sx={{ display: 'flex', alignItems: 'center', my: 3 }}>
              <Divider sx={{ flex: 1, borderColor: 'rgba(120, 219, 255, 0.2)' }} />
              <Typography sx={{ px: 2, color: 'rgba(255, 255, 255, 0.5)', fontSize: '0.9rem' }}>
                OR
              </Typography>
              <Divider sx={{ flex: 1, borderColor: 'rgba(120, 219, 255, 0.2)' }} />
            </Box>

            {/* Sign Up Form */}
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
                    '& fieldset': {
                      borderColor: 'rgba(120, 219, 255, 0.3)',
                    },
                    '&:hover fieldset': {
                      borderColor: 'rgba(120, 219, 255, 0.5)',
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: '#78dbff',
                    },
                  },
                  '& .MuiInputLabel-root': {
                    color: 'rgba(255, 255, 255, 0.7)',
                  },
                  '& .MuiInputLabel-root.Mui-focused': {
                    color: '#78dbff',
                  }
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
                    '& fieldset': {
                      borderColor: 'rgba(120, 219, 255, 0.3)',
                    },
                    '&:hover fieldset': {
                      borderColor: 'rgba(120, 219, 255, 0.5)',
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: '#78dbff',
                    },
                  },
                  '& .MuiInputLabel-root': {
                    color: 'rgba(255, 255, 255, 0.7)',
                  },
                  '& .MuiInputLabel-root.Mui-focused': {
                    color: '#78dbff',
                  }
                }}
              />

              <TextField
                fullWidth
                label="Password"
                name="password"
                type={showPassword ? 'text' : 'password'}
                value={formData.password}
                onChange={handleChange}
                margin="normal"
                required
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        onClick={handleClickShowPassword}
                        edge="end"
                        sx={{ color: 'rgba(255, 255, 255, 0.7)' }}
                      >
                        {showPassword ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
                sx={{
                  mb: 2,
                  '& .MuiOutlinedInput-root': {
                    color: 'white',
                    '& fieldset': {
                      borderColor: 'rgba(120, 219, 255, 0.3)',
                    },
                    '&:hover fieldset': {
                      borderColor: 'rgba(120, 219, 255, 0.5)',
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: '#78dbff',
                    },
                  },
                  '& .MuiInputLabel-root': {
                    color: 'rgba(255, 255, 255, 0.7)',
                  },
                  '& .MuiInputLabel-root.Mui-focused': {
                    color: '#78dbff',
                  }
                }}
              />

              <TextField
                fullWidth
                label="Confirm Password"
                name="confirmPassword"
                type={showConfirmPassword ? 'text' : 'password'}
                value={formData.confirmPassword}
                onChange={handleChange}
                margin="normal"
                required
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        onClick={handleClickShowConfirmPassword}
                        edge="end"
                        sx={{ color: 'rgba(255, 255, 255, 0.7)' }}
                      >
                        {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
                sx={{
                  mb: 2,
                  '& .MuiOutlinedInput-root': {
                    color: 'white',
                    '& fieldset': {
                      borderColor: 'rgba(120, 219, 255, 0.3)',
                    },
                    '&:hover fieldset': {
                      borderColor: 'rgba(120, 219, 255, 0.5)',
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: '#78dbff',
                    },
                  },
                  '& .MuiInputLabel-root': {
                    color: 'rgba(255, 255, 255, 0.7)',
                  },
                  '& .MuiInputLabel-root.Mui-focused': {
                    color: '#78dbff',
                  }
                }}
              />

              <FormControlLabel
                control={
                  <Checkbox
                    name="agreeToTerms"
                    checked={formData.agreeToTerms}
                    onChange={handleChange}
                    sx={{
                      color: 'rgba(120, 219, 255, 0.5)',
                      '&.Mui-checked': {
                        color: '#78dbff',
                      }
                    }}
                  />
                }
                label={
                  <Typography sx={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: '0.9rem' }}>
                    I agree to the{' '}
                    <Box component="span" sx={{ color: '#78dbff', cursor: 'pointer' }}>
                      Terms of Service
                    </Box>
                    {' '}and{' '}
                    <Box component="span" sx={{ color: '#78dbff', cursor: 'pointer' }}>
                      Privacy Policy
                    </Box>
                  </Typography>
                }
                sx={{ mb: 3, mt: 1 }}
              />

              <Button
                type="submit"
                fullWidth
                variant="contained"
                size="large"
                sx={{
                  py: 1.5,
                  mb: 2,
                  backgroundColor: '#78dbff',
                  color: '#0a0a2a',
                  fontWeight: 'bold',
                  fontSize: '1rem',
                  '&:hover': {
                    backgroundColor: '#a6e8ff',
                    boxShadow: '0 0 20px rgba(120, 219, 255, 0.5)'
                  }
                }}
              >
                Create Account
              </Button>

              <Box sx={{ textAlign: 'center', mt: 3 }}>
                <Typography sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  Already have an account?{' '}
                  <Link
                    to="/signin"
                    style={{
                      color: '#78dbff',
                      textDecoration: 'none',
                      fontWeight: 600
                    }}
                  >
                    Sign In
                  </Link>
                </Typography>
              </Box>
            </form>
          </CardContent>
        </Card>

        {/* Back to Home Link */}
        <Box sx={{ textAlign: 'center', mt: 3 }}>
          <Link
            to="/"
            style={{
              color: 'rgba(255, 255, 255, 0.6)',
              textDecoration: 'none',
              fontSize: '0.9rem'
            }}
          >
            ← Back to Home
          </Link>
        </Box>
      </Container>
    </Box>
  );
}

export default SignUp;