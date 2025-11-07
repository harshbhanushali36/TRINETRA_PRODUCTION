// ============================================
// Final Home.js - No Preloader
// ============================================

import React from 'react';
import { Link } from 'react-router-dom';
import { 
  Box, 
  Typography, 
  Button, 
  Container,
  AppBar,
  Toolbar,
  Card,
  CardContent,
  Grid,
  IconButton,
  TextField,
  Divider
} from '@mui/material';
import { SatelliteAlt, Warning, Notifications, Menu } from '@mui/icons-material';
import Galaxy from './Galaxy';

function Home() {
  return (
    <Box sx={{ 
      flexGrow: 1, 
      minHeight: '100vh', 
      color: 'white',
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

      {/* Navigation Bar */}
      <Box sx={{ 
        position: 'fixed', 
        top: 0, 
        left: 0, 
        right: 0, 
        zIndex: 1000,
        p: 2 
      }}>
        <AppBar 
          position="static" 
          elevation={0}
          sx={{ 
            backgroundColor: 'rgba(26, 26, 74, 0.4)',
            backdropFilter: 'blur(20px)',
            borderRadius: '16px',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            maxWidth: '1200px',
            margin: '0 auto',
            boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.5)'
          }}
        >
          <Toolbar sx={{ justifyContent: 'space-between', py: 1 }}>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <img 
                src="/trinetra_logo.png" 
                alt="Trinetra Logo" 
                style={{ 
                  height: '40px', 
                  width: 'auto',
                  marginRight: '12px',
                  backgroundColor: 'transparent',
                  filter: 'drop-shadow(0 0 0 transparent)'
                }} 
                onError={(e) => {
                  e.target.style.display = 'none';
                }}
              />
              <Typography variant="h6" component="div" sx={{ fontWeight: 'bold', color: 'white' }}>
                Trinetra
              </Typography>
            </Box>

            <Box sx={{ display: { xs: 'none', md: 'flex' }, gap: 3 }}>
              <Button color="inherit" sx={{ textTransform: 'none', fontWeight: 500, fontSize: '1rem',color: 'white' }}>Features</Button>
              <Button color="inherit" sx={{ textTransform: 'none', fontWeight: 500, fontSize: '1rem',color: 'white' }}>Testimonials</Button>
              <Button color="inherit" sx={{ textTransform: 'none', fontWeight: 500, fontSize: '1rem',color: 'white' }}>Highlights</Button>
              <Button color="inherit" sx={{ textTransform: 'none', fontWeight: 500, fontSize: '1rem',color: 'white' }}>Pricing</Button>
              <Button color="inherit" sx={{ textTransform: 'none', fontWeight: 500, fontSize: '1rem',color: 'white' }}>FAQ</Button>
              <Button color="inherit" sx={{ textTransform: 'none', fontWeight: 500, fontSize: '1rem',color: 'white' }}>Blog</Button>
            </Box>

            <Box sx={{ display: 'flex', gap: 2 }}>
              <Button 
                component={Link}
                to="/signup"
                variant="contained" 
                sx={{ 
                  textTransform: 'none',
                  backgroundColor: '#78dbff',
                  color: '#0a0a2a',
                  fontSize: '1rem',
                  borderRadius: '8px',
                  fontWeight: 'bold',
                  '&:hover': {
                    backgroundColor: '#a6e8ff',
                    boxShadow: '0 0 20px rgba(120, 219, 255, 0.5)'
                  }
                }}
              >
                Sign Up
              </Button>
              <Button 
                component={Link}
                to="/signin"
                variant="outlined" 
                sx={{ 
                  textTransform: 'none',
                  borderColor: 'rgba(120, 219, 255, 0.6)',
                  color: '#78dbff',
                  fontSize: '1rem',
                  borderRadius: '8px',
                  '&:hover': {
                    borderColor: '#a6e8ff',
                    backgroundColor: 'rgba(120, 219, 255, 0.1)'
                  }
                }}
              >
                Sign In
              </Button>
            </Box>
            <IconButton sx={{ display: { xs: 'block', md: 'none' }, color: 'white' }}>
              <Menu />
            </IconButton>
          </Toolbar>
        </AppBar>
      </Box>

      {/* Hero Section */}
      <Box sx={{ py: 12, textAlign: 'center', pt: 16, position: 'relative', zIndex: 1 }}>
        <Container maxWidth="lg">
          <Typography variant="h2" component="h1" gutterBottom sx={{ fontWeight: 'bold', mb: 3, textShadow: '0 0 20px rgba(120, 219, 255, 0.5)', letterSpacing: '2px' }}>
            Monitor Space Threats in Real-Time
          </Typography>
          <Typography variant="h5" sx={{ mb: 4, maxWidth: '600px', mx: 'auto', opacity: 0.9, fontSize: '1.5rem' }}>
            Advanced AI-powered Space Situational Awareness system for collision prediction, 
            orbital debris tracking, and emergency alert integration.
          </Typography>
          <Box sx={{ mt: 4 }}>
            <Button 
              component={Link}
              to="/enroll"
              variant="contained" 
              size="large" 
              sx={{ 
                mr: 2, 
                px: 4, 
                backgroundColor: '#78dbff',
                color: '#0a0a2a',
                fontWeight: 'bold',
                fontSize: '1.1rem',
                '&:hover': {
                  backgroundColor: '#a6e8ff',
                  boxShadow: '0 0 20px rgba(120, 219, 255, 0.5)'
                }
              }}
            >
              Enroll for Alerts
            </Button>
            <Button 
              component={Link}
              to="/dashboard"
              variant="outlined" 
              size="large" 
              sx={{ 
                px: 4, 
                borderColor: '#78dbff', 
                color: '#78dbff',
                fontSize: '1.1rem',
                '&:hover': {
                  borderColor: '#a6e8ff',
                  backgroundColor: 'rgba(120, 219, 255, 0.1)'
                }
              }}
            >
              Dashboard
            </Button>
          </Box>
        </Container>
      </Box>

      {/* Features Section */}
      <Container sx={{ py: 8, position: 'relative', zIndex: 1 }}>
        <Typography variant="h3" component="h2" textAlign="center" gutterBottom sx={{ mb: 6, textShadow: '0 0 15px rgba(120, 219, 255, 0.3)', letterSpacing: '1px' }}>
          Key Features
        </Typography>
        <Grid container spacing={4} justifyContent="center">
          <Grid item xs={12} md={4} sx={{ display: 'flex', justifyContent: 'center' }}>
            <Card sx={{ 
              background: 'rgba(26, 26, 74, 0.4)',
              backdropFilter: 'blur(15px)',
              border: '1px solid rgba(120, 219, 255, 0.3)',
              maxWidth: 350,
              width: '100%',
              transition: 'transform 0.3s, box-shadow 0.3s',
              '&:hover': {
                transform: 'translateY(-8px)',
                boxShadow: '0 10px 30px rgba(120, 219, 255, 0.2)',
                border: '1px solid rgba(120, 219, 255, 0.5)'
              }
            }}>
              <CardContent sx={{ textAlign: 'center', p: 4 }}>
                <SatelliteAlt sx={{ fontSize: 60, mb: 2, color: '#78dbff' }} />
                <Typography variant="h5" gutterBottom sx={{ fontSize: '1.5rem' }}>
                  Real-Time Orbit Tracking
                </Typography>
                <Typography sx={{ opacity: 0.8, fontSize: '1.1rem' }}>
                  Monitor 40,000+ space objects with SGP4 propagation and high-fidelity orbital models
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4} sx={{ display: 'flex', justifyContent: 'center' }}>
            <Card sx={{ 
              background: 'rgba(45, 26, 74, 0.4)',
              backdropFilter: 'blur(15px)',
              border: '1px solid rgba(255, 119, 198, 0.3)',
              maxWidth: 350,
              width: '100%',
              transition: 'transform 0.3s, box-shadow 0.3s',
              '&:hover': {
                transform: 'translateY(-8px)',
                boxShadow: '0 10px 30px rgba(255, 119, 198, 0.2)',
                border: '1px solid rgba(255, 119, 198, 0.5)'
              }
            }}>
              <CardContent sx={{ textAlign: 'center', p: 4 }}>
                <Warning sx={{ fontSize: 60, mb: 2, color: '#ff77c6' }} />
                <Typography variant="h5" gutterBottom sx={{ fontSize: '1.5rem' }}>
                  AI Collision Prediction
                </Typography>
                <Typography sx={{ opacity: 0.8, fontSize: '1.1rem' }}>
                  Machine Learning-powered risk assessment with 90%+ accuracy for high-risk events
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4} sx={{ display: 'flex', justifyContent: 'center' }}>
            <Card sx={{ 
              background: 'rgba(26, 26, 74, 0.4)',
              backdropFilter: 'blur(15px)',
              border: '1px solid rgba(120, 119, 198, 0.3)',
              maxWidth: 350,
              width: '100%',
              transition: 'transform 0.3s, box-shadow 0.3s',
              '&:hover': {
                transform: 'translateY(-8px)',
                boxShadow: '0 10px 30px rgba(120, 119, 198, 0.2)',
                border: '1px solid rgba(120, 119, 198, 0.5)'
              }
            }}>
              <CardContent sx={{ textAlign: 'center', p: 4 }}>
                <Notifications sx={{ fontSize: 60, mb: 2, color: '#7877c6' }} />
                <Typography variant="h5" gutterBottom sx={{ fontSize: '1.5rem' }}>
                  Emergency Alert System
                </Typography>
                <Typography sx={{ opacity: 0.8, fontSize: '1.1rem' }}>
                  Instant SMS, email, and mobile notifications for dangerous conjunction events
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Container>

      {/* Stats Section */}
      <Container sx={{ py: 8, position: 'relative', zIndex: 1 }}>
        <Grid container spacing={8} justifyContent="center" textAlign="center">
          <Grid item xs={6} md={3} sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Typography variant="h2" sx={{ color: '#78dbff', fontWeight: 'bold', mb: 1, textShadow: '0 0 10px rgba(120, 219, 255, 0.5)' }}>
              40K+
            </Typography>
            <Typography sx={{ opacity: 0.8, fontSize: '1.1rem' }}>
              Tracked Objects
            </Typography>
          </Grid>
          <Grid item xs={6} md={3} sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Typography variant="h2" sx={{ color: '#ff77c6', fontWeight: 'bold', mb: 1, textShadow: '0 0 10px rgba(255, 119, 198, 0.5)' }}>
              85%
            </Typography>
            <Typography sx={{ opacity: 0.8, fontSize: '1.1rem' }}>
              Prediction Accuracy
            </Typography>
          </Grid>
          <Grid item xs={6} md={3} sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Typography variant="h2" sx={{ color: '#78dbff', fontWeight: 'bold', mb: 1, textShadow: '0 0 10px rgba(120, 219, 255, 0.5)' }}>
              24/7
            </Typography>
            <Typography sx={{ opacity: 0.8, fontSize: '1.1rem' }}>
              Real-time Monitoring
            </Typography>
          </Grid>
          <Grid item xs={6} md={3} sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Typography variant="h2" sx={{ color: '#7877c6', fontWeight: 'bold', mb: 1, textShadow: '0 0 10px rgba(120, 119, 198, 0.5)' }}>
              10⁻¹²
            </Typography>
            <Typography sx={{ opacity: 0.8, fontSize: '1.1rem' }}>
              Low Probability Detection
            </Typography>
          </Grid>
        </Grid>
      </Container>

      {/* Newsletter Section */}
      <Container sx={{ py: 12, textAlign: 'center', position: 'relative', zIndex: 1 }}>
        <Typography variant="h3" component="h2" gutterBottom sx={{ textShadow: '0 0 15px rgba(120, 219, 255, 0.3)', letterSpacing: '1px' }}>
          Our Latest Products
        </Typography>
        <Typography variant="h6" sx={{ mb: 3, opacity: 0.9, maxWidth: '600px', mx: 'auto', fontSize: '1.3rem' }}>
          Explore our cutting-edge dashboard, delivering high-quality solutions tailored to your needs.
          <Box component="span" sx={{ display: 'block', mt: 1, fontWeight: 'bold' }}>
            Elevate your experience with top-tier features and services.
          </Box>
        </Typography>
        
        <Divider sx={{ my: 4, backgroundColor: 'rgba(120, 219, 255, 0.3)', width: '100px', mx: 'auto', height: '2px' }} />
        
        <Box sx={{ maxWidth: '400px', mx: 'auto', mt: 4 }}>
          <Typography variant="subtitle1" sx={{ mb: 2, textAlign: 'left', fontSize: '1.1rem' }}>
            Your email address
          </Typography>
          <TextField
            fullWidth
            placeholder="Enter your email"
            variant="outlined"
            sx={{ 
              mb: 2,
              '& .MuiOutlinedInput-root': {
                color: 'white',
                fontSize: '1.1rem',
                '& fieldset': {
                  borderColor: 'rgba(120, 219, 255, 0.3)',
                },
                '&:hover fieldset': {
                  borderColor: 'rgba(120, 219, 255, 0.5)',
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#78dbff',
                },
              }
            }}
          />
          <Button 
            variant="contained" 
            fullWidth
            size="large"
            sx={{ 
              backgroundColor: '#78dbff',
              color: '#0a0a2a',
              py: 1.5,
              fontWeight: 'bold',
              fontSize: '1.1rem',
              '&:hover': {
                backgroundColor: '#a6e8ff',
                boxShadow: '0 0 20px rgba(120, 219, 255, 0.5)'
              }
            }}
          >
            Start now
          </Button>
        </Box>
      </Container>

      {/* Footer */}
      <Box sx={{ 
        py: 8, 
        borderTop: '1px solid rgba(120, 219, 255, 0.15)', 
        position: 'relative', 
        zIndex: 2,
        background: 'rgba(10, 10, 42, 0.4)',
        backdropFilter: 'blur(10px)'
      }}>
        <Container>
          <Grid container spacing={4}>
            <Grid item xs={12} md={4}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <Box sx={{
                  width: 36,
                  height: 36,
                  borderRadius: '8px',
                  background: 'linear-gradient(135deg, #78dbff 0%, #7877c6 100%)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <SatelliteAlt sx={{ fontSize: 20, color: 'white' }} />
                </Box>
                <Typography variant="h6" fontWeight={700} sx={{ letterSpacing: '0.5px' }}>
                  TRINETRA SSA
                </Typography>
              </Box>
              <Typography sx={{ opacity: 0.7, mb: 3, lineHeight: 1.7, fontSize: '0.95rem' }}>
                Advanced Space Threat Monitoring and Situational Awareness System for next-generation orbital operations.
              </Typography>
            </Grid>
            
            <Grid item xs={6} md={2}>
              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 700, mb: 2 }}>
                Product
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <Typography sx={{ opacity: 0.7, cursor: 'pointer', '&:hover': { opacity: 1, color: '#78dbff' }, transition: 'all 0.2s' }}>Dashboard</Typography>
                <Typography sx={{ opacity: 0.7, cursor: 'pointer', '&:hover': { opacity: 1, color: '#78dbff' }, transition: 'all 0.2s' }}>API</Typography>
                <Typography sx={{ opacity: 0.7, cursor: 'pointer', '&:hover': { opacity: 1, color: '#78dbff' }, transition: 'all 0.2s' }}>Integrations</Typography>
                <Typography sx={{ opacity: 0.7, cursor: 'pointer', '&:hover': { opacity: 1, color: '#78dbff' }, transition: 'all 0.2s' }}>Pricing</Typography>
              </Box>
            </Grid>
            
            <Grid item xs={6} md={2}>
              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 700, mb: 2 }}>
                Resources
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <Typography sx={{ opacity: 0.7, cursor: 'pointer', '&:hover': { opacity: 1, color: '#78dbff' }, transition: 'all 0.2s' }}>Documentation</Typography>
                <Typography sx={{ opacity: 0.7, cursor: 'pointer', '&:hover': { opacity: 1, color: '#78dbff' }, transition: 'all 0.2s' }}>API Reference</Typography>
                <Typography sx={{ opacity: 0.7, cursor: 'pointer', '&:hover': { opacity: 1, color: '#78dbff' }, transition: 'all 0.2s' }}>Tutorials</Typography>
                <Typography sx={{ opacity: 0.7, cursor: 'pointer', '&:hover': { opacity: 1, color: '#78dbff' }, transition: 'all 0.2s' }}>Blog</Typography>
              </Box>
            </Grid>
            
            <Grid item xs={6} md={2}>
              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 700, mb: 2 }}>
                Company
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <Typography sx={{ opacity: 0.7, cursor: 'pointer', '&:hover': { opacity: 1, color: '#78dbff' }, transition: 'all 0.2s' }}>About</Typography>
                <Typography sx={{ opacity: 0.7, cursor: 'pointer', '&:hover': { opacity: 1, color: '#78dbff' }, transition: 'all 0.2s' }}>Careers</Typography>
                <Typography sx={{ opacity: 0.7, cursor: 'pointer', '&:hover': { opacity: 1, color: '#78dbff' }, transition: 'all 0.2s' }}>Contact</Typography>
                <Typography sx={{ opacity: 0.7, cursor: 'pointer', '&:hover': { opacity: 1, color: '#78dbff' }, transition: 'all 0.2s' }}>Partners</Typography>
              </Box>
            </Grid>
            
            <Grid item xs={6} md={2}>
              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 700, mb: 2 }}>
                Legal
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <Typography sx={{ opacity: 0.7, cursor: 'pointer', '&:hover': { opacity: 1, color: '#78dbff' }, transition: 'all 0.2s' }}>Privacy</Typography>
                <Typography sx={{ opacity: 0.7, cursor: 'pointer', '&:hover': { opacity: 1, color: '#78dbff' }, transition: 'all 0.2s' }}>Terms</Typography>
                <Typography sx={{ opacity: 0.7, cursor: 'pointer', '&:hover': { opacity: 1, color: '#78dbff' }, transition: 'all 0.2s' }}>Security</Typography>
                <Typography sx={{ opacity: 0.7, cursor: 'pointer', '&:hover': { opacity: 1, color: '#78dbff' }, transition: 'all 0.2s' }}>Compliance</Typography>
              </Box>
            </Grid>
          </Grid>
          
          <Divider sx={{ my: 4, borderColor: 'rgba(120, 219, 255, 0.15)' }} />
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
            <Typography sx={{ opacity: 0.6, fontSize: '0.9rem' }}>
              © 2024 TRINETRA SSA. SVKM's NMIMS STME. All rights reserved.
            </Typography>
            <Box sx={{ display: 'flex', gap: 3 }}>
              <Typography sx={{ opacity: 0.6, fontSize: '0.9rem', cursor: 'pointer', '&:hover': { opacity: 1 } }}>
                Status
              </Typography>
              <Typography sx={{ opacity: 0.6, fontSize: '0.9rem', cursor: 'pointer', '&:hover': { opacity: 1 } }}>
                Changelog
              </Typography>
            </Box>
          </Box>
        </Container>
      </Box>
    </Box>
  );
}

export default Home;