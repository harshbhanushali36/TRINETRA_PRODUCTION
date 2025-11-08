import React, { useState, useRef, useEffect } from 'react';
import { Box, Paper, TextField, IconButton, Typography, CircularProgress, Fade } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import CloseIcon from '@mui/icons-material/Close';
import ChatIcon from '@mui/icons-material/Chat';
import './chatbot.css';

const Chatbot = ({ selectedObject = null }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hello! I'm TRINETRA, your AI assistant for space object tracking. I can help you with questions about satellites, orbits, space debris, and more. How can I assist you today?"
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [apiConnected, setApiConnected] = useState(true); // Assume connected initially
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');
    
    // Add user message to chat
    const newMessages = [...messages, { role: 'user', content: userMessage }];
    setMessages(newMessages);
    setIsLoading(true);

    try {
      // Prepare conversation history for API
      const history = newMessages.slice(0, -1).map(msg => ({
        role: msg.role,
        content: msg.content
      }));

      // API URL - can be configured via environment variable
      const apiUrl = process.env.REACT_APP_CHATBOT_API_URL || 'http://localhost:5000/api/chat';
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          selectedObject: selectedObject,
          history: history
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `API error: ${response.status}`);
      }

      const data = await response.json();
      setApiConnected(true);
      
      // Add assistant response to chat
      setMessages([...newMessages, { role: 'assistant', content: data.response }]);
    } catch (error) {
      console.error('Error sending message:', error);
      setApiConnected(false);
      
      let errorMessage = 'Sorry, I encountered an error. ';
      if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        errorMessage += 'Please make sure the backend API is running on http://localhost:5000 and try again.';
      } else {
        errorMessage += error.message || 'Please try again later.';
      }
      
      setMessages([
        ...newMessages,
        {
          role: 'assistant',
          content: errorMessage
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      {/* Chat Toggle Button */}
      {!isOpen && (
        <Fade in={!isOpen}>
          <Box
            className="chatbot-toggle"
            onClick={() => setIsOpen(true)}
            sx={{
              position: 'fixed',
              bottom: 24,
              right: 24,
              width: 60,
              height: 60,
              borderRadius: '50%',
              backgroundColor: '#78dbff',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              boxShadow: '0 8px 32px rgba(120, 219, 255, 0.4)',
              zIndex: 1000,
              transition: 'all 0.3s ease',
              '&:hover': {
                backgroundColor: '#a6e8ff',
                transform: 'scale(1.1)',
                boxShadow: '0 12px 40px rgba(120, 219, 255, 0.6)',
              }
            }}
          >
            <ChatIcon sx={{ color: '#0a0a2a', fontSize: 28 }} />
          </Box>
        </Fade>
      )}

      {/* Chat Window */}
      <Fade in={isOpen}>
        <Paper
          elevation={24}
          sx={{
            position: 'fixed',
            bottom: 24,
            right: 24,
            width: 320,
            height: 480,
            maxHeight: '70vh',
            display: isOpen ? 'flex' : 'none',
            flexDirection: 'column',
            backgroundColor: 'rgba(26, 26, 74, 0.95)',
            backdropFilter: 'blur(20px)',
            border: '1px solid rgba(120, 219, 255, 0.3)',
            borderRadius: '12px',
            zIndex: 1000,
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)',
          }}
        >
          {/* Header */}
          <Box
            sx={{
              padding: 1.5,
              borderBottom: '1px solid rgba(120, 219, 255, 0.3)',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              backgroundColor: 'rgba(10, 10, 42, 0.5)',
              borderRadius: '12px 12px 0 0',
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Typography
                variant="h6"
                sx={{
                  color: '#78dbff',
                  fontFamily: "'Iceland', sans-serif",
                  fontWeight: 400,
                  fontSize: '16px',
                  textShadow: '0 0 15px rgba(120, 219, 255, 0.3)',
                }}
              >
                🤖 TRINETRA
              </Typography>
              <Box
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  backgroundColor: apiConnected ? '#44ff44' : '#ff4444',
                  boxShadow: apiConnected 
                    ? '0 0 8px rgba(68, 255, 68, 0.6)' 
                    : '0 0 8px rgba(255, 68, 68, 0.6)',
                }}
                title={apiConnected ? 'API Connected' : 'API Disconnected'}
              />
            </Box>
            {selectedObject && (
              <Typography
                variant="caption"
                sx={{
                  color: 'rgba(255, 255, 255, 0.7)',
                  fontFamily: "'Iceland', sans-serif",
                  fontSize: '10px',
                  marginRight: 1,
                  maxWidth: '100px',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {selectedObject}
              </Typography>
            )}
            <IconButton
              onClick={() => setIsOpen(false)}
              sx={{
                color: '#ff77c6',
                '&:hover': {
                  color: '#ffa6dd',
                  transform: 'rotate(90deg)',
                },
                transition: 'all 0.3s ease',
              }}
            >
              <CloseIcon />
            </IconButton>
          </Box>

          {/* Messages Container */}
          <Box
            sx={{
              flex: 1,
              overflowY: 'auto',
              padding: 2,
              display: 'flex',
              flexDirection: 'column',
              gap: 2,
            }}
            className="chatbot-messages"
          >
            {messages.map((msg, index) => (
              <Box
                key={index}
                sx={{
                  display: 'flex',
                  justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                }}
              >
                <Box
                  sx={{
                    maxWidth: '75%',
                    padding: 1.5,
                    borderRadius: '12px',
                    backgroundColor:
                      msg.role === 'user'
                        ? 'rgba(120, 219, 255, 0.2)'
                        : 'rgba(10, 10, 42, 0.6)',
                    border:
                      msg.role === 'user'
                        ? '1px solid rgba(120, 219, 255, 0.3)'
                        : '1px solid rgba(255, 255, 255, 0.1)',
                  }}
                >
                  <Typography
                    sx={{
                      color: '#ffffff',
                      fontFamily: "'Iceland', sans-serif",
                      fontSize: '13px',
                      lineHeight: 1.5,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                    }}
                  >
                    {msg.content}
                  </Typography>
                </Box>
              </Box>
            ))}
            {isLoading && (
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'flex-start',
                }}
              >
                <Box
                  sx={{
                    padding: 1.5,
                    borderRadius: '12px',
                    backgroundColor: 'rgba(10, 10, 42, 0.6)',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                  }}
                >
                  <CircularProgress size={20} sx={{ color: '#78dbff' }} />
                </Box>
              </Box>
            )}
            <div ref={messagesEndRef} />
          </Box>

          {/* Input Container */}
          <Box
            sx={{
              padding: 1.5,
              borderTop: '1px solid rgba(120, 219, 255, 0.3)',
              backgroundColor: 'rgba(10, 10, 42, 0.5)',
              borderRadius: '0 0 12px 12px',
              display: 'flex',
              gap: 1,
            }}
          >
            <TextField
              inputRef={inputRef}
              fullWidth
              multiline
              maxRows={4}
              placeholder="Ask about space objects..."
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
              sx={{
                '& .MuiOutlinedInput-root': {
                  color: '#ffffff',
                  fontFamily: "'Iceland', sans-serif",
                  fontSize: '13px',
                  backgroundColor: 'rgba(10, 10, 42, 0.5)',
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
                '& .MuiInputBase-input': {
                  fontSize: '13px',
                },
                '& .MuiInputBase-input::placeholder': {
                  color: 'rgba(255, 255, 255, 0.5)',
                  opacity: 1,
                  fontSize: '13px',
                },
              }}
            />
            <IconButton
              onClick={sendMessage}
              disabled={isLoading || !inputMessage.trim()}
              sx={{
                color: '#78dbff',
                backgroundColor: 'rgba(120, 219, 255, 0.2)',
                '&:hover': {
                  backgroundColor: 'rgba(120, 219, 255, 0.3)',
                  transform: 'scale(1.1)',
                },
                '&:disabled': {
                  color: 'rgba(120, 219, 255, 0.3)',
                  backgroundColor: 'rgba(120, 219, 255, 0.1)',
                },
                transition: 'all 0.3s ease',
              }}
            >
              <SendIcon />
            </IconButton>
          </Box>
        </Paper>
      </Fade>
    </>
  );
};

export default Chatbot;
