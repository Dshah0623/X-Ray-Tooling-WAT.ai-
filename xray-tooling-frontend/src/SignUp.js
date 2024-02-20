// SignUpForm.js
import React, { useState } from 'react';
import { TextField, Button, Container, Typography, CssBaseline } from '@mui/material';
import { createUserWithEmailAndPassword } from 'firebase/auth';
import { Route, useNavigate } from 'react-router-dom';
import { auth } from "./firebase";

const SignUpForm = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
    
  let navigate = useNavigate();

  const handleSignUp = async () => {
    try {
      await createUserWithEmailAndPassword(auth, email, password);
      console.log('User registered successfully!');
      navigate('/Stepone');
    } catch (error) {
      console.error('Error signing up:', error.message);
    }
    
  };

  return (
    <Container component="main" maxWidth="sm" sx={{ backgroundColor: 'white', borderRadius: 8, boxShadow: 2, p: 4 }}>
      <CssBaseline />
      <Typography variant="h4" align="center" color="textPrimary" gutterBottom>
        Sign Up
      </Typography>
      <Typography variant="subtitle1" align="center" color="textSecondary" gutterBottom>
        Create an account to get started
      </Typography>
      <form>
        <TextField
          label="Email"
          type="email"
          fullWidth
          margin="normal"
          variant="outlined"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <TextField
          label="Password"
          type="password"
          fullWidth
          margin="normal"
          variant="outlined"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <Button variant="contained" color="primary" fullWidth onClick={handleSignUp}>
          Sign Up
        </Button>
      </form>
    </Container>
  );
};

export default SignUpForm;
