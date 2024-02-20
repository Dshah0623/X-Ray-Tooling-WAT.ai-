import React, { useState } from 'react';
import {
  TextField,
  Typography,
  Button,
  Grid,
  Alert,
  AlertTitle,
  AppBar,
  Toolbar,
} from '@mui/material';
import { auth } from "../firebase";
import { getAuth, createUserWithEmailAndPassword } from "firebase/auth";
import { Route, useNavigate } from 'react-router-dom';

const Register = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showAlert, setShowAlert] = useState(false);
  const [alertMessage, setAlertMessage] = useState('');

  const handleEmailChange = (event) => {
    setEmail(event.target.value);
  };
  const handlePasswordChange = (event) => {
    setPassword(event.target.value);
  };
  const handleConfirmPasswordChange = (event) => {
    setConfirmPassword(event.target.value);
  };
  let navigate = useNavigate();

  const handleRegister = () => {
    const auth = getAuth();
    if (password != confirmPassword) {
      setAlertMessage("Passwords do not match");

      console.log(alertMessage);
      showAlert = true;
    }
    createUserWithEmailAndPassword(auth, email, password)
      .then((userCredential) => {
        // Signed up 
        console.log("User Created");
        console.log("Authenticated")
        navigate('/Stepone')
      })
      .catch((error) => {
        console.log(error.message)
        console.error('Firebase Error: ', error);
      });
  };

  return (
    <div style={{ backgroundColor: 'white', padding: '20px', borderRadius: '8px' }}>
       <AppBar position="static" style={{ background: 'Yellow' }}>
        <Toolbar>
          <Typography style={{ color: 'Black' }} variant="h6">Welcome to XRAY Tooling Project</Typography>
        </Toolbar>
      </AppBar>
      <Typography variant="h4">Register</Typography>
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <TextField
            label="Email"
            fullWidth
            variant="outlined"
            value={email}
            onChange={handleEmailChange}
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            type="password"
            label="Password"
            fullWidth
            variant="outlined"
            value={password}
            onChange={handlePasswordChange}
          />
        </Grid>
        <Grid item xs={12}>
          <TextField
            type="password"
            label="ConfirmPassword"
            fullWidth
            variant="outlined"
            value={confirmPassword}
            onChange={handleConfirmPasswordChange}
          />
        </Grid>
        <Grid item xs={12}>
          <Button variant="contained" color="primary" onClick={handleRegister}>
            Register
          </Button>
        </Grid>
      </Grid>

      {showAlert && (
        <Alert severity="info">
          <AlertTitle>Info</AlertTitle>
          {alertMessage}
        </Alert>
      )}
    </div>
  );
};

export default Register;
