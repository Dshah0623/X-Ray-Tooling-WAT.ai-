import React, { createContext, useState, useContext  } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Link, useNavigate } from 'react-router-dom'
import {
    TextField,
    Typography,
    Button,
    Alert,
    AlertTitle,
    Grid,
    FormControlLabel,
    ButtonBase,
    AppBar, Toolbar, IconButton, Menu
  } from '@mui/material';

const Login = () => {
    const [Email, setEmail] = useState('');
    const [Password, setPassword] = useState('');
    const [showAlert, setShowAlert] = useState(false);
    const [AlertMessage, setAlertMessage] = useState('');

    const handleEmailChange = (event) => {
        setEmail(event.target.value);
    };
    const handlePasswordChange = (event) => {
        setPassword(event.target.value);
    };
    let navigate = useNavigate(); 
    const routeChange = () =>{ 
        navigate('/');
    }

    return(
        <div>
            "HELLO"
        </div>
    )


};
export default Login;