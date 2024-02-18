import logo from './logo.svg';
import './App.css';
import React from 'react';
import Login from './Login/Login';
import Stepone from './Step1/Stepone';
import Steptwo from './Steptwo';
import ChatScreen from './chatpage';
import SignUp from './SignUp'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
function App() {
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <Routes>
            <Route path="/" element={<Login />} />
            <Route path="/Stepone" element={<Stepone />} />
            <Route path="/Steptwo" element={<Steptwo />} />
            <Route path="/RAG" element={<ChatScreen />} />
            <Route path="/SignUp" element={<SignUp />} />
          </Routes>
        </header>
      </div>
    </Router>
  );
}

export default App;
