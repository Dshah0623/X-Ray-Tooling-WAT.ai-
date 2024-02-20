import logo from "./logo.svg";
import "./App.css";
import React from "react";
import Login from "./Login/Login";
import Register from "./Register";
import Stepone from "./Step1/Stepone";
import Steptwo from "./Steptwo";
import Steptwo from "./Step3/Stepthree";

import ChatScreen from "./chatpage";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
function App() {
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <Routes>
            <Route path="/" element={<Stepone />} />
            <Route path="/Stepone" element={<Stepone />} />
            <Route path="/Steptwo" element={<Steptwo />} />
            <Route path="/Stepthree" element={<Stepthree />} />
            <Route path="/RAG" element={<ChatScreen />} />
            <Route path="/Register" element={<Register />} />
          </Routes>
        </header>
      </div>
    </Router>
  );
}

export default App;
