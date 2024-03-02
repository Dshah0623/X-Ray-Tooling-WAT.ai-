import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Typography,
  Button,
  AppBar, Toolbar, Card, CardContent, Paper, Select, MenuItem, Container, Box, InputLabel, FormControl, FormControlLabel, RadioGroup, Radio
} from '@mui/material';
import './chatpage.css';
import { useLocation } from 'react-router-dom';


const ChatScreen = () => {
  // Passing forward state from previous page
  const location = useLocation();
  const { phaseOneResult, phaseTwoResult } = location.state;
  
  const [activeFlow, setActiveFlow] = useState('Agent'); 
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const [injury, setInjury] = useState(phaseOneResult);
  const [injuryLocation, setInjuryLocation] = useState(phaseTwoResult);
  const [flowMessage, setFlowMessage] = useState('')

  const [model, setModel] = useState('openai');

  const [data, setData] = useState('');

  let navigate = useNavigate();
  const [selectedOption, setSelectedOption] = useState('base');

const sendQuery = async () => {
  if (input.trim() !== '') {
    const newMessage = { text: input, sender: 'user' };
    setMessages(messages => [...messages, newMessage]);
    setInput(''); 

    try {
      const response = await fetch('http://127.0.0.1:8000/rag/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: input.trim(), model: model }),
      });
      if (response.ok) {
        const data = await response.json();
        console.log('RAG run:', data);
        setData(data);
        const serverMessage = { text: data.response, sender: 'bot' };
        setMessages(messages => [...messages, serverMessage]); // Add new server message to the conversation
      }
    } catch (error) {
      console.error('Error running RAG:', error);
    }
  }
};


const sendFlowQuery = async (flow) => {
  if (injury.trim() == '' || injuryLocation.trim() == '') return;


  try {
    // set loading
    setFlowMessage("Loading...");
    const response = await fetch('http://127.0.0.1:8000/rag/flow', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ injury: injury, injury_location: injuryLocation, flow: flow, model: model }),
    });
    if (response.ok) {
      const data = await response.json();
      console.log('RAG run:', data);
      setData(data);
      const serverMessage = { text: data.response.content, sender: 'bot' };
      setFlowMessage(serverMessage.text); // Add new server message to the conversation
    }
  } catch (error) {
    console.error('Error running RAG:', error);
  }
  
};

const sendFlowStream = async (flow) => {
  if (injury.trim() == '' || injuryLocation.trim() == '') return;
  try {
    // set loading
    setFlowMessage("Loading...");
    const response = await fetch('http://127.0.0.1:8000/rag/flow', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ injury: injury, injury_location: injuryLocation, flow: flow, model: model }),
    });

    const reader = response.body.getReader();
    const chunks = [];
    

  } catch (error) {
    console.error('Error running RAG:', error);
  }
  
};

  const renderActiveFlow = () => {
    switch (activeFlow) {
      case 'Agent':
          return (
            <div className="chat-screen">
              <div className="messages">
                {messages.map((message, index) => (
                  <div key={index} className={`message ${message.sender}`}>
                    {message.text}
                  </div>
                ))}
              </div>
              <div className="input-area">
                <input
                  type="text"
                  placeholder="Type a message..."
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                />
                <button onClick={sendQuery}>Send</button>
              </div>
            </div>
          );

      case 'Flows':
        return (
          <div className="chat-screen" style={{width:"70%"}}>
            <div className="input-area">
              <input
                type="text"
                placeholder="Injury"
                value={injury}
                onChange={(e) => setInjury(e.target.value)}
              />
            </div>
            <div className="input-area">
              <input
                type="text"
                placeholder="Injury Location"
                value={injuryLocation}
                onChange={(e) => setInjuryLocation(e.target.value)}
              />
            </div>
            <div className="input-area">
              <button onClick={() => sendFlowQuery("base")}>Base Flow</button>
              <button onClick={() => sendFlowQuery("restriction")}>Restriction Flow</button>
              <button onClick={() => sendFlowQuery("heat_ice")}>Heat & Ice Flow</button>
              <button onClick={() => sendFlowQuery("expectation")}>Expectation Flow</button>
            </div>
            <div className="messages" style={{width: "fit"}}>
              <p style={{color:"black", flex: 1, flexWrap: 'wrap'}}>{flowMessage}</p>
            </div>
        </div>
        );

      default:
        return <div>Select a flow</div>;
    }
  };

  const handleRAG = () => {
    navigate('/RAG');
  };

  const handleLogin = () => {
    navigate('/Login');
  }

  const handleSelectChange = (event) => {
    setSelectedOption(event.target.value);
    sendFlowQuery(selectedOption);
  };

  const handleResults = ()=>{
    navigate('/Results')
  };


  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh', width: '100vw'}}>
      {/* <div className="top-bar">
        <button onClick={() => setActiveFlow('Agent')}>General Chat Agent</button>
        <button onClick={() => setActiveFlow('Flows')}>Injury-specific Flows</button>
        {/* add more flows here */}
      {/* </div>
      {renderActiveFlow()} */} 
      <AppBar position="static" sx={{ backgroundColor: 'white', height: '65px', width: '100%', borderBottom: 'none', boxShadow: 'none' }}>
          <Toolbar variant="dense" sx={{ display: 'flex', justifyContent: 'space-between', borderBottom: 'none' }}>
                <Typography variant="h6" component="div" sx={{ color: 'black', marginTop:'5px', fontWeight:'bold' }}>
                <span style={{ color: '#4686ee' }}>X-Ray</span><span style={{ color: 'black' }}>Tooling</span>
                </Typography>
                <div sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  <Button color="inherit" sx={{ color: 'black', marginRight: '16px'}} onClick={handleResults}>Results</Button>
                  <Button color="inherit" sx={{ color: 'black', marginLeft:'16px' }} onClick={handleRAG}>Rehabilitation</Button>
                </div>
                <Button color="inherit" onClick={handleLogin} sx={{ color: 'white', backgroundColor:'#4686ee', borderRadius:'20px', width: '100px','&:hover': { backgroundColor: 'grey'} }} >Log In</Button>
          </Toolbar>
        </AppBar>

        <Card sx={{boxShadow:'none', display:'flex', alignContent:'horizontal'}}>
          <CardContent sx={{'width': '60%'}}>
          <Typography variant="h5" sx={{boxShadow:'none', fontWeight:'Bold', marginRight:'70%', marginBottom:'2%'}}>ChatBot</Typography>
          <Box sx={{borderRadius: '8px',  border: '2px solid #ccc'}}>
          <FormControl component="fieldset" sx={{padding:'3%',display: 'flex', flexDirection: 'column',alignItems: 'flex-start'}}>
            <RadioGroup value={selectedOption} onChange={handleSelectChange} >
              <FormControlLabel value="base" control={<Radio sx={{ color: 'black', marginBottom:'2%' }} />} label={
                    <Typography variant="body1" sx={{ color: 'black', display: 'flex', flexDirection: 'column',alignItems: 'flex-start' }}>
                      <span style={{ fontSize: '1.0em', fontWeight:'bold' }}>Base</span>
                      <span style={{ fontSize: '0.8em', color:'grey' }}>Great for general diagnosis on the injury</span>
                    </Typography>
                  } />
              <FormControlLabel value="restriction" control={<Radio sx={{ color: 'black', marginBottom:'2%' }} />} label={
                    <Typography variant="body1" sx={{ color: 'black', flexDirection: 'column',alignItems: 'flex-start', display: 'flex', flexDirection: 'column' }}>
                      <span style={{ fontSize: '1.0em', fontWeight:'bold' }}>Restriction</span>
                      <span style={{ fontSize: '0.8em', color:'grey' }}>Describes things to avoid depending on the injury</span>
                    </Typography>
                  } />
              <FormControlLabel value="heat_ice" control={<Radio sx={{ color: 'black',marginBottom:'2%' }} />} label={
                    <Typography variant="body1" sx={{ color: 'black', flexDirection: 'column',alignItems: 'flex-start', display: 'flex', flexDirection: 'column' }}>
                      <span style={{ fontSize: '1.0em', fontWeight:'bold' }}>Heat & Ice</span>
                      <span style={{ fontSize: '0.8em', color:'grey' }}>Provides information best practices for heating and icing</span>
                    </Typography>
                  } />
              <FormControlLabel value="expectation" control={<Radio sx={{ color: 'black', marginBottom:'2%' }}/>} label={
                    <Typography variant="body1" sx={{ color: 'black', flexDirection: 'column',alignItems: 'flex-start', display: 'flex', flexDirection: 'column' }}>
                      <span style={{ fontSize: '1.0em', fontWeight:'bold' }}>Expectation</span>
                      <span style={{ fontSize: '0.8em', color:'grey' }}>Reports on the typical time of recovery as well as surgery expectations</span>
                    </Typography>
                  } />
            </RadioGroup>
          </FormControl>
          </Box>
          </CardContent>
          <div className="chat-container"> 
            <div className="chat-screen">
              <div className="messages">
                  {messages.map((message, index) => (
                      <div className="message-container" key={index}>
                          <div className={`message ${message.sender}`}>
                              {message.text}
                          </div>
                      </div>
                  ))}
              </div>
                <div className="input-area">
                  <input
                    type="text"
                    placeholder="Type a message and press 'Enter' to Chat!"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => {if (e.key === 'Enter') {sendQuery();}}}
                  />
                  <button onClick={sendQuery} ></button>
                </div>
              </div>
          </div>
          
        </Card>
    </div>
  );
};

export default ChatScreen;
