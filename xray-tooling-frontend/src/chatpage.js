import React, { useState } from 'react';
import './chatpage.css';

const ChatScreen = () => {
  const [activeFlow, setActiveFlow] = useState('Agent'); 
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [data, setData] = useState('');


const sendMessage = async () => {
  if (input.trim() !== '') {
    const newMessage = { text: input, sender: 'user' };
    setMessages(messages => [...messages, newMessage]);
    setInput(''); 

    try {
      const response = await fetch('http://127.0.0.1:8000/RAG', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: input.trim() }),
      });
      if (response.ok) {
        const data = await response.json();
        console.log('RAG run:', data);
        setData(data);
        const serverMessage = { text: data.results, sender: 'bot' };
        setMessages(messages => [...messages, serverMessage]); // Add new server message to the conversation
      }
    } catch (error) {
      console.error('Error running RAG:', error);
    }
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
                <button onClick={sendMessage}>Send</button>
              </div>
            </div>
          );

      case 'Flow1':
        return (
          <div>
            {/* placeholder for flow logic */}
            <p>Flow1 content goes here...</p>
          </div>
        );

      case 'Flow2':
        return (
          <div>
            {/* placeholder for flow logic */}
            <p>Flow2 content goes here...</p>
          </div>
        );
      default:
        return <div>Select a flow</div>;
    }
  };

  return (
    <div>
      <div className="top-bar">
        <button onClick={() => setActiveFlow('Agent')}>General Chat Agent</button>
        <button onClick={() => setActiveFlow('Flow1')}>Other Flow 1</button>
        <button onClick={() => setActiveFlow('Flow2')}>Other Flow 2</button>
        {/* add more flows here */}
      </div>
      {renderActiveFlow()}
    </div>
  );
};

export default ChatScreen;
