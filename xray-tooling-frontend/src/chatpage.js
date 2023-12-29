import React, { useState } from 'react';

const ChatScreen = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  // Function to handle sending messages
  const sendMessage = async () => {
    // Logic to send messages to backend or service
    // For demo purposes, we'll just add the message to the state
    if (input.trim() !== '') {
      setMessages([...messages, { text: input, sender: 'user' }]);
      setInput('');
    }
    const formData = new FormData();
    formData.append('text', input.trim());
    try {
      const response = await fetch('http://127.0.0.1:8000/RAG', {
        method: 'POST',
        body: formData,
      });
      if(response.ok) {
        const data = await response.json();
        console.log('RAG run:', data);
      }
    }
    catch (error) {
      console.error('Error running RAG:', error);
      // Handle error
    }
    
  };

  return (
    <div className="chat-screen">
      <div className="messages">
        {messages.map((message, index) => (
          <div key={index} className={message.sender === 'user' ? 'message user' : 'message'}>
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
};

export default ChatScreen;
