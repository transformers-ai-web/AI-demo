import React, { useState } from "react";
import { FaRobot, FaPaperPlane } from "react-icons/fa";
import "./App.css";
import "./pages/chatbot.css";

function App() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hello! Ask me anything." },
  ]);
  const [input, setInput] = useState("");
  const [activeTab, setActiveTab] = useState("chatbot");

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message to chat
    const userMsg = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMsg]);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: input }),
      });

      const data = await response.json();

      const botMsg = { sender: "bot", text: data.response };
      setMessages((prev) => [...prev, botMsg]);
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Error: could not reach server." },
      ]);
    }

    setInput("");
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <FaRobot className="icon" />
          <span>Chatbot</span>
        </div>
        <nav className="nav">
          <button
            onClick={() => setActiveTab("chatbot")}
            style={{
              backgroundColor:
                activeTab === "chatbot" ? "#2a2a2a" : "transparent",
              color: activeTab === "chatbot" ? "#fff" : "#aaa",
            }}
          >
            💬 Chatbot
          </button>
          {/* <button
            onClick={() => setActiveTab("rag")}
            style={{
              backgroundColor: activeTab === "rag" ? "#2a2a2a" : "transparent",
              color: activeTab === "rag" ? "#fff" : "#aaa",
            }}
          >
            📘 RAG Demo
          </button> */}
        </nav>
      </aside>

      {/* Chat area */}
      <main className="chat-section">
        <header className="chat-header">
          <h1>{activeTab === "chatbot" ? "AI Chat" : "RAG Demo"}</h1>
          <button className="header-btn">Documentation</button>
        </header>

        <div className="chat-messages">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`message ${msg.sender === "bot" ? "bot" : "user"}`}
              style={{
                backgroundColor:
                  msg.sender === "bot" ? "#2a2a2a" : "#3b82f6",
                color: msg.sender === "bot" ? "#fff" : "#fff",
                alignSelf:
                  msg.sender === "bot" ? "flex-start" : "flex-end",
              }}
            >
              {msg.text}
            </div>
          ))}
        </div>

        <footer className="chat-footer">
          <form className="chat-input" onSubmit={handleSend}>
            <input
              type="text"
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
            />
            <button type="submit">
              <FaPaperPlane />
            </button>
          </form>
        </footer>
      </main>
    </div>
  );
}

export default App;
