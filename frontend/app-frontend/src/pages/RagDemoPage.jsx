import React, { useState } from 'react';
import './rag.css';

function RagDemoPage() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [file, setFile] = useState(null);

  const uploadFile = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append('document', file);

    await fetch('http://localhost:8000/upload', {
      method: 'POST',
      body: formData,
    });
  };

  const handleQuery = async () => {
    const res = await fetch('http://localhost:8000/rag-query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    setResponse(data.answer);
  };

  return (
    <div className="rag-container">
      <h2>RAG Demo</h2>
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button onClick={uploadFile}>Upload Document</button>

      <div className="query-section">
        <input 
          value={query} 
          onChange={(e) => setQuery(e.target.value)} 
          placeholder="Ask something about your documents..."
        />
        <button onClick={handleQuery}>Ask</button>
      </div>

      {response && (
        <div className="rag-response">
          <h4>Answer:</h4>
          <p>{response}</p>
        </div>
      )}
    </div>
  );
}

export default RagDemoPage;
