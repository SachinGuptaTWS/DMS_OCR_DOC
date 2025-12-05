import React, { useState } from 'react';

const TestConnection = () => {
  const [status, setStatus] = useState('idle');
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  const testBackendConnection = async () => {
    setStatus('loading');
    setError(null);
    try {
      const res = await fetch('/api/health', {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
      });
      
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      
      const data = await res.json();
      setResponse(data);
      setStatus('success');
    } catch (err) {
      console.error('Connection test failed:', err);
      setError(err.message);
      setStatus('error');
    }
  };

  return (
    <div className="p-4 border rounded-lg shadow mb-4">
      <h2 className="text-lg font-semibold mb-3">Backend Connection Test</h2>
      <div className="flex items-center gap-4">
        <button
          onClick={testBackendConnection}
          disabled={status === 'loading'}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400 transition-colors"
        >
          {status === 'loading' ? 'Testing...' : 'Test Connection'}
        </button>
        
        <div className="flex items-center">
          {status === 'success' && (
            <span className="text-green-600 flex items-center">
              <svg className="w-5 h-5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              Connected successfully!
            </span>
          )}
          {status === 'error' && (
            <span className="text-red-600 flex items-center">
              <svg className="w-5 h-5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
              Connection failed
            </span>
          )}
        </div>
      </div>
      
      {(status === 'success' || status === 'error') && (
        <div className={`mt-3 p-3 rounded ${status === 'success' ? 'bg-green-50' : 'bg-red-50'}`}>
          {status === 'success' ? (
            <div>
              <p className="font-medium text-green-800">✓ Backend is responding!</p>
              <pre className="mt-2 p-2 bg-white text-xs text-green-700 rounded overflow-auto">
                {JSON.stringify(response, null, 2)}
              </pre>
            </div>
          ) : (
            <div>
              <p className="font-medium text-red-800">✗ Could not connect to backend</p>
              <p className="mt-1 text-sm text-red-700">{error}</p>
              <div className="mt-2 text-sm text-red-600">
                <p className="font-medium">Troubleshooting steps:</p>
                <ul className="list-disc pl-5 mt-1 space-y-1">
                  <li>Check if the backend server is running</li>
                  <li>Verify the backend URL is correct</li>
                  <li>Check browser console for detailed error messages</li>
                  <li>Ensure CORS is properly configured on the backend</li>
                </ul>
              </div>
            </div>
          )}
        </div>
      )}
      
      <div className="mt-4 pt-3 border-t border-gray-200">
        <h3 className="text-sm font-medium text-gray-700 mb-2">API Endpoints:</h3>
        <ul className="space-y-1 text-sm">
          <li className="flex items-center">
            <code className="bg-gray-100 px-2 py-1 rounded text-xs font-mono">GET /api/health</code>
            <span className="ml-2 text-gray-500">- Health check endpoint</span>
          </li>
          <li className="flex items-center">
            <code className="bg-gray-100 px-2 py-1 rounded text-xs font-mono">POST /api/process</code>
            <span className="ml-2 text-gray-500">- Process documents</span>
          </li>
          <li className="flex items-center">
            <code className="bg-gray-100 px-2 py-1 rounded text-xs font-mono">GET /docs</code>
            <span className="ml-2 text-gray-500">- API documentation</span>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default TestConnection;
