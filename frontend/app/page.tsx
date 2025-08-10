"use client";
import React, { useState, useEffect, useRef } from 'react';
import { Send, FileText, AlertCircle, CheckCircle, Loader2, Scale, BookOpen, Gavel, RefreshCw } from 'lucide-react';

interface Message {
  id: string;
  type: 'user' | 'bot' | 'system';
  content: string;
  timestamp: Date;
  isLoading?: boolean;
}

interface DocumentStatus {
  url: string;
  status: 'pending' | 'processing' | 'ready' | 'error';
  error?: string;
}

const LegalChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [documentUrls, setDocumentUrls] = useState('');
  const [documentsStatus, setDocumentsStatus] = useState<DocumentStatus[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isDocumentsReady, setIsDocumentsReady] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // API Configuration
  const API_BASE_URL = 'http://localhost:8000'; // Update this to your API URL
  const API_TOKEN = 'legal_doc_analyzer_token_2024';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Initialize with welcome message
    const welcomeMessage: Message = {
      id: 'welcome',
      type: 'system',
      content: '🏛️ Welcome to the Indian Legal Document Analyzer! Upload your legal documents (PDF URLs) and ask me questions about Indian law, constitutional provisions, criminal law (IPC, CrPC), civil law, company law, and more. I use InLegalBERT-2 for specialized legal analysis.',
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
  }, []);

  const validateUrls = (urls: string): string[] => {
    const urlList = urls.split(',').map(url => url.trim()).filter(url => url);
    const validUrls: string[] = [];
    
    for (const url of urlList) {
      try {
        new URL(url);
        if (url.toLowerCase().includes('.pdf') || url.includes('blob:') || url.includes('drive.google.com')) {
          validUrls.push(url);
        }
      } catch {
        // Invalid URL
      }
    }
    
    return validUrls;
  };

  const processDocuments = async () => {
    const validUrls = validateUrls(documentUrls);
    
    if (validUrls.length === 0) {
      const errorMessage: Message = {
        id: Date.now().toString(),
        type: 'system',
        content: '❌ Please provide valid PDF URLs. URLs should be comma-separated and point to PDF documents.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
      return;
    }

    // Initialize document status
    const initialStatus: DocumentStatus[] = validUrls.map(url => ({
      url,
      status: 'pending'
    }));
    setDocumentsStatus(initialStatus);
    setIsLoading(true);

    const systemMessage: Message = {
      id: Date.now().toString(),
      type: 'system',
      content: `📄 Processing ${validUrls.length} legal document(s)... This may take a moment as I analyze and index the legal content using InLegalBERT-2.`,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, systemMessage]);

    try {
      // Update status to processing
      setDocumentsStatus(prev => prev.map(doc => ({ ...doc, status: 'processing' })));

      // Test document processing by making a simple question
      const response = await fetch(`${API_BASE_URL}/legal/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${API_TOKEN}`
        },
        body: JSON.stringify({
          documents: validUrls.join(','),
          questions: ['What is the main topic of this legal document?']
        })
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();

      // Update status to ready
      setDocumentsStatus(prev => prev.map(doc => ({ ...doc, status: 'ready' })));
      setIsDocumentsReady(true);

      const successMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'system',
        content: '✅ Legal documents processed successfully! You can now ask questions about constitutional law, criminal law (IPC/CrPC), civil law, company law, tax law, family law, and other legal topics. I\'ll analyze the documents using specialized legal AI.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, successMessage]);

    } catch (error) {
      console.error('Document processing error:', error);
      
      setDocumentsStatus(prev => prev.map(doc => ({ 
        ...doc, 
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error'
      })));

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'system',
        content: `❌ Failed to process legal documents: ${error instanceof Error ? error.message : 'Unknown error'}. Please check your URLs and try again.`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;
    
    if (!isDocumentsReady) {
      const warningMessage: Message = {
        id: Date.now().toString(),
        type: 'system',
        content: '⚠️ Please upload and process legal documents first before asking questions.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, warningMessage]);
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    };

    const botLoadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: 'bot',
      content: 'Analyzing legal documents and preparing response...',
      timestamp: new Date(),
      isLoading: true
    };

    setMessages(prev => [...prev, userMessage, botLoadingMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const validUrls = validateUrls(documentUrls);
      
      const response = await fetch(`${API_BASE_URL}/legal/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${API_TOKEN}`
        },
        body: JSON.stringify({
          documents: validUrls.join(','),
          questions: [inputMessage.trim()]
        })
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      const legalAnalysis = result.legal_analysis[0] || 'No analysis available.';

      // Replace loading message with actual response
      setMessages(prev => prev.map(msg => 
        msg.isLoading ? {
          ...msg,
          content: legalAnalysis,
          isLoading: false
        } : msg
      ));

    } catch (error) {
      console.error('Chat error:', error);
      
      // Replace loading message with error
      setMessages(prev => prev.map(msg => 
        msg.isLoading ? {
          ...msg,
          content: `❌ Error analyzing your legal query: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`,
          isLoading: false
        } : msg
      ));
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([{
      id: 'welcome',
      type: 'system',
      content: '🏛️ Welcome to the Indian Legal Document Analyzer! Upload your legal documents (PDF URLs) and ask me questions about Indian law, constitutional provisions, criminal law (IPC, CrPC), civil law, company law, and more. I use InLegalBERT-2 for specialized legal analysis.',
      timestamp: new Date()
    }]);
    setDocumentUrls('');
    setDocumentsStatus([]);
    setIsDocumentsReady(false);
  };

  const MessageIcon = ({ type }: { type: 'user' | 'bot' | 'system' }) => {
    switch (type) {
      case 'user':
        return <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center text-white font-medium">U</div>;
      case 'bot':
        return <div className="w-8 h-8 bg-purple-500 rounded-full flex items-center justify-center text-white"><Gavel className="w-4 h-4" /></div>;
      case 'system':
        return <div className="w-8 h-8 bg-gray-500 rounded-full flex items-center justify-center text-white"><Scale className="w-4 h-4" /></div>;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      <div className="max-w-4xl mx-auto p-4">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <Scale className="w-8 h-8 text-purple-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-800">Indian Legal Document Analyzer</h1>
              <p className="text-gray-600">AI-powered legal analysis with InLegalBERT-2</p>
            </div>
          </div>

          {/* Document Upload Section */}
          <div className="border-t pt-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              📄 Legal Document URLs (PDF, comma-separated):
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={documentUrls}
                onChange={(e) => setDocumentUrls(e.target.value)}
                placeholder="https://example.com/document1.pdf, https://example.com/document2.pdf"
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                disabled={isLoading}
              />
              <button
                onClick={processDocuments}
                disabled={isLoading || !documentUrls.trim()}
                className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileText className="w-4 h-4" />}
                Process
              </button>
            </div>

            {/* Document Status */}
            {documentsStatus.length > 0 && (
              <div className="mt-3 space-y-2">
                {documentsStatus.map((doc, index) => (
                  <div key={index} className="flex items-center gap-2 text-sm">
                    {doc.status === 'pending' && <Loader2 className="w-4 h-4 animate-spin text-yellow-500" />}
                    {doc.status === 'processing' && <Loader2 className="w-4 h-4 animate-spin text-blue-500" />}
                    {doc.status === 'ready' && <CheckCircle className="w-4 h-4 text-green-500" />}
                    {doc.status === 'error' && <AlertCircle className="w-4 h-4 text-red-500" />}
                    <span className={`truncate max-w-xs ${
                      doc.status === 'ready' ? 'text-green-700' : 
                      doc.status === 'error' ? 'text-red-700' : 'text-gray-600'
                    }`}>
                      {doc.url}
                    </span>
                    {doc.error && <span className="text-red-500 text-xs">({doc.error})</span>}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Legal Specializations */}
          <div className="mt-4 p-3 bg-purple-50 rounded-md">
            <div className="flex items-center gap-2 mb-2">
              <BookOpen className="w-4 h-4 text-purple-600" />
              <span className="text-sm font-medium text-purple-800">Legal Specializations:</span>
            </div>
            <div className="text-xs text-purple-700 flex flex-wrap gap-2">
              <span className="bg-purple-100 px-2 py-1 rounded">Constitutional Law</span>
              <span className="bg-purple-100 px-2 py-1 rounded">Criminal Law (IPC/CrPC)</span>
              <span className="bg-purple-100 px-2 py-1 rounded">Civil Law</span>
              <span className="bg-purple-100 px-2 py-1 rounded">Company Law</span>
              <span className="bg-purple-100 px-2 py-1 rounded">Tax Law</span>
              <span className="bg-purple-100 px-2 py-1 rounded">Family Law</span>
              <span className="bg-purple-100 px-2 py-1 rounded">Labour Law</span>
            </div>
          </div>
        </div>

        {/* Chat Interface */}
        <div className="bg-white rounded-lg shadow-lg flex flex-col h-96">
          {/* Chat Header */}
          <div className="px-4 py-3 border-b bg-gray-50 rounded-t-lg flex justify-between items-center">
            <h2 className="font-semibold text-gray-800">Legal Analysis Chat</h2>
            <button
              onClick={clearChat}
              className="text-gray-500 hover:text-gray-700 flex items-center gap-1 text-sm"
            >
              <RefreshCw className="w-4 h-4" />
              Clear
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
              >
                <MessageIcon type={message.type} />
                <div className={`flex-1 max-w-xs sm:max-w-md lg:max-w-lg xl:max-w-xl ${
                  message.type === 'user' ? 'text-right' : 'text-left'
                }`}>
                  <div className={`inline-block p-3 rounded-lg ${
                    message.type === 'user' 
                      ? 'bg-blue-500 text-white rounded-br-sm' 
                      : message.type === 'system'
                      ? 'bg-gray-100 text-gray-800 rounded-bl-sm'
                      : 'bg-purple-100 text-purple-900 rounded-bl-sm'
                  }`}>
                    {message.isLoading ? (
                      <div className="flex items-center gap-2">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Analyzing legal documents...</span>
                      </div>
                    ) : (
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    )}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t p-4">
            <div className="flex gap-2">
              <input
                ref={inputRef}
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={
                  isDocumentsReady 
                    ? "Ask about constitutional articles, IPC sections, legal provisions..." 
                    : "Upload legal documents first to start chatting..."
                }
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                disabled={isLoading || !isDocumentsReady}
              />
              <button
                onClick={sendMessage}
                disabled={isLoading || !inputMessage.trim() || !isDocumentsReady}
                className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
                Send
              </button>
            </div>

            {/* Sample Questions */}
            {isDocumentsReady && (
              <div className="mt-3">
                <p className="text-xs text-gray-600 mb-2">💡 Sample Legal Questions:</p>
                <div className="flex flex-wrap gap-2">
                  {[
                    "What are the fundamental rights under Article 21?",
                    "Explain the procedure for filing an FIR",
                    "What constitutes a valid contract?",
                    "Grounds for divorce under Hindu Marriage Act"
                  ].map((question, index) => (
                    <button
                      key={index}
                      onClick={() => setInputMessage(question)}
                      className="text-xs px-2 py-1 bg-purple-50 text-purple-700 rounded hover:bg-purple-100 transition-colors"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-6 text-center text-xs text-gray-500">
          <p>⚖️ Indian Legal Document Analyzer v1.0.0 | Powered by InLegalBERT-2 & Hybrid Search</p>
          <p className="mt-1">🔒 Secure • 🚀 Fast • 🎯 Accurate Legal Analysis</p>
        </div>
      </div>
    </div>
  );
};

export default LegalChatInterface;