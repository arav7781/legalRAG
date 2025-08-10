"use client";
import React, { useState, useEffect, useRef } from 'react';
import { Send, FileText, AlertCircle, CheckCircle, Loader2, Scale, BookOpen, Gavel, RefreshCw, X, Copy, ThumbsUp, ThumbsDown } from 'lucide-react';

interface Message {
  id: string;
  type: 'user' | 'bot' | 'system';
  content: string;
  timestamp: Date;
  isLoading?: boolean;
  isError?: boolean;
}

interface DocumentStatus {
  url: string;
  status: 'ready' | 'error';
  error?: string;
}

const LegalChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [documentUrls, setDocumentUrls] = useState('');
  const [documentsStatus, setDocumentsStatus] = useState<DocumentStatus[]>([]);
  const [isDocumentsReady, setIsDocumentsReady] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [currentLoadingMessageId, setCurrentLoadingMessageId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // API Configuration
  const API_BASE_URL = 'http://localhost:8000';
  const API_TOKEN = 'legal_doc_analyzer_token_2024';

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Initialize with welcome message
    const welcomeMessage: Message = {
      id: 'welcome',
      type: 'system',
      content: '👋 Welcome to the Indian Legal Document Analyzer!\n\n🔸 First, provide your legal document URLs (PDF format)\n🔸 Then start asking questions about Indian law\n🔸 I specialize in Constitutional, Criminal, Civil, Company, Tax, Family, and Labour law\n\n✨ Powered by InLegalBERT-2 for accurate legal analysis',
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
  }, []);

  const validateUrls = (urls: string): string[] => {
    const urlList = urls.split(',').map(url => url.trim()).filter(url => url);
    const validUrls: string[] = [];
    
    for (const url of urlList) {
      try {
        const urlObj = new URL(url);
        if (url.toLowerCase().includes('.pdf') || 
            url.includes('blob:') || 
            url.includes('drive.google.com') ||
            url.includes('api.sci.gov.in') ||
            urlObj.hostname.includes('gov.in')) {
          validUrls.push(url);
        }
      } catch {
        // Invalid URL - skip
      }
    }
    
    return validUrls;
  };

  const setupDocuments = () => {
    const validUrls = validateUrls(documentUrls);
    
    if (validUrls.length === 0) {
      const errorMessage: Message = {
        id: Date.now().toString(),
        type: 'system',
        content: '❌ Please provide valid PDF URLs. Examples:\n• https://example.com/document.pdf\n• https://api.sci.gov.in/judgements/...\n\nURLs should be comma-separated if multiple documents.',
        timestamp: new Date(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
      return;
    }

    // Mark documents as ready without processing
    const documentStatus: DocumentStatus[] = validUrls.map(url => ({
      url,
      status: 'ready'
    }));
    
    setDocumentsStatus(documentStatus);
    setIsDocumentsReady(true);

    const successMessage: Message = {
      id: Date.now().toString(),
      type: 'system',
      content: `✅ ${validUrls.length} document(s) loaded successfully!\n\n🚀 You can now ask questions about:\n• Constitutional provisions & articles\n• Criminal law (IPC, CrPC, Evidence Act)\n• Civil procedures & contracts\n• Company law & regulations\n• Tax law & compliance\n• Family & matrimonial law\n• Labour & employment law\n\n💬 Start typing your legal question below!`,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, successMessage]);
    
    // Clear the URL input
    setDocumentUrls('');
    
    // Focus on chat input
    setTimeout(() => {
      inputRef.current?.focus();
    }, 500);
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;
    
    if (!isDocumentsReady) {
      const warningMessage: Message = {
        id: Date.now().toString(),
        type: 'system',
        content: '⚠️ Please add legal documents first using the document input above.',
        timestamp: new Date(),
        isError: true
      };
      setMessages(prev => [...prev, warningMessage]);
      return;
    }

    const userQuestion = inputMessage.trim();
    setInputMessage('');

    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: userQuestion,
      timestamp: new Date()
    };

    // Add typing indicator
    const loadingId = (Date.now() + 1).toString();
    const botLoadingMessage: Message = {
      id: loadingId,
      type: 'bot',
      content: '',
      timestamp: new Date(),
      isLoading: true
    };

    setMessages(prev => [...prev, userMessage, botLoadingMessage]);
    setIsTyping(true);
    setCurrentLoadingMessageId(loadingId);

    try {
      const validUrls = documentsStatus.map(doc => doc.url);
      
      const response = await fetch(`${API_BASE_URL}/legal/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${API_TOKEN}`
        },
        body: JSON.stringify({
          documents: validUrls.join(','),
          questions: [userQuestion]
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`${response.status}: ${errorText}`);
      }

      const result = await response.json();
      const legalAnalysis = result.legal_analysis?.[0] || 'No analysis available for this question.';

      // Replace loading message with actual response
      setMessages(prev => prev.map(msg => 
        msg.id === loadingId ? {
          ...msg,
          content: legalAnalysis,
          isLoading: false
        } : msg
      ));

    } catch (error) {
      console.error('Chat error:', error);
      
      const errorContent = `❌ I encountered an error analyzing your legal query.\n\nError: ${error instanceof Error ? error.message : 'Unknown error'}\n\n💡 Please try rephrasing your question or check the document URLs.`;
      
      // Replace loading message with error
      setMessages(prev => prev.map(msg => 
        msg.id === loadingId ? {
          ...msg,
          content: errorContent,
          isLoading: false,
          isError: true
        } : msg
      ));
    } finally {
      setIsTyping(false);
      setCurrentLoadingMessageId(null);
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
      id: 'welcome-new',
      type: 'system',
      content: '👋 Chat cleared! Add your legal documents above to start a new conversation.',
      timestamp: new Date()
    }]);
    setDocumentUrls('');
    setDocumentsStatus([]);
    setIsDocumentsReady(false);
    setIsTyping(false);
    setCurrentLoadingMessageId(null);
  };

  const removeDocument = (index: number) => {
    const newStatus = documentsStatus.filter((_, i) => i !== index);
    setDocumentsStatus(newStatus);
    
    if (newStatus.length === 0) {
      setIsDocumentsReady(false);
      const noDocsMessage: Message = {
        id: Date.now().toString(),
        type: 'system',
        content: '📄 All documents removed. Please add legal documents to continue chatting.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, noDocsMessage]);
    }
  };

  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content);
    // You could add a toast notification here
  };

  const MessageIcon = ({ type }: { type: 'user' | 'bot' | 'system' }) => {
    switch (type) {
      case 'user':
        return (
          <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-blue-600 rounded-full flex items-center justify-center text-white font-semibold shadow-sm">
            <span className="text-sm">U</span>
          </div>
        );
      case 'bot':
        return (
          <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-purple-600 rounded-full flex items-center justify-center text-white shadow-sm">
            <Gavel className="w-4 h-4" />
          </div>
        );
      case 'system':
        return (
          <div className="w-8 h-8 bg-gradient-to-r from-gray-500 to-gray-600 rounded-full flex items-center justify-center text-white shadow-sm">
            <Scale className="w-4 h-4" />
          </div>
        );
    }
  };

  const TypingIndicator = () => (
    <div className="flex items-center gap-1 text-purple-600">
      <div className="flex gap-1">
        <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
        <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
        <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
      </div>
      <span className="text-sm text-purple-600 ml-2">Analyzing legal documents...</span>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="max-w-5xl mx-auto p-4">
        {/* Enhanced Header */}
        <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6 mb-6">
          <div className="flex items-center gap-4 mb-6">
            <div className="w-12 h-12 bg-gradient-to-r from-purple-600 to-blue-600 rounded-xl flex items-center justify-center">
              <Scale className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                Legal Document Analyzer
              </h1>
              <p className="text-gray-600 mt-1">AI-powered Indian legal analysis with InLegalBERT-2</p>
            </div>
          </div>

          {/* Document Upload Section */}
          {!isDocumentsReady ? (
            <div className="border-2 border-dashed border-purple-200 rounded-lg p-6 bg-purple-50">
              <div className="text-center mb-4">
                <FileText className="w-12 h-12 text-purple-500 mx-auto mb-3" />
                <h3 className="text-lg font-semibold text-gray-800 mb-2">Add Legal Documents</h3>
                <p className="text-gray-600 text-sm">Upload PDF URLs to start legal analysis</p>
              </div>
              
              <div className="flex gap-3">
                <input
                  type="text"
                  value={documentUrls}
                  onChange={(e) => setDocumentUrls(e.target.value)}
                  placeholder="https://example.com/legal-document.pdf, https://api.sci.gov.in/judgement.pdf"
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent bg-white text-black"
                  onKeyPress={(e) => e.key === 'Enter' && setupDocuments()}
                />
                <button
                  onClick={setupDocuments}
                  disabled={!documentUrls.trim()}
                  className="px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:from-purple-700 hover:to-blue-700 disabled:from-gray-400 disabled:to-gray-400 disabled:cursor-not-allowed flex items-center gap-2 font-medium transition-all duration-200 shadow-md hover:shadow-lg"
                >
                  <FileText className="w-4 h-4" />
                  Add Documents
                </button>
              </div>
              
              <div className="mt-4 text-xs text-gray-500">
                💡 <strong>Tip:</strong> You can add multiple PDF URLs separated by commas
              </div>
            </div>
          ) : (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="font-medium text-green-800">Documents Ready ({documentsStatus.length})</span>
                </div>
                <button
                  onClick={clearChat}
                  className="text-gray-500 hover:text-gray-700 flex items-center gap-1 text-sm px-3 py-1 rounded-md hover:bg-gray-100 transition-colors"
                >
                  <RefreshCw className="w-4 h-4" />
                  Reset
                </button>
              </div>
              
              <div className="space-y-2">
                {documentsStatus.map((doc, index) => (
                  <div key={index} className="flex items-center justify-between bg-white rounded-md p-2 border">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />
                      <span className="text-sm text-gray-700 truncate" title={doc.url}>
                        {doc.url}
                      </span>
                    </div>
                    <button
                      onClick={() => removeDocument(index)}
                      className="text-gray-400 hover:text-red-500 p-1 rounded transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Legal Specializations - only show when documents are ready */}
          {isDocumentsReady && (
            <div className="mt-4 p-4 bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg border border-purple-100">
              <div className="flex items-center gap-2 mb-3">
                <BookOpen className="w-5 h-5 text-purple-600" />
                <span className="font-semibold text-purple-800">Legal Specializations Available</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {[
                  'Constitutional Law',
                  'Criminal Law (IPC/CrPC)',
                  'Civil Procedures',
                  'Company Law',
                  'Tax Law',
                  'Family Law',
                  'Labour Law',
                  'Property Law'
                ].map((area, index) => (
                  <span 
                    key={index}
                    className="bg-white text-purple-700 px-3 py-1 rounded-full text-xs font-medium border border-purple-200"
                  >
                    {area}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Enhanced Chat Interface */}
        <div className="bg-white rounded-xl shadow-lg border border-gray-100 flex flex-col" style={{ height: '600px' }}>
          {/* Chat Header */}
          <div className="px-6 py-4 border-b bg-gradient-to-r from-purple-50 to-blue-50 rounded-t-xl">
            <div className="flex justify-between items-center">
              <div>
                <h2 className="font-semibold text-gray-800 flex items-center gap-2">
                  <Gavel className="w-5 h-5 text-purple-600" />
                  Legal Analysis Chat
                </h2>
                <p className="text-xs text-gray-500 mt-1">
                  {isDocumentsReady 
                    ? `Ready to analyze ${documentsStatus.length} document(s)` 
                    : 'Add documents above to start chatting'
                  }
                </p>
              </div>
              {messages.length > 1 && (
                <button
                  onClick={clearChat}
                  className="text-gray-500 hover:text-gray-700 flex items-center gap-2 text-sm px-3 py-2 rounded-lg hover:bg-white/50 transition-all duration-200"
                >
                  <RefreshCw className="w-4 h-4" />
                  Clear Chat
                </button>
              )}
            </div>
          </div>

          {/* Messages */}
          <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-6 space-y-6 bg-gray-50/30">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-4 ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
              >
                <div className="flex-shrink-0">
                  <MessageIcon type={message.type} />
                </div>
                
                <div className={`flex-1 max-w-3xl ${message.type === 'user' ? 'text-right' : 'text-left'}`}>
                  <div className={`inline-block p-4 rounded-2xl shadow-sm ${
                    message.type === 'user' 
                      ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-br-md' 
                      : message.type === 'system'
                      ? `rounded-bl-md border ${message.isError ? 'bg-red-50 border-red-200 text-red-800' : 'bg-gray-50 border-gray-200 text-gray-800'}`
                      : `bg-white border border-gray-200 text-gray-800 rounded-bl-md ${message.isError ? 'border-red-200 bg-red-50' : ''}`
                  }`}>
                    {message.isLoading ? (
                      <TypingIndicator />
                    ) : (
                      <div>
                        <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                        
                        {/* Message actions for bot messages */}
                        {message.type === 'bot' && !message.isLoading && !message.isError && (
                          <div className="flex items-center gap-2 mt-3 pt-3 border-t border-gray-100">
                            <button
                              onClick={() => copyMessage(message.content)}
                              className="text-gray-400 hover:text-gray-600 p-1 rounded transition-colors"
                              title="Copy response"
                            >
                              <Copy className="w-4 h-4" />
                            </button>
                            <button
                              className="text-gray-400 hover:text-green-600 p-1 rounded transition-colors"
                              title="Good response"
                            >
                              <ThumbsUp className="w-4 h-4" />
                            </button>
                            <button
                              className="text-gray-400 hover:text-red-600 p-1 rounded transition-colors"
                              title="Poor response"
                            >
                              <ThumbsDown className="w-4 h-4" />
                            </button>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  
                  <div className="text-xs text-gray-400 mt-2 px-1">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Enhanced Input */}
          <div className="border-t bg-white p-6 rounded-b-xl">
            <div className="flex gap-3 items-end">
              <div className="flex-1">
                <textarea
                  // ref={inputRef}
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder={
                    isDocumentsReady 
                      ? "Ask about constitutional articles, IPC sections, legal precedents, or any legal question..." 
                      : "Add legal documents first to start asking questions..."
                  }
                  className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none bg-gray-50 focus:bg-white transition-all duration-200 text-black"
                  disabled={!isDocumentsReady || isTyping}
                  rows={1}
                  style={{ maxHeight: '120px', minHeight: '48px' }}
                />
              </div>
              <button
                onClick={sendMessage}
                disabled={!isDocumentsReady || !inputMessage.trim() || isTyping}
                className="px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl hover:from-purple-700 hover:to-blue-700 disabled:from-gray-300 disabled:to-gray-300 disabled:cursor-not-allowed flex items-center gap-2 font-medium transition-all duration-200 shadow-md hover:shadow-lg"
              >
                {isTyping ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
                Send
              </button>
            </div>

            {/* Sample Questions */}
            {isDocumentsReady && !isTyping && (
              <div className="mt-4">
                <p className="text-xs text-gray-500 mb-3 font-medium">💡 Sample Legal Questions:</p>
                <div className="flex flex-wrap gap-2">
                  {[
                    "What are the key constitutional provisions in this case?",
                    "Explain the legal precedents mentioned",
                    "What are the grounds for this legal decision?",
                    "Summarize the court's reasoning",
                    "What remedies are available to the parties?"
                  ].map((question, index) => (
                    <button
                      key={index}
                      onClick={() => setInputMessage(question)}
                      className="text-xs px-3 py-2 bg-purple-50 text-purple-700 rounded-lg hover:bg-purple-100 transition-colors border border-purple-100 font-medium"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Enhanced Footer */}
        <div className="mt-6 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-full shadow-sm border border-gray-100">
            <Scale className="w-4 h-4 text-purple-600" />
            <span className="text-sm text-gray-600">
              Powered by <strong>InLegalBERT-2</strong> • Hybrid Vector Search • Indian Legal AI
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LegalChatInterface;