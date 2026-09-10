import { useState, useEffect, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  Cpu,
  Activity,
  MessageSquare,
  Plus,
  Send,
  Bot,
  Tag,
  ShieldCheck,
  ShieldAlert,
  History,
} from 'lucide-react'
import './App.css'

function App() {
  const [conversations, setConversations] = useState([])
  const [activeConvId, setActiveConvId] = useState(null)
  const [messages, setMessages] = useState([])
  const [inputText, setInputText] = useState('')
  const [sending, setSending] = useState(false)
  const [analysis, setAnalysis] = useState(null)
  const [stats, setStats] = useState(null)
  const [backendOnline, setBackendOnline] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Health check
  useEffect(() => {
    fetch('/api/health')
      .then(() => setBackendOnline(true))
      .catch(() => setBackendOnline(false))
  }, [])

  // Load conversations
  const loadConversations = useCallback(async () => {
    try {
      const res = await fetch('/api/conversations')
      const data = await res.json()
      setConversations(data)
    } catch { /* offline */ }
  }, [])

  // Load stats
  const loadStats = useCallback(async () => {
    try {
      const res = await fetch('/api/stats')
      const data = await res.json()
      setStats(data)
    } catch { /* offline */ }
  }, [])

  useEffect(() => {
    loadConversations()
    loadStats()
  }, [loadConversations, loadStats])

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, sending])

  // Load a conversation
  const selectConversation = async (id) => {
    const safeId = parseInt(id, 10)
    if (!Number.isInteger(safeId) || safeId <= 0) return
    setActiveConvId(safeId)
    setAnalysis(null)
    try {
      const res = await fetch(`/api/conversations/${safeId}`)
      const data = await res.json()
      setMessages(data.messages || [])
      // Set analysis from last customer message with a decision
      const lastCustomer = [...(data.messages || [])].reverse().find(
        m => m.sender === 'customer' && m.decision
      )
      if (lastCustomer) {
        setAnalysis({
          intent: lastCustomer.intent,
          confidence: lastCustomer.confidence,
          escalate: lastCustomer.decision.decision === 'escalate',
          reason: lastCustomer.decision.escalation_reason,
          historical_evidence: [],
        })
      }
    } catch { /* offline */ }
  }

  // New conversation
  const handleNewConversation = async () => {
    try {
      const res = await fetch('/api/conversations', { method: 'POST' })
      const conv = await res.json()
      setActiveConvId(conv.id)
      setMessages([])
      setAnalysis(null)
      await loadConversations()
      inputRef.current?.focus()
    } catch { /* offline */ }
  }

  // Send message
  const handleSend = async (e) => {
    e.preventDefault()
    const text = inputText.trim()
    if (!text || !activeConvId || sending) return

    setInputText('')
    setSending(true)

    // Optimistically add customer message
    const tempMsg = {
      id: `temp-${Date.now()}`,
      sender: 'customer',
      message: text,
      timestamp: new Date().toISOString(),
    }
    setMessages(prev => [...prev, tempMsg])

    try {
      const res = await fetch(`/api/conversations/${activeConvId}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      })
      const result = await res.json()
      // Replace temp with real messages
      setMessages(prev => [
        ...prev.filter(m => m.id !== tempMsg.id),
        result.customer_message,
        result.agent_response,
      ])
      setAnalysis(result.analysis)
      await loadConversations()
      await loadStats()
    } catch (err) {
      // Remove temp on error
      setMessages(prev => prev.filter(m => m.id !== tempMsg.id))
      setInputText(text)
    } finally {
      setSending(false)
    }
  }

  // Handle Enter key
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend(e)
    }
  }

  const formatTime = (ts) => {
    if (!ts) return ''
    const d = new Date(ts)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  const getConfidenceColor = (c) => {
    if (c >= 0.8) return 'var(--green)'
    if (c >= 0.6) return 'var(--orange)'
    return 'var(--red)'
  }

  return (
    <div className="app-layout">
      {/* Header */}
      <header className="header">
        <div className="header-brand">
          <div className="logo">
            <Cpu size={24} />
          </div>
          <h1>AppleSupport AI Agent</h1>
        </div>
        <div className="header-status">
          <span className="status-dot" style={{
            background: backendOnline ? 'var(--green)' : 'var(--red)',
            animation: backendOnline ? 'pulse 2s infinite' : 'none',
          }} />
          <Activity size={16} />
          {backendOnline ? 'Agent Online' : 'Backend Offline'}
        </div>
      </header>

      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2>
            <MessageSquare size={14} />
            Conversations
          </h2>
          <button
            className="new-conversation-btn"
            onClick={handleNewConversation}
            disabled={!backendOnline}
          >
            <Plus size={18} />
            New Conversation
          </button>
        </div>
        <div className="conversation-list">
          {conversations.length === 0 && (
            <div className="analysis-empty">
              No conversations yet.<br />Start one above.
            </div>
          )}
          {conversations.map(c => (
            <div
              key={c.id}
              className={`conversation-item ${c.id === activeConvId ? 'active' : ''}`}
              onClick={() => selectConversation(c.id)}
            >
              <div className="conversation-item-header">
                <span className="conversation-item-id">#{c.id}</span>
                <span className="conversation-item-time">
                  {formatTime(c.created_at)}
                </span>
              </div>
              <div className="conversation-item-meta">
                <span className={`badge badge-${c.status}`}>
                  {c.status}
                </span>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  {c.message_count} msgs
                </span>
              </div>
            </div>
          ))}
        </div>
      </aside>

      {/* Main chat area */}
      <main className="chat-area">
        {/* Stats bar */}
        {stats && stats.total_conversations > 0 && (
          <div className="stats-row">
            <div className="stat-card">
              <div className="stat-value">{stats.total_conversations}</div>
              <div className="stat-label">Total</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{stats.active}</div>
              <div className="stat-label">Active</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{stats.escalated}</div>
              <div className="stat-label">Escalated</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {(stats.auto_handle_rate * 100).toFixed(0)}%
              </div>
              <div className="stat-label">Auto-Handle</div>
            </div>
          </div>
        )}

        {/* Messages */}
        <div className="chat-messages">
          {!activeConvId ? (
            <div className="chat-empty">
              <div className="chat-empty-icon">
                <Cpu size={64} />
              </div>
              <h3>Welcome to AppleSupport AI</h3>
              <p>
                Start a new conversation to see the AI agent in action. It classifies intent,
                retrieves historical evidence, makes escalation decisions, and generates
                contextual responses using Qwen 2.5.
              </p>
            </div>
          ) : messages.length === 0 && !sending ? (
            <div className="chat-empty">
              <div className="chat-empty-icon">
                <MessageSquare size={64} />
              </div>
              <h3>Conversation #{activeConvId}</h3>
              <p>Type a customer support message below to analyze.</p>
            </div>
          ) : (
            <>
              {messages.map(msg => (
                <div key={msg.id} className={`message message-${msg.sender}`}>
                  {msg.sender === 'agent' && (
                    <div className="message-agent-avatar">
                      <Bot size={18} />
                    </div>
                  )}
                  <div className="message-agent-content">
                    <div className="message-sender">
                      {msg.sender === 'customer' ? 'Customer' : 'AI Agent'}
                    </div>
                    <div className="message-text">
                      {msg.sender === 'agent'
                        ? <ReactMarkdown>{msg.message}</ReactMarkdown>
                        : msg.message
                      }
                    </div>
                    <div className="message-time">{formatTime(msg.timestamp)}</div>
                  </div>
                </div>
              ))}
              {sending && (
                <div className="typing-indicator">
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                  <div className="typing-dot" />
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input */}
        {activeConvId && (
          <div className="chat-input-container">
            <form className="chat-input-form" onSubmit={handleSend}>
              <textarea
                ref={inputRef}
                className="chat-input"
                placeholder="Type a customer message... (Enter to send)"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                disabled={sending || !backendOnline}
                rows={1}
              />
              <button
                type="submit"
                className="send-btn"
                disabled={!inputText.trim() || sending || !backendOnline}
              >
                <Send size={18} />
                {sending ? 'Sending...' : 'Send'}
              </button>
            </form>
          </div>
        )}
      </main>

      {/* Analysis panel */}
      <aside className="analysis-panel">
        <div className="analysis-panel-header">
          <h2>Agent Analysis</h2>
        </div>

        {!analysis ? (
          <div className="analysis-empty">
            Send a customer message to see the AI agent's analysis — intent
            classification, escalation decision, and historical evidence.
          </div>
        ) : (
          <>
            {/* Intent */}
            <div className="analysis-section">
              <h3>
                <Tag size={14} />
                Detected Intent
              </h3>
              <div className="intent-card">
                <div className="intent-name">
                  {analysis.intent?.replace(/_/g, ' ')}
                </div>
                <div className="confidence-bar-container">
                  <div className="confidence-bar">
                    <div
                      className="confidence-bar-fill"
                      style={{
                        width: `${(analysis.confidence * 100).toFixed(0)}%`,
                        background: getConfidenceColor(analysis.confidence),
                      }}
                    />
                  </div>
                  <span
                    className="confidence-value"
                    style={{ color: getConfidenceColor(analysis.confidence) }}
                  >
                    {(analysis.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Escalation Decision */}
            <div className="analysis-section">
              <h3>Escalation Decision</h3>
              <div className={`decision-card ${analysis.escalate ? 'decision-escalate' : 'decision-auto'}`}>
                <div className="decision-icon">
                  {analysis.escalate ? <ShieldAlert size={24} /> : <ShieldCheck size={24} />}
                </div>
                <div className="decision-details">
                  <div className="decision-label">
                    {analysis.escalate ? 'ESCALATE TO HUMAN' : 'AUTO-HANDLE'}
                  </div>
                  <div className="decision-reason">{analysis.reason}</div>
                </div>
              </div>
            </div>

            {/* Historical Evidence */}
            {analysis.historical_evidence?.length > 0 && (
              <div className="analysis-section">
                <h3>
                  <History size={14} />
                  Historical Evidence
                </h3>
                {analysis.historical_evidence.map((ev, i) => (
                  <div key={i} className="evidence-item">
                    <div className="evidence-header">
                      <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
                        Match #{i + 1}
                      </span>
                      <span className="evidence-similarity">
                        {(ev.similarity * 100).toFixed(1)}% similar
                      </span>
                    </div>
                    <div className="evidence-customer">
                      "{ev.customer?.slice(0, 120)}{ev.customer?.length > 120 ? '…' : ''}"
                    </div>
                    <div className="evidence-response">
                      {ev.response?.slice(0, 200)}{ev.response?.length > 200 ? '…' : ''}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </aside>
    </div>
  )
}

export default App
